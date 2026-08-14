from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from typing import Union, List
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from UltraSeg.core.augment import data_augment_pipeline

ROI = {
    "ymin": 70,
    "ymax": 920,
    "xmin": 16,
    "xmax": 1052
}


class Ultrasound_Dataset(Dataset):
    def __init__(self,
                 data_root: str,
                 img_dir: str,
                 mask_dir: str,
                 img_suffix: Union[str, List[str]] = ['.jpg', '.png', '.bmp', '.tif', '.tiff'],
                 mask_suffix: Union[str, List[str]] = ['.png'],
                 input_size: list = [512, 512],
                 mean: list = [0.485, 0.456, 0.406],
                 std: list = [0.229, 0.224, 0.225],
                 classes: list = None,
                 color_map: list = None,
                 num_classes: int = None,
                 mode: str = 'train',
                 augment_version: int = 2,
                 use_roi: bool = False,
                 rare_classes: list = None,
                 hard_samp: bool = False):
        super().__init__()
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        assert mode in ['train', 'val', 'test', 'all'], f"mode should be one of ['train', 'val', 'test', 'all'], but got {mode}"
        self.mode = mode

        def _glob_with_suffixes(folder: Path, suffixes: Union[str, List[str]]) -> List[Path]:
            """根据后缀（字符串或后缀列表）在 folder 中收集匹配的文件路径。"""
            if isinstance(suffixes, str):
                suffixes = [suffixes]
            paths: set = set()
            for suffix in suffixes:
                if not suffix.startswith('.'):
                    suffix = '.' + suffix
                paths.update(folder.glob(f'*{suffix}'))
            return sorted(paths)

        img_folder = Path(data_root) / img_dir / mode
        mask_folder = Path(data_root) / mask_dir / mode
        self.img_paths = _glob_with_suffixes(img_folder, img_suffix)
        self.mask_paths = _glob_with_suffixes(mask_folder, mask_suffix)
        self.mean = list(mean)
        self.std = list(std)
        assert len(self.img_paths) == len(self.mask_paths), "Number of images and masks should be the same !"

        if self.mode != 'train':
            self.hard_samp = False

        if num_classes is not None:
            self.num_classes = num_classes

        if classes is not None:
            self.classes = classes

        if color_map is not None:
            self.color_map = color_map

        self.use_roi = use_roi
        self.rare_classes = rare_classes if rare_classes is not None else None
        self.hard_samp = hard_samp if self.mode == 'train' else False

        self.compute_class_pixel_frequency()
        self.compute_class_rarity_weights(ignore_index=-1)
        self.compute_sample_weights(ignore_index=-1)

        train_pipeline, val_pipeline = data_augment_pipeline(input_size=input_size,
                                                             mean=mean,
                                                             std=std,
                                                             version=augment_version,
                                                             rare_classes=self.rare_classes)
        self.augment_pipeline = train_pipeline if mode == 'train' else val_pipeline

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = cv.imread(str(img_path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        if self.use_roi:
            img = img[ROI["ymin"]:ROI["ymax"], ROI["xmin"]:ROI["xmax"]]
            mask = mask[ROI["ymin"]:ROI["ymax"], ROI["xmin"]:ROI["xmax"]]

        # 数据增强
        augmented = self.augment_pipeline(image=img, mask=mask)

        image = augmented['image']
        mask = augmented['mask']
        # mask = mask.unsqueeze(0)

        return (image, mask)

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def compute_class_pixel_frequency(self):
        """
        统计整个训练集每个类别的像素频率。
        """
        class_counter = np.zeros(self.num_classes, dtype=np.int64)

        for mask_path in self.mask_paths:
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {mask_path}")

            binc = np.bincount(mask.reshape(-1), minlength=self.num_classes)
            class_counter += binc

        total_pixels = class_counter.sum()
        class_freq = class_counter / (total_pixels + 1e-12)

        self.class_freq = class_freq
        self.class_counter = class_counter

    def compute_class_rarity_weights(self, ignore_index=None):
        """
        根据全局类别频率，计算类别稀有性权重。
        这里用温和形式：1 / log(1.1 + freq)

        class_freq: np.ndarray, shape [num_classes]
        """
        rarity = 1.0 / np.log(1.1 + self.class_freq + 1e-12)

        if ignore_index is not None:
            rarity[ignore_index] = 0.0

        # 可选：归一化到均值为1附近
        valid = rarity > 0
        rarity[valid] = rarity[valid] / rarity[valid].mean()

        self.class_rarity = rarity

    def compute_sample_weights(self,
                               ignore_index=0,
                               alpha=2.0):

        if self.rare_classes is None:
            fg_freq = self.class_freq.copy()
            fg_freq[ignore_index] = np.inf
            threshold = np.median(fg_freq[np.isfinite(fg_freq)])
            rare_classes = set(np.where(self.class_freq < threshold)[0].tolist())
            rare_classes.discard(ignore_index)
        else:
            rare_classes = set(self.rare_classes)

        self.rare_classes = rare_classes
        sample_weights = []

        for mask_path in self.mask_paths:
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {mask_path}")

            classes_in_mask = set(np.unique(mask).tolist())
            weight = 1.0

            for c in classes_in_mask:
                if c in rare_classes:
                    weight += alpha * self.class_rarity[c]

            sample_weights.append(float(weight))

        sample_weights = np.array(sample_weights, dtype=np.float64)

        # 归一化，避免数值跨度过大
        sample_weights = sample_weights / sample_weights.mean()

        self.sample_weights = sample_weights.tolist()


def create_dataset(dataset_cfg,
                   input_size=None,
                   mean=None,
                   std=None,
                   split='train',
                   rare_classes=None,
                   hard_samp=False,
                   use_roi: bool = False,
                   augment_version: int = 2):

    dataset = Ultrasound_Dataset(data_root=dataset_cfg['data_root'],
                                 img_dir=dataset_cfg['img_dir'],
                                 mask_dir=dataset_cfg['mask_dir'],
                                 img_suffix=dataset_cfg['img_suffix'],
                                 mask_suffix=dataset_cfg['mask_suffix'],
                                 input_size=[input_size, input_size] if input_size is not None else dataset_cfg['input_size'],
                                 mean=mean,
                                 std=std,
                                 num_classes=dataset_cfg['num_classes'],
                                 classes=dataset_cfg['classes'],
                                 color_map=dataset_cfg['color_map'],
                                 mode=split,
                                 augment_version=augment_version,
                                 rare_classes=rare_classes,
                                 hard_samp=hard_samp,
                                 use_roi=use_roi)
    return dataset
