from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
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


class Wrist_Ultrasound_Dataset(Dataset):
    def __init__(self,
                 data_root: str,
                 img_dir: str,
                 mask_dir: str,
                 img_suffix: str = '.jpg',
                 mask_suffix: str = '.png',
                 input_size: list = [512, 512],
                 mean: list = [0.485, 0.456, 0.406],
                 std: list = [0.229, 0.224, 0.225],
                 classes: list = None,
                 color_map: list = None,
                 mode: str = 'train',
                 use_cutmix: bool = False,
                 use_roi: bool = False):
        super().__init__()
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        assert mode in ['train', 'val', 'test'], f"mode should be one of ['train', 'val', 'test'], but got {mode}"
        self.mode = mode
        self.img_paths = sorted((Path(data_root) / img_dir / mode).glob(f'*{img_suffix}'))
        self.mask_paths = sorted((Path(data_root) / mask_dir / mode).glob(f'*{mask_suffix}'))
        self.mean = tuple(mean)
        self.std = tuple(std)
        assert len(self.img_paths) == len(self.mask_paths), "Number of images and masks should be the same !"

        train_pipeline, val_pipeline = data_augment_pipeline(input_size=input_size,
                                                             mean=mean,
                                                             std=std,
                                                             use_roi=use_roi,
                                                             use_cutmix=use_cutmix)
        self.augment_pipeline = train_pipeline if mode == 'train' else val_pipeline

        if classes is not None:
            self.classes = classes

        if color_map is not None:
            self.color_map = color_map
        self.use_roi = use_roi
        self.use_cutmix = use_cutmix

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

        if self.use_cutmix:
            # 随机抽取另一张图像，用于CutMix增强
            idx2 = np.random.randint(0, len(self.img_paths))
            # print(f"CutMix: {img_path.name} <--> {self.img_paths[idx2].name}")
            img2 = cv.imread(str(self.img_paths[idx2]))
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            mask2 = cv.imread(str(self.mask_paths[idx2]), cv.IMREAD_GRAYSCALE)
            if self.use_roi:
                img2 = img2[ROI["ymin"]:ROI["ymax"], ROI["xmin"]:ROI["xmax"]]
                mask2 = mask2[ROI["ymin"]:ROI["ymax"], ROI["xmin"]:ROI["xmax"]]

            # 数据增强
            augmented = self.augment_pipeline(image=img,
                                              mask=mask,
                                              image2=img2,
                                              mask2=mask2)
        else:
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


def create_dataset(dataset_cfg,
                   input_size=None,
                   normal_data='imagenet',
                   split='train',
                   use_cutmix: bool = False,
                   use_roi: bool = False):
    dataset = Wrist_Ultrasound_Dataset(data_root=dataset_cfg['data_root'],
                                       img_dir=dataset_cfg['img_dir'],
                                       mask_dir=dataset_cfg['mask_dir'],
                                       input_size=[input_size, input_size] if input_size is not None else dataset_cfg['input_size'],
                                       mean=dataset_cfg['mean'][normal_data],
                                       std=dataset_cfg['std'][normal_data],
                                       classes=dataset_cfg['classes'],
                                       color_map=dataset_cfg['color_map'],
                                       mode=split,
                                       use_cutmix=use_cutmix,
                                       use_roi=use_roi)
    return dataset
