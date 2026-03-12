from torch.utils.data import Dataset
import cv2 as cv
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
                 mode: str = 'train'):
        super().__init__()
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        assert mode in ['train', 'val', 'test'], f"mode should be one of ['train', 'val', 'test'], but got {mode}"
        self.mode = mode
        self.img_paths = sorted((Path(data_root) / img_dir / mode).glob(f'*{img_suffix}'))
        self.mask_paths = sorted((Path(data_root) / mask_dir / mode).glob(f'*{mask_suffix}'))

        assert len(self.img_paths) == len(self.mask_paths), "Number of images and masks should be the same !"

        train_pipeline, val_pipeline = data_augment_pipeline(input_size=input_size,
                                                             mean=mean,
                                                             std=std)
        self.augment_pipeline = train_pipeline if mode == 'train' else val_pipeline

        if classes is not None:
            self.classes = classes

        if color_map is not None:
            self.color_map = color_map

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = cv.imread(str(img_path))
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        # 数据增强
        augmented = self.augment_pipeline(image=img, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        return (image, mask)


def create_dataset(dataset_cfg,
                   normal_data='imagenet',
                   split='train'):
    dataset = Wrist_Ultrasound_Dataset(data_root=dataset_cfg['data_root'],
                                       img_dir=dataset_cfg['img_dir'],
                                       mask_dir=dataset_cfg['mask_dir'],
                                       input_size=dataset_cfg['input_size'],
                                       mean=dataset_cfg['mean'][normal_data],
                                       std=dataset_cfg['std'][normal_data],
                                       classes=dataset_cfg['classes'],
                                       color_map=dataset_cfg['color_map'],
                                       mode=split)
    return dataset
