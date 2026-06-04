# 超声腕部数据集划分

import random
import shutil
import sys
import os
import torch
from pathlib import Path
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from UltraSeg.core.fileio import yaml_load
from UltraSeg.core.initialize import set_random_seed, init_random_seed


if __name__ == '__main__':
    device = torch.device("cpu")
    train_percent = 0.7
    seed = init_random_seed(seed=42, device=device)
    set_random_seed(seed=seed, deterministic=True)

    yaml_dir = str(ROOT / 'UltraSeg/config/dataset/wrist.yaml')
    dataset_cfg = yaml_load(yaml_dir)
    data_root = dataset_cfg['data_root']
    img_folder = dataset_cfg['img_dir']
    mask_folder = dataset_cfg['mask_dir']
    class_rgb_folder = dataset_cfg['class_rgb_dir']
    object_folder = dataset_cfg['object_dir']

    img_dir = Path(data_root) / img_folder / 'all'
    mask_dir = Path(data_root) / mask_folder / 'all'
    class_rgb_dir = Path(data_root) / class_rgb_folder / 'all'
    object_dir = Path(data_root) / object_folder / 'all'

    assert img_dir.exists(), f'{img_dir} does not exist'
    assert mask_dir.exists(), f'{mask_dir} does not exist'
    assert class_rgb_dir.exists(), f'{class_rgb_dir} does not exist'
    assert object_dir.exists(), f'{object_dir} does not exist'

    img_paths = list(img_dir.glob("*.jpg"))
    num_total = len(img_paths)
    num_train = int(num_total * train_percent)

    print(f'Total images: {num_total}, Train images: {num_train}, Val images: {num_total - num_train}')

    random.shuffle(img_paths)
    pbar = tqdm(img_paths, desc='Copying images')
    for idx, path in enumerate(pbar):
        img_path = str(path)
        mask_path = str(path).replace(img_folder, mask_folder).replace('.jpg', '.png')
        class_rgb_path = str(path).replace(img_folder, class_rgb_folder).replace('.jpg', '.png')
        object_path = str(path).replace(img_folder, object_folder).replace('.jpg', '.png')

        if idx < num_train:
            shutil.copy(img_path, str(Path(data_root) / img_folder / 'train' / Path(img_path).name))
            shutil.copy(mask_path, str(Path(data_root) / mask_folder / 'train' / Path(mask_path).name))
            shutil.copy(class_rgb_path, str(Path(data_root) / class_rgb_folder / 'train' / Path(class_rgb_path).name))
            shutil.copy(object_path, str(Path(data_root) / object_folder / 'train' / Path(object_path).name))
        else:
            shutil.copy(img_path, str(Path(data_root) / img_folder / 'val' / Path(img_path).name))
            shutil.copy(mask_path, str(Path(data_root) / mask_folder / 'val' / Path(mask_path).name))
            shutil.copy(class_rgb_path, str(Path(data_root) / class_rgb_folder / 'val' / Path(class_rgb_path).name))
            shutil.copy(object_path, str(Path(data_root) / object_folder / 'val' / Path(object_path).name))
