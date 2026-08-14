# 超声腕部数据集划分

# 说明：
# 该脚本有两个主要功能：
# 1. 将彩色标签图像转换为单通道标签图像
# 2. 划分数据集为训练集、验证集和测试集
# 使用方法：
# 1. 准备数据集配置信息yaml文件，包含数据集的路径、类别信息、颜色映射表等
# 2. 准备数据集；

import random
import shutil
import sys
import os
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from typing import List, Tuple

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from UltraSeg.core.fileio import yaml_load
from UltraSeg.core.initialize import set_random_seed, init_random_seed


def rgb_to_label(color_label: np.ndarray, color_map: List[List[int]]) -> np.ndarray:
    """
    将彩色标签图像转换为单通道标签图像

    Args:
        color_label: 彩色标签图像，形状为 [H, W, 3]，RGB格式
        color_map: 颜色映射表，每个元素是 [R, G, B] 列表，索引对应类别编号

    Returns:
        label: 单通道标签图像，形状为 [H, W]，像素值为类别索引
    """
    # 获取图像高度和宽度
    h, w = color_label.shape[:2]

    # 初始化单通道标签图像
    label = np.zeros((h, w), dtype=np.uint8)

    # 将颜色映射表转换为numpy数组，便于向量化操作
    color_map_np = np.array(color_map, dtype=np.uint8)

    # 遍历每个类别，找到匹配的像素并赋值
    for class_idx, color in enumerate(color_map_np):
        # 创建掩码：找到所有匹配当前颜色的像素
        mask = np.all(color_label == color, axis=-1)
        label[mask] = class_idx

    return label


if __name__ == '__main__':
    device = torch.device("cpu")

    discard_classes = [10, 11, 12]

    yaml_dir = str(ROOT / 'UltraSeg/config/dataset/wan_shortlong.yaml')
    dataset_cfg = yaml_load(yaml_dir)
    data_root = dataset_cfg['data_root']
    img_folder = dataset_cfg['img_dir']
    mask_folder = dataset_cfg['mask_dir']
    class_rgb_folder = dataset_cfg['class_rgb_dir']
    object_folder = dataset_cfg['object_dir']
    color_map = dataset_cfg['color_map']

    img_dir = Path(data_root) / img_folder / 'all'
    discard_img_dir = Path(data_root) / img_folder / 'discard'
    discard_img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(data_root) / mask_folder / 'all'
    discard_mask_dir = Path(data_root) / mask_folder / 'discard'
    discard_mask_dir.mkdir(parents=True, exist_ok=True)
    class_rgb_dir = Path(data_root) / class_rgb_folder / 'all'
    discard_rgb_dir = Path(data_root) / class_rgb_folder / 'discard'
    discard_rgb_dir.mkdir(parents=True, exist_ok=True)

    assert img_dir.exists(), f'{img_dir} does not exist'
    assert mask_dir.exists(), f'{mask_dir} does not exist'
    assert class_rgb_dir.exists(), f'{class_rgb_dir} does not exist'

    img_paths = list(img_dir.glob("*.*"))
    num_total = len(img_paths)

    mask_paths = list(mask_dir.glob("*.png"))
    assert len(mask_paths) == len(img_paths), 'mask_paths and img_paths have different lengths'

    random.shuffle(img_paths)
    pbar = tqdm(img_paths, desc='Moving discarded images')
    discard_count = 0
    for idx, path in enumerate(pbar):
        img_path = str(path)
        img_stem = Path(path).stem
        mask_path = str(mask_dir / f"{img_stem}.png")
        class_rgb_path = str(class_rgb_dir / f"{img_stem}.png")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            pbar.set_postfix_str('Skipped: cannot read mask')
            continue

        discard_mask = np.isin(mask, discard_classes)
        if np.any(discard_mask):
            shutil.move(img_path, str(discard_img_dir / Path(img_path).name))
            shutil.move(mask_path, str(discard_mask_dir / Path(mask_path).name))
            shutil.move(class_rgb_path, str(discard_rgb_dir / Path(class_rgb_path).name))
            discard_count += 1
            pbar.set_postfix_str(f'Discarded: {discard_count}')

    print(f'\nTotal images: {num_total}, Discarded: {discard_count}, Remaining: {num_total - discard_count}')
