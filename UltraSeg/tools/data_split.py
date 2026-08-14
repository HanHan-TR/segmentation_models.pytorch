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


def convert_rgb_labels_to_single_channel(
    input_dir: str,
    output_dir: str,
    color_map: List[List[int]],
    file_extensions: Tuple[str, ...] = ('.png')
) -> None:
    """
    批量将目录中的彩色标签图像转换为单通道标签图像

    Args:
        input_dir: 输入目录，存放彩色标签图像
        output_dir: 输出目录，保存转换后的单通道标签图像
        color_map: 颜色映射表
        file_extensions: 需要处理的文件扩展名
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有符合条件的文件
    image_files = []
    for ext in file_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))

    # 按文件名排序
    image_files.sort()

    print(f"发现 {len(image_files)} 个彩色标签图像")

    # 处理每个文件
    for img_file in image_files:
        # 读取彩色标签图像（注意：cv2默认读取为BGR格式，需要转换为RGB）
        color_label = cv2.imread(str(img_file))
        if color_label is None:
            print(f"警告：无法读取文件 {img_file}")
            continue

        # 转换为RGB格式
        color_label = cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB)

        # 转换为单通道标签
        label = rgb_to_label(color_label, color_map)

        # 构建输出路径（保持原文件名，使用png格式保存）
        output_file = output_path / f"{img_file.stem}.png"

        # 保存单通道标签图像
        cv2.imwrite(str(output_file), label)

        print(f"已处理: {img_file.name} -> {output_file.name}")

    print("转换完成！")


if __name__ == '__main__':
    device = torch.device("cpu")
    train_percent = 0.7
    val_percent = 0.2
    test_percent = 0.1

    sum = train_percent + val_percent + test_percent

    seed = init_random_seed(seed=42, device=device)
    set_random_seed(seed=seed, deterministic=True)

    yaml_dir = str(ROOT / 'UltraSeg/config/dataset/wan_shortlong.yaml')
    dataset_cfg = yaml_load(yaml_dir)
    data_root = dataset_cfg['data_root']
    img_folder = dataset_cfg['img_dir']
    mask_folder = dataset_cfg['mask_dir']
    class_rgb_folder = dataset_cfg['class_rgb_dir']
    object_folder = dataset_cfg['object_dir']
    color_map = dataset_cfg['color_map']

    img_dir = Path(data_root) / img_folder / 'all'
    mask_dir = Path(data_root) / mask_folder / 'all'
    class_rgb_dir = Path(data_root) / class_rgb_folder / 'all'
    object_dir = Path(data_root) / object_folder / 'all'

    assert img_dir.exists(), f'{img_dir} does not exist'
    assert mask_dir.exists(), f'{mask_dir} does not exist'
    assert class_rgb_dir.exists(), f'{class_rgb_dir} does not exist'

    img_paths = list(img_dir.glob("*.*"))
    num_total = len(img_paths)
    num_train = int(num_total * train_percent)

    print(f'Total images: {num_total}, Train images: {num_train}, Val images: {num_total - num_train}')

    mask_paths = list(mask_dir.glob("*.png"))
    if len(mask_paths) == 0:
        rgb_mask_paths = list(class_rgb_dir.glob("*.png"))
        assert len(rgb_mask_paths) == len(img_paths), 'rgb_mask_paths and img_paths have different lengths'
        convert_rgb_labels_to_single_channel(class_rgb_dir, mask_dir, color_map)
    else:
        assert len(mask_paths) == len(img_paths), 'mask_paths and img_paths have different lengths'

    mask_paths = list(mask_dir.glob("*.png"))
    assert len(mask_paths) == len(img_paths), 'mask_paths and img_paths have different lengths'

    random.shuffle(img_paths)
    pbar = tqdm(img_paths, desc='Copying images')
    for idx, path in enumerate(pbar):
        img_path = str(path)
        mask_path = str(path).replace(img_folder, mask_folder).replace('.png', '.png')
        class_rgb_path = str(path).replace(img_folder, class_rgb_folder).replace('.png', '.png')
        object_path = str(path).replace(img_folder, object_folder).replace('.png', '.png')
        if idx < num_train:
            shutil.copy(img_path, str(Path(data_root) / img_folder / 'train' / Path(img_path).name))
            shutil.copy(mask_path, str(Path(data_root) / mask_folder / 'train' / Path(mask_path).name))
            shutil.copy(class_rgb_path, str(Path(data_root) / class_rgb_folder / 'train' / Path(class_rgb_path).name))
            shutil.copy(object_path, str(Path(data_root) / object_folder / 'train' / Path(object_path).name)) if Path(object_path).exists() else None
        else:
            shutil.copy(img_path, str(Path(data_root) / img_folder / 'val' / Path(img_path).name))
            shutil.copy(mask_path, str(Path(data_root) / mask_folder / 'val' / Path(mask_path).name))
            shutil.copy(class_rgb_path, str(Path(data_root) / class_rgb_folder / 'val' / Path(class_rgb_path).name))
            shutil.copy(object_path, str(Path(data_root) / object_folder / 'val' / Path(object_path).name)) if Path(object_path).exists() else None
