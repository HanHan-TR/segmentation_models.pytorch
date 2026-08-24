# 单通道mask转RGB标签脚本
# 说明：该脚本将单通道标签图像转换为彩色标签图像，是data_split.py中rgb_to_label的反向过程
# 使用方法：python UltraSeg/tools/mask2rgb.py --yaml <配置文件路径>

import sys
import os
import argparse
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


def label_to_rgb(label: np.ndarray, color_map: List[List[int]]) -> np.ndarray:
    """
    将单通道标签图像转换为彩色标签图像（rgb_to_label的反向过程）

    Args:
        label: 单通道标签图像，形状为 [H, W]，像素值为类别索引
        color_map: 颜色映射表，每个元素是 [R, G, B] 列表，索引对应类别编号

    Returns:
        color_label: 彩色标签图像，形状为 [H, W, 3]，RGB格式
    """
    # 获取图像高度和宽度
    h, w = label.shape[:2]

    # 将颜色映射表转换为numpy数组
    color_map_np = np.array(color_map, dtype=np.uint8)

    # 初始化彩色标签图像
    color_label = np.zeros((h, w, 3), dtype=np.uint8)

    # 遍历每个类别，将对应索引的像素赋值为颜色
    for class_idx, color in enumerate(color_map_np):
        # 创建掩码：找到所有值等于当前类别索引的像素
        mask = (label == class_idx)
        color_label[mask] = color

    return color_label


def convert_single_channel_to_rgb(
    input_dir: str,
    output_dir: str,
    color_map: List[List[int]],
    file_extensions: Tuple[str, ...] = ('.png',)
) -> None:
    """
    批量将目录中的单通道标签图像转换为彩色标签图像

    Args:
        input_dir: 输入目录，存放单通道标签图像
        output_dir: 输出目录，保存转换后的彩色标签图像
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

    print(f"发现 {len(image_files)} 个单通道标签图像")

    # 处理每个文件
    for img_file in tqdm(image_files, desc=f"转换 {input_path.name}"):
        # 以灰度模式读取单通道标签图像
        label = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(f"警告：无法读取文件 {img_file}")
            continue

        # 转换为彩色标签
        color_label = label_to_rgb(label, color_map)

        # 构建输出路径（保持原文件名，使用png格式保存）
        output_file = output_path / f"{img_file.stem}.png"

        # 将RGB转换为BGR格式（cv2保存要求）然后保存
        color_label_bgr = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), color_label_bgr)

    print(f"{input_path.name} 转换完成！")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将单通道mask转换为RGB标签图像')
    parser.add_argument('--yaml', type=str, required=True, help='数据集配置yaml文件路径')
    args = parser.parse_args()

    # 读取数据集配置
    dataset_cfg = yaml_load(args.yaml)
    data_root = dataset_cfg['data_root']
    mask_folder = dataset_cfg['mask_dir']
    color_map = dataset_cfg['color_map']

    # 构建Masks目录和输出SegmentationClass目录路径
    masks_dir = Path(data_root) / mask_folder
    output_dir = Path(data_root) / 'SegmentationClass'

    # 检查Masks目录是否存在
    assert masks_dir.exists(), f'Masks目录不存在: {masks_dir}'

    # 创建输出根目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出根目录: {output_dir}")

    # 获取Masks下所有子文件夹（all、train、val、test等）
    subdirs = [d for d in masks_dir.iterdir() if d.is_dir()]
    print(f"发现子文件夹: {[d.name for d in subdirs]}")

    # 遍历每个子文件夹进行转换
    for subdir in subdirs:
        input_subdir = masks_dir / subdir.name
        output_subdir = output_dir / subdir.name
        print(f"\n处理子文件夹: {subdir.name}")
        convert_single_channel_to_rgb(str(input_subdir), str(output_subdir), color_map)

    print("\n所有转换完成！")
