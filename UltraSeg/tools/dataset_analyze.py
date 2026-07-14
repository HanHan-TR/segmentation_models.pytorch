import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# from prettytable import PrettyTable

from typing import Union
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from UltraSeg.core.fileio import yaml_load

# font_path = Path.home() / '.local/share/fonts/wqy-microhei/wqy-microhei.ttc'
# if font_path.exists():
#     fm.fontManager.addfont(str(font_path))


def analyze_dataset(dataset_cfg: Union[dict, str], split: str = 'train'):
    """
    统计踝关节超声图像语义分割数据集的信息

    参数:
        dataset_cfg: dict或str, 数据集配置
        split: str, 数据集分割，'train'或'val'

    返回:
        dict: 包含统计信息的字典
    """
    if isinstance(dataset_cfg, str):
        dataset_cfg = yaml_load(dataset_cfg)

    data_root = dataset_cfg['data_root']
    masks_dir = Path(data_root) / dataset_cfg['mask_dir'] / split

    # 获取所有mask文件
    mask_files = sorted(masks_dir.glob('*.png'))
    num_images = len(mask_files)

    num_classes = dataset_cfg['num_classes']

    class_names = dataset_cfg['classes']

    image_height = dataset_cfg['image_height']
    image_width = dataset_cfg['image_width']
    image_area = image_height * image_width

    if num_images == 0:
        print("未找到任何mask文件")
        return None

    # 统计变量
    class_object_count = [0] * num_classes
    class_total_area = [0] * num_classes
    class_avg_area = [0.0] * num_classes
    # 遍历所有mask文件
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"无法读取mask文件: {mask_file}")
            continue

        # 获取图像中所有的类别
        unique_classes = np.unique(mask)

        for class_id in unique_classes:
            class_id = int(class_id)
            area = int(np.sum(mask == class_id))
            if area > 0:
                class_object_count[class_id] += 1
                class_total_area[class_id] += area

    # 计算平均面积

    for class_id in range(num_classes):
        class_avg_area[class_id] = (class_total_area[class_id] / class_object_count[class_id]) / image_area * 100

    # 准备结果
    result = {
        'num_images': num_images,
        'class_names': class_names,
        'class_object_count': class_object_count,
        'class_total_area': class_total_area,
        'class_avg_area': class_avg_area
    }

    return result


def plot_statistics(train_result, val_result, stem: str, msg=''):
    """
    使用柱状图展示统计结果

    参数:
        train_result: dict, 训练集统计结果
        val_result: dict, 验证集统计结果
        stem: str, 文件名前缀
        msg: str, 数据集描述
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    datasets = [
        (train_result, '训练集', axes[0]),
        (val_result, '验证集', axes[1]),
    ]

    for result, split_name, (ax_count, ax_area) in datasets:
        if result is None:
            ax_count.set_title(f'{split_name}：无数据')
            ax_count.axis('off')
            ax_area.axis('off')
            continue

        num_images = result['num_images']
        class_names = result['class_names']
        object_counts = result['class_object_count']
        avg_areas = result['class_avg_area']

        bars1 = ax_count.bar(class_names, object_counts, color='skyblue')
        ax_count.set_title(f'{msg} {split_name}（{num_images}张图像）\n每个组织类别的数量统计')
        ax_count.set_xlabel('组织类别')
        ax_count.set_ylabel('数量')
        ax_count.tick_params(axis='x', rotation=45, labelsize=8)

        for bar in bars1:
            height = bar.get_height()
            ax_count.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{int(height)}', ha='center', va='bottom')

        bars2 = ax_area.bar(class_names, avg_areas, color='lightgreen')
        ax_area.set_title(f'{msg} {split_name}（{num_images}张图像）\n每个组织类别的平均面积 (占图像总面积的百分比)')
        ax_area.set_xlabel('组织类别')
        ax_area.set_ylabel('平均面积 (%)')
        ax_area.tick_params(axis='x', rotation=45, labelsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax_area.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{stem}_dataset_statistics.png')
    plt.close()


# 示例用法
if __name__ == '__main__':
    # 设置您的mask文件夹路径
    dataset_cfg = "UltraSeg/config/dataset/wrist.yaml"
    stem = Path(dataset_cfg).stem
    if stem == 'wrist':
        msg = '腕部超声图像数据集'
    elif stem == 'huai':
        msg = '踝部超声图像数据集'
    elif stem == 'wan_guan':
        msg = '腕管部位超声图像数据集'
    # 分析数据集
    train_stats = analyze_dataset(dataset_cfg, split='train')
    val_stats = analyze_dataset(dataset_cfg, split='val')

    if train_stats or val_stats:
        plot_statistics(train_stats, val_stats, stem, msg)
    print(f'统计结果已保存至 {stem}_dataset_statistics.png')
