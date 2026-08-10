import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union
import sys
from scipy.ndimage import label

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from UltraSeg.core.fileio import yaml_load
from prettytable import PrettyTable
from UltraSeg.logger.logger import log_write


def collect_object_areas(dataset_cfg: Union[dict, str], cca=False, split: str = 'train'):
    """
    收集每个类别所有独立连通区域的面积（像素数）

    参数:
        dataset_cfg: dict或str, 数据集配置
        split: str, 数据集分割，'train'或'val'

    返回:
        dict: {class_id: [area1, area2, ...], ...}
    """
    if isinstance(dataset_cfg, str):
        dataset_cfg = yaml_load(dataset_cfg)

    data_root = dataset_cfg['data_root']
    masks_dir = Path(data_root) / dataset_cfg['mask_dir'] / split
    mask_suffix = dataset_cfg['mask_suffix']
    num_classes = dataset_cfg['num_classes']

    mask_files = sorted(masks_dir.glob(f'*{mask_suffix}'))

    class_areas = {i: [] for i in range(num_classes)}

    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取mask文件: {mask_file}")
            continue

        for class_id in range(num_classes):
            binary_mask = (mask == class_id).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            if cca:
                # 标记该类别所有的独立连通域
                # 返回值：
                #   labeled_mask: 标记后的连通域掩码，每个连通域被标记为不同的整数，不属于该类别的像素为0，属于该类别的连通域像素为1,2,3,...等
                #   num_features: 连通域数量
                labeled_mask, num_features = label(binary_mask)
                for obj_id in range(1, num_features + 1):
                    area = int(np.sum(labeled_mask == obj_id))
                    # if area < 10:
                    #     name = mask_file.stem.with_name(f'_cls_{class_id}{mask_suffix}')
                    #     cv2.imwrite(str(name), (labeled_mask == obj_id) * 255) if not name.exists() else None
                    class_areas[class_id].append(area)  # 记录该类别所有独立连通域的面积
            else:
                area = int(np.sum(binary_mask))
                class_areas[class_id].append(area)  # 记录该类别所有独立连通域的面积

    return class_areas, len(mask_files)


def compute_statistics(class_areas: dict, num_classes: int):
    """
    计算每个类别的面积统计信息

    参数:
        class_areas: {class_id: [area1, area2, ...], ...}
        num_classes: 类别总数

    返回:
        dict: 统计信息
    """
    stats = {}
    for class_id in range(num_classes):
        areas = class_areas.get(class_id, [])
        if len(areas) == 0:
            stats[class_id] = {
                'count': 0,
                'max': 0,
                'min': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'total_area': 0
            }
        else:
            areas_array = np.array(areas)
            stats[class_id] = {
                'count': len(areas),
                'max': int(np.max(areas_array)),
                'min': int(np.min(areas_array)),
                'mean': float(np.mean(areas_array)),
                'median': float(np.median(areas_array)),
                'std': float(np.std(areas_array)),
                'total_area': int(np.sum(areas_array))
            }
    return stats


def print_statistics_table(class_names: list, train_stats: dict, val_stats: dict, num_classes: int, logfile: Optional[str] = None):
    """
    使用prettytable输出统计表格

    参数:
        class_names: 类别名称列表
        train_stats: 训练集统计信息
        val_stats: 验证集统计信息
        num_classes: 类别总数
    """
    log_write("\n" + "=" * 120 + "\n", logfile=logfile)
    log_write("训练集 - 每个类别目标面积统计（单位：像素）\n", logfile=logfile)
    log_write("=" * 120 + "\n", logfile=logfile)

    train_table = PrettyTable()
    train_table.field_names = ["类别", "目标数量", "最大值", "最小值", "平均值", "中位数", "标准差", "总面积"]

    for class_id in range(num_classes):
        stat = train_stats[class_id]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        train_table.add_row([
            class_name,
            stat['count'],
            stat['max'],
            stat['min'],
            f"{stat['mean']:.2f}",
            f"{stat['median']:.2f}",
            f"{stat['std']:.2f}",
            stat['total_area']
        ])

    log_write(str(train_table), logfile=logfile)

    log_write("\n" + "=" * 120 + "\n", logfile=logfile)
    log_write("验证集 - 每个类别目标面积统计（单位：像素）\n", logfile=logfile)
    log_write("=" * 120 + "\n", logfile=logfile)

    val_table = PrettyTable()
    val_table.field_names = ["类别", "目标数量", "最大值", "最小值", "平均值", "中位数", "标准差", "总面积"]

    for class_id in range(num_classes):
        stat = val_stats[class_id]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        val_table.add_row([
            class_name,
            stat['count'],
            stat['max'],
            stat['min'],
            f"{stat['mean']:.2f}",
            f"{stat['median']:.2f}",
            f"{stat['std']:.2f}",
            stat['total_area']
        ])

    log_write(str(val_table), logfile=logfile)


def rgb_to_color(rgb_list):
    """将RGB列表[0-255]转换为matplotlib颜色字符串"""
    return f'#{rgb_list[0]:02x}{rgb_list[1]:02x}{rgb_list[2]:02x}'


def plot_area_histograms(class_areas: dict, class_names: list, num_classes: int,
                         split: str, stem: str, image_area: int = None, color_map: list = None):
    """
    绘制每个类别的面积分布直方图

    参数:
        class_areas: {class_id: [area1, area2, ...], ...}
        class_names: 类别名称列表
        num_classes: 类别总数
        split: 数据集分割名称
        stem: 文件名前缀
        image_area: 图像总面积（可选，用于计算百分比）
        color_map: 类别颜色映射列表
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    all_classes = [i for i in range(num_classes)]
    num_cols = 3
    num_rows = (len(all_classes) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()

    for idx, class_id in enumerate(all_classes):
        ax = axes[idx]
        areas = class_areas.get(class_id, [])
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"

        if color_map and class_id < len(color_map):
            color = rgb_to_color(color_map[class_id])
        else:
            color = 'skyblue'

        if len(areas) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{class_name}')
            ax.axis('off')
        else:
            areas_array = np.array(areas)

            if image_area is not None:
                areas_percent = (areas_array / image_area) * 100
                ax.hist(areas_percent, bins=30, color=color, edgecolor='black', alpha=0.7)
                ax.set_xlabel('面积占图像百分比 (%)', fontsize=10)
            else:
                ax.hist(areas_array, bins=30, color=color, edgecolor='black', alpha=0.7)
                ax.set_xlabel('面积（像素）', fontsize=10)

            ax.set_ylabel('目标数量', fontsize=10)
            ax.set_title(f'{class_name}\n(数量={len(areas)}, 均值={np.mean(areas_array):.1f}, 中位数={np.median(areas_array):.1f})',
                         fontsize=11)
            ax.grid(True, alpha=0.3)

    for idx in range(len(all_classes), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'{split} - 各类别目标面积分布直方图', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = f'{stem}_{split}_area_histograms.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'直方图已保存至 {output_path}')


def plot_combined_area_histograms(train_class_areas: dict, val_class_areas: dict,
                                  class_names: list, num_classes: int, stem: str, image_area: int = None,
                                  color_map: list = None):
    """
    绘制训练集和验证集的面积分布对比直方图

    参数:
        train_class_areas: 训练集面积数据
        val_class_areas: 验证集面积数据
        class_names: 类别名称列表
        num_classes: 类别总数
        stem: 文件名前缀
        image_area: 图像总面积（可选）
        color_map: 类别颜色映射列表
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    all_classes = [i for i in range(num_classes)]
    num_cols = 3
    num_rows = (len(all_classes) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()

    for idx, class_id in enumerate(all_classes):
        ax = axes[idx]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"

        if color_map and class_id < len(color_map):
            color = rgb_to_color(color_map[class_id])
        else:
            color = 'skyblue'

        train_areas = np.array(train_class_areas.get(class_id, []))
        val_areas = np.array(val_class_areas.get(class_id, []))

        if len(train_areas) == 0 and len(val_areas) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{class_name}')
            ax.axis('off')
        else:
            if image_area is not None:
                train_areas_plot = (train_areas / image_area) * 100 if len(train_areas) > 0 else []
                val_areas_plot = (val_areas / image_area) * 100 if len(val_areas) > 0 else []
                xlabel = '面积占图像百分比 (%)'
            else:
                train_areas_plot = train_areas
                val_areas_plot = val_areas
                xlabel = '面积（像素）'

            if len(train_areas_plot) > 0:
                ax.hist(train_areas_plot, bins=30, color=color, edgecolor='black', alpha=0.6, label='训练集')
            if len(val_areas_plot) > 0:
                ax.hist(val_areas_plot, bins=30, color=color, edgecolor='black', alpha=0.3, label='验证集', hatch='///')

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel('目标数量', fontsize=10)
            ax.set_title(f'{class_name}', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    for idx in range(len(all_classes), len(axes)):
        axes[idx].axis('off')

    fig.suptitle('训练集与验证集 - 各类别目标面积分布对比', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = f'{stem}_area_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'对比直方图已保存至 {output_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='超声图像语义分割数据集面积统计分析')
    parser.add_argument('--config', type=str, default='wan_guan',
                        help='数据集配置文件名称（位于UltraSeg/config/dataset/目录下）')
    parser.add_argument('--cca', action='store_true',
                        help='是否进行连通域(connected component analysis)分析，默认不进行，图像中的每个类别仅存在一个连通域')
    args = parser.parse_args()

    dataset_cfg_path = f'UltraSeg/config/dataset/{args.config}.yaml'
    dataset_cfg = yaml_load(dataset_cfg_path)

    stem = Path(dataset_cfg_path).stem
    num_classes = dataset_cfg['num_classes']
    class_names = dataset_cfg['classes']
    color_map = dataset_cfg.get('color_map', None)
    image_height = dataset_cfg['image_height']
    image_width = dataset_cfg['image_width']
    image_area = image_height * image_width

    logfile = f'{stem}_dataset_area_statistics.log'
    log_write(f"\n数据集配置文件: {dataset_cfg_path}\n", logfile=logfile)
    log_write(f"数据集名: {stem}\n", logfile=logfile)
    log_write(f"类别数量: {num_classes}\n", logfile=logfile)
    log_write(f"图像尺寸: {image_width} x {image_height}\n", logfile=logfile)
    log_write(f"图像总面积: {image_area} 像素\n", logfile=logfile)

    log_write("正在分析训练集...\n", logfile=logfile)
    train_class_areas, train_num_images = collect_object_areas(dataset_cfg, split='train')
    log_write(f"训练集图像数量: {train_num_images}\n", logfile=logfile)

    log_write("正在分析验证集...\n", logfile=logfile)
    val_class_areas, val_num_images = collect_object_areas(dataset_cfg, split='val')
    log_write(f"验证集图像数量: {val_num_images}\n", logfile=logfile)

    train_stats = compute_statistics(train_class_areas, num_classes)
    val_stats = compute_statistics(val_class_areas, num_classes)

    print_statistics_table(class_names, train_stats, val_stats, num_classes, logfile)

    plot_area_histograms(train_class_areas, class_names, num_classes, '训练集', stem, image_area, color_map)
    plot_area_histograms(val_class_areas, class_names, num_classes, '验证集', stem, image_area, color_map)

    plot_combined_area_histograms(train_class_areas, val_class_areas, class_names, num_classes, stem, image_area, color_map)

    print("\n分析完成！")
