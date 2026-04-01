import torch
import numpy as np
import cv2
from typing import List


def visualize_predictions(images: torch.Tensor,
                          targets: torch.Tensor,
                          pred: torch.Tensor,
                          mean: List[float] = None,
                          std: List[float] = None,
                          color_map: List[List[int]] = None,
                          alpha: float = 0.6,
                          max_images: int = 16,
                          save_path: str = None):
    """
    可视化语义分割推理结果与真实标签

    Args:
        images: 输入图像张量，形状为 [B, C, H, W]
        targets: 真实标签张量，形状为 [B, H, W]
        pred: 预测标签张量，形状为 [B, H, W]
        mean: 均值列表，用于反归一化
        std: 标准差列表，用于反归一化
        original_size: 原始图像尺寸 [height, width]
        color_map: 类别颜色映射表
        max_images: 最多显示的图像数量
        save_path: 保存路径，如果为None则不保存
    """
    if color_map is None:
        color_map = [[0, 0, 0], [32, 32, 185], [102, 245, 102], [214, 41, 69],
                     [218, 70, 218], [177, 70, 92], [156, 63, 156], [165, 32, 59],
                     [204, 204, 59], [194, 87, 140]]

    num_images = min(images.shape[0], max_images)
    num_rows = (num_images + 1) // 2

    canvas_width = 1920
    canvas_height = 1390
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    cell_width = canvas_width // 4
    cell_height = canvas_height // num_rows

    for i in range(num_images):
        # 计算当前图像在画布中的位置
        row = i // 2  # 每行显示2个图像
        img_idx_in_row = i % 2  # 当前行中的第几个图像

        # 计算预测和标签的列位置
        pred_col = img_idx_in_row * 2
        gt_col = img_idx_in_row * 2 + 1

        # 计算坐标
        pred_x_start = pred_col * cell_width
        pred_y_start = row * cell_height
        pred_x_end = pred_x_start + cell_width
        pred_y_end = pred_y_start + cell_height

        gt_x_start = gt_col * cell_width
        gt_y_start = row * cell_height
        gt_x_end = gt_x_start + cell_width
        gt_y_end = gt_y_start + cell_height

        # 处理当前图像
        img = images[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        pred_i = pred[i].cpu().numpy()

        if mean is not None and std is not None:
            mean_arr = np.array(mean).reshape(3, 1, 1)
            std_arr = np.array(std).reshape(3, 1, 1)
            img = img * std_arr + mean_arr

        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))

        img_resized = cv2.resize(img, (cell_width, cell_height))

        # 处理标签
        target_colored = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(color_map):
            mask = target == class_id
            target_colored[mask] = color
        target_resized = cv2.resize(target_colored, (cell_width, cell_height))
        target_overlay = cv2.addWeighted(img_resized, (1 - alpha), target_resized, alpha, 0)

        # 处理预测
        pred_colored = np.zeros((pred_i.shape[0], pred_i.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(color_map):
            mask = pred_i == class_id
            pred_colored[mask] = color
        pred_resized = cv2.resize(pred_colored, (cell_width, cell_height))
        pred_overlay = cv2.addWeighted(img_resized, (1 - alpha), pred_resized, alpha, 0)

        # 绘制预测
        canvas[pred_y_start:pred_y_end, pred_x_start:pred_x_end] = pred_overlay
        cv2.putText(canvas, f'Pred {i+1}', (pred_x_start + 10, pred_y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 绘制标签
        canvas[gt_y_start:gt_y_end, gt_x_start:gt_x_end] = target_overlay
        cv2.putText(canvas, f'GT {i+1}', (gt_x_start + 10, gt_y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if save_path is not None:
        cv2.imwrite(save_path, canvas)
