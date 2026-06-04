import os
import numpy as np
from PIL import Image
from pathlib import Path


def calculate_mean_std(image_root_folder):
    # 初始化统计量
    pixel_sum = np.zeros(3)  # 各通道的像素值之和
    pixel_sq_sum = np.zeros(3)  # 各通道的像素值的平方和
    num_pixels = 0  # 像素总数
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_paths = []

    # 递归遍历所有子文件夹
    for ext in image_extensions:
        image_paths.extend(Path(image_root_folder).glob(f"**/*{ext}"))
        image_paths.extend(Path(image_root_folder).glob(f"**/*{ext.upper()}"))  # 大写扩展名

    # 转换为字符串路径并去重
    image_paths = list(set(str(p) for p in image_paths))

    # 遍历图像文件夹
    for im in image_paths:
        img = Image.open(im).convert('RGB')
        img_array = np.array(img).astype(np.float64)  # 形状为(H, W, 3)

        # 累加统计量[6,7](@ref)
        pixel_sum += img_array.sum(axis=(0, 1))
        im2 = np.square(img_array)
        pixel_sq_sum += (im2).sum(axis=(0, 1))
        num_pixels += img_array.shape[0] * img_array.shape[1]

    # 计算最终均值和标准差
    mean = pixel_sum / num_pixels
    a = pixel_sq_sum / num_pixels
    b = mean**2
    std = np.sqrt(a - b)  # σ = sqrt(E[X²] - (E[X])²)

    return mean, std


if __name__ == "__main__":
    image_root_folder = "/home/wanghan/work/dataset/wrist/JPEGImages/train"  # 替换为实际路径
    mean, std = calculate_mean_std(image_root_folder)
    # 打印结果
    print("\n" + "=" * 50)
    print(f"统计结果（基于{image_root_folder}及其所有子文件夹）:")
    print("=" * 50)
    print(f"均值 (R, G, B): [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}] float: [{mean[0]/255:.3f}, {mean[1]/255:.3f}, {mean[2]/255:.3f}]")
    print(f"标准差 (R, G, B): [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}] float: [{std[0]/255:.3f}, {std[1]/255:.3f}, {std[2]/255:.3f}]")
    print("=" * 50)
