"""
为指定 image 文件夹中的每一张图像创建一个同尺寸的全 0 单通道 mask，
并以相同文件名（扩展名为 .png）保存到指定的 mask 文件夹中。

用法：
    python UltraSeg/tools/create_zero_masks.py --image_dir UltraSeg/background/images/img --mask_dir UltraSeg/background/masks
    # 或使用默认路径（相对于项目根目录）：
    python UltraSeg/tools/create_zero_masks.py
"""

import argparse
import sys
import os
import shutil
from pathlib import Path, PosixPath
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 把 ROOT 加入 sys.path


IMAGE_EXTS = {".jpg", ".png", ".bmp", ".tif", ".tiff"}


def create_zero_masks(image_dir: Path, mask_dir: Path) -> None:
    """
    遍历 image_dir 下的图像，在 mask_dir 中生成同尺寸、单通道、像素全 0 的 png 掩码。

    Args:
        image_dir: 原始图像所在文件夹
        mask_dir: 输出 mask 所在文件夹（若不存在则自动创建）
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir 不存在或不是文件夹: {image_dir}")

    src_train = image_dir / "train"
    src_val = image_dir / "val"

    src_train.mkdir(parents=True, exist_ok=True)
    src_val.mkdir(parents=True, exist_ok=True)

    mask_train = mask_dir / "train"
    mask_val = mask_dir / "val"

    mask_train.mkdir(parents=True, exist_ok=True)
    mask_val.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        [p for p in image_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )

    random.shuffle(image_files)  # 随机打乱图像文件列表

    split_ratio = 0.7  # 训练集占比

    num_train = int(len(image_files) * split_ratio)

    if not image_files:
        print(f"[警告] 在 {image_dir} 中未找到任何图像文件，支持的扩展名: {sorted(IMAGE_EXTS)}")
        return

    print(f"共发现 {len(image_files)} 张图像，开始生成全 0 mask ...")
    for n, img_path in tqdm(enumerate(image_files), desc="生成 mask", unit="img"):
        with Image.open(img_path) as im:
            width, height = im.size

        # 生成单通道 (H, W)、全 0、uint8 的数组
        mask = np.zeros((height, width), dtype=np.uint8)

        # 文件名保持与原图像一致，扩展名改为 .png
        mask_name = img_path.stem + ".png"

        if n < num_train:
            shutil.move(str(img_path), str(src_train / img_path.name))
            Image.fromarray(mask, mode="L").save(mask_train / mask_name, format="PNG")
        else:
            shutil.move(str(img_path), str(src_val / img_path.name))
            Image.fromarray(mask, mode="L").save(mask_val / mask_name, format="PNG")

    print(f"完成！mask 已保存到: {mask_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 image 文件夹中的每张图像生成同尺寸的全 0 单通道 PNG mask。"
    )
    parser.add_argument(
        "--image_dir",
        type=PosixPath,
        default=ROOT / "UltraSeg/background/images",
        help="原始图像所在文件夹（默认: UltraSeg/background/images）",
    )
    parser.add_argument(
        "--mask_dir",
        type=PosixPath,
        default=ROOT / "UltraSeg/background/masks",
        help="输出 mask 所在文件夹（默认: UltraSeg/background/masks）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    create_zero_masks(image_dir, mask_dir)


if __name__ == "__main__":
    main()
