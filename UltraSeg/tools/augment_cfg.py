# 使用albumentations库生成数据增强配置yaml文件
import albumentations as A
import argparse
import sys
import os
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

defautlt_train_transform = A.Compose([A.Resize(height=640, width=640),
                                      A.OneOf([A.Rotate(limit=(-90, 90), p=0.5),
                                               A.HorizontalFlip(p=0.5),
                                               A.VerticalFlip(p=0.5),
                                               ], p=1),
                                      # 模糊
                                      A.OneOf([A.MotionBlur(blur_limit=13, p=0.5),
                                               A.GlassBlur(sigma=2.5, max_delta=4, p=0.5),
                                               A.Blur(p=0.5),
                                               A.Defocus(p=0.5)
                                               ], p=1),
                                      # 颜色调整
                                      A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.2,
                                                                          contrast_limit=0.2, p=0.5),  # 随机亮度与对比度
                                               A.RandomGamma(gamma_limit=(60, 140), p=0.5),  # 随机Gamma调整
                                               A.HueSaturationValue(hue_shift_limit=5,
                                                                    sat_shift_limit=10,
                                                                    val_shift_limit=10, p=0.5),  # HSV随机改变
                                               ], p=1),

                                      # 场景模拟
                                      A.ISONoise(color_shift=(0.05, 0.2), intensity=(0.1, 0.5), p=0.1),

                                      # 归一化与转换成 Tensor 模式
                                      A.Normalize(mean=(0.563, 0.328, 0.244), std=(0.315, 0.222, 0.190)),  # !! mean 与 std 的数值需要根据不同的数据集改变
                                      # Convert image and mask to PyTorch tensors
                                      A.ToTensorV2(),  # !! 请使用最新版的albumentations
                                      ])

defautlt_val_transform = A.Compose([  # 归一化与转换成 Tensor 模式
    A.Resize(height=640, width=640),
    A.Normalize(mean=(0.563, 0.328, 0.244), std=(0.315, 0.222, 0.190)),  # !! mean 与 std 的数值需要根据不同的数据集改变
    # Convert image and mask to PyTorch tensors
    A.ToTensorV2(),  # !! 请使用最新版的albumentations （2.0.6版本是OK的）
])


def parse_args():
    parser = argparse.ArgumentParser(description='Create augmentation setting file.')
    parser.add_argument('--file_type', type=str,
                        default='yaml', choices=['json', 'yaml'],
                        help='augmentation setting file type')
    parser.add_argument('--save_dir', type=str,
                        default=ROOT / 'UltraSeg/config/hyper',
                        help='dir to save augmentation setting file')
    parser.add_argument('--img_mean', type=tuple,
                        default=(0.563, 0.328, 0.244),
                        help='image rgb mean')
    parser.add_argument('--img_std', type=tuple,
                        default=(0.315, 0.222, 0.190),
                        help='image rgb std')
    parser.add_argument('--name', type=str,
                        default='kvasir_train_transform',
                        help='name of augmentation setting file')
    args = parser.parse_args()
    return args


# Requires: pip install pyyaml
if __name__ == '__main__':
    args = parse_args()

    # transform = defautlt_training_transform
    transform = defautlt_train_transform

    if args.file_type == 'json':
        pass
    elif args.file_type == 'yaml':
        try:
            import yaml  # Check if PyYAML is installed
        except ImportError:
            print("PyYAML not installed, skipping YAML example.")
    else:
        raise TypeError('augmentation setting file type should be "json" or "yaml". ')

    file_name = args.name + '.' + args.file_type
    file = Path(args.save_dir) / file_name
    A.save(transform, file, data_format=args.file_type)

    print("YAML serialization successful.")

    # Verify loaded_transform_yaml works similarly...
    # loaded_transform = A.load(file, data_format=args.file_type)
