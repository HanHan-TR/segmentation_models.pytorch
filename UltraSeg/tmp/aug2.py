import albumentations as A
import numpy as np


def add_speckle_noise(image, **kwargs):
    img_float = image.astype(np.float32)
    noise = np.random.randn(*img_float.shape)
    noisy_img = img_float + img_float * 0.08 * noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def get_train_augmentation(INPUT_SIZE):
    height, width = INPUT_SIZE

    return A.Compose([
        # ==============================
        # 第一阶段：空间几何形态 (Spatial)
        # ==============================
        A.OneOf([
            A.RandomResizedCrop(
                size=(height, width),
                scale=(0.7, 1.0),
                ratio=(0.85, 1.15),
                p=1.0,
            ),
            A.Resize(height=height, width=width, p=1.0),
        ], p=1.0),

        A.Affine(
            scale=(0.7, 1.2),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-15, 15),
            p=0.4
        ),
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.ElasticTransform(alpha=40, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=1.0),
            A.OpticalDistortion(distort_limit=(-0.3, 0.3), p=1.0)
        ], p=0.3),

        # ==============================
        # 第二阶段：物理成像退化
        # ==============================
        # 1. 探头底噪 (Noise)
        A.OneOf([
            A.GaussNoise(std_range=(10.0 / 255, 30.0 / 255), p=1.0),
            A.Lambda(image=add_speckle_noise, p=1.0),
        ], p=0.2),

        # 2. 声束扩散模糊 (Blur - 模糊会晕染上面的噪声)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),

        # 3. 机器基础增益 (Brightness/Contrast)
        A.RandomBrightnessContrast(
            brightness_limit=(-0.15, 0.15),
            contrast_limit=(-0.15, 0.15),
            p=0.3
        ),

        # 4. 探头物理分辨率极限 (Simulate Low Res)
        # 🌟 必须在 Gamma 之前执行插值！
        A.OneOf([
            A.Downscale(scale_range=(0.5, 0.85), p=1.0),
            A.Downscale(scale_range=(0.25, 0.5), p=1.0),
        ], p=0.2),

        # 5. 显示器非线性渲染 (Gamma)
        A.OneOf([
            # 正常 Gamma
            A.RandomGamma(gamma_limit=(80, 150), p=1.0),
            # ⭐ nnU-Net Invert Gamma（真正版本）
            A.Compose([
                A.InvertImg(p=1.0),
                A.RandomGamma(gamma_limit=(80, 150), p=1.0),
                A.InvertImg(p=1.0),
            ])
        ], p=0.2),

        # ==============================
        # 第三阶段：防遮挡与正则化
        # ==============================
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(1, max(2, height // 12)),
            hole_width_range=(1, max(2, width // 12)),
            fill=128,
            p=0.1
        ),
    ])
