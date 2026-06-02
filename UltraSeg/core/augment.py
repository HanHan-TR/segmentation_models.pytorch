import cv2
import numpy as np
import albumentations as A


def add_speckle_noise(image, **kwargs):
    img_float = image.astype(np.float32)
    noise = np.random.randn(*img_float.shape)
    noisy_img = img_float + img_float * 0.08 * noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def data_augment_pipeline(input_size=[512, 512],
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225],
                          seed=42,
                          version=2):

    train_pipeline = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Affine(rotate=(-5.0, 5.0),
                 scale=(1.0 - 0.1, 1.0 + 0.1),
                 translate_percent=(-0.05, 0.05),
                 shear=(-10, 10),
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST,
                 p=0.3),
        A.OneOf([A.ElasticTransform(alpha=10.0,
                                    sigma=6.0,
                                    interpolation=cv2.INTER_LINEAR,
                                    mask_interpolation=cv2.INTER_NEAREST,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    p=1.0),
                 A.GridDistortion(num_steps=5,
                                  interpolation=cv2.INTER_LINEAR,
                                  mask_interpolation=cv2.INTER_NEAREST,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  p=1.0),
                 ],
                p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15,
                                   p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),

        # 乘性噪声：对强度做乘性扰动（更像 speckle 的扰动方式）
        A.MultiplicativeNoise(multiplier=(0.85, 1.15), p=0.2),
        A.GaussNoise(std_range=(0.0, 0.06), p=0.08),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.2, 1.0), p=0.1),

        A.Normalize(mean=tuple(mean), std=tuple(std)),
        A.ToTensorV2()
    ], seed=seed)

    train_pipeline2 = A.Compose([
        # ==============================
        # 第一阶段：空间几何形态 (Spatial)
        # ==============================
        A.OneOf([
            A.RandomResizedCrop(
                size=(input_size[0], input_size[1]),
                scale=(0.7, 1.0),
                ratio=(0.85, 1.15),
                p=1.0,
            ),
            A.Resize(height=input_size[0], width=input_size[1], p=1.0),
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
            hole_height_range=(1, max(2, input_size[0] // 12)),
            hole_width_range=(1, max(2, input_size[1] // 12)),
            fill=128,
            p=0.1
        ),
        A.Normalize(mean=tuple(mean), std=tuple(std)),
        A.ToTensorV2()
    ], seed=seed)

    val_pipeline = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(mean=tuple(mean), std=tuple(std)),
        A.ToTensorV2()
    ], seed=seed)

    if version == 2:
        return train_pipeline2, val_pipeline
    else:
        return train_pipeline, val_pipeline
