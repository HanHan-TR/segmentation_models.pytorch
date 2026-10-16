import cv2
import random
import numpy as np
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
from albumentations.core.transforms_interface import DualTransform


class LabelGuidedCropAlbumentations(DualTransform):
    """
    基于 Albumentations 框架的标签指导裁剪组件
    """

    def __init__(self, height, width, p_rare=0.9, rare_classes=None, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.p_rare = p_rare
        self.rare_classes = rare_classes

    def apply(self, img, ymin=0, xmin=0, **params):
        return img[ymin:ymin + self.height, xmin:xmin + self.width]

    def apply_to_mask(self, mask, ymin=0, xmin=0, **params):
        return mask[ymin:ymin + self.height, xmin:xmin + self.width]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        h, w = mask.shape[:2]
        ch, cw = min(self.height, h), min(self.width, w)

        # 概率性触发标签指导裁剪
        if random.random() < self.p_rare and self.rare_classes is not None:
            present_rare_classes = [c for c in self.rare_classes if c in mask]
            if present_rare_classes:
                target_cls = random.choice(present_rare_classes)  # 随机选择一个罕见类别
                coords = np.argwhere(mask == target_cls)
                if len(coords) > 0:
                    # 计算该组织的质心坐标，由于将组织区域看成均匀分布，所以质心坐标就是形心坐标
                    cy, cx = coords.mean(axis=0).astype(int)

                    # 计算裁剪框左上角，并做边界安全防越界处理
                    ymin = max(0, cy - ch // 2)
                    if ymin + ch > h:
                        ymin = max(0, h - ch)
                    xmin = max(0, cx - cw // 2)
                    if xmin + cw > w:
                        xmin = max(0, w - cw)

                    return {"ymin": ymin, "xmin": xmin}

        # 兜底逻辑：传统的随机裁剪
        ymin = random.randint(0, h - ch)
        xmin = random.randint(0, w - cw)
        return {"ymin": ymin, "xmin": xmin}

    @property
    def targets_as_params(self):
        # 核心：声明本组件的参数计算依赖于 mask 传入
        return ["mask"]

    def get_transform_init_args_names(self):
        return ("height", "width", "p_rare", "rare_classes")


def add_speckle_noise(image, **kwargs):
    img_float = image.astype(np.float32)
    noise = np.random.randn(*img_float.shape)
    noisy_img = img_float + img_float * 0.08 * noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def data_augment_pipeline(input_size=[512, 512],
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225],
                          seed=42,
                          version=2,
                          rare_classes=None,
                          augment_cfg_save_path=None):

    if version == 3:
        assert rare_classes is not None, "rare_classes must be provided for version 3"

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

        A.Affine(scale=(0.8, 1.8),
                 translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 rotate=(-15, 15),
                 p=0.4),
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
            A.Lambda(name='add_speckle_noise', image=add_speckle_noise, p=1.0),
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

    train_pipeline3 = A.Compose([
        # ==============================
        # 第一阶段：空间几何形态 (Spatial)
        # ==============================
        A.OneOf([
            LabelGuidedCropAlbumentations(height=input_size[1],
                                          width=input_size[0],
                                          p_rare=0.95,
                                          rare_classes=rare_classes,
                                          p=0.85),
            A.Resize(height=input_size[0], width=input_size[1], p=0.15),
        ], p=1.0),

        A.Affine(scale=(0.8, 1.6),
                 translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 rotate=(-10, 10),
                 p=0.4),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=20, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=(-0.2, 0.2), p=1.0),
            A.OpticalDistortion(distort_limit=(-0.2, 0.2), p=1.0)
        ], p=0.3),

        # ==============================
        # 第二阶段：物理成像退化
        # ==============================
        # 1. 探头底噪 (Noise)
        A.OneOf([
            A.GaussNoise(std_range=(10.0 / 255, 30.0 / 255), p=1.0),
            A.Lambda(name='add_speckle_noise', image=add_speckle_noise, p=1.0),
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
            num_holes_range=(1, 2),
            hole_height_range=(1, max(2, input_size[0] // 24)),
            hole_width_range=(1, max(2, input_size[1] // 24)),
            fill=128,
            p=0.05
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
        try:
            A.save(train_pipeline2, augment_cfg_save_path) if augment_cfg_save_path is not None else None
        except (TypeError, AttributeError):
            pass
        return train_pipeline2, val_pipeline
    elif version == 3:
        try:
            A.save(train_pipeline3, augment_cfg_save_path) if augment_cfg_save_path is not None else None
        except (TypeError, AttributeError):
            pass
        return train_pipeline3, val_pipeline
    else:
        try:
            A.save(train_pipeline, augment_cfg_save_path) if augment_cfg_save_path is not None else None
        except (TypeError, AttributeError):
            pass
        return train_pipeline, val_pipeline
