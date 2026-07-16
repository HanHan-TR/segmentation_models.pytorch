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


def medical_in_place_copy_paste(src_image, src_mask, dst_image, dst_mask, target_class, blend=True):
    if (target_class not in src_mask) or (target_class in dst_mask):
        # 两个跳过条件：
        # • 源掩码中不存在目标类别 → 无东西可复制
        # • 目标掩码中已存在目标类别 → 避免重叠/冲突
        # 满足任一条件则直接返回目标图像的副本，不做任何修改
        return dst_image.copy(), dst_mask.copy()

    obj_mask = (src_mask == target_class).astype(np.uint8)
    out_image = dst_image.copy()
    out_mask = dst_mask.copy()
    # 将源图像标签中的目标类别的像素复制到目标图像的标签中的相同位置
    out_mask[obj_mask == 1] = target_class

    # 图像粘贴，分两种模式：blend=True（边缘混合粘贴）和blend=False（直接粘贴）模式
    if blend:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask = cv2.dilate(obj_mask, kernel, iterations=1)
        blur_mask = cv2.GaussianBlur(dilated_mask.astype(float), (5, 5), 0)
        if len(out_image.shape) == 3:
            blur_mask = np.expand_dims(blur_mask, axis=-1)
            obj_mask_3d = np.expand_dims(obj_mask, axis=-1)
        else:
            obj_mask_3d = obj_mask
        mask_fg = obj_mask_3d.astype(float) + (1 - obj_mask_3d.astype(float)) * blur_mask
        out_image = (mask_fg * src_image + (1.0 - mask_fg) * out_image).astype(np.uint8)
    else:
        # 直接粘贴：将源图像中的目标类别的像素直接粘贴到目标图像中相同位置，边界会有明显的接缝
        out_image[obj_mask == 1] = src_image[obj_mask == 1]
    return out_image, out_mask


def ultrasound_in_place_copy_paste(src_image,
                                   src_mask,
                                   dst_image,
                                   dst_mask,
                                   target_class,
                                   blend=True,
                                   intensity_correction=True,
                                   shadow_extension=True,
                                   shadow_pixels=15,
                                   speckle_uniform=True,
                                   speckle_std=0.04):
    if (target_class not in src_mask) or (target_class in dst_mask):
        # 两个跳过条件：
        # • 源掩码中不存在目标类别 → 无东西可复制
        # • 目标掩码中已存在目标类别 → 避免重叠/冲突
        # 满足任一条件则直接返回目标图像的副本，不做任何修改
        return dst_image.copy(), dst_mask.copy()

    obj_mask = (src_mask == target_class).astype(np.uint8)
    out_image = dst_image.copy()
    out_mask = dst_mask.copy()

    paste_mask = obj_mask.copy()
    if shadow_extension:
        shadow = np.zeros_like(obj_mask)
        rows = np.any(obj_mask, axis=1)
        if rows.any():
            top_row = np.argmax(rows)
            bottom_row = len(rows) - np.argmax(rows[::-1])
            shadow_top = bottom_row
            shadow_bottom = min(obj_mask.shape[0], bottom_row + shadow_pixels)
            if shadow_bottom > shadow_top:
                cols_in_obj = np.any(obj_mask[top_row:bottom_row], axis=0)
                col_start = np.argmax(cols_in_obj)
                col_end = len(cols_in_obj) - np.argmax(cols_in_obj[::-1])
                shadow[shadow_top:shadow_bottom, col_start:col_end] = 1
                shadow = cv2.GaussianBlur(shadow, (5, 5), 0)
                shadow = (shadow > 0.3).astype(np.uint8)
                shadow[obj_mask == 1] = 0
                paste_mask = np.clip(obj_mask + shadow, 0, 1).astype(np.uint8)

    src_adjusted = src_image.copy()
    if intensity_correction:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        neighborhood = cv2.dilate(paste_mask, kernel_dilate, iterations=10)
        neighborhood[paste_mask == 1] = 0

        src_pixels = src_image[paste_mask == 1].astype(np.float64)
        dst_pixels = dst_image[neighborhood == 1].astype(np.float64)

        if len(src_pixels) > 0 and len(dst_pixels) > 0:
            src_mean = src_pixels.mean()
            src_std = max(src_pixels.std(), 1e-6)
            dst_mean = dst_pixels.mean()
            dst_std = max(dst_pixels.std(), 1e-6)

            src_float = src_image.astype(np.float64)
            normalized = (src_float - src_mean) / src_std
            adjusted = normalized * dst_std + dst_mean
            src_adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    out_mask[paste_mask == 1] = target_class

    if blend:
        use_seamless = (
            len(out_image.shape) == 3
            and out_image.shape[2] == 3
            and paste_mask.sum() > 0
        )
        if use_seamless:
            try:
                coords = np.argwhere(paste_mask > 0)
                center_y = int(coords[:, 0].mean())
                center_x = int(coords[:, 1].mean())
                center = (center_x, center_y)

                mask_255 = (paste_mask * 255).astype(np.uint8)
                out_image = cv2.seamlessClone(
                    src_adjusted, out_image, mask_255, center, cv2.NORMAL_CLONE
                )
            except cv2.error:
                use_seamless = False

        if not use_seamless:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated_mask = cv2.dilate(paste_mask, kernel, iterations=1)
            blur_mask = cv2.GaussianBlur(dilated_mask.astype(float), (5, 5), 0)
            if len(out_image.shape) == 3:
                blur_mask = np.expand_dims(blur_mask, axis=-1)
                paste_mask_3d = np.expand_dims(paste_mask, axis=-1)
            else:
                paste_mask_3d = paste_mask
            mask_fg = paste_mask_3d.astype(float) + (
                1 - paste_mask_3d.astype(float)
            ) * blur_mask
            out_image = (
                mask_fg * src_adjusted + (1.0 - mask_fg) * out_image
            ).astype(np.uint8)
    else:
        out_image[paste_mask == 1] = src_adjusted[paste_mask == 1]

    if speckle_uniform and paste_mask.sum() > 0:
        paste_region = out_image[paste_mask == 1].astype(np.float32)
        if len(paste_region) > 0:
            noise = np.random.randn(*paste_region.shape).astype(np.float32)
            paste_region = paste_region + paste_region * speckle_std * noise
            out_image[paste_mask == 1] = np.clip(paste_region, 0, 255).astype(np.uint8)

    return out_image, out_mask


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
                          rare_classes=None):

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

        A.Affine(scale=(0.7, 1.2),
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

    train_pipeline3 = A.Compose([
        # ==============================
        # 第一阶段：空间几何形态 (Spatial)
        # ==============================
        A.OneOf([
            LabelGuidedCropAlbumentations(height=input_size[1],
                                          width=input_size[0],
                                          p_rare=0.9,
                                          rare_classes=rare_classes,
                                          p=1.0),
            A.Resize(height=input_size[0], width=input_size[1], p=1.0),
        ], p=1.0),

        A.Affine(scale=(0.7, 1.2),
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
    elif version == 3:
        return train_pipeline3, val_pipeline
    else:
        return train_pipeline, val_pipeline
