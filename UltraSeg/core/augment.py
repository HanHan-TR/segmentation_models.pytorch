import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


class CutMix(DualTransform):
    def __init__(self, p=0.5, always_apply=False, cut_ratio_range=(0.45, 0.75)):
        super().__init__(always_apply, p)
        self.cut_ratio_range = cut_ratio_range

    def _get_cut_coordinates(self, h, w):
        """计算裁剪区域的坐标

        Args:
            h: 图像高度
            w: 图像宽度

        Returns:
            x1, x2, y1, y2: 裁剪区域的边界坐标
        """
        min_ratio, max_ratio = self.cut_ratio_range
        cut_h = np.random.randint(int(h * min_ratio), int(h * max_ratio))
        cut_w = np.random.randint(int(w * min_ratio), int(w * max_ratio))
        cx = np.random.randint(cut_h // 2, h - cut_h // 2)
        cy = np.random.randint(cut_w // 2, w - cut_w // 2)
        x1, x2 = cx - cut_h // 2, cx + cut_h // 2
        y1, y2 = cy - cut_w // 2, cy + cut_w // 2
        return x1, x2, y1, y2

    def apply(self, img, **params):
        img2 = params['image2']
        h, w = img.shape[:2]
        x1, x2, y1, y2 = self._get_cut_coordinates(h, w)
        # 区域替换
        img[x1:x2, y1:y2] = img2[x1:x2, y1:y2]
        return img

    def apply_to_mask(self, mask, **params):
        # 掩码同步区域替换
        mask2 = params['mask2']
        h, w = mask.shape[:2]
        x1, x2, y1, y2 = self._get_cut_coordinates(h, w)
        mask[x1:x2, y1:y2] = mask2[x1:x2, y1:y2]
        return mask


def data_augment_pipeline(input_size=[512, 512],
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225],
                          seed=42,
                          use_roi=False,
                          use_cutmix=False):
    if use_roi and use_cutmix:
        cut_mix_p = 0.45
    else:
        cut_mix_p = 0.0

    train_pipeline = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        CutMix(p=cut_mix_p),
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

        A.ToFloat(max_value=255.0),
        A.Normalize(mean=tuple(mean), std=tuple(std), max_pixel_value=1.0),
        A.ToTensorV2()
    ], seed=seed)

    val_pipeline = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.ToFloat(max_value=255.0),
        A.Normalize(mean=tuple(mean), std=tuple(std), max_pixel_value=1.0),
        A.ToTensorV2()
    ], seed=seed)

    return train_pipeline, val_pipeline
