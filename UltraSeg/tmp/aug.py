import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

# Albumentations / AlbumentationsX
import albumentations as A
from albumentations.pytorch import ToTensorV2  # 需要安装 albumentationsx + torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# 可复现性：全局随机种子（PyTorch层面）
# 注意：Albumentations 的复现主要靠 A.Compose(seed=...)，而不是 numpy/random 全局种子
# 参考其 reproducibility 指南：Compose 的 seed 才是关键
# ----------------------------
def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让 CUDA 算子更确定性（会变慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# 自定义：非零区域 z-score 归一化（对超声扇形黑边/背景更鲁棒）
# 参考 nnU-Net 预处理思想：通常会基于非零区域做统计，并强调归一化与重采样顺序
# ----------------------------
class ZScoreNormalizeNonZero(A.ImageOnlyTransform):
    def __init__(self, eps: float = 1e-6, nonzero_threshold: float = 0.0, p: float = 1.0):
        super().__init__(p=p)
        self.eps = eps
        self.nonzero_threshold = nonzero_threshold

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # img: HxWxC, float32 in [0,1] recommended
        assert img.ndim == 3, "Expect HxWxC"
        x = img[..., 0]  # grayscale
        mask = x > self.nonzero_threshold
        if mask.sum() < 16:
            # fallback：整图统计
            mu = float(x.mean())
            sigma = float(x.std())
        else:
            mu = float(x[mask].mean())
            sigma = float(x[mask].std())
        x = (x - mu) / (sigma + self.eps)
        img[..., 0] = x
        return img


# ----------------------------
# 自定义：Time-Gain Compensation (TGC) / 深度增益变化
# 说明：Speckle & Shadows 将 TGC 作为模仿传统超声图像质量参数的一部分
# 这里只实现一个轻量“随深度线性/指数增益”版本
# ----------------------------
class RandomTGC(A.ImageOnlyTransform):
    def __init__(
        self,
        gain_range: Tuple[float, float] = (0.7, 1.4),
        mode: str = "exp",  # "linear" or "exp"
        p: float = 0.15,
    ):
        super().__init__(p=p)
        self.gain_range = gain_range
        self.mode = mode

    def apply(self, img: np.ndarray, gain: float = 1.0, **params) -> np.ndarray:
        h, w, c = img.shape
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]  # (H,1)
        if self.mode == "linear":
            curve = 1.0 + (gain - 1.0) * (y - 0.5) * 2.0
        else:
            # exp：更像深度衰减/补偿
            k = np.log(max(gain, 1e-3))
            curve = np.exp(k * (y - 0.5) * 2.0).astype(np.float32)

        out = img.copy()
        out[..., 0] = np.clip(out[..., 0] * curve, 0.0, 1.0)
        return out

    def get_params(self) -> Dict[str, Any]:
        # AlbumentationsX 推荐使用 transform 内部随机状态（而不是全局 random）
        rng = getattr(self, "py_random", None)
        if rng is None:
            # fallback（旧版本兼容）：会弱化严格复现
            rng = random
        gain = rng.uniform(self.gain_range[0], self.gain_range[1])
        return {"gain": gain}


# ----------------------------
# 自定义：声影（Acoustic Shadow）仿真（轻量版）
# Speckle & Shadows 强调声影是超声最常见伪影之一，并提出人工声影增强
# 这里用“随机竖向带 + 随深度衰减”的简化模型
# ----------------------------
class RandomAcousticShadow(A.ImageOnlyTransform):
    def __init__(
        self,
        strength_range: Tuple[float, float] = (0.3, 0.8),  # 强度（越大越黑）
        width_frac_range: Tuple[float, float] = (0.05, 0.20),  # 阴影带宽占比
        p: float = 0.12,
    ):
        super().__init__(p=p)
        self.strength_range = strength_range
        self.width_frac_range = width_frac_range

    def apply(self, img: np.ndarray, x0: int = 0, w: int = 10, strength: float = 0.5, **params) -> np.ndarray:
        h, W, c = img.shape
        out = img.copy()

        # 横向 gaussian 形状（阴影最强在中心，边缘渐弱）
        xs = np.arange(W, dtype=np.float32)
        center = x0 + w / 2.0
        sigma = max(w / 3.0, 1.0)
        gx = np.exp(-0.5 * ((xs - center) / sigma) ** 2).astype(np.float32)  # (W,)
        gx = gx[None, :]  # (1,W)

        # 随深度增强阴影：越深越黑
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]  # (H,1)
        depth = y ** 1.2  # 可调
        atten = 1.0 - strength * (gx * depth)  # (H,W)
        atten = np.clip(atten, 0.0, 1.0)

        out[..., 0] = np.clip(out[..., 0] * atten, 0.0, 1.0)
        return out

    def get_params_dependent_on_data(self, params: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        img = data["image"]
        h, W, _ = img.shape

        rng = getattr(self, "py_random", None)
        if rng is None:
            rng = random

        width_frac = rng.uniform(self.width_frac_range[0], self.width_frac_range[1])
        w = int(max(3, width_frac * W))
        x0 = rng.randint(0, max(0, W - w))
        strength = rng.uniform(self.strength_range[0], self.strength_range[1])
        return {"x0": x0, "w": w, "strength": strength}


@dataclass
class AugConfig:
    # 输入/裁剪尺寸（未指定：按任务改）
    out_size: Tuple[int, int] = (320, 320)

    # 复现：给 Compose 一个固定 seed；若设为 None 则每次运行序列不同
    albumentations_seed: Optional[int] = 137

    # ROI/不平衡：优先用 mask 引导裁剪
    use_foreground_crop: bool = True

    # 几何（保守范围，适配多数超声分割）
    rotate_limit_deg: float = 15.0
    scale_limit: float = 0.15
    translate_limit: float = 0.05
    p_affine: float = 0.70
    p_flip: float = 0.30  # 超声左右含义依任务调整（心脏某些视角需谨慎）

    # 弹性形变：低概率、轻量
    p_elastic: float = 0.10

    # 强度/对比度
    p_brightness_contrast: float = 0.25
    brightness_limit: float = 0.15
    contrast_limit: float = 0.15

    p_gamma: float = 0.20
    gamma_limit: Tuple[int, int] = (80, 120)

    # 噪声：乘性噪声优先（贴近 speckle 乘性模型）
    p_multiplicative_noise: float = 0.20
    multiplier: Tuple[float, float] = (0.85, 1.15)

    p_gauss_noise: float = 0.08
    gauss_std_range: Tuple[float, float] = (0.0, 0.06)  # 对 float[0,1] 更温和

    # 质量退化
    p_blur: float = 0.10
    p_downscale: float = 0.10

    # 超声伪影
    p_tgc: float = 0.15
    p_shadow: float = 0.12

    # 输出归一化（训练更稳定）
    use_zscore_nonzero: bool = True


def build_train_transform(cfg: AugConfig) -> A.Compose:
    h, w = cfg.out_size

    # 关键：mask_interpolation 使用最近邻，否则类别会被插值污染
    # 参考 Albumentations 的 mask_interpolation 参数
    transform_list: List[A.BasicTransform] = []

    # 1) 尺寸规范化 + ROI裁剪（优先解决小目标/不平衡）
    # CropNonEmptyMaskIfExists：mask非空就围绕mask裁剪，否则随机裁剪
    if cfg.use_foreground_crop:
        transform_list.append(A.CropNonEmptyMaskIfExists(height=h, width=w, p=1.0))
    else:
        transform_list.append(A.RandomCrop(height=h, width=w, p=1.0))

    # 2) 几何增强
    transform_list.extend([
        A.HorizontalFlip(p=cfg.p_flip),
        A.Affine(rotate=(-cfg.rotate_limit_deg, cfg.rotate_limit_deg),
                 scale=(1.0 - cfg.scale_limit, 1.0 + cfg.scale_limit),
                 translate_percent=(-cfg.translate_limit, cfg.translate_limit),
                 shear=(-10, 10),
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST,
                 mode=cv2.BORDER_CONSTANT,
                 cval=0,
                 cval_mask=0,
                 p=cfg.p_affine),
        A.OneOf([A.ElasticTransform(alpha=10.0,
                                    sigma=6.0,
                                    interpolation=cv2.INTER_LINEAR,
                                    mask_interpolation=cv2.INTER_NEAREST,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    value=0,
                                    mask_value=0,
                                    approximate=True,
                                    p=1.0),
                 A.GridDistortion(num_steps=5,
                                  interpolation=cv2.INTER_LINEAR,
                                  mask_interpolation=cv2.INTER_NEAREST,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  value=0, mask_value=0,
                                  p=1.0),
                 ],
                p=cfg.p_elastic),
    ])

    # 3) 强度/噪声/质量
    transform_list.extend([
        A.RandomBrightnessContrast(
            brightness_limit=cfg.brightness_limit,
            contrast_limit=cfg.contrast_limit,
            p=cfg.p_brightness_contrast,
        ),
        A.RandomGamma(gamma_limit=cfg.gamma_limit, p=cfg.p_gamma),

        # 乘性噪声：对强度做乘性扰动（更像 speckle 的扰动方式）
        A.MultiplicativeNoise(multiplier=cfg.multiplier, p=cfg.p_multiplicative_noise),
        A.GaussNoise(std_range=cfg.gauss_std_range, p=cfg.p_gauss_noise),

        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.2, 1.0), p=cfg.p_blur),
        A.Downscale(scale_range=(0.5, 0.9), interpolation_pair=None, p=cfg.p_downscale),

        # 4) 超声特定伪影（低概率）
        RandomTGC(p=cfg.p_tgc),
        RandomAcousticShadow(p=cfg.p_shadow),
    ])

    # 5) 归一化与张量化
    # 先把 uint8 转 float[0,1]，再做可选 z-score（非零区域更鲁棒）
    transform_list.extend([
        A.ToFloat(max_value=255.0, p=1.0),
        ZScoreNormalizeNonZero(p=1.0) if cfg.use_zscore_nonzero else A.NoOp(),
        ToTensorV2(transpose_mask=True),
    ])

    return A.Compose(transform_list, seed=cfg.albumentations_seed)


def build_val_transform(cfg: AugConfig) -> A.Compose:
    # 验证/测试：只做确定性步骤（裁剪/resize/归一化），避免引入随机性
    h, w = cfg.out_size
    t = A.Compose([
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.ToFloat(max_value=255.0, p=1.0),
        ZScoreNormalizeNonZero(p=1.0) if cfg.use_zscore_nonzero else A.NoOp(),
        ToTensorV2(transpose_mask=True),
    ], seed=cfg.albumentations_seed)
    return t


class UltrasoundSegDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transform: Optional[A.Compose] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.filenames = sorted([
            fn for fn in os.listdir(image_dir)
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fn = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fn)
        mask_path = os.path.join(self.mask_dir, fn)

        # 读取灰度图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        img = img[..., None]  # HxWx1

        # 读取mask（单通道，像素值=类别ID）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img_t = aug["image"]          # torch.FloatTensor: (C,H,W)
            mask_t = aug["mask"].long()   # torch.LongTensor: (H,W)
        else:
            # 最小fallback（不建议用于训练）
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t, fn


# 一个极简 U-Net（可运行示例，未针对性能优化）
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def save_offline_augmented_samples(
    dataset: UltrasoundSegDataset,
    out_image_dir: str,
    out_mask_dir: str,
    num_aug_per_sample: int = 2,
) -> None:
    """
    可选：离线增强保存（用于审计/复现实验/小数据快速扩充）
    注意：会占用磁盘；建议只保存“增强后样本”或保存少量用于可视化检查
    """
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # dataset.transform 内含随机；若 cfg.albumentations_seed 固定且遍历顺序固定，可复现
    for i in range(len(dataset)):
        _, _, fn = dataset[i]  # 仅取文件名
        # 重新读取原图、原mask，以便多次增强
        img_path = os.path.join(dataset.image_dir, fn)
        mask_path = os.path.join(dataset.mask_dir, fn)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[..., None]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for k in range(num_aug_per_sample):
            aug = dataset.transform(image=img, mask=mask)
            img_t = aug["image"].cpu().numpy()  # (C,H,W)
            mask_t = aug["mask"].cpu().numpy().astype(np.uint8)  # (H,W)

            # 反归一化到可视化范围：这里只做一个简化映射（z-score -> min-max）
            x = img_t[0]
            x = (x - x.min()) / (x.max() - x.min() + 1e-6)
            x_u8 = (x * 255.0).clip(0, 255).astype(np.uint8)

            base = os.path.splitext(fn)[0]
            out_fn = f"{base}_aug{k:02d}.png"
            cv2.imwrite(os.path.join(out_image_dir, out_fn), x_u8)
            cv2.imwrite(os.path.join(out_mask_dir, out_fn), mask_t)


def train_one_epoch(model, loader, optimizer, device, num_classes: int):
    model.train()
    ce = nn.CrossEntropyLoss()

    total = 0.0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)        # (B,1,H,W)
        masks = masks.to(device)      # (B,H,W)

        logits = model(imgs)          # (B,C,H,W)
        loss = ce(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
    return total / max(len(loader), 1)


def main():
    # ----------------------------
    # 未指定：路径与类别数（按你数据修改）
    # ----------------------------
    image_dir = "./images"
    mask_dir = "./masks"
    num_classes = 2

    cfg = AugConfig(out_size=(320, 320), albumentations_seed=137)
    set_torch_seed(137)

    train_tf = build_train_transform(cfg)
    val_tf = build_val_transform(cfg)

    dataset = UltrasoundSegDataset(image_dir, mask_dir, transform=train_tf)

    # DataLoader 的 num_workers 会影响增强序列；A/B 对比务必固定
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=1, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(3):
        # 若希望“每个 epoch 用不同但可复现的增强序列”，可以：
        # train_tf.set_random_seed(cfg.albumentations_seed + epoch)
        loss = train_one_epoch(model, loader, optimizer, device, num_classes)
        print(f"Epoch {epoch}: loss={loss:.4f}")

    # 可选：离线增强保存
    # save_offline_augmented_samples(dataset, "./aug_images", "./aug_masks", num_aug_per_sample=2)


if __name__ == "__main__":
    main()
