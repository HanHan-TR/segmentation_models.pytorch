import albumentations as A
import cv2


def data_augment_pipeline(input_size=[512, 512],
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225],
                          seed=42):
    train_pipeline = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Affine(rotate=(-15.0, 15.0),
                 scale=(1.0 - 0.15, 1.0 + 0.15),
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
        # A.Downscale(scale_range=(0.5, 0.9), interpolation_pair=None, p=0.1),

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
