import cv2
import numpy as np


# 1. 固定 ROI 参数

ROI = {
    "x1": 0,
    "y1": 70,
    "x2": 1152,
    "y2": 920
}

MODEL_INPUT_SIZE = (512, 512)   # (w, h)


def crop_ultrasound_roi(image, roi=ROI):
    """
    从原图中裁剪超声有效区域
    image: H x W x C
    return:
        roi_img: 裁剪后的 ROI 图像
        roi_box: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    roi_img = image[y1:y2, x1:x2].copy()
    return roi_img, (x1, y1, x2, y2)


def preprocess_for_model(image_path, roi=ROI, model_input_size=MODEL_INPUT_SIZE):
    """
    预处理：
    1) 读取原图
    2) 裁剪 ROI
    3) resize 到模型输入尺寸
    4) 归一化
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]
    if (orig_w, orig_h) != (1270, 920):
        print(f"[警告] 当前图像尺寸为 {(orig_w, orig_h)}，与预期 (1270, 920) 不同，请检查 ROI 是否仍然适用。")

    roi_img, roi_box = crop_ultrasound_roi(image_bgr, roi)

    # resize 到模型输入大小
    input_img = cv2.resize(roi_img, model_input_size, interpolation=cv2.INTER_LINEAR)

    # 归一化到 [0,1]
    input_img = input_img.astype(np.float32) / 255.0

    # HWC -> CHW
    input_tensor = np.transpose(input_img, (2, 0, 1))

    # 增加 batch 维度: 1 x C x H x W
    input_tensor = np.expand_dims(input_tensor, axis=0)

    meta = {
        "orig_size": (orig_h, orig_w),
        "roi_box": roi_box,
        "roi_size": (roi_img.shape[0], roi_img.shape[1]),  # (h, w)
        "model_input_size": (model_input_size[1], model_input_size[0])  # (h, w)
    }

    return input_tensor, image_bgr, meta
