import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import os
import sys
from PIL import Image, ImageDraw, ImageFont

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from segmentation_models_pytorch import create_model
from UltraSeg.core.fileio import yaml_load


def load_model(checkpoint_path):
    """
    加载训练好的模型
    """
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path)

    if 'meta_info' in checkpoint:
        meta_info = checkpoint['meta_info']
    else:
        meta_info = {}
        raise ValueError("Checkpoint is missing meta_info")

    arch = meta_info.get('arch')
    encoder_name = meta_info.get('encoder_name')
    decoder_attention_type = meta_info.get('decoder_attention_type', None)
    in_channels = meta_info.get('in_channels', 3)
    num_classes = meta_info.get('num_classes')

    input_size = meta_info.get('input_size')
    mean = meta_info.get('mean')
    std = meta_info.get('std')

    model = create_model(arch=arch,
                         encoder_name=encoder_name,
                         encoder_weights=None,  # 不加载预训练权重，使用训练好的权重
                         decoder_attention_type=decoder_attention_type,
                         in_channels=in_channels,
                         classes=num_classes
                         )
    # 处理不同格式的 checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)

    return model, input_size, mean, std


def preprocess_image(image_path, input_size, mean, std):
    """
    预处理图像：resize、归一化、转换为tensor
    """
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 保存原始图像用于可视化
    original_img = img.copy()

    # resize到模型输入尺寸
    if isinstance(input_size, int):
        input_size = [input_size, input_size]
    elif isinstance(input_size, list):
        assert len(input_size) == 2, "input_size must be a list of two integers"

    img_resized = cv2.resize(img, (input_size[1], input_size[0]))

    assert len(mean) == 3 and len(std) == 3, "mean and std must be lists of three floats"
    # 归一化
    img_resized = img_resized / 255.0
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)
    img_normalized = (img_resized - mean_arr) / std_arr

    # 转换为tensor
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().unsqueeze(0)

    return original_img, img_tensor


def inference(model, img_tensor, device):
    """
    执行推理
    """
    model.to(device).eval()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).long()

    return pred.cpu().numpy()[0]


CHINESE_FONT_PATH = os.path.expanduser("~/.local/share/fonts/wqy-microhei/wqy-microhei.ttc")


def _draw_class_labels(overlay, mask, class_names, color_map, orig_h, orig_w):
    """
    在每个类别区域的形心处绘制类别名称（支持中文）
    """
    mask_h, mask_w = mask.shape[:2]
    scale_x = orig_w / mask_w
    scale_y = orig_h / mask_h

    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)

    try:
        font = ImageFont.truetype(CHINESE_FONT_PATH, 24)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for class_id in range(len(color_map)):
        if class_id == 0:
            continue
        if class_names is None or class_id >= len(class_names):
            continue
        class_mask = (mask == class_id)
        if not np.any(class_mask):
            continue
        ys, xs = np.where(class_mask)
        cy = int(np.mean(ys) * scale_y)
        cx = int(np.mean(xs) * scale_x)

        class_name = class_names[class_id]
        draw.text((cx, cy), class_name, font=font, fill=(255, 255, 255), anchor="mm")

    overlay[:] = np.array(overlay_pil)


def visualize_result(original_img, pred_mask, gt_mask, color_map, class_names=None, alpha=0.6):
    """
    将预测结果和真实标签绘制到原图像上，并水平拼接，并在每个类别区域形心标注类别名称
    """
    h, w = original_img.shape[:2]

    # 处理预测mask
    pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(color_map):
        mask = pred_mask == class_id
        pred_colored[mask] = color
    pred_resized = cv2.resize(pred_colored, (w, h))
    pred_overlay = cv2.addWeighted(original_img, (1 - alpha), pred_resized, alpha, 0)

    # 添加标签文字
    cv2.putText(pred_overlay, 'Pred', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # 在预测mask上添加类别名称
    _draw_class_labels(pred_overlay, pred_mask, class_names, color_map, h, w)

    # 处理真实标签mask（如果提供）
    if gt_mask is not None:
        gt_colored = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(color_map):
            mask = gt_mask == class_id
            gt_colored[mask] = color
        gt_resized = cv2.resize(gt_colored, (w, h))
        gt_overlay = cv2.addWeighted(original_img, (1 - alpha), gt_resized, alpha, 0)
        cv2.putText(gt_overlay, 'GT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # 在GT mask上添加类别名称
        _draw_class_labels(gt_overlay, gt_mask, class_names, color_map, h, w)
    else:
        # 如果没有gt_mask，显示原图
        gt_overlay = original_img.copy()
        cv2.putText(gt_overlay, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 水平拼接
    combined = np.hstack((pred_overlay, gt_overlay))
    return combined


def main():
    parser = argparse.ArgumentParser(description='语义分割模型推理')
    parser.add_argument('--input', type=str,
                        default='/home/t_wanghan/work/data/wan_shortlong/JPEGImages/all',
                        help='输入图像路径或包含图像的文件夹路径')
    parser.add_argument('--gt', type=str,
                        default='/home/t_wanghan/work/data/wan_shortlong/Masks/all',
                        help='真实标签图像路径或包含标签的文件夹路径')
    parser.add_argument('--output', default='inference_results', help='输出文件夹路径')
    parser.add_argument('--checkpoint', type=str,
                        default='res/wrist-seg/best/timm-tf_efficientnet_lite1-p22-tuneWithBG/weights/ema_best.pth',
                        help='训练好的模型checkpoint路径')
    parser.add_argument('--dataset_cfg', default='UltraSeg/config/dataset/wan/wan_shortlong.yaml', help='数据集配置文件路径')
    parser.add_argument('--split', type=str, default='all', help='数据集划分，选择all/train/val/test')
    parser.add_argument('--input_size', type=int, default=384, help='模型输入尺寸')
    parser.add_argument('--device', default='0', help='cuda设备，如 0 或 cpu')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != 'cpu' else "cpu")

    # 加载模型
    model, input_size, mean, std = load_model(args.checkpoint)
    dataset_cfg = yaml_load(args.dataset_cfg)
    dataset_name = Path(args.dataset_cfg).stem
    class_names = dataset_cfg.get('classes', None)
    color_map = dataset_cfg.get('color_map', None)
    assert color_map is not None, "color_map must be provided in dataset_cfg"
    color_map = np.array(color_map, dtype=np.uint8)

    # 收集输入图像列表
    input_path = Path(args.input)
    if input_path.is_file():
        # 单张图像
        image_paths = [input_path]
    elif input_path.is_dir():
        # 文件夹中的所有图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        image_paths = sorted([p for p in input_path.iterdir()
                             if p.suffix.lower() in image_extensions])
    else:
        raise ValueError(f"输入路径不存在: {args.input}")

    # 获取mask路径（如果存在）
    mask_dir = Path(args.gt)

    # 创建输出文件夹
    model_name = Path(args.checkpoint).stem
    model_type = Path(args.checkpoint).suffix.replace('.', '_')
    out_folder_name = f"{args.output}_{model_name}{model_type}_{dataset_name}"
    output_dir = Path(args.checkpoint).parents[1] / out_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每张图像
    for img_path in image_paths:
        print(f"Processing {img_path}...")

        # 预处理图像
        input_size = args.input_size if args.input_size is not None else input_size
        original_img, img_tensor = preprocess_image(str(img_path), input_size, mean, std)

        # 执行推理
        pred_mask = inference(model, img_tensor, device)

        # 尝试加载对应的gt mask
        mask_path = mask_dir / img_path.name.replace(img_path.suffix, '.png')
        if mask_path.exists():
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = None

        # 可视化结果
        result = visualize_result(original_img, pred_mask, gt_mask, color_map, class_names)

        # 保存结果
        output_name = img_path.stem + '_pred_gt.jpg'
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Result saved to {output_path}")


if __name__ == '__main__':
    main()
