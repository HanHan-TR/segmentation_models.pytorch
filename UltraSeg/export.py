import argparse
import torch
import torch.nn as nn
import openvino as ov
import nncf
import os
import sys
from pathlib import Path, PosixPath
from typing import Union, Optional, Callable, Any

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from UltraSeg.core.fileio import yaml_load
from UltraSeg.core.dataset import create_dataset
from UltraSeg.inference import load_model


def convert_to_onnx(model: nn.Module,
                    output_path: Union[str, PosixPath],
                    input_size: int = 512,
                    opset_version: int = 11):
    """
    将PyTorch模型转换为ONNX格式

    Args:
        model: PyTorch模型
        output_path: ONNX模型保存路径
        input_size: 输入图像尺寸（默认512）
        opset_version: ONNX opset版本（默认11）
    """
    model.eval()
    example_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(model,
                      example_input,
                      str(output_path),
                      opset_version=opset_version,
                      dynamo=False,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None
                      )
    print(f"ONNX模型已保存到: {output_path}")


def convert_onnx_to_openvino(onnx_path: Union[str, PosixPath],
                             output_path: Union[str, PosixPath],
                             compress_to_fp16: bool = True):
    """
    将ONNX模型转换为OpenVINO格式

    Args:
        onnx_path: ONNX模型路径
        output_path: OpenVINO模型保存路径
        compress_to_fp16: 是否压缩为FP16格式
    """
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(output_path), compress_to_fp16=compress_to_fp16)
    print(f"OpenVINO模型已保存到: {output_path}")
    return ov_model


def create_calibration_dataset(dataset_cfg: dict,
                               input_size: int,
                               split: str = 'val',
                               max_samples: int = 300) -> nncf.Dataset:
    """
    创建用于量化校准的数据集

    Args:
        dataset_cfg: 数据集配置字典
        input_size: 输入图像尺寸
        split: 数据集分割（train/val）
        max_samples: 最大样本数

    Returns:
        NNCF Dataset对象
    """
    dataset = create_dataset(
        dataset_cfg,
        input_size=input_size,
        split=split,
        augment_version=2
    )

    # 限制样本数量
    class LimitedDataset:
        def __init__(self, dataset, limit):
            self.dataset = dataset
            self.limit = min(limit, len(dataset))

        def __len__(self):
            return self.limit

        def __getitem__(self, index):
            image, _ = self.dataset[index]
            return image

    limited_dataset = LimitedDataset(dataset, max_samples)

    # 定义转换函数：将tensor转换为numpy并添加batch维度
    def transform_func(data_item):
        image = data_item
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        # 添加batch维度，使形状从(3, H, W)变为(1, 3, H, W)
        image = image[None, ...]
        return image

    return nncf.Dataset(limited_dataset, transform_func)


def quantize_int8(ov_model: ov.Model,
                  calibration_dataset: nncf.Dataset,
                  output_path: Union[str, PosixPath]):
    """
    使用NNCF进行INT8量化

    Args:
        ov_model: OpenVINO模型
        calibration_dataset: 校准数据集
        output_path: 量化后模型保存路径

    Returns:
        量化后的OpenVINO模型
    """
    quantized_model = nncf.quantize(ov_model, calibration_dataset)
    ov.save_model(quantized_model, str(output_path))
    print(f"INT8量化模型已保存到: {output_path}")
    return quantized_model


def quantize_accuracy_aware(ov_model: ov.Model,
                            calibration_dataset: nncf.Dataset,
                            validation_dataset: nncf.Dataset,
                            output_path: Union[str, PosixPath],
                            max_drop: float = 0.01,
                            evaluator_fn: Optional[Callable[..., float]] = None):
    """
    使用NNCF进行Accuracy-aware量化

    Args:
        ov_model: OpenVINO模型
        calibration_dataset: 校准数据集
        validation_dataset: 验证数据集
        output_path: 量化后模型保存路径
        max_drop: 最大精度下降容忍度
        evaluator_fn: 评估函数（可选）

    Returns:
        量化后的OpenVINO模型
    """
    # 如果未提供评估函数，使用默认的评估函数
    if evaluator_fn is None:
        def default_evaluator(model: ov.Model, dataset: nncf.Dataset) -> float:
            compiled_model = ov.compile_model(model)
            correct = 0
            total = 0
            for data_item in dataset:
                input_data = data_item
                if isinstance(input_data, torch.Tensor):
                    input_data = input_data.numpy()
                input_data = input_data[None, ...]  # 添加batch维度
                output = compiled_model(input_data)[0]
                pred = output.argmax(axis=1)
                # 默认评估：计算非背景类别的准确率
                correct += (pred != 0).sum()
                total += pred.size
            return correct / total if total > 0 else 0.0

        evaluator_fn = default_evaluator

    # 创建量化参数
    params = nncf.AwareQuantizationParameters(
        max_drop=max_drop,
        evaluator=evaluator_fn
    )

    # 执行Accuracy-aware量化
    quantized_model = nncf.quantize_with_accuracy_aware(
        ov_model,
        calibration_dataset,
        validation_dataset,
        params
    )

    ov.save_model(quantized_model, str(output_path))
    print(f"Accuracy-aware量化模型已保存到: {output_path}")
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description='模型转换与量化导出工具')

    # 基础参数
    parser.add_argument('--checkpoint', type=str,
                        default='/home/t_wanghan/work/segmentation_models.pytorch/res/wrist-seg/no-att-no-roi-hard-samp-384-p/weights/ema_best.pth',
                        help='PyTorch模型权重文件路径（.pth）')
    parser.add_argument('--dataset_cfg', type=str,
                        default='UltraSeg/config/dataset/wrist.yaml',
                        help='数据集配置文件路径')
    # ONNX转换参数
    parser.add_argument('--opset_version', type=int, default=13,
                        help='ONNX opset版本')

    # OpenVINO转换参数
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='是否将OpenVINO模型压缩为FP16格式')

    # 量化参数
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='是否对模型进行量化')
    parser.add_argument('--quantization_type', type=str, default='int8',
                        choices=['int8', 'accuracy_aware'],
                        help='量化类型：int8（默认）或accuracy_aware')
    parser.add_argument('--max_drop', type=float, default=0.01,
                        help='Accuracy-aware量化的最大精度下降容忍度')
    parser.add_argument('--calibration_samples', type=int, default=300,
                        help='校准数据集样本数')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.checkpoint).parent
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载PyTorch模型
    print("Loading PyTorch model...")
    model, input_size, _, _ = load_model(args.checkpoint)
    model_name = Path(args.checkpoint).stem

    # 2. 转换为ONNX
    print("Converting to ONNX...")
    name = model_name + '.onnx'
    onnx_path = Path(output_dir) / name
    convert_to_onnx(model, onnx_path, input_size, args.opset_version)

    # 3. 转换为OpenVINO
    print("Converting ONNX to OpenVINO...")
    name = model_name + '_fp16.xml' if args.fp16 else model_name + '_fp32.xml'
    ov_path = Path(output_dir) / name
    ov_model = convert_onnx_to_openvino(onnx_path, ov_path, args.fp16)

    # 4. 量化（如果需要）
    if args.quantize:
        print(f"Quantizing model with {args.quantization_type}...")

        # 创建校准数据集
        dataset_cfg = yaml_load(args.dataset_cfg)
        calibration_dataset = create_calibration_dataset(
            dataset_cfg,
            input_size,
            split='val',
            max_samples=args.calibration_samples
        )

        if args.quantization_type == 'int8':
            name = model_name + '_int8.xml'
            quantized_path = Path(output_dir) / name
            quantize_int8(ov_model, calibration_dataset, quantized_path)

        elif args.quantization_type == 'accuracy_aware':
            # 创建验证数据集（使用训练集的一部分）
            validation_dataset = create_calibration_dataset(
                dataset_cfg,
                input_size,
                split='val',
                max_samples=min(100, args.calibration_samples)
            )
            name = model_name + '_accuracy_aware.xml'
            quantized_path = Path(output_dir) / name
            quantize_accuracy_aware(
                ov_model,
                calibration_dataset,
                validation_dataset,
                quantized_path,
                args.max_drop
            )

    print("Export completed successfully!")


if __name__ == '__main__':
    main()
