import argparse
import openvino as ov
import torch
import torch.nn as nn
import os
import sys
from pathlib import Path, PosixPath
from typing import Union, Optional

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from segmentation_models_pytorch import create_model
from UltraSeg.core.fileio import yaml_load


def convert_model(model: nn.Module,
                  ckpt_path: Optional[Union[str, PosixPath]] = None,
                  output_path: Optional[Union[str, PosixPath]] = None,
                  compress_to_fp16: bool = True,
                  save_ov_model: bool = True,
                  input_size: int = 512):
    """
    将PyTorch模型转换为OpenVINO模型

    Args:
        model (nn.Module): 要转换的PyTorch模型
        ckpt_path (Optional[Union[str, PosixPath]]): 模型权重文件路径
            如果为None，则不加载权重文件
        output_path (Optional[Union[str, PosixPath]]): 转换后模型的保存路径
            如果为None，则根据ckpt_path或使用默认名称
        compress_to_fp16 (bool): 是否将模型压缩为FP16格式
        save_ov_model (bool): 是否保存转换后的OpenVINO模型

    Returns:
        ov.Model: 转换后的OpenVINO模型
    """
    # 将模型设置为评估模式
    model.eval()

    # 如果提供了权重文件路径，加载模型参数
    if Path(ckpt_path).is_file():
        # 加载模型权重，使用CPU作为设备
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # 检查权重文件中是否包含'model_state_dict'键
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            # 直接加载权重（适用于简单的权重文件）
            model.load_state_dict(ckpt)

    # 使用OpenVINO转换模型
    example_input = torch.randn((1, 3, input_size, input_size))  # 根据需要调整输入大小
    ov_model = ov.convert_model(model, example_input=example_input)

    # 如果需要保存转换后的模型
    if save_ov_model:
        # 根据不同情况确定保存路径
        if output_path is not None:
            # 使用用户指定的输出路径
            ov.save_model(ov_model,
                          str(output_path),
                          compress_to_fp16=compress_to_fp16)
        elif Path(ckpt_path).is_file():
            # 使用与权重文件相同的路径，将后缀改为.xml
            ov.save_model(ov_model,
                          str(Path(ckpt_path).with_suffix('.xml')),
                          compress_to_fp16=compress_to_fp16)
        else:
            # 使用默认名称保存
            ov.save_model(ov_model,
                          'ov_model.xml',
                          compress_to_fp16=compress_to_fp16)

    # 返回转换后的OpenVINO模型
    return ov_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--input_size', type=int, default=512, help='input size for training and validation, e.g. 512')
    parser.add_argument('--model_cfg', type=str, default='UltraSeg/config/network/unet-mobilenetv2.yaml', help='model config file')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes for training')
    parser.add_argument('--att_type', type=str, default='scse', help='decoder attention type for training, none or scse')
    parser.add_argument('--load_from_ckpt', type=str, default='res/saved/scse-no-roi-512-p6/weights/ema_best.pth', help='load from checkpoint')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    opts = parse_args()
    model_cfg = yaml_load(opts.model_cfg)
    model = create_model(arch=model_cfg['arch'],
                         encoder_name=model_cfg['encoder_name'],
                         encoder_weights=model_cfg['encoder_weights'],
                         decoder_attention_type=opts.att_type,
                         in_channels=3,
                         classes=opts.num_classes)
    convert_model(model,
                  ckpt_path=opts.load_from_ckpt,
                  compress_to_fp16=False,
                  save_ov_model=True,
                  input_size=opts.input_size)
