import nncf
import torch
import torch.nn as nn
import openvino as ov
from typing import Optional, Callable, Any, Union
from .convert import convert_model
from pathlib import PosixPath


def quantize_model(model: nn.Module,
                   calibration_dataset: Union[nncf.Dataset, torch.utils.data.Dataset],
                   ckpt_path: Optional[Union[str, PosixPath]] = None,
                   transform_func: Optional[Callable[..., Any]] = None,):
    ov_model = convert_model(model, ckpt_path=ckpt_path, save_ov_model=False)
    if isinstance(calibration_dataset, torch.utils.data.Dataset):
        calibration_dataset = nncf.Dataset(calibration_dataset, transform_func)

    quantized_model = nncf.quantize(ov_model, calibration_dataset)
    model_name = f"quantized_{ckpt_path.name if ckpt_path else 'model'}.xml"
    model_name = str(PosixPath(ckpt_path).with_name(model_name)) if ckpt_path else model_name
    ov.save_model(quantized_model, model_name)
    return quantized_model
