import platform
import sys
import os
import torch
from datetime import datetime
from typing import Optional, List, Union
from pathlib import Path, PosixPath
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from UltraSeg.core.fileio import yaml_load, object_to_dict

try:
    from thop import profile, clever_format
except ImportError:
    raise ImportError("Please install thop: pip install thop")

TQDM_BAR_FORMAT = '{l_bar}{bar:16}{r_bar}'


def log_write(message='', logfile: Optional[str] = None):
    if logfile is None:
        print(message)
    else:
        with open(logfile, 'a') as f:
            f.write(message)


import io
import contextlib


def log_model_info(model: torch.nn.Module,
                   input_size: Union[int, List[int]] = [384, 384],
                   logfile: Optional[str] = None):
    """
    测量并记录模型的参数量、计算量和模型大小

    Args:
        model (torch.nn.Module): 要测量的模型
        input_size (Union[int, List[int]]): 输入图像的尺寸 [height, width]
        logfile (Optional[str]): 日志文件路径，默认为None（打印到控制台）
    """
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(input_size, int):
        input_size = [input_size, input_size]

    input_tensor = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    # 深拷贝模型，避免 thop.profile() 在原始模型上添加临时属性
    model_copy = copy.deepcopy(model)
    model_copy.to(device).eval()

    # 1. 使用 thop 计算总参数量和总 FLOPs
    # 通过临时重定向 stdout 和 stderr 来禁用 thop 的 INFO 输出
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        flops_total, params_total = profile(model_copy, inputs=(input_tensor,))
    flops_total, params_total = clever_format([flops_total, params_total], "%.3f")

    # 2. 计算模型大小（MB）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # 参数数量 * 每个参数的字节数
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2  # 转换为 MB

    # 输出信息
    log_write(f"{'input size:':<28} {input_size[0]}x{input_size[1]}", logfile=logfile)
    log_write(f"{'number of parameters:':<28} {params_total}", logfile=logfile)
    log_write(f"{'computation cost:':<28} {flops_total}FLOPs", logfile=logfile)
    log_write(f"{'model size:':<28} {size_all_mb:.2f}MB\n", logfile=logfile)


def get_environment_info(logfile=None):
    """收集并返回训练环境信息字典"""
    env_info = {}

    # 1. 操作系统信息
    env_info["Platform"] = platform.system()  # Windows/Linux/macOS
    env_info["OS Version"] = platform.platform()

    # 2. Python环境
    env_info["Python Version"] = sys.version.split()[0]  # 取主版本号

    # 3. PyTorch环境
    env_info["PyTorch Version"] = torch.__version__

    # 4. GPU和CUDA信息
    if torch.cuda.is_available():
        env_info["CUDA Available"] = "Yes"
        env_info["CUDA Version"] = torch.version.cuda
        env_info["Number of GPUs"] = torch.cuda.device_count()
        # env_info["Current Device"] = torch.cuda.current_device()
        env_info["Device Name"] = torch.cuda.get_device_name()
        env_info["GPU Memory (MB)"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f}"
    else:
        env_info["CUDA Available"] = "No"

    result = "\n".join(f"{key + ':':<25} {value}" for key, value in env_info.items())
    result += '\n'

    title = "=============================== environment info ====================================== \n"
    log_write(title, logfile=logfile)
    log_write(result, logfile=logfile)


def get_experiment_info(cfg_dir: PosixPath,
                        model: torch.nn.Module,
                        input_size: Union[int, List[int]] = [384, 384],
                        logfile: Optional[str] = None):
    dataset_cfg = yaml_load(cfg_dir / 'dataset.yaml')
    title = "=============================== dataset info ====================================== \n"
    log_write(title, logfile=logfile)

    for key, value in dataset_cfg.items():
        if isinstance(value, list):
            log_write(f"{key}:", logfile=logfile)
            for item in value:
                if isinstance(item, list):
                    item_str = ', '.join(str(i) for i in item)
                    log_write(f"  - {item_str}", logfile=logfile)
                else:
                    log_write(f"  - {item}", logfile=logfile)
        else:
            log_write(f"{key  + ':':<20} {value}", logfile=logfile)

    title = "=============================== model info ====================================== \n"
    model_cfg = yaml_load(cfg_dir / 'model.yaml')
    log_write(title, logfile=logfile)
    for key, value in model_cfg.items():
        log_write(f"{key  + ':':<28} {value}", logfile=logfile)

    log_model_info(model, input_size=input_size, logfile=logfile)

    config_dict = yaml_load(cfg_dir / 'hyper.yaml')
    title = "=============================== training hyperparameters ====================================== \n"
    log_write(title, logfile=logfile)
    config_dict.pop('model_cfg')
    config_dict.pop('dataset_cfg')
    for key, value in config_dict.items():
        if key != 'exp_dir':
            log_write(f"{key  + ':':<28} {value}", logfile=logfile)

    log_write(f"\nThis running is saved at: {config_dict['exp_dir']}\n", logfile=logfile)

    exp_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_write(f"Training Start Time:   {exp_time}\n", logfile=logfile)
