import platform
import sys
import torch

TQDM_BAR_FORMAT = '{l_bar}{bar:16}{r_bar}'


def log_write(file, message=''):
    with open(file, 'a') as file:
        file.write(message)


def get_environment_info(logfile):
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
        env_info["Current Device"] = torch.cuda.current_device()
        env_info["Device Name"] = torch.cuda.get_device_name()
        env_info["GPU Memory (MB)"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f}"
    else:
        env_info["CUDA Available"] = "No"

    result = "\n".join(f"{key + ':':<20} {value}" for key, value in env_info.items())
    result += '\n'

    title = "=============================== env info ====================================== \n"
    log_write(logfile, title)
    log_write(logfile, result)
