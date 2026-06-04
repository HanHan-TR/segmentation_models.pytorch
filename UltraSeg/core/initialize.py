import torch
import random
import numpy as np
from typing import Optional

from torch import distributed as dist
import torch.nn as nn
import os.path as osp


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def init_random_seed(seed: Optional[int] = None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    seed = np.random.randint(2**31)  # 生成随机种子（0 ~ 2^31-1）

    # Make sure all ranks share the same random seed to prevent some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    # 分布式环境下通过广播同步种子
    rank, world_size = get_dist_info()
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_state_dict(module, state_dict, prefix='', strict=False, logger=None):
    """将状态字典加载到模块中。

    该方法修改自 :meth:torch.nn.Module.load_state_dict。
    默认将 strict 参数设为 False，且即使严格模式为 False 也会显示参数不匹配的提示信息。

    Args:
        module (Module): 接收权重的模型.
        state_dict (OrderedDict): 权重字典
        strict (bool): 是否严格检查state_dict的键与接收权重文件的模型的
            :meth:`~torch.nn.Module.state_dict` 方法返回的键完全相同。默认值: ``False``，即默认关闭严格模式
        logger (:obj:`logging.Logger`, optional): 用于记录错误信息的日志记录器。若未指定，将使用 print 函数输出

    Note:
        - unexpected_keys (list of str) 用于记录非预期的参数键，指存在于state_dict中但未被模型使用的参数键，即模型当前结构不需要的多余参数
        - all_missing_keys (list of str) 用于记录所有缺失的参数键，指模型需要的但state_dict中缺少的参数键，即当前模型结构中未被初始化的必需参数
        需要注意的是， all_missing_keys和unexpected_keys只会反映出权重的键（名称）不同的情况，而不能反映权重向量维度不匹配的情况.
        有关向量维度不匹配的信息将由PyTorch底层的加载方法自动填充到err_msg中打印输出到终端。
        - err_msg (list of str) 错误信息缓存
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)  # 获取模型元数据
    state_dict = state_dict.copy()  # 创建副本避免污染原始数据
    if metadata is not None:
        state_dict._metadata = metadata  # 保持元数据完整

    def load(module, prefix=prefix):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        # 调用PyTorch底层加载方法
        module._load_from_state_dict(state_dict,
                                     prefix,
                                     local_metadata,
                                     True,
                                     all_missing_keys,
                                     unexpected_keys,
                                     err_msg)

        for name, child in module._modules.items():  # 递归处理子模块
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # 打破 load->load 的引用循环

    missing_keys = [
        # ignore "num_batches_tracked" of BN layers
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in model state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_ckpt(model, ckpt, weights_only=True):
    filename = ckpt
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, weights_only=weights_only)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict)
