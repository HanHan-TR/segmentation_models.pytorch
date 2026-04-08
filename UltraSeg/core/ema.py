import torch
import torch.nn as nn
import copy


class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        """
        model: 正在训练的模型
        decay: EMA 衰减系数，越接近 1 越平滑
        """
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()  # 影子模型

        # EMA 模型不需要梯度
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """
        在每次 optimizer.step() 之后调用
        """
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()

        for k, v in ema_state.items():
            # assert k in model_state, f"EMA model parameter {k} not found in model state dict"
            if v.dtype.is_floating_point:
                # 对浮点参数/缓冲区做 EMA
                v.mul_(self.decay).add_(model_state[k], alpha=1.0 - self.decay)
            else:
                # 对非浮点类型（如整型计数）直接拷贝
                v.copy_(model_state[k])

    def to(self, device):
        self.ema_model.to(device)
        return self.ema_model

    def state_dict(self):
        return self.ema_model.state_dict()

    def model(self):
        return self.ema_model
