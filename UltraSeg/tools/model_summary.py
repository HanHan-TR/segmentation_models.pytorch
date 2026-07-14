import sys
import os
sys.path.insert(0, '/home/t_wanghan/work/segmentation_models.pytorch')
import torch
from segmentation_models_pytorch import create_model
from torchinfo import summary
from torchviz import make_dot
# 按用户指定配置：arch=unet, encoder_name=timm-tf_efficientnet_lite1
model = create_model(
    arch='unet',
    encoder_name='timm-tf_efficientnet_lite1',
    encoder_weights=None,     # 不加载预训练权重，仅打印架构
    in_channels=3,
    classes=10,                # 示例类别数
    decoder_attention_type=None,
)

print("=" * 80)
print("模型总体结构 (print(model))")
print("=" * 80)
print(model)

print()
print("=" * 80)
print("模型子模块逐层展开（named_modules）")
print("=" * 80)
for name, module in model.named_modules():
    # 只打印叶子节点（不含子模块的模块），同时也打印一级子模块名
    if len(list(module.children())) == 0:
        print(f"{name:80s}  {module.__class__.__name__}")

print()
print("=" * 80)
print("参数量统计")
print("=" * 80)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量:    {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

summary(model, input_size=(1, 3, 512, 512), depth=4, col_names=['input_size', 'output_size', 'num_params', 'mult_adds'])

x = torch.randn(1, 3, 512, 512).to('cuda')
model.to('cuda')

with torch.no_grad():
    y = model(x)

make_dot(y, params=dict(list(model.named_parameters()))).render("unet_model", format="png")
