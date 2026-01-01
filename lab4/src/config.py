import torch
from pathlib import Path

# 模型配置
image_size = 32
timesteps = 1000

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练超参数
batch_size = 128
epochs = 30
lr = 1e-4

# 条件生成配置
num_classes = 10

# CFG 配置
label_dropout = 0.1
cfg_scale = 3.0

# 默认检查点配置
default_config = 'sinusoidal_linear'
default_epoch = 29

# 目录配置
outputs_dir = Path('outputs')
ckpt_uncond = outputs_dir / 'checkpoints_uncond'
ckpt_cond = outputs_dir / 'checkpoints_cond'


def get_checkpoint_path(
	conditional: bool,
	config_name: str = default_config,
	epoch: int = default_epoch,
) -> Path:
	"""
	获取检查点路径。

	Args:
		conditional: 是否为条件生成模型
		config_name: 配置名称 (如 'sinusoidal_linear')
		epoch: epoch 编号

	Returns:
		检查点文件路径
	"""
	base = ckpt_cond if conditional else ckpt_uncond
	return base / config_name / f'epoch_{epoch}.pt'
