"""
CFG (Classifier-Free Guidance) 对比实验。
对比有无 CFG 对条件生成质量的影响。
"""

import torch

from config import device, num_classes, cfg_scale, get_checkpoint_path, outputs_dir
from util import load_diffusion_model
from record import save_image_grid
from sample import sample_images


ckpt_path = get_checkpoint_path(conditional=True)
output_dir = outputs_dir / 'cfg_comparison'
batch_size = 80


def main():
	diffusion = load_diffusion_model(ckpt_path, num_classes=num_classes)
	print(f'Loaded checkpoint from {ckpt_path}')

	diffusion.model.eval()

	output_dir.mkdir(parents=True, exist_ok=True)

	# 每个数字生成 8 个样本
	labels = torch.arange(10, device=device).repeat_interleave(8)

	# 固定初始噪声以便公平对比
	start_noise = torch.randn(
		batch_size,
		1,
		diffusion.image_size,
		diffusion.image_size,
		device=device,
	)

	# 1. 无 CFG 采样 (cfg_scale=1.0)
	print('Sampling without CFG (cfg_scale=1.0)...')
	samples_no_cfg = sample_images(
		diffusion,
		batch_size,
		device,
		labels,
		start_noise=start_noise,
		cfg_scale=1.0,
	)
	save_image_grid(samples_no_cfg, output_dir / 'no_cfg.png', nrow=8)
	print(f'Saved to {output_dir / "no_cfg.png"}')

	# 2. 有 CFG 采样
	print(f'Sampling with CFG (cfg_scale={cfg_scale})...')
	samples_with_cfg = sample_images(
		diffusion,
		batch_size,
		device,
		labels,
		start_noise=start_noise,
		cfg_scale=cfg_scale,
	)
	save_image_grid(samples_with_cfg, output_dir / 'with_cfg.png', nrow=8)
	print(f'Saved to {output_dir / "with_cfg.png"}')

	print('CFG comparison completed.')


if __name__ == '__main__':
	main()
