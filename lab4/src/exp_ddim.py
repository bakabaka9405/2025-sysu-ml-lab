"""
DDPM 与 DDIM 性能对比实验。
对比不同采样步数下的生成速度和质量。
"""

import time
import torch

from config import device, num_classes, get_checkpoint_path, outputs_dir
from util import load_diffusion_model
from record import save_image_grid
from sample import sample_images

cond = True

ckpt_path = get_checkpoint_path(conditional=cond)
output_dir = outputs_dir / 'comparison' / ('conditional' if cond else 'unconditional')
batch_size = 16


def main():
	num_classes_param = num_classes if cond else None
	diffusion = load_diffusion_model(ckpt_path, num_classes=num_classes_param)
	print(f'Loaded checkpoint from {ckpt_path}')

	diffusion.model.eval()

	output_dir.mkdir(parents=True, exist_ok=True)

	labels = torch.arange(10, device=device).repeat(2)[:batch_size] if cond else None

	# 固定初始噪声
	start_noise = torch.randn(
		batch_size,
		1,
		diffusion.image_size,
		diffusion.image_size,
		device=device,
	)
	# 1. DDPM 采样
	print('Running DDPM sampling...')
	start_time = time.time()
	samples = sample_images(
		diffusion,
		batch_size,
		device,
		labels,
		start_noise=start_noise,
	)
	ddpm_time = time.time() - start_time
	print(f'DDPM sampling time: {ddpm_time:.2f}s')

	save_image_grid(samples, output_dir / 'ddpm_1000_steps.png', nrow=4)

	# 2. 不同步数的 DDIM 采样
	for steps in [10, 20, 50, 100]:
		start_time = time.time()
		samples = sample_images(
			diffusion,
			batch_size,
			device,
			labels,
			ddim=True,
			ddim_timesteps=steps,
			ddim_eta=0.0,
			start_noise=start_noise,
		)
		ddim_time = time.time() - start_time
		print(f'DDIM {steps} steps sampling time: {ddim_time:.2f}s')

		save_image_grid(samples, output_dir / f'ddim_{steps}_steps.png', nrow=4)

	# 3. DDIM eta=1.0
	start_time = time.time()
	samples = sample_images(
		diffusion,
		batch_size,
		device,
		labels,
		ddim=True,
		ddim_timesteps=50,
		ddim_eta=1.0,
	)
	ddim_time = time.time() - start_time
	print(f'DDIM 50 steps (eta=1.0) sampling time: {ddim_time:.2f}s')

	save_image_grid(samples, output_dir / 'ddim_50_steps_eta1.png', nrow=4)


if __name__ == '__main__':
	main()
