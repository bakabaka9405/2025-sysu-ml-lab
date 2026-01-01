"""
图像修复实验。
使用扩散模型修复图像缺失部分。
"""

import torch
from typing import Literal

from config import device, num_classes, get_checkpoint_path, outputs_dir
from util import load_diffusion_model, get_dataloader, inverse_transform
from diffusion import Diffusion
from record import save_image_grid

ckpt_path = get_checkpoint_path(conditional=True)
output_dir = outputs_dir / 'inpainting'


def get_mask(
	shape: tuple[int, ...],
	mask_type: Literal['center', 'random', 'right_half'] = 'center',
) -> torch.Tensor:
	"""
	生成掩码。

	Args:
		shape: 图像形状。
		mask_type: 掩码类型

	Returns:
		torch.Tensor: 掩码张量。1 表示已知，0 表示缺失。
	"""
	mask = torch.ones(shape)
	if mask_type == 'center':
		h, w = shape[-2], shape[-1]
		cx, cy = h // 2, w // 2
		mask[..., cx - 4 : cx + 4, cy - 4 : cy + 4] = 0
	elif mask_type == 'random':
		mask = torch.bernoulli(torch.full(shape, 0.5))
	elif mask_type == 'right_half':
		w = shape[-1]
		mask[..., :, w // 2 :] = 0
	return mask


@torch.no_grad()
def inpaint(
	diffusion: Diffusion,
	image: torch.Tensor,
	mask: torch.Tensor,
	labels: torch.Tensor | None = None,
) -> torch.Tensor:
	"""
	使用扩散模型修复图像缺失部分。

	Args:
		diffusion: 扩散模型实例。
		image: 原始图像 (batch_size, channels, height, width)。
		mask: 掩码 (batch_size, channels, height, width)。1 表示已知，0 表示缺失。
		labels: 标签（用于条件生成）

	Returns:
		torch.Tensor: 修复后的图像。
	"""
	batch_size = image.shape[0]
	img = torch.randn_like(image)

	for i in reversed(range(0, diffusion.timesteps)):
		t = torch.full((batch_size,), i, device=device, dtype=torch.long)

		# 1. 预测 x_{t-1}
		img = diffusion.p_sample(img, t, i, y=labels, cfg_scale=3.0)

		# 2. 获取已知部分在 t-1 步的噪声版本
		if i > 0:
			t_prev = torch.full((batch_size,), i - 1, device=device, dtype=torch.long)
			known_part = diffusion.q_sample(image, t_prev)
		else:
			known_part = image

		# 3. 合并已知部分和生成部分
		img = mask * known_part + (1 - mask) * img

	return img


def main():
	print(f'Using device: {device}')

	diffusion = load_diffusion_model(ckpt_path, num_classes=num_classes)
	print(f'Loaded checkpoint from {ckpt_path}')

	# 获取数据
	dataloader = get_dataloader(device, batch_size=10)
	images, labels = next(iter(dataloader))
	images = images[:10].to(device)
	labels = labels[:10].to(device)

	output_dir.mkdir(parents=True, exist_ok=True)

	mask_types: list[Literal['center', 'random', 'right_half']] = ['center', 'random', 'right_half']

	for mask_type in mask_types:
		print(f'current: {mask_type}')

		mask = get_mask(images.shape, mask_type=mask_type).to(device)

		inpainted_images = inpaint(diffusion, images, mask, labels=labels)

		images_vis = inverse_transform(images)

		masked_images_vis = images.clone()
		masked_images_vis[mask == 0] = -1
		masked_images_vis = inverse_transform(masked_images_vis)

		inpainted_images_vis = inverse_transform(inpainted_images)

		# 拼接：原图 | 遮挡图 | 修复图
		results = []
		for i in range(10):
			results.extend([images_vis[i], masked_images_vis[i], inpainted_images_vis[i]])

		results_tensor = torch.stack(results)
		save_image_grid(results_tensor, output_dir / f'{mask_type}.png', nrow=3)

	print(f'Inpainting results saved to {output_dir}')


if __name__ == '__main__':
	main()
