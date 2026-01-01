"""
标签嵌入插值实验。
在条件生成模型的嵌入空间中进行插值，展示数字间的平滑过渡。
"""

import torch

from config import device, image_size, num_classes, get_checkpoint_path, outputs_dir
from util import load_diffusion_model, inverse_transform
from diffusion import Diffusion
from record import save_image_grid

ckpt_path = get_checkpoint_path(conditional=True)
output_dir = outputs_dir / 'interpolation'


def interpolate(diffusion: Diffusion, n_rows: int = 8, n_steps: int = 10):
	"""
	执行插值生成。

	Args:
		diffusion: 扩散模型实例。
		n_rows: 行数
		n_steps: 插值步数
	"""
	diffusion.eval()
	output_dir.mkdir(parents=True, exist_ok=True)

	start_digits = torch.randint(0, 10, (n_rows,), device=device)
	end_digits = torch.randint(0, 10, (n_rows,), device=device)

	for i in range(n_rows):
		while start_digits[i] == end_digits[i]:
			end_digits[i] = torch.randint(0, 10, (1,), device=device).item()

	label_emb = diffusion.model.label_emb
	start_embs = label_emb(start_digits)  # type: ignore
	end_embs = label_emb(end_digits)  # type: ignore

	noise = torch.randn((n_rows, 1, image_size, image_size), device=device)

	all_samples = []

	for step in range(n_steps):
		alpha = step / (n_steps - 1)
		embs = (1 - alpha) * start_embs + alpha * end_embs

		samples = diffusion.sample(
			batch_size=n_rows,
			device=device,
			y=embs,
			start_img=noise,
			cfg_scale=3.0,
		)
		samples = inverse_transform(samples)
		all_samples.append(samples)

	grid = torch.stack(all_samples)  # (n_steps, n_rows, 1, H, W)
	grid = grid.permute(1, 0, 2, 3, 4)  # (n_rows, n_steps, 1, H, W)
	grid = grid.reshape(-1, 1, image_size, image_size)  # (n_rows * n_steps, 1, H, W)

	save_image_grid(grid, output_dir / 'interpolation.png', nrow=n_steps)


def main():
	diffusion = load_diffusion_model(ckpt_path, num_classes=num_classes)
	interpolate(diffusion)


if __name__ == '__main__':
	main()
