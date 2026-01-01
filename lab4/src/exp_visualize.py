"""
可视化 DDPM 采样过程。
记录每个指定步数的图像状态，展示从噪声到清晰图像的演变过程。
"""

import torch
import torchvision

from config import device, timesteps, image_size, num_classes, get_checkpoint_path, outputs_dir
from util import load_diffusion_model
from diffusion import Diffusion
from record import save_image_grid

# 设置为 False 进行无条件生成可视化，True 进行条件生成可视化
CONDITIONAL = False

ckpt_path = get_checkpoint_path(conditional=CONDITIONAL)
output_dir = outputs_dir / 'sampling_visualization' / ('conditional' if CONDITIONAL else 'unconditional')


@torch.no_grad()
def visualize_sampling_process(
	diffusion: Diffusion,
	batch_size: int = 10,
	labels: torch.Tensor | None = None,
) -> list[torch.Tensor]:
	"""
	可视化 DDPM 采样过程，记录每个指定步数的图像状态。

	Args:
		diffusion: 扩散模型实例。
		batch_size: 批量大小
		save_interval: 保存图像的间隔步数
		labels: 条件标签

	Returns:
		tuple[list[torch.Tensor], list[int]]: 采样过程中保存的图像列表和对应的步数列表。
	"""
	diffusion.eval()

	img = torch.randn((batch_size, 1, image_size, image_size), device=device)

	res = []

	# 反向采样过程
	with torch.autocast('cuda'):
		for j, i in enumerate(reversed(range(0, timesteps))):
			t = torch.full((batch_size,), i, device=device, dtype=torch.long)
			img = diffusion.p_sample(img, t, i, labels)

			if (j + 1) % 100 == 0:
				res.append(img.clone())

	return res


def main():
	if output_dir.exists():
		for i in output_dir.iterdir():
			if i.is_file():
				i.unlink()
	output_dir.mkdir(parents=True, exist_ok=True)

	num_classes_param = num_classes if CONDITIONAL else None
	diffusion = load_diffusion_model(ckpt_path, num_classes=num_classes_param)
	print(f'Loaded checkpoint from {ckpt_path}')

	batch_size = 10
	labels = torch.arange(10, device=device) if CONDITIONAL else None

	saved_images = visualize_sampling_process(
		diffusion,
		batch_size=batch_size,
		labels=labels,
	)

	grid = torch.stack(saved_images)  # (T, B, C, H, W)
	grid = grid.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)

	grid = grid.reshape(-1, 1, image_size, image_size)  # (B*T, 1, H, W)
	grid = torchvision.utils.make_grid(
		grid,
		nrow=10,
		padding=0,
		normalize=True,
		value_range=(-1, 1),
	)

	save_image_grid(grid.unsqueeze(0), output_dir / 'sampling_10x10.png', nrow=1, padding=0)
	print(f'Sampling visualization saved to {output_dir / "sampling_10x10.png"}')


if __name__ == '__main__':
	main()
