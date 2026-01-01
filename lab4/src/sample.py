import torch
from util import inverse_transform
from diffusion import Diffusion


@torch.no_grad()
def sample_images(
	diffusion_model: Diffusion,
	batch_size: int = 16,
	device: torch.device = torch.device('cpu'),
	labels: torch.Tensor | None = None,
	ddim: bool = False,
	ddim_eta: float = 0.0,
	ddim_timesteps: int = 50,
	start_noise: torch.Tensor | None = None,
	cfg_scale: float = 1.0,
) -> torch.Tensor:
	"""
	使用扩散模型生成图像样本。

	Args:
		diffusion_model: 扩散模型实例。
		batch_size: 批量大小
		device: 设备
		labels: 标签（用于条件生成）
		ddim: 是否使用 DDIM 采样
		ddim_eta: DDIM 采样的 eta 参数
		ddim_timesteps: DDIM 采样步数
		start_noise: 初始噪声，用于控制采样起点以进行公平对比
		cfg_scale: CFG 引导强度

	Returns:
		torch.Tensor: 生成的图像样本，并且经过反归一化处理。
	"""
	diffusion_model.eval()
	if labels is not None:
		batch_size = labels.shape[0]
	if ddim:
		samples = diffusion_model.ddim_sample(
			batch_size,
			device,
			ddim_timesteps,
			ddim_eta,
			labels,
			start_noise,
			cfg_scale,
		)
	else:
		samples = diffusion_model.sample(
			batch_size,
			device,
			labels,
			start_noise,
			cfg_scale,
		)
	return inverse_transform(samples)
