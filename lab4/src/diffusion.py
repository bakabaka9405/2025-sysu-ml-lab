import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal


def linear_beta_schedule(
	timesteps: int,
) -> torch.Tensor:
	"""
	生成线性的 beta 调度。

	Args:
		timesteps: 总的时间步数。

	Returns:
		torch.Tensor: beta 调度张量。
	"""
	return torch.linspace(0.0001, 0.02, timesteps)


def cosine_beta_schedule(
	timesteps: int,
	s: float = 0.008,
) -> torch.Tensor:
	"""
	生成余弦 beta 调度。

	Args:
		timesteps: 总的时间步数。
		s: 偏移量

	Returns:
		torch.Tensor: beta 调度张量。
	"""
	steps = timesteps + 1
	x = torch.linspace(0, timesteps, steps)
	alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return torch.clip(betas, 0.0001, 0.9999)


class Diffusion(nn.Module):
	"""
	扩散模型核心类，包含前向扩散过程和反向采样过程。
	"""

	betas: torch.Tensor
	alphas_cumprod: torch.Tensor
	alphas_cumprod_prev: torch.Tensor
	sqrt_recip_alphas: torch.Tensor
	sqrt_alphas_cumprod: torch.Tensor
	sqrt_one_minus_alphas_cumprod: torch.Tensor
	posterior_variance: torch.Tensor
	posterior_mean_coef1: torch.Tensor
	posterior_mean_coef2: torch.Tensor

	def __init__(
		self,
		model: nn.Module,
		timesteps: int = 1000,
		image_size: int = 28,
		schedule_type: Literal['linear', 'cosine'] = 'linear',
	):
		"""
		初始化扩散模型。

		Args:
			model: 用于预测噪声的神经网络模型。
			timesteps: 扩散步数
			image_size: 图像尺寸
			schedule_type: beta 调度类型
		"""
		super().__init__()
		self.model = model
		self.timesteps = timesteps
		self.image_size = image_size

		if schedule_type == 'linear':
			betas = linear_beta_schedule(timesteps)
		elif schedule_type == 'cosine':
			betas = cosine_beta_schedule(timesteps)
		else:
			raise ValueError(f'Unknown schedule type: {schedule_type}')

		alphas = 1.0 - betas
		alphas_cumprod = torch.cumprod(alphas, dim=0)
		alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

		# Register buffers
		self.register_buffer('betas', betas)
		self.register_buffer('alphas_cumprod', alphas_cumprod)
		self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
		self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
		self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
		self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
		self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

		# calculations for posterior q(x_{t-1} | x_t, x_0)
		self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
		self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

	def q_sample(
		self,
		x_start: torch.Tensor,
		t: torch.Tensor,
		noise: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""
		前向扩散过程：从 x_0 采样 x_t。

		Args:
			x_start: 初始图像 x_0。
			t: 时间步 t。
			noise: 噪声

		Returns:
			torch.Tensor: 加噪后的图像 x_t。
		"""
		if noise is None:
			noise = torch.randn_like(x_start)

		sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
		sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

		return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

	def p_losses(
		self,
		x_start: torch.Tensor,
		t: torch.Tensor,
		noise: torch.Tensor | None = None,
		y: torch.Tensor | None = None,
		label_dropout: float = 0.0,
	) -> torch.Tensor:
		"""
		计算损失函数。

		Args:
			x_start: 初始图像 x_0。
			t: 时间步 t。
			noise: 噪声
			y: 标签
			label_dropout: CFG 训练时的条件丢弃率

		Returns:
			torch.Tensor: 损失值。
		"""
		if noise is None:
			noise = torch.randn_like(x_start)

		# CFG: 随机丢弃部分条件
		if y is not None and label_dropout > 0:
			drop_mask = torch.rand(y.shape[0], device=y.device) < label_dropout
			y = y.clone()
			y[drop_mask] = -1

		x_noisy = self.q_sample(x_start, t, noise)
		pred = self.model(x_noisy, t, y)

		loss = F.mse_loss(pred, noise)
		return loss

	@torch.no_grad()
	def p_sample(
		self,
		x: torch.Tensor,
		t: torch.Tensor,
		t_index: int,
		y: torch.Tensor | None = None,
		cfg_scale: float = 1.0,
	) -> torch.Tensor:
		"""
		反向采样过程的一步：从 x_t 采样 x_{t-1}。

		Args:
			x: 当前图像 x_t。
			t: 时间步 t。
			t_index: 时间步索引。
			y: 标签
			cfg_scale: CFG 引导强度

		Returns:
			torch.Tensor: 上一步图像 x_{t-1}。
		"""
		sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
		sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)

		# CFG: 同时预测有条件和无条件噪声
		if cfg_scale > 1.0 and y is not None:
			y_null = torch.full_like(y, -1)
			noise_uncond = self.model(x, t, y_null)
			noise_cond = self.model(x, t, y)
			noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
		else:
			noise_pred = self.model(x, t, y)

		# Predict x_start (x_0)
		x_recon = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
		x_recon.clamp_(-1.0, 1.0)

		# Calculate mean using posterior formula
		posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x.shape)
		posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x.shape)

		model_mean = posterior_mean_coef1_t * x_recon + posterior_mean_coef2_t * x

		if t_index == 0:
			return model_mean
		else:
			posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
			noise = torch.randn_like(x)
			return model_mean + torch.sqrt(posterior_variance_t) * noise

	@torch.no_grad()
	def sample(
		self,
		batch_size: int,
		device: torch.device,
		y: torch.Tensor | None = None,
		start_img: torch.Tensor | None = None,
		cfg_scale: float = 1.0,
	) -> torch.Tensor:
		"""
		生成图像样本。

		Args:
			batch_size: 批量大小。
			device: 设备。
			y: 标签
			start_img: 初始噪声图像
			cfg_scale: CFG 引导强度

		Returns:
			torch.Tensor: 生成的图像样本。
		"""
		shape = (batch_size, 1, self.image_size, self.image_size)
		if start_img is not None:
			img = start_img
		else:
			img = torch.randn(shape, device=device)

		with torch.autocast(device_type=device.type):
			for i in reversed(range(0, self.timesteps)):
				t = torch.full((batch_size,), i, device=device, dtype=torch.long)
				img = self.p_sample(img, t, i, y, cfg_scale)

		return img

	@torch.no_grad()
	def ddim_sample(
		self,
		batch_size: int,
		device: torch.device,
		timesteps: int = 50,
		eta: float = 0.0,
		y: torch.Tensor | None = None,
		start_img: torch.Tensor | None = None,
		cfg_scale: float = 1.0,
	) -> torch.Tensor:
		"""
		使用 DDIM 采样生成图像。

		Args:
			batch_size: 批量大小
			device: 设备
			timesteps: 采样步数
			eta: 控制采样随机性
			y: 标签
			start_img: 初始噪声图像
			cfg_scale: CFG 引导强度

		Returns:
			torch.Tensor: 生成的图像样本
		"""
		shape = (batch_size, 1, self.image_size, self.image_size)
		if start_img is not None:
			img = start_img
		else:
			img = torch.randn(shape, device=device)

		time_seq = [int((1 - i / timesteps) ** 2 * (self.timesteps - 1)) for i in range(timesteps)]

		with torch.autocast(device.type):
			for i, t_step in enumerate(time_seq):
				prev_t_step = time_seq[i + 1] if i < len(time_seq) - 1 else -1

				t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

				alpha_cumprod_t = self._extract(self.alphas_cumprod, t, img.shape)

				if prev_t_step >= 0:
					prev_t = torch.full((batch_size,), prev_t_step, device=device, dtype=torch.long)
					alpha_cumprod_prev = self._extract(self.alphas_cumprod, prev_t, img.shape)
				else:
					alpha_cumprod_prev = torch.ones_like(alpha_cumprod_t)

				if cfg_scale > 1.0 and y is not None:
					y_null = torch.full_like(y, -1)
					noise_uncond = self.model(img, t, y_null)
					noise_cond = self.model(img, t, y)
					noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
				else:
					noise_pred = self.model(img, t, y)

				pred_x0 = (img - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
				pred_x0 = torch.clamp(pred_x0, -1, 1)

				sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev))

				dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma_t**2) * noise_pred

				noise = sigma_t * torch.randn_like(img)

				img = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + noise

				if t_step <= 10:
					break

		return img

	def _extract(
		self,
		a: torch.Tensor,
		t: torch.Tensor,
		x_shape: tuple[int, ...],
	) -> torch.Tensor:
		"""
		从张量 a 中提取索引 t 对应的值，并重塑为 x_shape 的形状。

		Args:
			a: 源张量。
			t: 索引张量。
			x_shape: 目标形状。

		Returns:
			torch.Tensor: 提取并重塑后的张量。
		"""
		batch_size = t.shape[0]
		out = a.gather(-1, t)
		return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
