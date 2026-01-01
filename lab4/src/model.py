import math
import torch
from torch import nn, Tensor
from typing import Literal


class SinusoidalPositionEmbeddings(nn.Module):
	"""
	正弦位置编码层。
	"""

	def __init__(self, dim: int):
		"""
		初始化正弦位置编码层。

		Args:
			dim: 嵌入维度。
		"""
		super().__init__()
		self.dim = dim

	def forward(self, time: Tensor) -> Tensor:
		"""
		前向传播。

		Args:
			time: 时间步张量。

		Returns:
			Tensor: 位置编码张量。
		"""
		device = time.device
		half_dim = self.dim // 2
		embeddings = math.log(10000) / (half_dim - 1)
		embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
		embeddings = time[:, None] * embeddings[None, :]
		embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
		return embeddings


class LearnablePositionEmbeddings(nn.Module):
	"""
	可学习的位置编码层。
	"""

	def __init__(self, dim: int, max_seq_len: int = 1000):
		"""
		初始化可学习位置编码层。

		Args:
			dim: 嵌入维度。
			max_seq_len: 最大序列长度
		"""
		super().__init__()
		self.embeddings = nn.Embedding(max_seq_len, dim)
		nn.init.normal_(self.embeddings.weight, 0, 0.1)

	def forward(self, time: Tensor) -> Tensor:
		"""
		前向传播。

		Args:
			time: 时间步张量。

		Returns:
			Tensor: 位置编码张量。
		"""
		return self.embeddings(time)


class Block(nn.Module):
	"""
	基础卷积块，包含 Conv2d, GroupNorm 和 SiLU。
	"""

	def __init__(self, in_channel: int, out_channel: int):
		"""
		初始化基础卷积块。

		Args:
			in_channel: 输入通道数。
			out_channel: 输出通道数。
		"""
		super().__init__()
		self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
		self.norm = nn.GroupNorm(8, out_channel)
		self.silu = nn.SiLU()

	def forward(self, x: Tensor) -> Tensor:
		return self.silu(self.norm(self.conv(x)))


class ResnetBlock(nn.Module):
	"""
	残差块，支持时间嵌入。
	"""

	def __init__(self, in_channel: int, out_channel: int, time_emb_dim: int | None = None):
		"""
		初始化残差块。

		Args:
			in_channel: 输入通道数。
			out_channel: 输出通道数。
			time_emb_dim: 时间嵌入维度
		"""
		super().__init__()
		self.mlp = (
			nn.Sequential(
				nn.SiLU(),
				nn.Linear(time_emb_dim, out_channel),
			)
			if time_emb_dim
			else None
		)
		self.block1 = Block(in_channel, out_channel)
		self.block2 = Block(out_channel, out_channel)
		self.res_conv = nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

	def forward(self, x: Tensor, time_emb: Tensor | None = None) -> Tensor:
		"""
		前向传播。

		Args:
			x: 输入张量。
			time_emb: 时间嵌入张量

		Returns:
			Tensor: 输出张量。
		"""
		h = self.block1(x)
		if self.mlp is not None and time_emb is not None:
			h += self.mlp(time_emb)[:, :, None, None]
		h = self.block2(h)
		return h + self.res_conv(x)


class UNet(nn.Module):
	"""
	U-Net 模型，用于扩散模型。
	"""

	def __init__(
		self,
		n_channels: int = 64,
		ch_mults: tuple[int, ...] = (1, 2, 4),
		n_blocks: int = 2,
		num_classes: int | None = None,
		embedding_type: Literal['sinusoidal', 'learnable'] = 'sinusoidal',
		timesteps: int = 1000,
	):
		"""
		初始化 U-Net 模型。

		Args:
			n_channels: 基础通道数
			ch_mults: 通道倍增系数
			n_blocks: 每个分辨率层级的块数
			num_classes: 类别数量（用于条件生成）
			embedding_type: 嵌入类型
			timesteps: 总时间步数
		"""
		super().__init__()
		n_resolutions = len(ch_mults)

		# Time Embedding
		time_emb_dim = n_channels * 4
		if embedding_type == 'sinusoidal':
			emb_layer = SinusoidalPositionEmbeddings(n_channels)
		elif embedding_type == 'learnable':
			emb_layer = LearnablePositionEmbeddings(n_channels, max_seq_len=timesteps)
		else:
			raise ValueError(f'Unknown embedding type: {embedding_type}')

		self.time_mlp = nn.Sequential(
			emb_layer,
			nn.Linear(n_channels, time_emb_dim),
			nn.SiLU(),
			nn.Linear(time_emb_dim, time_emb_dim),
		)

		if num_classes is not None:
			self.label_emb = nn.Embedding(num_classes, time_emb_dim)

		self.down_blocks = nn.ModuleList()
		self.up_blocks = nn.ModuleList()

		cur_channel = n_channels
		self.init_conv = nn.Conv2d(1, cur_channel, kernel_size=3, padding=1)

		# Down
		# We need to track channels for skip connections
		# Initial conv output is the first skip
		self.down_block_ch = [cur_channel]

		for i in range(n_resolutions):
			out_channel = n_channels * ch_mults[i]
			for _ in range(n_blocks):
				self.down_blocks.append(ResnetBlock(cur_channel, out_channel, time_emb_dim))
				cur_channel = out_channel
				self.down_block_ch.append(cur_channel)

			if i != n_resolutions - 1:
				self.down_blocks.append(nn.Conv2d(cur_channel, cur_channel, 4, 2, 1))
				self.down_block_ch.append(cur_channel)

		# Middle
		self.mid_block1 = ResnetBlock(cur_channel, cur_channel, time_emb_dim)
		self.mid_block2 = ResnetBlock(cur_channel, cur_channel, time_emb_dim)

		# Up
		for i in reversed(range(n_resolutions)):
			out_channel = n_channels * ch_mults[i]
			for _ in range(n_blocks + 1):
				skip_ch = self.down_block_ch.pop()
				self.up_blocks.append(ResnetBlock(cur_channel + skip_ch, out_channel, time_emb_dim))
				cur_channel = out_channel

			if i != 0:
				self.up_blocks.append(nn.ConvTranspose2d(cur_channel, cur_channel, 4, 2, 1))

		self.out_conv = nn.Sequential(
			nn.GroupNorm(8, cur_channel),
			nn.SiLU(),
			nn.Conv2d(cur_channel, 1, 3, padding=1),
		)

	def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
		"""
		前向传播。

		Args:
			x: 输入图像张量
			t: 时间步张量
			y: 标签张量，-1 表示无条件（用于 CFG）

		Returns:
			Tensor: 输出张量（预测的噪声）
		"""
		t_emb = self.time_mlp(t)
		if y is not None and hasattr(self, 'label_emb'):
			if y.dtype in (torch.long, torch.int):
				# CFG: -1 表示无条件，对应位置不添加标签嵌入
				valid_mask = y >= 0
				y_safe = y.clamp(min=0)
				label_emb = self.label_emb(y_safe)
				label_emb = label_emb * valid_mask[:, None].float()
				t_emb = t_emb + label_emb
			else:
				t_emb = t_emb + y

		x = self.init_conv(x)

		skips = [x]
		for layer in self.down_blocks:
			if isinstance(layer, ResnetBlock):
				x = layer(x, t_emb)
			else:
				x = layer(x)
			skips.append(x)

		x = self.mid_block1(x, t_emb)
		x = self.mid_block2(x, t_emb)

		for layer in self.up_blocks:
			if isinstance(layer, ResnetBlock):
				skip = skips.pop()
				x = torch.cat([x, skip], dim=1)
				x = layer(x, t_emb)
			else:
				x = layer(x)

		return self.out_conv(x)
