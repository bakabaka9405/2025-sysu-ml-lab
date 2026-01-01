import torch
import torchvision
import torchvision.transforms.v2 as transforms
from timm.data.loader import MultiEpochsDataLoader as DataLoader
from pathlib import Path
from typing import Literal

from diffusion import Diffusion
from model import UNet


class Normalize:
	"""
	将 [0, 1] 映射到 [-1, 1]。
	"""

	def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
		return tensor * 2 - 1


class Dataset(torch.utils.data.Dataset):
	def __init__(self, data, targets):
		self.data = data
		self.targets = targets

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.targets[idx]


def get_dataloader(device: torch.device, batch_size: int = 128) -> DataLoader:
	"""
	获取 MNIST 数据加载器。

	Args:
		batch_size: 批量大小

	Returns:
		DataLoader: 数据加载器。
	"""
	transform = transforms.Compose(
		[
			transforms.ToImage(),
			transforms.ToDtype(torch.float32, scale=True),
			transforms.Pad(2, fill=0),
			transforms.Normalize((0.5,), (0.5,)),
		]
	)

	data_path = Path('./data')
	data_path.mkdir(parents=True, exist_ok=True)

	dataset = torchvision.datasets.MNIST(
		root=data_path,
		train=True,
		download=True,
	)

	img = []
	labels = []

	for i in range(len(dataset)):
		image, label = dataset[i]
		img.append(transform(image).to(device))
		labels.append(label)

	dataset = Dataset(
		torch.stack(img).to(device),
		torch.tensor(labels).to(device),
	)

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
	)


def inverse_transform(tensors: torch.Tensor) -> torch.Tensor:
	"""Convert tensors from [-1, 1] to [0, 1]"""
	return (tensors.clamp(-1, 1) + 1.0) / 2.0


def save_checkpoint(
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	path: str | Path,
):
	"""
	保存检查点。

	Args:
		model: 模型。
		optimizer: 优化器。
		epoch: 当前 epoch。
		path: 保存路径。
	"""
	data: dict = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
	}
	torch.save(data, path)


def infer_config_from_path(path: Path) -> tuple[str, str]:
	"""
	从检查点路径推断 embedding 和 schedule 类型。

	Args:
		path: 检查点文件路径

	Returns:
		(embedding_type, schedule_type) 元组
	"""
	config_name = path.parent.name  # 如 'sinusoidal_linear'
	parts = config_name.split('_')
	embedding_type = parts[0]  # 'sinusoidal' 或 'learnable'
	schedule_type = parts[1]  # 'linear' 或 'cosine'
	return embedding_type, schedule_type


def load_diffusion_model(
	checkpoint_path: Path,
	num_classes: int | None = None,
	embedding_type: Literal['sinusoidal', 'learnable'] | None = None,
	schedule_type: Literal['linear', 'cosine'] | None = None,
) -> Diffusion:
	"""
	加载扩散模型。

	Args:
		checkpoint_path: 检查点路径。
		num_classes: 类别数量
		embedding_type: 嵌入类型（若为 None 则从路径推断）
		schedule_type: 调度类型（若为 None 则从路径推断）

	Returns:
		Diffusion: 加载好的扩散模型实例。
	"""
	from config import image_size, timesteps, device

	# 自动推断配置
	if embedding_type is None or schedule_type is None:
		inferred_emb, inferred_sch = infer_config_from_path(checkpoint_path)
		embedding_type = embedding_type or inferred_emb  # type: ignore
		schedule_type = schedule_type or inferred_sch  # type: ignore

	model = UNet(
		n_channels=64,
		ch_mults=(1, 2, 4),
		num_classes=num_classes,
		embedding_type=embedding_type,  # type: ignore
		timesteps=timesteps,
	).to(device)

	diffusion = Diffusion(
		model,
		timesteps=timesteps,
		image_size=image_size,
		schedule_type=schedule_type,  # type: ignore
	).to(device)

	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
	model.load_state_dict(checkpoint['model_state_dict'])

	return diffusion
