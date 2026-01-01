"""
可视化模块。
提供图像保存和标量数据记录功能，替代 TensorBoard。
"""

import csv
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field


def save_image_grid(
	tensor: torch.Tensor,
	path: Path,
	nrow: int = 8,
	padding: int = 2,
) -> None:
	"""
	将图像张量保存为网格图片。

	Args:
		tensor: 图像张量 (N, C, H, W)
		path: 保存路径
		nrow: 每行图像数量
		normalize: 是否归一化到 [0, 1]
		value_range: 归一化前的值范围
		padding: 图像间距
	"""
	path.parent.mkdir(parents=True, exist_ok=True)

	grid = torchvision.utils.make_grid(
		tensor,
		nrow=nrow,
		padding=padding,
	)

	# (C, H, W) -> (H, W, C)
	grid = grid.permute(1, 2, 0).cpu().numpy()

	plt.imsave(path, grid)


@dataclass
class ScalarRecord:
	"""单个标量记录"""

	step: int
	value: float


@dataclass
class ScalarTracker:
	"""
	标量数据追踪器。
	记录训练过程中的标量数据，支持导出 CSV 和绘制折线图。
	"""

	data: dict[str, list[ScalarRecord]] = field(default_factory=dict)

	def add(
		self,
		tag: str,
		value: float,
		step: int,
	) -> None:
		"""
		记录单个标量值。

		Args:
			tag: 标签名称
			value: 标量值
			step: 步数
		"""
		if tag not in self.data:
			self.data[tag] = []
		self.data[tag].append(ScalarRecord(step=step, value=value))

	def save_csv(
		self,
		path: Path,
	) -> None:
		"""
		导出所有数据为 CSV 文件。

		Args:
			path: 保存路径
		"""
		path.parent.mkdir(parents=True, exist_ok=True)

		with open(path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(['tag', 'step', 'value'])
			for tag, records in self.data.items():
				for record in records:
					writer.writerow([tag, record.step, record.value])

	def save_plot(
		self,
		path: Path,
		tag: str | None = None,
	) -> None:
		"""
		将数据绘制为折线图并保存。

		Args:
			path: 保存路径
			tag: 指定标签，若为 None 则绘制所有标签
		"""
		path.parent.mkdir(parents=True, exist_ok=True)

		plt.figure(figsize=(10, 6))

		tags_to_plot = [tag] if tag else list(self.data.keys())

		for t in tags_to_plot:
			if t not in self.data:
				continue
			records = self.data[t]
			steps = [r.step for r in records]
			values = [r.value for r in records]
			plt.plot(steps, values, label=t, marker='o', markersize=4)

		plt.xlabel('Step')
		plt.ylabel('Value')
		plt.title('Training Metrics')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(path, dpi=150)
		plt.close()
