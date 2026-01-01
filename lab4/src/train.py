import shutil
import torch
from typing import Literal
from tqdm import tqdm
from torch.optim import AdamW
from itertools import product

from config import (
	image_size,
	timesteps,
	device,
	batch_size,
	epochs,
	lr,
	ckpt_uncond,
	ckpt_cond,
	outputs_dir,
	label_dropout,
)
from model import UNet
from diffusion import Diffusion
from util import get_dataloader, save_checkpoint
from sample import sample_images
from record import save_image_grid, ScalarTracker


def train(conditional: bool):
	"""
	训练扩散模型。

	Args:
		conditional: 条件生成开关
	"""
	print(f'Using device: {device}')

	num_classes = 10 if conditional else None
	ckpt_dir_base = ckpt_cond if conditional else ckpt_uncond
	output_dir_base = outputs_dir / 'train' / ('conditional' if conditional else 'unconditional')

	shutil.rmtree(output_dir_base, ignore_errors=True)

	dataloader = get_dataloader(device, batch_size=batch_size)

	embedding_types: list[Literal['sinusoidal', 'learnable']] = ['sinusoidal', 'learnable']
	schedule_types: list[Literal['linear', 'cosine']] = ['linear', 'cosine']

	tracker = ScalarTracker()
	for emb, sch in product(embedding_types, schedule_types):
		cfg = f'{emb}_{sch}'

		print(f'Training configuration: {cfg}')

		output_dir = output_dir_base / cfg
		output_dir.mkdir(parents=True, exist_ok=True)

		model = UNet(
			num_classes=num_classes,
			embedding_type=emb,
		).to(device)

		diffusion = Diffusion(
			model,
			timesteps,
			image_size,
			sch,
		).to(device)

		optimizer = AdamW(diffusion.parameters(), lr=lr)
		scaler = torch.GradScaler()

		ckpt_dir = ckpt_dir_base / cfg
		ckpt_dir.mkdir(parents=True, exist_ok=True)

		for epoch in range(epochs):
			model.train()
			losses = 0.0

			dropout = label_dropout if conditional else 0.0

			for images, labels in tqdm(dataloader, desc=f'Epoch {epoch}'):
				y = labels if conditional else None
				t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device).long()

				with torch.autocast('cuda'):
					loss = diffusion.p_losses(images, t, y=y, label_dropout=dropout)

				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()

				losses += loss.item()

			avg_loss = losses / len(dataloader)
			print(f'Epoch {epoch} Loss: {avg_loss:.4f}')

			tracker.add(f'{cfg} loss', avg_loss, epoch)

			save_checkpoint(model, optimizer, epoch, ckpt_dir / f'epoch_{epoch}.pt')

			labels = torch.randint(0, 10, (4,), device=device) if conditional else None
			samples = sample_images(diffusion, device=device, labels=labels)
			save_image_grid(samples, output_dir / f'samples_epoch_{epoch}.png', nrow=2, padding=0)

		labels = torch.arange(10, device=device).repeat_interleave(8) if conditional else None
		samples = sample_images(diffusion, device=device, labels=labels)
		save_image_grid(samples, output_dir / 'final_samples.png', nrow=8, padding=0)
		if conditional:
			samples = sample_images(diffusion, device=device, labels=labels, cfg_scale=3.0)
			save_image_grid(samples, output_dir / 'final_samples_cfg.png', nrow=8, padding=0)

	tracker.save_csv(output_dir_base / 'loss.csv')
	tracker.save_plot(output_dir_base / 'loss.png')


if __name__ == '__main__':
	train(True)
