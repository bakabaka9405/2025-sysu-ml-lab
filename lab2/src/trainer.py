import torch
import torchvision.transforms.v2 as transforms
from timm.data.loader import MultiEpochsDataLoader as DataLoader
from typing import Any, Literal


def load_data(dir):
	import pickle
	import numpy as np

	X_train = []
	Y_train = []
	for i in range(1, 6):
		with open(dir + r'/data_batch_' + str(i), 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		X_train.append(dict[b'data'])
		Y_train += dict[b'labels']
	X_train = np.concatenate(X_train, axis=0)
	with open(dir + r'/test_batch', 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	X_test = dict[b'data']
	Y_test = dict[b'labels']

	X_train = X_train.reshape(-1, 3, 32, 32)
	X_test = X_test.reshape(-1, 3, 32, 32)

	return X_train, Y_train, X_test, Y_test


class Dataset(torch.utils.data.Dataset):
	def __init__(self, data, labels, transform: transforms.Compose | None = None):
		self.data = data
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		image = self.data[idx]
		label = self.labels[idx]
		if self.transform:
			image = self.transform(image)
		return image, label


def train(
	device: torch.device,
	model: Any,
	epochs: int,
	batch_size: int,
	lr: float,
	criterion: torch.nn.Module,
	optimizer: Literal['sgd', 'sgd_momentum', 'adam'],
	transform_train: transforms.Compose | None,
	transform_test: transforms.Compose | None,
) -> list[list[float]]:
	X_train, Y_train, X_test, Y_test = load_data(r'C:\Temp\material2-CIFAR10\data')

	X_train = torch.tensor(X_train).float().to(device)
	X_test = torch.tensor(X_test).float().to(device)
	Y_train = torch.tensor(Y_train).long().to(device)
	Y_test = torch.tensor(Y_test).long().to(device)
	dataset_train = Dataset(X_train, Y_train, transform=transform_train)
	dataset_test = Dataset(X_test, Y_test, transform=transform_test)
	train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

	match optimizer:
		case 'sgd':
			optim = torch.optim.SGD(model.parameters(), lr=lr)
		case 'sgd_momentum':
			optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
		case 'adam':
			optim = torch.optim.Adam(model.parameters(), lr=lr)
		case _:
			raise ValueError(f'Unknown optimizer: {optimizer}')

	model.to(device)

	print(f'Training on {device}')
	print(f'Model: {model.__class__.__name__}')
	print(f'Optimizer: {optimizer}, Learning Rate: {lr}')
	print(f'Batch Size: {batch_size}, Epochs: {epochs}')
	print('-' * 70)

	stat = [[], [], [], []]

	test_acc = 0.0

	for epoch in range(epochs):
		model.train()
		train_loss = 0.0
		train_correct = 0
		train_total = 0

		for images, y_true in train_loader:
			optim.zero_grad()
			with torch.autocast('cuda'):
				outputs = model(images)
				loss = criterion(outputs, y_true)
			loss.backward()
			optim.step()

			train_loss += loss.item()
			y_pred = torch.argmax(outputs.data, 1)
			train_total += y_true.size(0)
			train_correct += (y_pred == y_true).sum().item()

		train_acc = 100 * train_correct / train_total
		train_loss = train_loss / len(train_loader)

		model.eval()
		test_correct = 0
		test_total = 0
		test_loss = 0.0

		with torch.no_grad():
			for images, y_true in test_loader:
				with torch.autocast('cuda'):
					outputs = model(images)
					loss = criterion(outputs, y_true)
				test_loss += loss.item()
				y_pred = torch.argmax(outputs.data, 1)
				test_total += y_true.size(0)
				test_correct += (y_pred == y_true).sum().item()

		test_acc = 100 * test_correct / test_total
		test_loss = test_loss / len(test_loader)

		print(
			f'Epoch [{epoch + 1:>2}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%'
		)

		stat[0].append(train_loss)
		stat[1].append(train_acc)
		stat[2].append(test_loss)
		stat[3].append(test_acc)

	print(f'Training completed. Final Test Accuracy: {test_acc:.2f}%')

	return stat
