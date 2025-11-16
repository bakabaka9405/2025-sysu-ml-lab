from trainer import train
from models import LinearModel, MLP, LeNet
import torch
import torchvision.transforms.v2 as transforms
import plot
import time


class ZeroOneNormalize:
	def __call__(self, tensor: torch.Tensor):
		return tensor.float().div(255)


def mlp_test(device, transform_train, transform_test):
	stats = []
	mlp_params = [256, 512, 1024, 2048, [256, 1024, 256], [1024, 4096, 1024]]
	for size in mlp_params:
		mlp_model = MLP(input_size=32 * 32 * 3, hidden_size=size, num_classes=10)
		mlp_stat = train(
			device=device,
			model=mlp_model,
			epochs=20,
			batch_size=128,
			lr=0.001,
			criterion=torch.nn.CrossEntropyLoss(),
			optimizer='adam',
			transform_train=transform_train,
			transform_test=transform_test,
		)
		stats.append(mlp_stat)
		print()

	plot.plot_stat(20, stats, [str(size) for size in mlp_params], filename='mlp_results.png')


def cnn_test(device, transform_train, transform_test):
	stats = []

	params = [
		([6, 16], True),
		([16, 32], True),
		([32, 64], True),
		([6, 16, 32], True),
		([16, 32, 64], True),
		([6, 16], False),
		([16, 32], False),
		([32, 64], False),
		([6, 16, 32], False),
		([16, 32, 64], False),
	]

	for param in params:
		model = LeNet(10, *param)
		stat = train(
			device=device,
			model=model,
			epochs=20,
			batch_size=128,
			lr=0.001,
			criterion=torch.nn.CrossEntropyLoss(),
			optimizer='adam',
			transform_train=transform_train,
			transform_test=transform_test,
		)
		stats.append(stat)
		print()

	plot.plot_stat(20, stats, [str(param[0]) + ('(P)' if param[1] else '(nP)') for param in params], filename='cnn_results.png')


def optim_test(device, transform_train, transform_test):
	stats = []
	for optim in ['sgd', 'sgd_momentum', 'adam']:
		model = LeNet(10)
		start = time.time()
		stat = train(
			device=device,
			model=model,
			epochs=20,
			batch_size=128,
			lr=0.001,
			criterion=torch.nn.CrossEntropyLoss(),
			optimizer=optim,  # type:ignore
			transform_train=transform_train,
			transform_test=transform_test,
		)

		stats.append(stat)
		end = time.time()
		print(f'Optimizer: {optim}, Time: {end - start:.2f}s')

	plot.plot_stat(20, stats, ['SGD', 'SGD with Momentum', 'Adam'], filename='optim_results.png')


def all_test(device, transform_train, transform_test):
	stats = []
	models = [LinearModel(32 * 32 * 3, 10), MLP(32 * 32 * 3, 256, 10), LeNet(10, [16, 32, 64], True)]
	names = ['Linear Model', 'MLP', 'LeNet']
	for model in models:
		stat = train(
			device=device,
			model=model,
			epochs=20,
			batch_size=128,
			lr=0.001,
			criterion=torch.nn.CrossEntropyLoss(),
			optimizer='adam',
			transform_train=transform_train,
			transform_test=transform_test,
		)
		stats.append(stat)
		print()

	plot.plot_stat(20, stats, names, filename='all_models_results.png')


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	transform_train = transforms.Compose(
		[
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		]
	)

	transform_test = transforms.Compose(
		[
			ZeroOneNormalize(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		]
	)

	# mlp_test(device, transform_train, transform_test)

	# cnn_test(device, transform_train, transform_test)

	# optim_test(device, transform_train, transform_test)

	all_test(device, transform_train, transform_test)


if __name__ == '__main__':
	main()
