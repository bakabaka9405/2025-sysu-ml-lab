from matplotlib import pyplot as plt


def plot_stat(epochs: int, stats: list[list[list[float]]], names: list[str], filename: str = 'result.png') -> None:
	plt.figure(figsize=(18, 4))

	x = list(range(1, epochs + 1))

	for i, (title, ylabel) in enumerate(
		zip(
			['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'],
			['Loss', 'Accuracy (%)', 'Loss', 'Accuracy (%)'],
		)
	):
		plt.subplot(1, 4, i + 1)
		for name, stat in zip(names, stats):
			plt.plot(x, stat[i], label=name)
		plt.title(title)
		plt.xlabel('Epoch')
		plt.ylabel(ylabel)
		plt.grid(True)
		plt.legend()

	plt.savefig(filename, bbox_inches='tight')
