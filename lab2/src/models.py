import torch


class LinearModel(torch.nn.Module):
	def __init__(self, input_size: int, num_classes: int):
		super().__init__()
		self.linear = torch.nn.Linear(input_size, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.view(x.size(0), -1)
		return self.linear(x)


class MLP(torch.nn.Module):
	def __init__(self, input_size: int, hidden_size: list[int] | int, num_classes: int):
		super().__init__()
		if isinstance(hidden_size, int):
			hidden_size = [hidden_size]
		self.layers = torch.nn.ModuleList()
		in_features = input_size
		for hidden in hidden_size:
			self.layers.append(torch.nn.Linear(in_features, hidden))
			self.layers.append(torch.nn.ReLU())
			in_features = hidden
		self.layers.append(torch.nn.Linear(in_features, num_classes))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.view(x.size(0), -1)
		for layer in self.layers:
			x = layer(x)
		return x


class LeNet(torch.nn.Module):
	def __init__(
		self,
		num_classes: int,
		conv_channels: list[int] = [6, 16],
		pooling: bool = True,
	):
		super().__init__()
		conv = []
		in_channels = 3
		stride = 1 if pooling else 2
		for out_channels in conv_channels:
			conv.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2))
			conv.append(torch.nn.ReLU())
			if pooling:
				conv.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
			in_channels = out_channels
		self.conv = torch.nn.Sequential(*conv)
		sz = 32 // (2 ** len(conv_channels))
		self.fc1 = torch.nn.Linear(in_channels * sz * sz, 120)
		self.fc2 = torch.nn.Linear(120, 84)
		self.fc3 = torch.nn.Linear(84, num_classes)
		self.relu = torch.nn.ReLU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv(x)
		x = x.view(x.size(0), -1)
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		return x
