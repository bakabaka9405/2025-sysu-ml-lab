import torch
from typing import Literal


class LinearClassifierModel(torch.nn.Module):
	def __init__(self, input_dim):
		super(LinearClassifierModel, self).__init__()
		self.linear = torch.nn.Linear(input_dim, 1)
		torch.nn.init.kaiming_normal_(self.linear.weight)

	def forward(self, x):
		return self.linear(x)


def lcm_train(
	X_train,
	y_train,
	epochs=100,
	lr=0.01,
	criterion: Literal['hinge', 'cross_entropy'] = 'hinge',
):
	if criterion == 'hinge':
		y_train = y_train * 2 - 1
	model = LinearClassifierModel(X_train.shape[1])
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

	if criterion == 'hinge':
		loss_fn = lambda y_pred, y_true: torch.mean(torch.clamp(1 - y_true * y_pred, min=0))
	elif criterion == 'cross_entropy':
		loss_fn = torch.nn.BCEWithLogitsLoss()
	else:
		raise ValueError('Unsupported criterion')

	model.train()
	for epoch in range(epochs):
		optimizer.zero_grad()
		outputs = model(X_train).squeeze()

		loss = loss_fn(outputs, y_train)

		loss.backward()
		optimizer.step()

	return model


def lcm_test(model, X_test, y_test, criterion: Literal['hinge', 'cross_entropy'] = 'hinge'):
	model.eval()
	with torch.no_grad():
		outputs = model(X_test).squeeze()
		if criterion == 'hinge':
			predictions = torch.where(outputs >= 0, torch.tensor(1.0), torch.tensor(0.0))
		else:
			probs = torch.sigmoid(outputs)
			predictions = torch.where(probs > 0.5, torch.tensor(1.0), torch.tensor(0.0))

		accuracy = torch.mean((predictions == y_test).float()).item()
	return accuracy
