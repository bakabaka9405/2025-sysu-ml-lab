import torch
from sklearn.svm import SVC
import lcm
from pathlib import Path

def main():
	dataset_root = Path(r'C:\Temp\processed_8_9')
	training_dataset = torch.load(dataset_root / 'training.pt')
	testing_dataset = torch.load(dataset_root / 'test.pt')

	X_train, y_train = training_dataset
	X_test, y_test = testing_dataset

	X_train = X_train.reshape(X_train.shape[0], -1).float()  # (N, 784)
	X_test = X_test.reshape(X_test.shape[0], -1).float()

	torch.nn.functional.normalize(X_train, p=2.0, dim=1, out=X_train)
	torch.nn.functional.normalize(X_test, p=2.0, dim=1, out=X_test)

	y_train = (y_train - 8).float()  # (N,)
	y_test = (y_test - 8).float()

	clf = SVC(C=9, kernel='linear')  # linear kernel
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(f'Test accuracy (SVM linear kernel): {accuracy * 100:.2f}%')

	clf = SVC(C=10, kernel='rbf')  # rbf kernel
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(f'Test accuracy (SVM rbf kernel): {accuracy * 100:.2f}%')

	model = lcm.lcm_train(
		X_train,
		y_train,
		epochs=100,
		lr=0.02,
		criterion='hinge',
	)
	accuracy = lcm.lcm_test(model, X_test, y_test, criterion='hinge')
	print(f'Test accuracy (LCM hinge loss): {accuracy * 100:.2f}%')

	model = lcm.lcm_train(
		X_train,
		y_train,
		epochs=200,
		lr=0.01,
		criterion='cross_entropy',
	)
	accuracy = lcm.lcm_test(model, X_test, y_test, criterion='cross_entropy')
	print(f'Test accuracy (LCM cross entropy): {accuracy * 100:.2f}%')

if __name__ == '__main__':
	main()