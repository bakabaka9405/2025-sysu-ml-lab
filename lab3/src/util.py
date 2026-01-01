import config
import pandas
import random
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def reset_random_seed():
	random.seed(config.random_seed)


def load_data():
	train_csv = config.dataset_root / 'mnist_train.csv'
	test_csv = config.dataset_root / 'mnist_test.csv'
	train_data = pandas.read_csv(train_csv, header=0).to_numpy()
	test_data = pandas.read_csv(test_csv, header=0).to_numpy()
	X_train = train_data[:, 1:].astype(np.float64) / 255.0
	X_test = test_data[:, 1:].astype(np.float64) / 255.0
	Y_train = train_data[:, 0].astype(np.int64)
	Y_test = test_data[:, 0].astype(np.int64)
	return X_train, X_test, Y_train, Y_test


def clustering_accuracy(n: int, y_cluster: NDArray[np.int64], y_true: NDArray[np.int64]) -> float:
	matrix = np.zeros((n, n), dtype=np.int64)
	np.add.at(matrix, (y_cluster, y_true), 1)
	rows, cols = linear_sum_assignment(matrix, maximize=True)
	return float(matrix[rows, cols].sum()) / len(y_true)
