import numpy as np
from typing import Literal
from scipy.spatial.distance import cdist

import util


def kmeans(
	K: int = 10,
	max_iter: int = 100,
	init: Literal['random', 'kmeans++'] = 'kmeans++',
	tol: float = 1e-4,
):	
	util.reset_random_seed()
	X_train, X_test, Y_train, Y_test = util.load_data()

	n = X_train.shape[0]
	if init == 'random':
		mu = X_train[np.random.choice(n, K, replace=False)]
	else:
		mu = np.empty((K, X_train.shape[1]), dtype=np.float64)
		mu[0] = X_train[np.random.choice(n)]
		for i in range(1, K):
			d = cdist(X_train, mu[:i])
			p = np.square(np.min(d, axis=1))
			p /= p.sum()
			mu[i] = X_train[np.random.choice(n, p=p)]
	
	d = cdist(X_train, mu)
	r = np.argmin(d, axis=1)
	j = float(np.sum(d[np.arange(n), r] ** 2))

	i = 0
	for i in range(1, max_iter + 1):
		mu1 = mu.copy()
		for k in range(K):
			mask = r == k
			if np.any(mask):
				mu1[k] = X_train[mask].mean(axis=0)
			else:
				mu1[k] = X_train[np.random.randint(0, n)]

		shift = np.linalg.norm(mu1 - mu)
		mu = mu1
		d = cdist(X_train, mu)
		r = np.argmin(d, axis=1)
		j = float(np.sum(d[np.arange(n), r] ** 2))

		print(f'i={i:03d} | J={j:.2f} | shift={shift:.6f}')

		if shift < tol:
			break

	train_acc = util.clustering_accuracy(10, r, Y_train)
	test_r = np.argmin(cdist(X_test, mu), axis=1)
	test_acc = util.clustering_accuracy(10, test_r, Y_test)

	print(f'Converged in {i} iterations')
	print(f'Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%')