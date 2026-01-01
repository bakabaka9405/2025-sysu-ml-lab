import numpy as np
from scipy.special import logsumexp
import scipy.linalg
import util
from scipy.spatial.distance import cdist

def log_gaussian(X, mu, Sigma, cov, reg=1e-6):
	n, d = X.shape
	K = mu.shape[0]
	TWO_PI = 2.0 * np.pi

	log_p = np.zeros((n, K))

	for k in range(K):
		diff = X - mu[k]  # (N, d)

		if cov == 'spherical':
			var = Sigma[k]
			log_det = d * np.log(var)
			mahalanobis = np.sum(diff**2, axis=1) / var

		elif cov == 'diag':
			log_det = np.sum(np.log(Sigma[k]))
			mahalanobis = np.sum(diff**2 / Sigma[k], axis=1)

		elif cov == 'full':
			L = np.linalg.cholesky(Sigma[k])

			log_det = 2 * np.sum(np.log(np.diag(L)))
			y = scipy.linalg.solve_triangular(L, diff.T, lower=True)
			mahalanobis = np.sum(y**2, axis=0)
		else:
			assert False

		log_p[:, k] = -0.5 * (d * np.log(TWO_PI) + log_det + mahalanobis)

	return log_p


def em_gmm(
	K=10,
	max_iter=100,
	tol=1e-3,
	reg=1e-3,
	cov='diag',
	init='random',
):
	X_train, X_test, Y_train, Y_test = util.load_data()
	n, d = X_train.shape

	util.reset_random_seed()
	rng = np.random.default_rng(util.config.random_seed)

	# Init mu
	if init == 'random':
		mu = X_train[rng.choice(n, K, replace=False)].copy()
	elif init == 'kmeans++':
		mu = np.empty((K, X_train.shape[1]), dtype=np.float64)
		mu[0] = X_train[np.random.choice(n)]
		for i in range(1, K):
			d = cdist(X_train, mu[:i])
			p = np.square(np.min(d, axis=1))
			p /= p.sum()
			mu[i] = X_train[np.random.choice(n, p=p)]
	else:
		assert False

	if cov == 'spherical':
		Sigma = np.full(K, np.mean(np.var(X_train, axis=0)) + reg)
	elif cov == 'diag':
		Sigma = np.tile(np.var(X_train, axis=0) + reg, (K, 1))
	elif cov == 'full':
		Sigma = np.tile(np.diag(np.var(X_train, axis=0) + reg), (K, 1, 1))
	else:
		assert False

	pi = np.full(K, 1.0 / K)

	hist = []
	gamma = 0
	lb = -np.inf
	i = 0

	for i in range(1, max_iter + 1):
		# E-Step
		if cov != 'full':
			Sigma = np.maximum(Sigma, reg)

		log_p_x_z = log_gaussian(X_train, mu, Sigma, cov, reg) + np.log(pi)
		log_p_x = logsumexp(log_p_x_z, axis=1, keepdims=True)
		gamma = np.exp(log_p_x_z - log_p_x)

		ll = float(np.sum(log_p_x))
		hist.append(ll)
		print(f'iter {i:03d} ll={ll:.2f} d={ll - lb:.6f}')

		if abs(ll - lb) < tol:
			break
		lb = ll

		# M-Step
		N_k = np.sum(gamma, axis=0)

		mu = (gamma.T @ X_train) / N_k[:, None]

		if cov == 'spherical':
			Sigma1 = np.zeros(K)
			for k in range(K):
				diff = X_train - mu[k]
				dist_sq = np.sum(diff**2, axis=1)
				Sigma1[k] = np.sum(gamma[:, k] * dist_sq) / (d * N_k[k]) + reg
			Sigma = Sigma1

		elif cov == 'diag':
			Sigma1 = np.zeros((K, d))
			for k in range(K):
				diff = X_train - mu[k]
				Sigma1[k] = np.sum(gamma[:, k][:, None] * (diff**2), axis=0) / N_k[k] + reg
			Sigma = Sigma1

		elif cov == 'full':
			Sigma1 = np.zeros((K, d, d))
			for k in range(K):
				diff = X_train - mu[k]
				Sigma1[k] = (diff.T * gamma[:, k]) @ diff / N_k[k]
				Sigma1[k].flat[:: d + 1] += reg
			Sigma = Sigma1

		pi = N_k / n
		pi /= pi.sum()

	train_idx = np.argmax(gamma, axis=1).astype(np.int64)
	train_acc = util.clustering_accuracy(K, train_idx, Y_train)

	log_p_x_z_test = log_gaussian(X_test, mu, Sigma, cov, reg) + np.log(pi)
	log_p_x_test = logsumexp(log_p_x_z_test, axis=1, keepdims=True)

	test_idx = np.argmax(log_p_x_z_test, axis=1).astype(np.int64)
	test_acc = util.clustering_accuracy(K, test_idx, Y_test)

	print(f'iters={i}')
	print(f'Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%')
	print(f'll train: {hist[-1]:.2f} | test: {float(np.sum(log_p_x_test)):.2f}')


if __name__ == '__main__':
	em_gmm(cov='spherical', init='random')
	em_gmm(cov='spherical', init='kmeans++')
	em_gmm(cov='diag', init='random')
	em_gmm(cov='diag', init='kmeans++')
	em_gmm(cov='full', init='random')
	em_gmm(cov='full', init='kmeans++')
