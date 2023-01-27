

class FiniteDimensionalBasis(object):
	def __init__(self, X_train, basis_config):
		self.type = 'abstract'

	def transform1D(self, x1D):
		pass

	def transform(self, X):
		p = self.p
		return jnp.concat([self.transform1D(X[:, covariate_ix]) for covariate_ix in range(p)])


class LinearBasis(object):
	def __init__(self, X_train):
		self._mean = X_train.mean(axis=0)
		self._sd = X_train.std(axis=0)

	def transform(self, X):
		N, p = X.shape
		X_feat = (X - self._mean) / self._sd
		return X_feat.reshape((N, p, 1))


class TreeBasis(object):
	def __init__(self, X_train, n_cuts, quantile):
		self._mean = X_train.mean(axis=0)
		self._sd = X_train.std(axis=0)

	def transform(self, X):
		N, p = X.shape
		X_feat = (X - self._mean) / self._sd
		return X_feat.reshape((N, p, 1))


@jax.jit
def transform1D(x1d, n_cuts, quantile):


@jax.jit
def transform1D(n_cuts, ):