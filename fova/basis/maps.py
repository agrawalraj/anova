
import jax.numpy as jnp
from jax import vmap


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

	def transform(self, X, normalize=True):
		if normalize:
			N, p = X.shape
			X_feat = (X - self._mean) / self._sd
			return X_feat.reshape((N, p, 1))
		return X.copy().reshape((N, p, 1))


class RepeatedFiniteBasis(object):
	def __init__(self, X_train, basis1d):
		self.basis1d = basis1d
		self.basis_mapped = vmap(basis1d)
		X_feat = jnp.transpose(self.basis_mapped(X_train), axes=(0, 2, 1))
		self._basis_mean = X_feat.mean(axis=0)
		self._basis_sd = X_feat.std(axis=0)

	def transform(self, X, normalize=True):
		X_feat = jnp.transpose(self.basis_mapped(X), axes=(0, 2, 1))
		if normalize:
			return (X_feat - self._basis_mean) / self._basis_sd
		else:
			return X_feat


class BasisComposer(object):
	# LinearBasis (zero mean input) ---> repeated finite basis 
	pass


# @jax.jit
# def transform1D(x1d, n_cuts, quantile):


# @jax.jit
# def transform1D(n_cuts, ):