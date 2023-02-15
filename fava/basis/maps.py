
import numpy as np
import jax.numpy as jnp
from jax import vmap
from sklearn.preprocessing import KBinsDiscretizer


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


class TreeBasis(object):
	def __init__(self, X_train, **kwargs):
		self.transformer = KBinsDiscretizer(encode='onehot-dense', **kwargs)
		N, p = X_train.shape
		X_feat = jnp.array(self.transformer.fit_transform(np.array(X_train)))
		assert len(set(self.transformer.n_bins_)) == 1, "Different number of bins!"
		nbins = self.transformer.n_bins_[0]
		X_feat = X_feat.reshape((N, p, nbins))
		self.nbins = nbins
		self._basis_mean = X_feat.mean(axis=0)
		self._basis_sd = X_feat.std(axis=0)

	def transform(self, X, normalize=True):
		N, p = X.shape
		nbins = self.nbins
		X_feat = jnp.array(self.transformer.transform(np.array(X))).reshape((N, p, nbins))
		if normalize:
			return (X_feat - self._basis_mean) / self._basis_sd
		else:
			return X_feat


class Identity(object):
	def __init__(self):
		pass

	def transform(self, X):
		return X


class BasisComposer(object):
	# LinearBasis (zero mean input) ---> repeated finite basis 
	pass


# @jax.jit
# def transform1D(x1d, n_cuts, quantile):


# @jax.jit
# def transform1D(n_cuts, ):