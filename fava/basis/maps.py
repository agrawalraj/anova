
import numpy as np
import jax.numpy as jnp
from jax import vmap
from sklearn.preprocessing import KBinsDiscretizer
from fava.util import is_valid_mask


class AbstractBasis(object):
	def __init__(self, X_train):
		pass

	def transform1D(self, X, covariate_ix):
		pass

	def transform(self, X):
		pass


class BasisUnion(AbstractBasis):
	def __init__(self, basis_list, covariate_masks):
		assert is_valid_mask(covariate_masks), "Invalid covariate masks"
		self.covariate2basis = # TODO ADD THIS
		self.max_basis_dim = # TODO ADD THIS
		self.basis_list = basis_list
		self.covariate_masks = covariate_masks

	def transform1D(self, X, covariate_ix):
		basis = self.covariate2basis[covariate_ix]		
		return self.basis.transform1D(X, covariate_ix)

	def transform(self, X):
		X_feat = jnp.zeros((N, p, self.max_basis_dim))
		for basis, covariate_mask in zip(self.basis_list, self.covariate_masks):
			X_feat[:, covariate_mask, :] = basis.transform(X[:, covariate_mask])
		return X_feat


class ScaleInputBasis(AbstractBasis):
	def __init__(self, X_train, featscaling):
		self.featscaling = featscaling.fit(X_train)

	def transform1D(self, X, covariate_ix):
		assert covariate_ix >= 0
		assert covariate_ix < self.p, f"{covariate_ix} not in range({self.p})"
		return self.featscaling.transform(X[:, covariate_ix])

	def transform(self, X):
		return self.featscaling.transform(X)


class Identity(AbstractBasis):
	def __init__(self):
		pass
	
	def transform1D(self, X, covariate_ix):
		return X[:, covariate_ix]

	def transform(self, X):
		return X


class FiniteDimensionalBasis(AbstractBasis):
	def __init__(self, X_train):
		self.p = X_train.shape[1]

	def transform1D(self, X, covariate_ix):
		pass

	def transform(self, X):
		p = self.p
		return jnp.concat([self.transform1D(X[:, covariate_ix]) for covariate_ix in range(p)])


class LinearBasis(FiniteDimensionalBasis):
	def __init__(self, X_train):
		super().__init__(X_train)
		self._mean = X_train.mean(axis=0)
		self._sd = X_train.std(axis=0)

	def transform1D(self, X, covariate_ix):
		assert covariate_ix >= 0
		assert covariate_ix < self.p, f"{covariate_ix} not in range({self.p})"
		return X[:, covariate_ix] - self._mean[covariate_ix]
	
	def transform(self, X):
		N, p = X.shape
		X_feat = (X - self._mean) / self._sd
		return X_feat.reshape((N, p, 1))


class RepeatedFiniteBasis(FiniteDimensionalBasis):
	def __init__(self, X_train, basis1d, scale_basis):
		self.p = X_train.shape[1]
		self.basis1d = basis1d
		self.basis_mapped = vmap(basis1d)
		self.scale_basis = scale_basis
		X_feat = self.scale_basis.transform(X_train)
		assert X_feat.shape[1] == self.p, "Scaling must not change the number of input features"
		X_feat = jnp.transpose(self.basis_mapped(X_train), axes=(0, 2, 1))
		self._basis_mean = X_feat.mean(axis=0)
		self._basis_sd = X_feat.std(axis=0)

	def transform1D(self, X, covariate_ix):
		pass

	def transform(self, X, normalize=True):
		X_feat = self.scale_basis.transform(X)
		X_feat = jnp.transpose(self.basis_mapped(X_feat), axes=(0, 2, 1))
		if normalize:
			return (X_feat - self._basis_mean) / self._basis_sd
		else:
			return X_feat


class TreeBasis(FiniteDimensionalBasis):
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
	
	def transform1D(self, X, covariate_ix):
		return pass

	def transform(self, X, normalize=True):
		N, p = X.shape
		nbins = self.nbins
		X_feat = jnp.array(self.transformer.transform(np.array(X))).reshape((N, p, nbins))
		if normalize:
			return (X_feat - self._basis_mean) / self._basis_sd
		else:
			return X_feat
