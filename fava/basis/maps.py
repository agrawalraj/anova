
import numpy as np
import jax.numpy as jnp
from jax import vmap
from sklearn.preprocessing import KBinsDiscretizer, SplineTransformer
from fava.basis.utils import WrapProcessor, is_valid_mask, covariate2basis


class AbstractBasis:
	def __init__(self, X_train):
		N_train, p = X_train.shape
		self.N_train = N_train
		self.p = p

	def transform1D(self, X, covariate_ix):
		pass

	def transform(self, X):
		pass


class BasisUnion(AbstractBasis):
	def __init__(self, basis_list, covariate_masks):
		assert is_valid_mask(covariate_masks), "Invalid covariate masks"
		self.covariate2basis = covariate2basis(covariate_masks)
		self.max_basis_dim = max([basis.max_basis_dim for basis in basis_list])
		self.basis_list = basis_list
		self.covariate_masks = covariate_masks

	def transform1D(self, X, covariate_ix):
		basis = self.covariate2basis[covariate_ix]		
		return self.basis.transform1D(X, covariate_ix)

	def transform(self, X):
		N, p = X.shape
		X_feat = jnp.zeros((N, p, self.max_basis_dim))
		for basis, covariate_mask in zip(self.basis_list, self.covariate_masks):
			# Update X_feat with zero padding (if necessary)
			X_feat = X_feat.at[:, covariate_mask, :].set(jnp.pad(basis.transform(X[:, covariate_mask]), ((0, 0), (0, 0), (0, self.max_basis_dim - basis.max_basis_dim))))
		return X_feat


# TODO: fix this
class Identity(AbstractBasis):
	def __init__(self):
		self.max_basis_dim = 1
	
	def transform1D(self, X, covariate_ix):
		return X[:, covariate_ix]

	def transform(self, X):
		return X


class FiniteDimensionalBasis(AbstractBasis):
	def transform1D(self, X, covariate_ix):
		pass

	def transform(self, X):
		p = self.p
		# 0 pad to match the last basis dimension
		X_feat = jnp.zeros((self.N_train, p, self.max_basis_dim))


class LinearBasis(FiniteDimensionalBasis):
	def __init__(self, X_train):
		super().__init__(X_train)
		self._mean = X_train.mean(axis=0)
		self._sd = X_train.std(axis=0)
		self.max_basis_dim = 1

	def transform1D(self, X, covariate_ix):
		assert covariate_ix >= 0
		assert covariate_ix < self.p, f"{covariate_ix} not in range({self.p})"
		return (X[:, covariate_ix] - self._mean[covariate_ix]).reshape((-1, 1)) / self._sd[covariate_ix]
	
	def transform(self, X):
		N, p = X.shape
		X_feat = (X - self._mean) / self._sd
		return X_feat.reshape((N, p, 1))


class RepeatedFiniteBasis(FiniteDimensionalBasis):
	def __init__(self, X_train, basis1d, scale_input=Identity()):
		super().__init__(X_train)
		self.basis1d = basis1d
		self.basis_mapped = vmap(basis1d)
		self.scale_input = scale_input
		X_train = self.scale_input.transform(X_train)
		assert X_train.shape[1] == self.p, "Scaling must not change the number of input features"
		X_feat = jnp.transpose(self.basis_mapped(X_train), axes=(0, 2, 1))
		self.max_basis_dim = X_feat.shape[2]
		self._basis_mean = X_feat.mean(axis=0)
		self._basis_sd = X_feat.std(axis=0)

	def transform1D(self, X, covariate_ix, normalize=True):
		x = self.scale_input.transform1D(X, covariate_ix)
		x = self.basis1d(x)
		if normalize:
			return (x - self._basis_mean[covariate_ix, :]) / self._basis_sd[covariate_ix, :]
		else:
			return x

	def transform(self, X, normalize=True):
		X_feat = self.scale_input.transform(X)
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
		self.max_basis_dim = X_feat.shape[2]
		self.nbins = nbins
		self._basis_mean = X_feat.mean(axis=0)
		self._basis_sd = X_feat.std(axis=0)
	
	def transform1D(self, X, covariate_ix):
		pass

	def transform(self, X, normalize=True):
		N, p = X.shape
		nbins = self.nbins
		X_feat = jnp.array(self.transformer.transform(np.array(X))).reshape((N, p, nbins))
		if normalize:
			return (X_feat - self._basis_mean) / self._basis_sd
		else:
			return X_feat


class SplineBasis(FiniteDimensionalBasis):
	"""
	Args:
		X_train: Jax ndarray of shape (N, p)
		kwargs: Arguments passed to SplineTransformer
	
	Returns:
		SplineBasis object

	Example:
		spline = SplineBasis(X_train, n_knots=10, degree=3)
	"""
	def __init__(self, X_train, **kwargs):
		assert X_train.ndim == 2, "X_train must be 2D"
		super().__init__(X_train)
		spline = SplineTransformer(**kwargs)
		X_feat = jnp.array(spline.fit_transform(X_train))
		X_feat = X_feat.reshape((X_train.shape[0], X_train.shape[1], -1))
		self.basis = WrapProcessor(spline)
		self.max_basis_dim = X_feat.shape[2]
		self._basis_mean = X_feat.mean(axis=0)
		self._basis_sd = X_feat.std(axis=0)
		
	def transform1D(self, X, covariate_ix, normalize=True):
		return self.transform(X, normalize)[:, covariate_ix, :]

	def transform(self, X, normalize=True):
		X_feat = self.basis.transform(X).reshape((X.shape[0], self.p, self.max_basis_dim))
		if normalize:
			return (X_feat - self._basis_mean) / self._basis_sd
		else:
			return X_feat


class AutoBasis(FiniteDimensionalBasis):
	def __init__(self, X_train, scale_input=Identity(), min_unique_cont=10, **kwargs):
		super().__init__(X_train)
		self.scale_input = scale_input
		self.min_unique_cont = min_unique_cont
		X_train = self.scale_input.transform(X_train)
		assert X_train.shape[1] == self.p, "Scaling must not change the number of input features"
		n_unique_by_covariate = [len(set(X_train[:, covariate_ix].tolist())) for covariate_ix in range(self.p)]
		self.basis_types = ['categorical' if n_unique < self.min_unique_cont else 'continuous' for n_unique in n_unique_by_covariate]
		continuous_mask = jnp.array([basis_type == 'continuous' for basis_type in self.basis_types])
		categorical_mask = jnp.array([basis_type == 'categorical' for basis_type in self.basis_types])
		self.covariate_masks = []
		self.basis_list = []
		if jnp.any(continuous_mask).item():
			self.covariate_masks.append(continuous_mask)
			self.basis_list.append(SplineBasis(X_train[:, continuous_mask], **kwargs))
		if jnp.any(categorical_mask).item():
			self.covariate_masks.append(categorical_mask)
			self.basis_list.append(LinearBasis(X_train[:, categorical_mask]))
		self.max_basis_dim = max([basis.max_basis_dim for basis in self.basis_list])
		self.basis = BasisUnion(self.basis_list, self.covariate_masks)
	
	def transform1D(self, X, covariate_ix):
		X = self.scale_input.transform(X)
		return self.basis.transform1D(X, covariate_ix)
	
	def transform(self, X):
		X = self.scale_input.transform(X)
		return self.basis.transform(X)


if __name__	== "__main__":
	# generate a random dataset
	N = 1000
	p = 2
	X = np.random.randn(N, p)
	X = jnp.array(X)
	# create a basis
	basis = SplineBasis(X)
	# transform the data
	X_feat = basis.transform(X)
	basis.transform1D(X, 0)
	# print(X_feat)
	# print(X_feat.shape)

	# Add a binary covariate to X
	X = np.random.randn(N, p)
	X = np.hstack((X, np.random.binomial(1, 0.5, size=(N, 1))))
	X = jnp.array(X)
	# create a basis
	basis = AutoBasis(X)
	# transform the data
	X_feat = basis.transform(X)
	# print(X_feat)
	print(X_feat.shape)
	# print(X_feat[:, 2, :])
	# print('\n')
	# print((X[:, 2] - X[:, 2].mean()) / X[:, 2].std())
