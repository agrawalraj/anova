
import numpy as np
import jax.numpy as jnp
from jax import vmap
from sklearn.preprocessing import KBinsDiscretizer, SplineTransformer
# from fava.util import is_valid_mask


class WrapProcessor:
	def __init__(self, processor):
		# TODO: check that processor is a valid processor
		self.processor = processor

	def transform1D(self, X, covariate_ix):
		return jnp.array(self.processor.transform1D(X[:, covariate_ix]))

	def transform(self, X):
		return jnp.array(self.processor.transform(X))


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
		self.covariate2basis = 0 # TODO ADD THIS
		self.max_basis_dim = max([basis.max_basis_dim for basis in self.basis_list])
		self.basis_list = basis_list
		self.covariate_masks = covariate_masks

	def transform1D(self, X, covariate_ix):
		basis = self.covariate2basis[covariate_ix]		
		return self.basis.transform1D(X, covariate_ix)

	def transform(self, X):
		N, p = X.shape
		X_feat = jnp.zeros((N, p, self.max_basis_dim))
		for basis, covariate_mask in zip(self.basis_list, self.covariate_masks):
			# TODO: fix array update
			# Need to zero pad the basis dimension
			X_feat[:, covariate_mask, :] = basis.transform(X[:, covariate_mask])
		return X_feat


class ScaleInputBasis(AbstractBasis):
	def __init__(self, X_train, featscaling):
		self.featscaling = featscaling.fit(X_train)
		self.max_basis_dim = 1

	def transform1D(self, X, covariate_ix):
		assert covariate_ix >= 0
		assert covariate_ix < self.p, f"{covariate_ix} not in range({self.p})"
		return self.featscaling.transform(X)[:, covariate_ix]

	def transform(self, X):
		return self.featscaling.transform(X)


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
	def __init__(self, X_train, basis1d, scale_basis=Identity()):
		super().__init__(X_train)
		self.basis1d = basis1d
		self.basis_mapped = vmap(basis1d)
		self.scale_basis = scale_basis
		X_train = self.scale_basis.transform(X_train)
		assert X_train.shape[1] == self.p, "Scaling must not change the number of input features"
		X_feat = jnp.transpose(self.basis_mapped(X_train), axes=(0, 2, 1))
		self.max_basis_dim = X_feat.shape[2]
		self._basis_mean = X_feat.mean(axis=0)
		self._basis_sd = X_feat.std(axis=0)

	def transform1D(self, X, covariate_ix):
		x = self.scale_basis.transform1D(X, covariate_ix)
		x = self.basis1d(x)
		return (x - self._basis_mean[covariate_ix]) / self._basis_sd[covariate_ix]

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


class AutoBasis(FiniteDimensionalBasis):
	def __init__(self, X_train, scale_basis=Identity(), max_basis_dim=10, min_unique_cont=10):
		# Check each of the types in X_train and predict what type they are
		# Then create a basis for each type
		super().__init__(X_train)
		self.scale_basis = scale_basis
		self.max_basis_dim = max_basis_dim
		self.min_unique_cont = min_unique_cont
		X_train = self.scale_basis.transform(X_train)
		assert X_train.shape[1] == self.p, "Scaling must not change the number of input features"
		self.basis_list = []
		self.covariate_masks = []
		self.basis_types = []
		for covariate_ix in range(self.p):
			basis, covariate_mask, basis_type = self._get_basis(X_train[:, covariate_ix], covariate_ix)
			self.basis_list.append(basis)
			self.covariate_masks.append(covariate_mask)
		self.basis = BasisUnion(self.basis_list, self.covariate_masks)
	
	def _get_basis(self, x1d, covariate_ix):
		covariate_mask = jnp.zeros(self.p)
		covariate_mask[covariate_ix] = 1
		if len(set(x1d)) < self.min_unique_cont: # X is categorical
			basis_type = "categorical"
			basis = LinearBasis(x1d.reshape(-1, 1))
			# Do linear basis
		else: # X is continuous
			# power basis (trend) + wavelet basis (seasonality)
			basis_type = "continuous"

		return basis, covariate_mask, basis_type 

	def transform1D(self, X, covariate_ix):
		x = self.scale_basis.transform1D(X, covariate_ix)
		return self.basis_list[covariate_ix].transform1D(x, 0)
	
	def transform(self, X):
		X = self.scale_basis.transform(X)
		return self.basis.transform(X)


class SplineBasis(AbstractBasis):
	def __init__(self, X_train, n_knots=5, degree=3, **kwargs):
		spline = SplineTransformer(
						degree=degree, 
			    		n_knots=n_knots, 
						include_bias=False, **kwargs)
		spline.fit(X_train)
		self.basis = WrapProcessor(spline)
		
	def transform1D(self, X, covariate_ix):
		return self.basis.transform1D(X, covariate_ix)
	
	def transform(self, X):
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
	print(X_feat)
	print(X_feat.shape)