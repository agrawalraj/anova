
import numpy as np
import jax.numpy as jnp


class WrapProcessor:
	"""
	Args:
		preprocessor: sklearn preprocessor
	Returns:
		preprocessor: sklearn preprocessor with transform method that returns jax array
	
	Example:
		from sklearn.preprocessing import StandardScaler
		from fava.basis.maps import WrapProcessor
		X = jnp.array(np.random.normal(size=(100, 3)))
		preprocessor = WrapProcessor(StandardScaler()).preprocessor
	"""
	def __init__(self, preprocessor):
		self.preprocessor = preprocessor

	def transform(self, X):
		return jnp.array(self.preprocessor.transform(X))


class ScaleInput:
	def __init__(self, X_train, featscaling):
		self.featscaling = featscaling.fit(X_train)

	def transform1D(self, X, covariate_ix):
		assert covariate_ix >= 0
		assert covariate_ix < self.p, f"{covariate_ix} not in range({self.p})"
		return self.featscaling.transform(X)[:, covariate_ix]

	def transform(self, X):
		return self.featscaling.transform(X)


def is_valid_mask(covariate_masks):
	"""
	Args:
		covariate_masks: list of boolean jax arrays of length p
	Returns:
		bool: True if each covariate is in exactly one basis. Else False.
	Examples:
		covariate_masks = [jnp.array([True, False, True]), jnp.array([False, True, False])]
		is_valid_mask(covariate_masks) = True

		covariate_masks = [jnp.array([True, False, True]), jnp.array([False, True, True])]
		is_valid_mask(covariate_masks) = False
	"""
	# Each covariate must be in exactly one basis
	for cov_ix in range(len(covariate_masks[0])):
		if sum([covariate_mask[cov_ix] for covariate_mask in covariate_masks]).item() != 1:
			return False
	return True


def covariate2basis(covariate_masks):
	"""
	Args:
		covariate_masks: list of boolean jax arrays of length p
	Returns:
		dict: covariate_ix -> basis_ix
	Example:
		covariate_masks = [jnp.array([True, False, True]), jnp.array([False, True, False])]
		covariate2basis(covariate_masks) = {0: 0, 1: 1, 2: 0}
	"""
	covariate2basis = {}
	for basis_ix, covariate_mask in enumerate(covariate_masks):
		for cov_ix in jnp.where(covariate_mask)[0].tolist():
			covariate2basis[cov_ix] = basis_ix
	return covariate2basis
