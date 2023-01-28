
import numpy as np
import jax.numpy as jnp
from jax import random
from sklearn.preprocessing import PolynomialFeatures

from fova.kernels.skim import *
from fova.basis.maps import RepeatedFiniteBasis


def make_feature_map(X, Q, include_bias=True):
	# Generate all interaction of order up to Q
	feat_map = PolynomialFeatures(degree=Q, interaction_only=True, 
									include_bias=include_bias)
	return jnp.array(feat_map.fit_transform(np.array(X)))


def relative_error(K_est, K_true):
	return jnp.abs(K_est - K_true).sum() / jnp.abs(K_true).sum()


def two_dimensional_feat_map_helper(X):
	assert X.shape[1] == 2
	N = X.shape[0]
	ones = jnp.ones((N, 1))
	X1 = X[:, 0].reshape((N, 1))
	X2 = X[:, 1].reshape((N, 1))
	X1_sq = (X1 * X1).reshape((N, 1))
	X2_sq = (X2 * X2).reshape((N, 1))
	X1_X2 = (X1 * X2).reshape((N, 1))
	X1_sq_X2_sq = (X1_sq * X2_sq).reshape((N, 1))
	X1_X2_sq = (X1 * X2_sq).reshape((N, 1))
	X1_sq_X2 = (X1_sq * X2).reshape((N, 1))
	return jnp.concatenate([ones, X1, X2, X1_sq, X2_sq, X1_X2, 
							X1_sq_X2_sq, X1_X2_sq, X1_sq_X2], axis=1)


def test_skim_matches_weight_space_view_linear():
	# Create random data
	N = 100
	p = 5
	key = random.PRNGKey(0)
	X = random.normal(key, shape=(N, p))

	for Q in [0, 1, 2, 3, 4]: # Test formula for multiple interaction orders
		# SKIM kernel
		kernel_params = dict()
		c = 0.
		kernel_params['U_tilde'] = 1e6 * jnp.ones(p)

		# Choice of u_tilde and c should make kappa basically all ones
		assert jnp.abs(get_kappa(kernel_params['U_tilde'], c) - jnp.ones(p)).sum() < 1e-5

		kernel_params['eta'] = jnp.ones(Q+1)
		K_skim = skim_kernel_matrix(X.reshape(N, p, 1), 
									X.reshape(N, p, 1), c, kernel_params)

		# Kernel matrix from explicit map
		X_explicit = make_feature_map(X, Q)
		K_explicit = X_explicit.dot(X_explicit.T)

		assert relative_error(K_skim, K_explicit) < 1e-5


def test_skim_matches_weight_space_view_basis():
	# Create random data
	N = 100
	p = 2
	key = random.PRNGKey(0)
	X = random.normal(key, shape=(N, p))
	basis1d = lambda x: jnp.array([x, x ** 2])
	basis = RepeatedFiniteBasis(X, basis1d)

	for Q in [2]:
		# SKIM kernel
		kernel_params = dict()
		c = 0.
		kernel_params['U_tilde'] = 1e6 * jnp.ones(p)

		# Choice of u_tilde and c should make kappa basically all ones
		assert jnp.abs(get_kappa(kernel_params['U_tilde'], c) - jnp.ones(p)).sum() < 1e-5

		kernel_params['eta'] = jnp.ones(Q+1)
		K_skim = skim_kernel_matrix(basis.transform(X, False), 
									basis.transform(X, False), c, kernel_params)

		# Kernel matrix from explicit map
		X_explicit = two_dimensional_feat_map_helper(X)
		K_explicit = X_explicit.dot(X_explicit.T)

		assert relative_error(K_skim, K_explicit) < 1e-5

