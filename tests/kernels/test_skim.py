
import numpy as np
import jax.numpy as jnp
from jax import random
from fova.kernels.skim import *
from sklearn.preprocessing import PolynomialFeatures


def make_feature_map(X, Q):
	# Generate all interaction of order up to Q
	feat_map = PolynomialFeatures(degree=Q, interaction_only=True)
	return jnp.array(feat_map.fit_transform(np.array(X)))


def relative_error(K_est, K_true):
	return jnp.abs(K_est - K_true).sum() / jnp.abs(K_true).sum()


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




