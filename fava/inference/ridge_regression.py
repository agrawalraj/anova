
import jax.numpy as jnp
from jax import jit
import jax


@jax.jit
def ridge_weights_cg(K, Y, sigma_sq, tol=.01):
	N = Y.shape[0]
	return  jax.scipy.sparse.linalg.cg(K + sigma_sq*jnp.eye(N), Y,
	                                 tol=tol)[0]


@jax.jit
def ridge_weights(K, Y, sigma_sq):
	N = Y.shape[0]
	return jax.scipy.linalg.inv(K + sigma_sq*jnp.eye(N)).dot(Y)


@jax.jit
def ridge_predict(K, alpha):
	return K.dot(alpha)


def kernel_ridge(K, Y, sigma_sq, opt_params):
	cg = opt_params['cg']
	if cg == True:
		# alpha_prev = opt_params['alpha_prev']
		tol = opt_params['cg_tol']
		return ridge_weights_cg(K, Y, sigma_sq, tol)
	else:
		return ridge_weights(K, Y, sigma_sq)
