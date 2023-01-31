
import jax.numpy as jnp
from jax import jit
import jax


@jax.jit
def ridge_weights_cg(K, Y, sigma_sq, tol=.01, x0=None):
	N = Y.shape[0]
	return  jax.scipy.sparse.linalg.cg(K + sigma_sq*jnp.eye(N), Y,
	                                 tol=tol, x0=x0)[0]


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
		if 'alpha_prev' in opt_params.keys():
			alpha_prev = opt_params['alpha_prev']
		else:
			alpha_prev = None
		tol = opt_params['cg_tol']
		return ridge_weights_cg(K, Y, sigma_sq, tol, x0=alpha_prev)
	else:
		return ridge_weights(K, Y, sigma_sq)
