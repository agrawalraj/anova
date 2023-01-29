
import jax as jnp
from fava.kernels.skim import *
from fava.inference.ridge_regression import *


@jax.jit
def mean_squared_error(y_pred, y_actual):
	return jnp.mean(jnp.square(y_pred - y_actual))


def fit_predict_new(
		X_train, Y_train, X_new, Y_new, hyperparams, kernel_params, opt_params
	):
	c = hyperparams['c']
	sigma_sq = hyperparams['sigma_sq']
	K_XX = skim_kernel_matrix(X_train, X_train, c, kernel_params)
	K_ZX = skim_kernel_matrix(X_new, X_train, c, kernel_params)

	alpha_hat = kernel_ridge(K_XX, Y_train, sigma_sq, opt_params)
	Y_pred = ridge_predict(K_ZX, alpha_hat)

	return mean_squared_error(Y_pred, Y_new), alpha_hat


def ridge_stochastic_cv_loss(key, X, Y, hyperparams, kernel_params, opt_params):
	# Sample batch of data points of size N - M for fitting ridge regression
	# Evaluate fit on a batch of M random datapoints
	N = X.shape[0]
	M = opt_params['M']
	indcs = jnp.arange(N)
	jax.random.permutation(key, indcs, axis=0)
	train_indcs = indcs[:(N - M)]
	cv_indcs = indcs[(N - M):]
	return fit_predict_new(X[train_indcs, :], Y[train_indcs], X[cv_indcs, :], 
			Y[cv_indcs], hyperparams, kernel_params, opt_params)[0]


def update_kernel(key, X, Y, loss, hyperparams, kernel_params, opt_params):
	grads = jax.grad(loss, argnums=4)(key, X, Y, hyperparams, 
										kernel_params, opt_params)
	gamma = opt_params['gamma'] # learning rate
	return jax.tree_map(
		lambda p, g: p - gamma * g, kernel_params, grads
		)
