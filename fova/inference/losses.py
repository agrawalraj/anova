
import jax as jnp
from fova.kernels.skim import *
from ridge_regression import *


def mean_squared_error(y_pred, y_actual):
	return jnp.mean(jnp.square(y_pred - y_actual))


def ridge_stochastic_cv_loss(key, X, Y, hyperparams, kernel_params, opt_params):
	# Sample batch of data points of size N - M for fitting ridge regression
	# Evaluate fit on a batch of M random datapoints
	N = X.shape[0]
	M = opt_params['M']
	indcs = jnp.arange(N)
	jax.random.permutation(key, indcs, axis=0)
	train_indcs = indcs[:(N - M)]
	cv_indcs = indcs[(N - M):]

	c = hyperparams['c']
	sigma_sq = hyperparams['sigma_sq']
	K_XX = kernel_matrix(X[train_indcs, :], X[train_indcs, :], c, kernel_params)
	K_ZX = kernel_matrix(X[cv_indcs, :], X[train_indcs, :], c, kernel_params)

	alpha_hat = kernel_ridge(K_XX, Y[train_indcs], sigma_sq, opt_params)
	Y_pred_cv = ridge_predict(K_ZX, alpha_hat)
	Y_true_cv = Y[cv_indcs]

	return mean_squared_error(Y_pred_cv, Y_true_cv)


def update_kernel(key, X, Y, loss, hyperparams, kernel_params, opt_params):
	grads = jax.grad(loss, argnums=4)(key, X, Y, hyperparams, 
										kernel_params, opt_params)
	gamma = opt_params['gamma'] # learning rate
	return jax.tree_map(
		lambda p, g: p - gamma * g, kernel_params, grads
		)
