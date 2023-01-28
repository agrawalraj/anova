
import numpy as np
import jax.numpy as jnp
from jax import random
from sklearn.preprocessing import PolynomialFeatures

from fova.kernels.skim import *
from fova.basis.maps import LinearBasis
from fova.decomposers.tensor_product import TensorProductKernelANOVA, LinearANOVA
from fova.inference.fit import GaussianSKIMFA
from fova.misc.scheduler import truncScheduler
from fova.misc.logger import GausLogger
from sklearn.linear_model import Ridge


def make_feature_map(X, Q, include_bias=True):
	# Generate all interaction of order up to Q
	feat_map = PolynomialFeatures(degree=Q, interaction_only=True, 
									include_bias=include_bias)

	X_feat = feat_map.fit_transform(np.array(X))
	return feat_map.get_feature_names_out(), X_feat


def test_anova():
	key = random.PRNGKey(0)
	N = 200
	p = 5
	X = random.normal(key, shape=(N, p))
	Y = X[:, 0] + X[:, 1] + X[:, 2] * X[:, 3]

	X_train = X[:100, :]
	Y_train = Y[:100]

	X_valid = X[100:, :]
	Y_valid = Y[100:]

	kernel_params = dict()
	Q = 2
	kernel_params['U_tilde'] = 1e6 * jnp.ones(p)
	kernel_params['eta'] = jnp.ones(Q+1)

	hyperparams = dict()
	hyperparams['sigma_sq'] = .5 * jnp.var(Y)
	hyperparams['c'] = 0.

	opt_params = dict()
	opt_params['cg'] = True
	opt_params['cg_tol'] = .001
	opt_params['M'] = 20
	opt_params['gamma'] = 0 # don't update params
	opt_params['T'] = 1

	featprocessor = LinearBasis(X_train)
	scheduler = truncScheduler()
	logger = GausLogger(100)

	X_normed = featprocessor.transform(X)
	X_normed = X_normed.reshape((N, p))

	opt_params['scheduler'] = scheduler

	skim = GaussianSKIMFA(X_train, Y_train, X_valid, Y_valid, featprocessor)

	skim.fit(key, hyperparams, kernel_params, opt_params, 
	            logger=GausLogger())

	decompose = TensorProductKernelANOVA(skim)
	anova = decompose.get_decomposition(X_train)

	linear_decompose = LinearANOVA(skim)
	lanova = linear_decompose.get_decomposition(X_train)

	print('Testing SKIM formula vs. special linear SKIM formula')
	for V, effect in anova.items():
		leffect = lanova[V]
		assert jnp.abs((effect - leffect)).sum() < 1e-4

	print('\n')

	# Compare difference with explicit regression
	print('Testing linear SKIM formula vs. explicit regression')
	feat_names, X_explicit = make_feature_map(np.array(X_normed), Q)
	ridge = Ridge(alpha=hyperparams['sigma_sq'].item(), fit_intercept=False)
	ridge.fit(X_explicit[:100, :], Y[:100])

	for V, __ in lanova.items():
		beta = linear_decompose.get_coef(V)
		if len(V) == 0:
			effect_ix = np.where(feat_names == '1')[0][0]
		if len(V) == 1:
			cov_ix = V[0]
			effect_ix = np.where(feat_names == f'x{cov_ix}')[0][0]
		if len(V) == 2:
			cov_ix1, cov_ix2 = sorted(V)
			effect_ix = np.where(feat_names == f'x{cov_ix1} x{cov_ix2}')[0][0]

		print(V, ridge.coef_[effect_ix], beta)
		assert jnp.abs((ridge.coef_[effect_ix] - beta)).sum() < 1e-2

