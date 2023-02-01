
import pandas as pd
import numpy as np

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import math

from fava.inference.fit import GaussianSKIMFA
from fava.basis.maps import LinearBasis, RepeatedFiniteBasis, TreeBasis
from fava.misc.scheduler import constantScheduler
from fava.misc.logger import GausLogger
from fava.decomposers.tensor_product import TensorProductKernelANOVA, LinearANOVA
from sklearn.ensemble import RandomForestRegressor


data = pd.read_excel('./data/Concrete_Data.xls') 
data = data.sample(frac=1) # shuffle all the data

# # Remove day variable so that all features are continuous
# data = data.drop(['Age (day)'], axis=1)

X = data.values[:, :-1].copy()

X = (X - X.mean(axis=0)) / X.std(axis=0)

p = X.shape[1]
Y = data.values[:, -1].copy()
avg_Y = Y.mean()
std_Y = Y.std()
Y = (Y - avg_Y) / std_Y

f = RandomForestRegressor(n_estimators=5000, oob_score=True)
f.fit(X, Y)



print(f.oob_score_)

np.random.seed(42)
N_fake = 1000
X_fake = np.random.uniform(low=-5, high=5, size=(N_fake, X.shape[1]))

f_X = f.predict(X_fake)

print(f'Var(f(X)): {f.predict(X).var()}')

#### Project onto two-way linear interaction model

X_fake = jnp.array(X)

# Generate random data
key = random.PRNGKey(0)
N, p = X_fake.shape
Z = jnp.array(X_fake)
frac_train = .8
N_train = int(N * frac_train)

# Y_syn = jnp.array(f_X)
Y_syn = jnp.array(Y)

Z_train = Z[:N_train, :]
Y_train = Y_syn[:N_train]

# Z_valid = Z[N_train:, :]
# Y_valid = Y_syn[N_train:]

Z_valid = Z[N_train:, :]
Y_valid = Y[N_train:]


# Ridge regression
def make_feature_map(X, Q, include_bias=True):
	# Generate all interaction of order up to Q
	feat_map = PolynomialFeatures(degree=Q, interaction_only=True, 
									include_bias=include_bias)

	trans = feat_map.fit_transform(np.array(X))
	print(feat_map.get_feature_names_out())
	return trans



from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100]).fit(make_feature_map(Z_train, 2), np.array(Y_train))
r2_score(Y_valid, ridge.predict(make_feature_map(Z_valid, 2)))

# Z_valid = jnp.array(X)
# Y_valid = jnp.array(f.predict(X))


kernel_params = dict()
Q = 2
kernel_params['U_tilde'] = .1 * jnp.ones(p)
kernel_params['eta'] = jnp.ones(Q+1)

hyperparams = dict()
hyperparams['sigma_sq'] = .5 # "ridgeless regression"
hyperparams['c'] = 0.

opt_params = dict()
opt_params['cg'] = True
opt_params['cg_tol'] = .01
# opt_params['alpha_prev'] = jnp.zeros(N - 50)
opt_params['M'] = 100
opt_params['gamma'] = .1
opt_params['T'] = 1000


# CV noise, scale start (bottom up or top down), 

# featprocessor = LinearBasis(Z_train)
# featprocessor = TreeBasis(Z, n_bins=10)
featprocessor = LinearBasis(Z_train)

scheduler = constantScheduler()
logger = GausLogger(100)

opt_params['scheduler'] = scheduler

skim = GaussianSKIMFA(Z_train, Y_train, Z_valid, Y_valid, featprocessor)

skim.fit(key, hyperparams, kernel_params, opt_params, 
            logger=GausLogger())

decompose = TensorProductKernelANOVA(skim)

anova = decompose.get_decomposition(jnp.array(X)) # Look at training

for V, effect in anova.items():
	print(V, round(effect.var(), 3))

