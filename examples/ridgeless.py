
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

from fava.inference.fit import GaussianSKIMFA
from fava.basis.maps import LinearBasis
from fava.misc.scheduler import constantScheduler
from fava.misc.logger import GausLogger
from fava.decomposers.tensor_product import TensorProductKernelANOVA, LinearANOVA

# Generate random data
key = random.PRNGKey(0)
N = 1000
p = 750
X = random.normal(key, shape=(N, p))
frac_train = .8
N_train = int(N * frac_train)

# theta = 1 / jnp.sqrt(p) * random.normal(key, shape=(p, 1))
# Y = X.dot(theta) ** 3
Y = X[:, 0] + X[:, 1] + X[:, 2] * X[:, 3]
# print(Y.std())
# X[:, 0] + X[:, 1] + X[:, 2] * X[:, 3] # No noise

X_train = X[:N_train, :]
Y_train = Y[:N_train]

X_valid = X[N_train:, :]
Y_valid = Y[N_train:]

kernel_params = dict()
Q = 2
kernel_params['U_tilde'] = jnp.ones(p) * 1 / jnp.sqrt(p - 1)
kernel_params['eta'] = jnp.ones(Q+1)

hyperparams = dict()
hyperparams['sigma_sq'] = 0. # "ridgeless regression"
hyperparams['c'] = 0.

opt_params = dict()
opt_params['cg'] = True
opt_params['cg_tol'] = .01
# opt_params['alpha_prev'] = jnp.zeros(N - 50)
opt_params['M'] = 20
opt_params['gamma'] = .1
opt_params['T'] = 200


featprocessor = LinearBasis(X_train)
scheduler = constantScheduler()
logger = GausLogger(100)

opt_params['scheduler'] = scheduler

skim = GaussianSKIMFA(X_train, Y_train, X_valid, Y_valid, featprocessor)

skim.fit(key, hyperparams, kernel_params, opt_params, 
            logger=GausLogger())

lanova = LinearANOVA(skim)

for V, effect in anova.items():
	print(V, round(lanova.get_coef(V), 3))

