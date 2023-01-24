
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

from fova.inference.fit import GaussianSKIMFA
from fova.basis.maps import LinearBasis
from fova.misc.scheduler import truncScheduler
from fova.misc.logger import GausLogger
from fova.decomposers.tensor_product import TensorProductKernelANOVA, LinearANOVA

# Generate random data
key = random.PRNGKey(0)
N = 200
p = 5
X = random.normal(key, shape=(N, p))
Y = X[:, 0] + X[:, 1] + X[:, 2] * X[:, 3]

X_train = X[:100, :]
Y_train = Y[:100]

X_valid = X[:100, :]
Y_valid = Y[:100]

kernel_params = dict()
Q = 2
kernel_params['U_tilde'] = jnp.ones(p)
kernel_params['eta'] = jnp.ones(Q+1)

hyperparams = dict()
hyperparams['sigma_sq'] = .5 * jnp.var(Y)
hyperparams['c'] = .1

opt_params = dict()
opt_params['cg'] = True
opt_params['cg_tol'] = .01
# opt_params['alpha_prev'] = jnp.zeros(N - 50)
opt_params['M'] = 20
opt_params['gamma'] = .1
opt_params['T'] = 100


featprocessor = LinearBasis(X_train)
scheduler = truncScheduler()
logger = GausLogger(100)

opt_params['scheduler'] = scheduler

skim = GaussianSKIMFA(X_train, Y_train, X_valid, Y_valid, featprocessor)

skim.fit(key, hyperparams, kernel_params, opt_params, 
            logger=GausLogger())


decompose = TensorProductKernelANOVA(skim)

anova = decompose.get_decomposition(X_train)

for V, effect in anova.items():
	print(V, round(effect.var(), 3))

print('\n')

lanova = LinearANOVA(skim)

for V, effect in anova.items():
	print(V, round(lanova.get_coef(V), 3))


X_train_feat = skim.X_train_feat
check1_linear = lanova.get_coef([2, 3]) * X_train_feat[:, 2, 0] * X_train_feat[:, 3, 0]
check1_kernel = anova[(2, 3)]

print(jnp.abs(check1_linear - check1_kernel).sum())



