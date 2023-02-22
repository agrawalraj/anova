
from jax import random
import jax.numpy as jnp
from fava.decomposers.sobol import Sobol


class ToyRegressionFunc:
    def predict(self, X):
        return X[:, 0] + X[:, 1] + X[:, 0]*X[:, 1]


def test_sobol():
    model = ToyRegressionFunc()
    N = 100
    p = 2
    key = random.PRNGKey(0)
    X = random.normal(key, shape=(N, p))

    # Standardize X to be mean-zero and unit variance
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    sobol_decomposer = Sobol(X, model, 2)

    # Make sure the sum of the effects in the decomposition add up to 
    # the prediction from the regression function

    decomp = sobol_decomposer.get_decomposition(X)
    sum_check = jnp.zeros(N)
    for val in decomp.values():
        sum_check += val

    assert jnp.abs(sum_check - model.predict(X)).sum() < 1e3

    # Since the covariates are standardized the main effect for covariate 0
    # should match the first column of X
    assert jnp.abs(X[:, 0] - decomp[(0,)]).sum() < 1e3
