
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

from fava.basis.maps import Identity
from fava.decomposers.decomposer import all_subsets, Decomposer


class Sobol(Decomposer):
    def __init__(self, X_dist, model, Q, key=random.PRNGKey(0)):
        self.featprocessor = Identity()
        self.X_dist = jax.random.permutation(key, X_dist.copy())
        self.p = X_dist.shape[1]
        self.model = model
        self.intercept = model.predict(X_dist).mean() # TODO: replace with predict_proba if needed
        self.selected_covs = jnp.arange(self.p).tolist()
        self.Q = Q

    def __get_effect_uncorrected(self, X, V):
        if len(V) == 0:
            return self.intercept * jnp.ones((X.shape[0],))
        else:
            preds = []
            for xv in X[:, V]:
                X_probe = self.X_dist.copy()
                X_probe = X_probe.at[:, V].set(xv)
                preds.append(model.predict(X_probe).mean()) # TODO: replace with predict_proba if needed
            return jnp.array(preds)

    def get_effect(self, X, V):
        assert len(X.shape) == 2
        effects = dict()
        sorted_subsets = all_subsets(list(V), len(V), True)
        for U in sorted_subsets:
            effect = self.__get_effect_uncorrected(X, U)
            if len(U) > 0:
                lower_order_subsets = all_subsets(U, len(U), True)
                for W in lower_order_subsets:
                    if len(W) < len(U):
                        effect -= effects[W]
            effects[U] = effect
        return effects[tuple(V)]


if __name__ == "__main__":
    class ExampleModel:
        def predict(self, X):
            return X[:, 0] + X[:, 1] + X[:, 0]*X[:, 1]

    model = ExampleModel()
    N = 100
    p = 2
    key = random.PRNGKey(0)
    X = random.normal(key, shape=(N, p))

    sobol_decomposer = Sobol(X, model, 2)
    # print(sobol_decomposer.get_effect(X, [0]))

    decomp = sobol_decomposer.get_decomposition(X)
    sum_check = jnp.zeros(N)
    for val in decomp.values():
        sum_check += val

    print(jnp.abs(sum_check - model.predict(X)).sum())

    print(X[:, 0])
    print(decomp[(0,)])



