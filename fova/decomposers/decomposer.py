
"""
TODO: say what this does
"""

from abc import ABC
import jax.numpy as jnp
from tqdm import tqdm
from itertools import chain, combinations


def all_subsets(selected, q, include_lower=True):
    s = list(selected)
    if include_lower:
        return list(chain.from_iterable(combinations(s, r) for r in range(q+1)))
    else:
        return list(combinations(s, q))


class Decomposer(ABC):

    """Abstract class for decomposing a regression function into additive
    and interaction effects.
    """
    def __init__(self, model):
        hyperparams, kernel_params, alpha = model.logger.get_final_params()
        self.featprocessor = model.featprocessor
        self.hyperparams = hyperparams
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.selected_covs = model.selected_covariates()
        self.Q = kernel_params['eta'].shape[0] - 1
        self.p = model.p
        self.X_train_feat = model.X_train_feat.copy()

    def get_effect(self, X_feat, V):
        pass

    def get_variation_at_order(self, X_feat, order):
        assert order <= self.Q, f"Model fit only contains {self.Q} interactions"
        V_order = all_subsets(self.selected_covs, self.Q, False)
        variation = jnp.zeros(X_feat.shape[0])
        for V in tqdm(V_order):
            variation += self.get_effect(X_feat, list(V))
        return variation

    def get_decomposition(self, X, max_effects=1e4):
        err1_msg = f"{X.shape[1]} different than {self.p} input covariates"
        assert X.shape[1] == self.p, err1_msg
        V_all = all_subsets(self.selected_covs, self.Q, True)
        num_effects = len(V_all)
        err2_msg = f"{num_effects} larger than {max_effects} effects threshold"
        assert num_effects < max_effects, err2_msg
        decomposition = dict()
        X_feat = self.featprocessor.transform(X)
        for V in tqdm(list(V_all)):
            f_V = self.get_effect(X_feat, list(V))
            decomposition[V] = f_V
        return decomposition


if __name__ == "__main__":
    import pytest

    def test_all_subsets():
        selected = [1, 4, 0]
        q = 1
        V_q = all_subsets(selected, q, False)
        V_q = list(V_q)
        assert len(V_q) == 3
        V_all = all_subsets(selected, 2, True)
        print([list(e) for e in V_all])

    test_all_subsets()
