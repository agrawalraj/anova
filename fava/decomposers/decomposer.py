
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
        assert isinstance(self.selected_covs, list)
        self.Q = kernel_params['eta'].shape[0] - 1
        self.p = model.p
        self.X_train_feat = model.X_train_feat.copy()

    def get_effect(self, X_feat, V):
        pass

    def get_variation_at_order(self, X, order, verbose=False):
        err1_msg = f"{X.shape[1]} different than {self.p} input covariates"
        assert X.shape[1] == self.p, err1_msg
        assert order <= self.Q, f"Model fit only contains {self.Q} interactions"
        X_feat = self.featprocessor.transform(X)
        V_order = all_subsets(self.selected_covs, order, False) # TODO: avoid explicity generating all interactions
        variation = jnp.zeros(X_feat.shape[0])
        if verbose:
            for V in tqdm(V_order):
                variation += self.get_effect(X_feat, list(V))
        else:
            for V in V_order:
                variation += self.get_effect(X_feat, list(V))
        return variation

    def get_variation_at_covariate(self, X, cov_ix, verbose=False):
        err1_msg = f"{X.shape[1]} different than {self.p} input covariates"
        assert X.shape[1] == self.p, err1_msg
        assert isinstance(cov_ix, int)
        assert cov_ix >= 0
        assert cov_ix < self.p
        assert self.Q >= 1
        X_feat = self.featprocessor.transform(X)
        V_all = all_subsets(sorted(list(set(self.selected_covs) - {cov_ix})), self.Q - 1, True)
        variation = jnp.zeros(X_feat.shape[0])
        if verbose:
            for V in tqdm(V_all):
                variation += self.get_effect(X_feat, sorted(list(V) + [cov_ix]))
        else:
            for V in V_all:
                variation += self.get_effect(X_feat, sorted(list(V) + [cov_ix]))
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
