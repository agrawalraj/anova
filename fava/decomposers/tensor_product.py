
"""
TODO: say what this does
"""

import jax.numpy as jnp
from fava.decomposers.decomposer import Decomposer
from fava.kernels.skim import kernel_matrix, kernel_V, get_kappa
from fava.inference.ridge_regression import ridge_predict


class TensorProductKernelANOVA(Decomposer):
    def get_effect(self, X_feat, V):
        eta = self.kernel_params['eta']
        kappa = get_kappa(self.kernel_params['U_tilde'], self.hyperparams['c'])
        alpha = self.alpha
        if len(V) == 0: # Constant / intercept term
            return eta[0] * alpha.sum() * jnp.ones(X_feat.shape[0])
        else:
            X_train_feat = self.X_train_feat
            q = len(V)
            eta_q = eta[q]
            theta_V = eta_q * jnp.product(kappa[jnp.array(V)]).item()
            kernel_fn = lambda x, z: kernel_V(x, z, V)
            K_ZX = kernel_matrix(X_feat, X_train_feat, kernel_fn)
            return theta_V * ridge_predict(K_ZX, alpha)


class LinearANOVA(TensorProductKernelANOVA):
    def get_effect(self, X_feat, V):
        if len(V) == 0: # Constant / intercept term
            eta = self.kernel_params['eta']
            alpha = self.alpha
            return eta[0] * alpha.sum() * jnp.ones(X_feat.shape[0])
        else:
            beta = self.get_coef(V)
            N, p, __ = X_feat.shape
            return beta * jnp.product(X_feat.reshape((N, p))[:, jnp.array(V)], axis=1)

    def get_coef(self, V):
        if len(set(V) - set(self.selected_covs)) > 0:
            return 0.
        assert self.p == self.X_train_feat.shape[1], 'Not linear model!'
        
        X_probe = jnp.zeros((1, self.p, 1))
        if len(V) == 0:
            return super().get_effect(X_probe, V).item()

        # f_V(x_v) = \theta_V \prod_{i \in V} x_i
        # Set \prod_{i \in V} x_i = 1 by setting x_i = 1
        # Then, f_V(x_v at probe) = theta_V
        X_probe = X_probe.at[:, jnp.array(V)].set(1.)
        f_V = super().get_effect(X_probe, V)
        return f_V.item()
