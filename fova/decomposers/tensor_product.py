
"""
TODO: say what this does
"""

import jax.numpy as jnp
from fova.decomposers.decomposer import Decomposer
from fova.kernels.skim import kernel_matrix, kernel_V, get_kappa
from fova.inference.ridge_regression import ridge_predict


class TensorProductKernelANOVA(Decomposer):
    def get_effect(self, X_feat, V):
        eta = self.kernel_params['eta']
        kappa = get_kappa(self.kernel_params['U_tilde'], self.hyperparams['c'])
        alpha = self.alpha
        if len(V) == 0: # Constant / intercept term
            return eta[0] * jnp.ones(X_feat.shape[0])
        else:
            X_train_feat = self.X_train_feat
            q = len(V)
            eta_q = eta[q]
            theta_V = eta_q * jnp.product(kappa[jnp.array(V)]).item()
            kernel_fn = lambda x, z: kernel_V(x, z, V)
            K_ZX = kernel_matrix(X_feat, X_train_feat, kernel_fn)
            return theta_V * ridge_predict(K_ZX, alpha)
