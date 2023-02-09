
import time
import numpy as np
import jax.numpy as jnp

from tqdm import tqdm
from math import floor
from jax import random

from fava.inference.losses import ridge_stochastic_cv_loss, update_kernel
from fava.inference.ridge_regression import ridge_predict
from fava.misc.logger import GausLogger
from fava.kernels.skim import skim_kernel_matrix, get_kappa
from sklearn.linear_model import LogisticRegression


class SKIMFA(object):
    def __init__(self, X_train, Y_train, X_valid, Y_valid, featprocessor):
        self.p = X_train.shape[1] # Number of covariates
        self.X_train_feat = featprocessor.transform(X_train)
        self.Y_train = Y_train
        self.X_valid_feat = featprocessor.transform(X_valid)
        self.Y_valid = Y_valid
        self.featprocessor = featprocessor

    def fit(
            self, key, loss, hyperparams_init, 
            kernel_params_init, opt_params, logger
        ):
        start_time = time.time()
        hyperparams = hyperparams_init.copy()
        kernel_params = kernel_params_init.copy()
        T = opt_params['T'] # Number gradient steps
        scheduler = opt_params['scheduler']
        c = hyperparams['c']

        # Training loop
        for t in tqdm(range(T)):
            # Update SKIM-FA kernel parameters
            key, subkey = random.split(key)
            kernel_params = update_kernel(subkey, self.X_train_feat, 
                                            self.Y_train, loss, hyperparams, 
                                            kernel_params, opt_params)

            # Update c
            hyperparams['c'] = scheduler.update(t, c, kernel_params)

            # Keep track of parameter changes
            logger.update(t, loss, hyperparams, kernel_params, opt_params, 
                            self.X_train_feat, self.Y_train, 
                            self.X_valid_feat, self.Y_valid)
        
        end_time = time.time()
        self.logger = logger
        self.fitting_time_minutes = (end_time - start_time) / 60.


class GaussianSKIMFA(SKIMFA):
    def fit(
            self, key, hyperparams_init, kernel_params_init, opt_params, 
            logger=GausLogger()
        ):
        loss = ridge_stochastic_cv_loss
        super(GaussianSKIMFA, self).fit(key, loss, hyperparams_init, 
                                        kernel_params_init, opt_params, logger)

    def predict(self, X_test):
        err_msg = f"There are {X_test.shape[1]} instead of {self.p} covariates"
        assert X_test.shape[1] == self.p, err_msg
        hyperparams, kernel_params, alpha = self.logger.get_final_params()
        c = hyperparams['c']
        X_test_feat = self.featprocessor.transform(X_test)
        K = skim_kernel_matrix(X_test_feat, self.X_train_feat, c, kernel_params)
        return ridge_predict(K, alpha)

    def selected_covariates(self):
        hyperparams, kernel_params, __ = self.logger.get_final_params()
        kappa = get_kappa(kernel_params['U_tilde'], hyperparams['c'])
        return sorted(jnp.where(kappa > 0)[0].tolist())


class PlattSKIMFA(GaussianSKIMFA):
    def fit(
            self, key, hyperparams_init, kernel_params_init, opt_params, 
            logger=BernLogger()
        ):
        super(PlattSKIMFA).fit(key, hyperparams_init, 
                                    kernel_params_init, opt_params, logger)
        # Platt scaling
        skim_train_unscaled = np.array(super(PlattSKIMFA).predict(self.X_train))
        N_train = self.X_train.shape[0]
        skim_train_unscaled = skim_train_unscaled.reshape((N_train,1))
        Y_train = np.array(self.Y_train)
        platt = LogisticRegression(penalty=None, 
                                    fit_intercept=True, random_state=0)
        platt.fit(skim_train_unscaled, Y_train)
        self.platt = platt

    def predict(self, X_test):
        skim_unscaled = np.array(super(PlattSKIMFA).predict(X_test))
        skim_unscaled = skim_unscaled.reshape((skim_unscaled.shape[0],1))
        return jnp.array(self.platt.predict_proba(skim_unscaled)[:, 1])


if __name__ == "__main__":
    from fava.basis.maps import LinearBasis
    from fava.misc.scheduler import truncScheduler
    from fava.misc.logger import GausLogger

    featprocessor = LinearBasis()
    scheduler = truncScheduler()
    logger = GausLogger(100)

