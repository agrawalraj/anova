
import jax.numpy as jnp
from fova.inference.losses import fit_predict_new
from fova.kernels.skim import get_kappa


class GausLogger(object):
	def __init__(self, freq=100):
		self.freq = freq
		self.val_losses = []
		self.all_alpha = []
		self.all_hyperparams = []
		self.all_kernel_params = []

	def update(
			self, t, loss, hyperparams, kernel_params, opt_params, 
			X_train_feat, Y_train, X_valid_feat, Y_valid
		):
		print('='*30 + f" Iteration {t}/{opt_params['T']} " + '='*30)
		
		c = hyperparams['c']
		kappa = get_kappa(kernel_params['U_tilde'], c)
		n_selected = len(jnp.where(kappa > 0)[0])
		print(f'There are {n_selected} selected.')

		if t % self.freq == 0:
			# Compute loss
			mse, alpha_hat = fit_predict_new(X_train_feat, Y_train, X_valid_feat, 
								Y_valid, hyperparams, kernel_params, opt_params)
			self.val_losses.append(mse)

			# Print MSE
			print(f'MSE (Validation)={mse}.')

			# Cache parameters
			self.all_alpha.append(alpha_hat)
			self.all_hyperparams.append(hyperparams.copy()) 
			self.all_kernel_params.append(kernel_params.copy())

	def get_final_params(self):
		return self.all_hyperparams[-1], self.all_kernel_params[-1], self.all_alpha[-1]
