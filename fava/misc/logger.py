
import numpy as np
import jax.numpy as jnp
from fava.inference.losses import fit_predict_new
from fava.kernels.skim import get_kappa
from sklearn.metrics import roc_auc_score


def smallest_validation_error_index(val_losses):
	return np.argmin(np.array(val_losses))


class GausLogger(object):
	def __init__(self, freq=100):
		self.freq = freq
		self.val_losses = []
		self.all_alpha = []
		self.all_hyperparams = []
		self.all_kernel_params = []

	def update(
			self, t, loss, hyperparams, kernel_params, opt_params, 
			X_train_feat, Y_train, X_valid_feat, Y_valid,
		):

		if t % self.freq == 0:
			print('='*30 + f" Iteration {t}/{opt_params['T']} " + '='*30)
		
			c = hyperparams['c']
			kappa = get_kappa(kernel_params['U_tilde'], c)
			n_selected = len(jnp.where(kappa > 0)[0])
			print(f'There are {n_selected} covariates selected.')

			# Compute loss
			mse, Y_pred, alpha_hat = fit_predict_new(X_train_feat, Y_train, X_valid_feat, 
								Y_valid, hyperparams, kernel_params, opt_params)
			mse = mse.item()
			self.val_losses.append(mse)

			# Print metrics
			print(f'MSE (Validation)={round(mse, 4)}.')
			print(f'R2 (Validation)={round(1 - mse/Y_valid.var().item(), 4)}.')
			print(f'eta={kernel_params["eta"]}')
			print(f'c={round(c, 4)}')
			if kappa.shape[0] < 100:
				print(f'kappa={kappa}') # TODO: instead report top most important covariates

			# Cache parameters
			self.all_alpha.append(alpha_hat)
			self.all_hyperparams.append(hyperparams.copy()) 
			self.all_kernel_params.append(kernel_params.copy())

	def get_final_params(self):
		# Return parameter set that yields smallest error on validation set
		best_index = smallest_validation_error_index(self.val_losses)
		return self.all_hyperparams[best_index], self.all_kernel_params[best_index], self.all_alpha[best_index]


class BernLogger(object):
	def __init__(self, freq=100):
		self.freq = freq
		self.val_losses = []
		self.all_alpha = []
		self.all_hyperparams = []
		self.all_kernel_params = []

	def update(
			self, t, loss, hyperparams, kernel_params, opt_params, 
			X_train_feat, Y_train, X_valid_feat, Y_valid,
		):

		if t % self.freq == 0:
			print('='*30 + f" Iteration {t}/{opt_params['T']} " + '='*30)
		
			c = hyperparams['c']
			kappa = get_kappa(kernel_params['U_tilde'], c)
			n_selected = len(jnp.where(kappa > 0)[0])
			print(f'There are {n_selected} covariates selected.')

			# Compute loss
			mse, Y_pred, alpha_hat = fit_predict_new(X_train_feat, Y_train, X_valid_feat, 
								Y_valid, hyperparams, kernel_params, opt_params)
			self.val_losses.append(mse)
			auroc = roc_auc_score(Y_valid, Y_pred)

			# Print metrics
			print(f'Brier Loss (Validation)={round(mse, 4)}.')
			print(f'AUC-ROC (Validation)={round(auroc, 4)}.')
			print(f'eta={kernel_params["eta"]}')
			print(f'c={round(c, 4)}')
			if kappa.shape[0] <= 100:
				print(f'kappa={kappa}')

			# Cache parameters
			self.all_alpha.append(alpha_hat)
			self.all_hyperparams.append(hyperparams.copy()) 
			self.all_kernel_params.append(kernel_params.copy())

	def get_final_params(self):
		# Return parameter set that yields smallest error on validation set
		best_index = smallest_validation_error_index(self.val_losses)
		return self.all_hyperparams[best_index], self.all_kernel_params[best_index], self.all_alpha[best_index]
