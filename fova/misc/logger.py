

class GausLogger(object):
	def __init__(self, freq=100):
		self.freq = []
		self.val_losses = []
		self.all_alpha = []
		self.all_hyperparams = []
		self.all_kernel_params = []

	def update(self, t, loss, hyperparams, kernel_params, opt_params, X_train_feat, Y_train, X_valid_feat, Y_valid):
		if t % self.freq == 0:
			# Compute loss
			c = hyperparams['c']
			sigma_sq = hyperparams['sigma_sq']
			K_XX = kernel_matrix(X_train_feat, X_train_feat, c, kernel_params)
			K_ZX = kernel_matrix(X_valid_feat, X_train_feat, c, kernel_params)

			alpha_hat = kernel_ridge(K_XX, Y, sigma_sq, opt_params)
			alpha_hat = kernel_ridge(K_XX, Y[train_indcs], sigma_sq, opt_params)
			Y_pred = ridge_predict(K_ZX, alpha_hat)
			Y_true = Y_valid[cv_indcs]
			mse = mean_squared_error(Y_pred, Y_true)
			self.val_losses.append(mse)

			# TODO: print MSE

			# Cache parameters
			self.all_hyperparams.append(hyperparams.copy()) 
			self.kernel_params.append(kernel_params.copy())

		# TODO: print current sparsity level, k strongest variables

	def get_final_params(self):
		return self.all_hyperparams[-1], self.all_kernel_params[-1], self.all_alpha[-1]
