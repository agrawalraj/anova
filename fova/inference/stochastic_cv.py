
from math import floor
from tqdm import tqdm
import time
import jax as jnp
from jax import random
from sklearn.model_selection import train_test_split


# This takes kernel and updates params using stochastic CV loss

# Randomly subsample M datapoints
# Compute loss on M subsample
# for each trainable parameter, update

# Could take as input a tracer that specifies what functions of parameters 
# should be printed


kernel, loss, init params, dyynamic hyperarams


# how to handle setting c


def init_params():
	pass


def update_hyperparams(params, **params):
	# Run this at the start of gradient step or after
	pass

def update_kernel_params():
	pass


# maybe this should be a class

class Inference(object):
	def __init__():

	def update():

	def 

	X_cv_train, X_cv_test, Y_cv_train, Y_cv_test = train_test_split(X_feat_train, Y_train, test_size=n_test)




# model params = parameters to take gradients
# model hyperparams = parameters not to take gradients (dont learn)
	# - c
	# - noise_var
# tracer - stores parameters, prints stuff, etc. logic
# kernel has a mode = training to know to used cached values; mode used to see if compute exact kernel or approx kernel
# subset to non-zero values of kappa to compute kernel to avoid reindexing issues

kernel_config:
	- init_params
	- init_hyperparams
	- basis


class GradientDescentSKIMFA(object):
	def __init__(self, X_train, Y_train, kernel_config):
		self.X_train = X_train # Training points needed to make new predictions
		self.Y_train = Y_train
		self.kernel_config = kernel_config.copy()
		self.skimfa_kernel = make_skimfa_kernel(X_train, kernel_config)

	def fit(params, hyperparams, pred_loss, tracer, opt_config):
		start_time = time.time()
		# Load in gradient descent optimization specs
		N = X_train.shape[0] # Number training datapoints
		M = optimization_config['M'] # Number of datapoints for CV loss
		T = optimization_config['T'] # Number gradient steps
		truncScheduler = optimization_config['truncScheduler']
		lr = optimization_config['lr'] # Learning rate
		X_train = self.X_train
		Y_train = self.Y_train
		for t in tqdm(range(T)):
			# NEED key, make_prediction
			parameter_gradients = grad(stochastic_cv_loss)(params, key, 
									hyperparams, X_train, Y_train, pred_loss, 
									make_prediction, hyperarams)
			for param_name, param_grad in parameter_gradients.items():
				params[param_name] = params[param_name] - lr * param_grad
			tracer(params, hyperparams, t)
		end_time = time.time()
		self.tracer = tracer
		self.fitting_time_minutes = (end_time - start_time) / 60.

	def predict(self, X_test):
		hyperparams, params, alpha = self.tracer.get_final_params()
		kernel = self.skimfa_kernel()
		return kernel_prediction(kernel, params, hyperparams, alpha, self.X_train, X_test)

	@property
	def selected_covariates(self):
		hyperparams, params = self.tracer.get_final_params()
		kappa = params['kappa']
		return jnp.where(kappa > 0)[0]

make_prediction(kernel, params, hyperarams, training_cv_batch)

def make_krr_prediction(kernel, params, hyperarams, training_cv_batch):
	K_XX = 
	alpha = 
	K_XZ = 
	return alpha.dot(K_XZ) 



grad(stochastic_cv_loss)(params, key, )



# grad(loss)(params, hyperarams, make_prediction, training_cv_batch)




def make_skimfa_kernel(X_train, kernel_config):
	# Initialize skimfa kernel hyperparams
	p = X_train.shape[1] # Total number of covariates
	Q = kernel_config['Q'] # Highest order interaction
	eta = torch.ones(Q + 1, requires_grad=True) # Intialize global interaction variances to 1
	U_tilde = torch.ones(p, requires_grad=True) # Unconstrained parameter to generate kappa
	noise_var = torch.tensor(noise_var_init, requires_grad=train_noise)
	c = 0.
	skimfa_kernel_fn = skimfa_kernel(train_valid_data=train_valid_data, kernel_config=kernel_config)
	self.skimfa_kernel_fn = 


class SKIMFA(object):
	def __init__(self, X_train, Y_train, kernel_config):
		self.X_train = X_train # Need to save training points to make new predictions
		self.Y_train = Y_train
		self.kernel_config = kernel_config.copy()
		self.skimfa_kernel = make_skimfa_kernel(X_train, kernel_config)

	def fit(self, optimization_config):
		start_time = time.time()
		skimfa_kernel = self.skimfa_kernel

		# Load in training data
		X_train = self.X_train
		Y_train = self.Y_train

		M = optimization_config['M'] # Number of datapoints for CV loss




		# Load in optimization specs
		N = X_train.shape[0] # Number training datapoints
		T = optimization_config['T'] # Number gradient steps
		M = optimization_config['M'] # Number of datapoints for CV loss
		truncScheduler = optimization_config['truncScheduler']
		param_save_freq = optimization_config['param_save_freq']
		valid_report_freq = optimization_config['valid_report_freq']
		lr = optimization_config['lr']
		train_noise = optimization_config['train_noise']
		noise_var_init = optimization_config['noise_var_init']
		truncScheduler = optimization_config['truncScheduler']

		training_losses = []
		validation_losses = []
		saved_params = dict()
	
		# Gradient descent training loop
		for t in tqdm(range(T)):
			random_indcs = torch.randperm(N)
			cv_train_indcs = random_indcs[M:]
			cv_test_indcs = random_indcs[:M]
	
			X_cv_train = X_train[cv_train_indcs, :]
			X_cv_test = X_train[cv_test_indcs, :]
			Y_cv_train = Y_train[cv_train_indcs]
			Y_cv_test = Y_train[cv_test_indcs]

			kappa = make_kappa(U_tilde, c)
			c = truncScheduler(t, U_tilde, c)
			K_train = skimfa_kernel_fn.kernel_matrix(X1=X_cv_train, X2=X_cv_train, kappa=kappa, eta=eta, 
									   X1_info=cv_train_indcs, 
									   X2_info=cv_train_indcs)

			K_test_train = skimfa_kernel_fn.kernel_matrix(X1=X_cv_test, X2=X_cv_train, kappa=kappa, eta=eta,
										    X1_info=cv_test_indcs, X2_info=cv_train_indcs)

			alpha = kernel_ridge_weights(K_train, Y_cv_train, noise_var)
			L = cv_mse_loss(alpha, K_test_train, Y_cv_test)

			# Perform gradient decent step
			L.backward()
			U_tilde.data = U_tilde.data - lr * U_tilde.grad.data
			U_tilde.grad.zero_()

			if eta.requires_grad:
				eta.data = eta.data - lr * eta.grad.data
				eta.grad.zero_()

			if train_noise:
				noise_var.data = noise_var.data - lr * noise_var.grad.data
				noise_var.grad.zero_()

			if (t % valid_report_freq == 0) or (t == (T-1)):
				K_valid_train = skimfa_kernel_fn.kernel_matrix(X1=X_valid, X2=X_cv_train, kappa=kappa, eta=eta)
				valid_mse = cv_mse_loss(alpha, K_valid_train, Y_valid)
				validation_losses.append(valid_mse)
				print(f'Mean-Squared Prediction Error on Validation (Iteration={t}): {round(valid_mse.item(), 3)}')
				print(f'Number Covariates Selected={torch.sum(kappa > 0).item()}')

			if (t % param_save_freq == 0) or (t == (T-1)):
				saved_params[t] = dict()
				saved_params['U_tilde'] = U_tilde
				saved_params['eta'] = eta
				saved_params['noise_var'] = noise_var
				saved_params['c'] = c

		end_time = time.time()
		self.fitting_time_minutes = (end_time - start_time) / 60.

		# Store final fitted parameters
		last_iter_params = dict()
		last_iter_params['kappa'] = make_kappa(U_tilde, c)
		last_iter_params['eta'] = eta
		last_iter_params['noise_var'] = noise_var

		# Get kernel ridge weights using full training set
		K_train = skimfa_kernel_fn.kernel_matrix(X1=X_train, X2=X_train, kappa=kappa, eta=eta, 
									   X1_info=torch.arange(N), 
									   X2_info=torch.arange(N))

		alpha = kernel_ridge_weights(K_train, Y_train, noise_var)
		last_iter_params['alpha'] = alpha
		last_iter_params['X_train'] = X_train
		self.last_iter_params = last_iter_params

		# Store selected covariates
		self.selected_covariates = torch.where(self.last_iter_params['kappa'] > 0)[0]

	def predict(self, X_test):
		kappa = self.last_iter_params['kappa']
		eta = self.last_iter_params['eta']
		alpha = self.last_iter_params['alpha']
		X_train = self.last_iter_params['X_train']
		skimfa_kernel = self.skimfa_kernel
		kernel_config = self.kernel_config.copy()
		kernel_config['cache'] = False
		skimfa_kernel_fn = skimfa_kernel(train_valid_data=None, kernel_config=kernel_config)
		K_test_train = skimfa_kernel_fn.kernel_matrix(X1=X_test, X2=X_train, kappa=kappa, eta=eta)
		return K_test_train.mv(alpha)

	@property
	def selected_covariates(self):
		return self.selected_covariates.clone()

