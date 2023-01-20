
"""
TODO: say what this does
"""

import jax as jnp
from fova.decomposer import Decomposer

# Linear regression coeffs case

# Product measure, job done at specifying fdbasis

class TensorProductKernelANOVA(Decomposer):
	# Just run kernel ridge regression
	# TODO: say this only workds for product measure. 
	def __init__(self, k_theta, alpha):
		# TODO: k_theta - needs to be a model selection kernel
		self.__k_theta = k_theta
		self.__kernels1d = k_theta.kernels1d
		self.__Q = k_theta.Q
		self.__alpha = alpha

	def get_effect(self, X_train, X, V):
		""" 

		References: Lemma 2 from
		https://arxiv.org/pdf/2106.12408.pdf 
	    Attributes
	    ----------
	    X_train : jax ndarray
	        	  Fitted regression function with a predict() method
	    """
	   	alpha = self.__alpha
	   	k_theta = self.__k_theta
	    kernels1d = self.__kernels1d
	    if len(V) == 0: # Constant / intercept term
	    	k_intercept = k_theta.intercept() # Value for intercept term
	    	return alpha.sum() * k_intercept * jnp.ones(X.shape[0])
		else:
			kernel_matrix = 1. 
			theta_V = k_theta.get_theta(V)
			for covariate_ix in V:
				Xi = X[:, covariate_ix]
				Xi_train = X_train[:, covariate_ix]
				kernel_matrix *= kernels1d.kernel_matrix1D(Xi, Xi_train, 
													covariate_ix=covariate_ix)
			return theta_V * kernel_matrix.dot(alpha)


class EmpiricalKernelANOVA(Decomposer):
	def get_effect(self, X_train, X, V, decomposition=None):
		if decomposition == None:
			# just return

		# kernels1d should just be tthe raw basis (no skim params)

	def get_decomposition():









class FiniteTensorBasisANOVA(Decomposer):
	def __init__(self, fdbasis, theta_fit):
		# TODO: k_theta - needs to be a model selection kernel
		self.__fdbasis = fdbasis
		self.__Q = fdbasis.Q
		self.__theta_fit = theta_fit

	def get_effect(self, X_train, X, V, mu):
		pass








	# Need 1-dimensional basis functions
	def fit(self, fdbasis, mu_project, mu_target, samp_dim_ratio=10, 
			max_samps=1e6):
		f = self.f
		basis_dim = fdbasis.dimension
		Q = fdbasis.Q
		N_project = basis_dim * samp_dim_ratio # TOOD: if mu_project is empirical distribution, should throw error if exceeds
		assert N_project < max_samps, "Too many samples. Chance ratio or increase threshold" # TODO: throw better error message
		X_raw = mu_project.sample(N_project)
		Y = f.predict(X_raw)
		X_feat = fdbasis.transform(X_raw)
		theta_ridge = RidgeRegression(X_feat, Y)

		# TODO: assert basis type is of XYZ
		f = self.f


		# Approximate f in a finite dimensional product space 


# Don't extrapolate outside of training
	# uniform between min and max of train
	# get min and max of train - uniform sample x points
	# then sample points will be tensor

# generate tensor product space
# basis
	# fit
	# featdim