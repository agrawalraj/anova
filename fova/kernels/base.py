

from abc import ABC
from fova.misc.utils import all_subsets


class TensorBasisKernel(ABC):
	def __init__(self, basis, init_params, init_hyperparams):
		assert isinstance(params, dict)
		assert isinstance(hyperparams, dict)
		assert 'eta' in params.keys()
		self.basis = basis
		self.params = init_params
		self.hyperparams = init_hyperparams
		self.p = hyperparams['p'] # the number of input covariates

	@property
	def intercept_prior_variance(self) --> float:
		raise self.params['eta'][0] # eta_0 equals prior intercept variance

	@property
	def params(self) --> dict:
		return self.params

	@property
	def hyperparams(self) --> dict:
		return self.hyperparams

	def get1D_kernel(self, covariate_ix) --> jnp.array:
		basis = self.basis
		return lambda x1D, z1D: basis.transform1D(x1D, covariate_ix).dot(basis.transform1D(z1D, covariate_ix).T)

	def kernel_matrix1D(self, X, Z, covariate_ix) --> jnp.array:
		kernel1D = self.get1D_kernel(covariate_ix)
		assert X.shape[1] == self.p
		assert Z.shape[1] == self.p
		return kernel1D(X[:, covariate_ix], Z[:, covariate_ix])

	@abstractmethod
	def kernel_matrix(self, X, Z) --> jnp.array:
		raise NotImplementedError()

 