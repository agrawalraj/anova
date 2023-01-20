

class FiniteDimensionalBasis(object):
	def __init__(self, X_train, basis_config):
		self.type = 'abstract'

	def transform1D(self, x1D):
		pass

	def transform(self, X):
		p = self.p
		return jnp.concat([self.transform1D(X[:, covariate_ix]) for covariate_ix in range(p)])