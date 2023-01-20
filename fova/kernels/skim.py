

import jax.numpy as jnp


def kernel_trick_two(X, Z, kernel):
	# See Theorem 2 of https://arxiv.org/pdf/2106.12408.pdf
	# O(N^2pQ) time to compute kernel matrix; can be slow since not vectorized
	N1 = X.shape[0]
	N2 = Z.shape[0]
	# Get parameters needed to compute the kernel
	kappa = kernel.params['kappa']
	eta = kernel.params['eta']
	Q = kernel.hyperparams['kappa']
	p = kappa.shape[0]
	# Check parameters dimensions are correct
	assert X.shape[1] == p
	assert Z.shape[1] == p
	assert eta.shape[0] == Q+1
	assert Q > 0
	kernel_hat_cache = dict()
	kernel_hat_cache[0] = jnp.ones((N1, N2))
	kernel_spower_cache = dict()
	# Cache k^s terms in Theorem 2
	for s in range(1, Q+1): # Intialize to 0 matrix
		kernel_spower_cache[s] = jnp.zeros((N1, N2))
	for i in range(p): # TODO: vectorize this loop w/ vmap
		if kappa[i] > 0: # maybe vmap this part
			K_Xi_Zi = kernel.kernel_matrix1D(X, Z, i)
			for s in range(1, Q+1):
				kernel_spower_cache[s] += kappa[i]**(2*s) * (K_Xi_Zi ** s)
	# Cache \hat{k}^q terms in Theorem 2
	for q in range(1, Q+1):
		kernel_hat_cache[q] = jnp.zeros((N1, N2))
		for s in range(1, q+1)
			kernel_hat_cache[q] += 1/q * (-1)**(s+1) * kernel_hat_cache[q-s] * kernel_spower_cache[s]
	# Compute SKIM-FA kernel from cached kernels
	K_X_Z = jnp.ones((N1, N2))
	for q in range(Q+1):
		K_X_Z += eta[q]**2 * kernel_hat_cache[q]
	return K_X_Z


def linear_kernel_trick_two_vectorized(X, Z, kernel):
	# See Theorem 2 of https://arxiv.org/pdf/2106.12408.pdf
	# O(N^2pQ) time to compute kernel matrix; can be slow since not vectorized
	# Exploit additional structure for linear kernel for vectorization
	N1 = X.shape[0]
	N2 = Z.shape[0]
	# Get parameters needed to compute the kernel
	kappa = kernel.params['kappa']
	eta = kernel.params['eta']
	Q = kernel.hyperparams['kappa']
	p = kappa.shape[0]
	# Check parameters dimensions are correct
	assert X.shape[1] == p
	assert Z.shape[1] == p
	assert eta.shape[0] == Q+1
	assert Q > 0
	kernel_hat_cache = dict()
	kernel_hat_cache[0] = jnp.ones((N1, N2))
	kernel_spower_cache = dict()
	# Cache k^s terms in Theorem 2
	S = jnp.where(kappa > 0)[0] # Set of selected covariates 
	for s in range(1, Q+1):
		kernel_spower_cache[s] = ((kappa[S] * X[:, S]) ** s).dot(((kappa[S] * Z[:, S]) ** s).T)
	# Cache \hat{k}^q terms in Theorem 2
	for q in range(1, Q+1):
		kernel_hat_cache[q] = jnp.zeros((N1, N2))
		for s in range(1, q+1)
			kernel_hat_cache[q] += 1/q * (-1)**(s+1) * kernel_hat_cache[q-s] * kernel_spower_cache[s]
	# Compute SKIM-FA kernel from cached kernels
	K_X_Z = jnp.ones((N1, N2))
	for q in range(Q+1):
		K_X_Z += eta[q]**2 * kernel_hat_cache[q]
	return K_X_Z


class SKIMFAKernel(TensorBasisKernel):
	def __init__(self, basis, init_params, init_hyperparams):
		super().__init__(basis, init_params, init_hyperparams)
		# Additional sanity checks here
	
	def kernel_matrix(self, X, Z) --> jnp.array:
		if self.basis.type == 'linear':
			X_tilde = basis.transform(X)
			Z_tilde = basis.transform(Z)
			return linear_kernel_trick_two_vectorized(X_tilde, Z_tilde, self)
		else:
			return kernel_trick_two(X, Z, self)

