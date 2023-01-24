
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax


@jax.jit
def get_kappa(u_tilde, c):
    return 1 / (1 - c) * jax.nn.relu((u_tilde ** 2 / (1 + u_tilde ** 2)) - c)


@jax.jit
def linear_kernel(x, y):
    return jnp.dot(x, y)


def kernel_matrix(X, Z, kernel):
    mapx1 = jax.vmap(lambda x, z: kernel(x, z), 
                     in_axes=(0, None), out_axes=0)
    
    mapx2 = jax.vmap(lambda x, z: mapx1(x, z), in_axes=(None, 0), out_axes=1)
    return mapx2(X, Z)


def skim_kernel_matrix(X, Z, c, kernel_params):
    kernel_fn = lambda x, z: skim_fa_kernel(x, z, c, kernel_params)
    return kernel_matrix(X, Z, kernel_fn)


@jax.jit
def kernel_pow_s_1d(xi_feat, zi_feat, kappa_i, s):
    return (kappa_i**2 * linear_kernel(xi_feat, zi_feat)) ** s


@jax.jit
def kernel_pow_s(x, z, kappa, s):
    mapped_kernel = vmap(kernel_pow_s_1d, in_axes=(0, 0, 0, None))
    return jnp.sum(mapped_kernel(x, z, kappa, s))


@jax.jit
def kernel_V(x, z, V):
    mapped_kernel = vmap(kernel_pow_s_1d, in_axes=(0, 0, None, None))
    return jnp.prod(mapped_kernel(x[V], z[V], 1., 1))


def skim_fa_kernel(x, z, c, kernel_params):
    # See Theorem 2 of https://arxiv.org/pdf/2106.12408.pdf
    # O(N^2pQ) time to compute kernel matrix

    # Get kernel hyperparameters / parameters
    kappa = get_kappa(kernel_params['U_tilde'], c)
    eta = kernel_params['eta']
    Q = eta.shape[0] - 1

    # Dictionaries to cache kernels
    kernel_hat_cache = dict()
    kernel_hat_cache[0] = 1.
    kernel_spower_cache = dict()
 
    # Cache k^s terms in Theorem 2
    for s in range(1, Q+1):
        kernel_spower_cache[s] = kernel_pow_s(x, z, kappa, s)
    
    # Cache \hat{k}^q terms in Theorem 2
    for q in range(1, Q+1):
        kernel_hat_cache[q] = 0.
        for s in range(1, q+1):
            kernel_hat_cache[q] += 1/q * (-1)**(s+1) * kernel_hat_cache[q-s] * kernel_spower_cache[s]
    
    # Compute SKIM-FA kernel from cached kernels
    k_skim_fa = 0.
    for q in range(Q+1):
        k_skim_fa += eta[q]**2 * kernel_hat_cache[q]
    return k_skim_fa

