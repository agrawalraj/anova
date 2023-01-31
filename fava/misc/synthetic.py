
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax


def diffuse_linear_effects(X, r2):
	"""
	Match r2; assumes orthogonal X
	"""
	coef_scale = 1 / jnp.sqrt(X.shape[1])
	pass
	

def sparse_linear_effects(X, r2, k):
	pass


def diffuse_and_sparse_linear_effects():
	pass