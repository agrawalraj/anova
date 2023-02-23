
import jax.numpy as jnp
from math import floor
from fava.kernels.skim import get_kappa


def get_percentile_thresh(U_tilde, percentile=.25):
	U = jnp.sort(get_kappa(U_tilde, c=0))
	pos = floor(percentile * U.shape[0])
	return U[pos].item()


def trunc_scheduler_helper(
		t, U_tilde, c_prev, r=.005, gamma=.5, 
		iter_cut=500, pcut=.1
	):
	if t < iter_cut:
		return 0.
	if t == iter_cut:
		# remove 25% of the covariates
		c = get_percentile_thresh(U_tilde, percentile=pcut)
		return c
	return max(min((1+r)*c_prev, gamma), c_prev)


class truncScheduler(object):
	def __init__(self):
		self.states = []

	def update(self, t, c, kernel_params, **kwargs):
		if t == 0:
			self.states.append(c)
		c_prev = self.states[-1]
		U_tilde = kernel_params['U_tilde']
		c = trunc_scheduler_helper(t, U_tilde, c_prev, **kwargs)
		self.states.append(c)
		return c


class constantScheduler(object):
	def __init__(self):
		self.states = []

	def update(self, t, c, kernel_params, **kwargs):
		self.states.append(c)
		return c