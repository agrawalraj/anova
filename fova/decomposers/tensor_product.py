
"""
TODO: say what this does
"""

import jax as np
from fova.decomposer import Decomposer


class TensorProductANOVA(Decomposer):

	# Need 1-dimensional basis functions
	def fit(self, basis, measure):
		# TODO: assert basis type is of XYZ


# generate tensor product space
