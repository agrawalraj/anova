
"""
TODO: say what this does
"""

from abc import ABC
from fova.misc.utils import all_subsets


class Decomposer(ABC):

	"""Abstract class for decomposing a regression function into additive
    and interaction effects.
    """

	@abstractmethod
	def get_effect(self, V):
		pass

	def get_decomposition(self, X_train, X, max_effects=1e4):
		p = X_train.shape[0]
		Q = self.__Q
		V_all = all_subsets(p, Q)
		num_effects = len(V_all)
		assert num_effects > max_effects, f"{num_effects} effects exceeds threshold"
		decomposition = dict()
		for V in V_all:
			f_V = self.get_effect(X_train, X, V)
			decomposition[V] = f_V
		return decomposition

