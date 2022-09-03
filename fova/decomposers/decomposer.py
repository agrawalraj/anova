
"""
TODO: say what this does
"""

from abc import ABC


class Decomposer(ABC):

	"""Abstract class for decomposing a regression function into additive
    and interaction effects.
    """

	@abstractmethod
	def get_effect(self, V):
		pass

	@abstractmethod
	def get_decomposition(self, X):
		pass

