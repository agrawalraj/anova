
"""
TODO: say what this does
"""

from abc import ABC


class Decomposer(ABC):

	"""Abstract class for decomposing a regression function into additive
    and interaction effects.

    Attributes
    ----------
    f : object
        Fitted regression function with a predict() method
    """

	def __init__(self, f):
		self.f = f # Regression function

	@abstractmethod
	def fit_decomposition(self):
		pass

	@abstractmethod
	def get_decomposition(self):
		pass

	@abstractmethod
	def covariate_importance(self):
		pass

