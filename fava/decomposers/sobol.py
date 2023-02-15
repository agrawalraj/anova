



class Sobol(Decomposer):
	def __init__(self, X_train, model):
		self.featprocessor = Identity()
		self.X_train_shuffled = shuffle(X_train.copy())
		self.model = model
		self.intercept = mode.predict(X_train).mean() # TODO: replace with predict_proba if needed

	def __get_effect_uncorrected(self, X, V):
		if len(V) == 0:
			return self.intercept * jnp.ones((X.shape[0],))
		else:
			preds = []
			for xv in X[: V]:
				X_probe = self.X_train_shuffled.copy()
				X_probe[:, V] = xv
				preds.append(model.predict(X_probe).mean()) # TODO: replace with predict_proba if needed
			return jnp.array(preds)

	def get_effect(self, X, V):
		effects = dict()
		sorted_subsets = all_subsets(V, len(V), True)
		for U in sorted_subsets:
			effect = self.__get_effect_uncorrected(X, U)
			if len(U) > 0:
				lower_order_subsets = all_subsets(U, len(U), True)
				for W in lower_order_subsets:
					if len(W) < len(U):
						effect -= effects[W]		
		return effect[V]


