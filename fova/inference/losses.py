

import jax as jnp


def mean_squared_pred_loss(y_pred, y_actual):
	return ((y_pred - y_actual) ** 2).mean()


def cv_mse_pred_loss(
		params, hyperarams, kernel, 
		make_prediction, training_cv_batch):
	y_pred = make_prediction(kernel, params, hyperarams, training_cv_batch)
	y_actual = training_cv_batch['Y_cv_test']
	return mean_squared_pred_loss(y_pred, y_actual)


def stochastic_cv_loss(
		params, key, M, X_train, Y_train, 
		loss, make_prediction, hyperarams, kernel):
	
	assert isinstance(params, dict), 
		"Parameters need to be stored in a dictionary!"
	
	# Subsample M datapoints from training dataset
	random_indcs = random.permutation(key, X_train.shape[0])
	cv_train_indcs = random_indcs[M:]
	cv_test_indcs = random_indcs[:M]
	training_cv_batch = dict() 
	training_cv_batch['X_cv_train'] = X_train[cv_train_indcs, :]
	training_cv_batch['X_cv_test'] = X_train[cv_test_indcs, :]
	training_cv_batch['Y_cv_train'] = Y_train[cv_train_indcs]
	training_cv_batch['Y_cv_test'] = Y_train[cv_test_indcs]

	return loss(params, hyperarams, kernel, make_prediction, training_cv_batch)
