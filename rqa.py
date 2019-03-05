
def rqa(training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number):
	'''
	@Parameters:
	training_well_ts -- numpy array 2D:
		the 1st column of type string (name of well)
		the last column of type integer: facies_class
		the another column: features

		Example:
		array([['RD-1P'		2555.4432	2434.7698	108.8463	0.2312	2.5599	84.4916	0.6982	0.036	5],
				['RD-1P'	2555.5956	2434.9184	101.5264	0.2011	2.586	81.334	0.617	0.0333	5],
				['RD-1P'	2557.7292	2436.9983	74.2481		0.1072	2.5488	68.2637	0.3139	0		3]])


	testing_well_ts -- numpy array 2d like training_well_ts but containing only data of one well

	dim : the dimension -- type: integer, greater than or equal to 2
	tau : the step -- type: integer, greater than or equal to 1
	epsilon: type of float greater than 0
	lambd: the positive integer
	percent: the float number between 0 and 1
	curve_number: the positive integer is index of column feature in training_well_ts
	facies_class_number: the integer greater than or equal to 0-- the name of class to detect

	@Return:
	predict_label-- numpy array 1D of shape (the length of testing_well_ts, ) containg only 0, 1:
		0: predict not facies_class_number
		1: predict belong to facies_class_number
	'''
	if not (percent > 0 and percent <= 1.0):
		print('percent must > 0 and <= 1.0')
		raise AssertionError


	vectors_train = create_train(training_well_ts, dim, tau, curve_number, facies_class_number)

	vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)
	r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1)

	r = np.sum(r_dist < epsilon, axis=0)
	if __debug__:
		print(np.sum(r))

# 	"""____________________________________________________"""

	predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)
	indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
	# print('indx_vectors_timeseries_test.shape: ', indx_vectors_timeseries_test)

	index = indx_vectors_timeseries_test[r > lambd, :]
	index = index[:, 0]
	add_index = list(np.arange(0, dim*percent, dtype=int))

#	index = index[:, :int(dim*percent)].ravel()
	for i in add_index:
		predict_label[(index+i).ravel()] = 1

	return predict_label.ravel().tolist()

