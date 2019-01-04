import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import json
import os, sys
import math
from matplotlib import colors as mcolors
# from config import BASEDIR
# FILE_PATH_MODEL = BASEDIR + 'files\\crp\\train_data'
FILE_PATH_MODEL = './files/crp/train_data'

def create_table_index(dim , tau, m):
	"""
	Arguments:
	m-- length of time series

	Returns:
	indx-- the indx of each vector
	"""
	num_vectors = m - (dim - 1)*tau
	indx = np.zeros((num_vectors, dim))
	indx[0,:] = np.arange(0, dim*tau, tau).astype(int)

	for i in range(1,num_vectors):
		indx[i,:] = indx[i-1,:]+1

	return indx.ravel()

def filter(data_well, label_well, label, dim ,tau):
	length = []
	vector_data_well_label = []
	index = np.where(label_well == label)[0]
	b = index[0]
	e = index[0]

	for i in range(len(index)-1):
		# khong lien tiep
		if(index[i+1] - index[i] != 1):
			if((e + 1 - b) > ((dim-1)*tau)):
				vector_data_well_label.append(data_well[b:e+1, 1:])

			# statistic length of sub timeseries
			if e+1-b > 0:
				length.append(e+1-b)

			b = index[i+1]
		# lien tiep
		else:
			e = index[i+1]

	if((e + 1 - b) > ((dim-1)*tau)):
		vector_data_well_label.append(data_well[b:e+1, 1:])
	# print('vector_data_well_label.shape: ', vector_data_well_label[0].shape)

	# # compute length of each sub timeseries
	# print('length: ', sorted(Counter(length).items(), key=lambda i: i[0]))

	return vector_data_well_label

def get_vector_each_vector_timeseries(vector_data_well_label, dim, tau):
	# vector train for feature in column 1
	timeseries_train = np.array(vector_data_well_label[:, 0], copy=True)

	indx_vectors_timeseries_train = create_table_index(dim, tau, timeseries_train.shape[0]).astype(int)
	vectors_train = timeseries_train[indx_vectors_timeseries_train].reshape((timeseries_train.shape[0] - (dim -1)*tau, dim))
	#vector train for another feature
	for i in range(1, vector_data_well_label.shape[1]):
		timeseries_train = np.array(vector_data_well_label[:, i], copy=True)
		vectors_train = np.concatenate((vectors_train, timeseries_train[indx_vectors_timeseries_train].reshape((timeseries_train.shape[0] - (dim -1)*tau, dim))), axis=1)
	return vectors_train

def extract_vector_train_each_well(X, y, dim, tau, label, train_well):
	data_well = X[np.where(X[:, 0] == train_well)[0], :]

	'''________________________________________________'''

	'''________________________________________________'''
	label_well = y[np.where(X[:, 0] == train_well)[0]]

	vector_data_well_label = filter(data_well, label_well, label, dim, tau)

	vectors_train = None
	if(len(vector_data_well_label) > 0):
		for i in range(len(vector_data_well_label)):
			if(vectors_train is None):
				vectors_train = get_vector_each_vector_timeseries(vector_data_well_label[0], dim, tau)
			else:
				vectors_train = np.concatenate((vectors_train, get_vector_each_vector_timeseries(vector_data_well_label[i], dim, tau)))

	else: return None
			

	return vectors_train


def extract_vector_test(X, dim, tau):
	data_well_label = X

	timeseries_test = np.array(data_well_label[:, 1], copy=True)
	#_________________#
	# print('timeseries_test.shape: ', timeseries_test.shape)
	#_________________#
	indx_vectors_timeseries_test = create_table_index(dim, tau, timeseries_test.shape[0]).astype(int)
	vectors_test = timeseries_test[indx_vectors_timeseries_test].reshape((timeseries_test.shape[0] - (dim -1)*tau, dim))

	# # vector train for other feature
	# for i in range(2, X.shape[1]):
	# 	timeseries_test = np.array(data_well_label[:, i], copy=True)
	# 	vectors_test = np.concatenate((vectors_test, timeseries_test[indx_vectors_timeseries_test].reshape((timeseries_test.shape[0] - (dim -1)*tau, dim))), axis=1)

	return vectors_test, indx_vectors_timeseries_test


def minMaxScalerPreprocessing(X, minScaler = 0.0, maxScaler = 253):
	return (X - minScaler)/(maxScaler - minScaler)



def hasFaciesClassNumber(facies, facies_class_number):
	return facies_class_number in set(facies)

def create_train(training_well_ts, dim, tau, curve_number, facies_class_number):
	# from sklearn.preprocessing import LabelEncoder
	# label_enc = LabelEncoder()
	# training_well_ts[:,0] = label_enc.fit_transform(training_well_ts[:, 0])

	#__debug__#
	# print('training_well_ts: ', training_well_ts.shape)
	# print('training_well_ts: ', training_well_ts[:, [curve_number]].shape)
	#_________#

	X = training_well_ts[:, [0, curve_number]]
	# print("X banwgf: ",X)
	y = training_well_ts[:, -1].astype(int)
	# print(type(y))
	train_well = list(set(X[:, 0]))

	# print('y: ')
	# print(y)
	# print('train_well', train_well)

	""" vector train of train_well[0]"""

	# if(hasFaciesClassNumber(y[np.where(X[:, 0] == train_well)[0]], facies_class_number)):
	# 	vectors_train = extract_vector_train_each_well(X, y, dim, tau, facies_class_number, train_well[0])
	vectors_train = None
	"""_____________"""
	""" vector train for another train well"""
	for i in range(1, len(train_well)):
		if(hasFaciesClassNumber(y[np.where(X[:, 0] ==  train_well[i])[0]], facies_class_number)):
			if vectors_train is None:
				new_vectors_train = extract_vector_train_each_well(X, y, dim, tau, facies_class_number, train_well[i])
				if(not (new_vectors_train is None)):

					vectors_train = new_vectors_train
			else:
				new_vectors_train = extract_vector_train_each_well(X, y, dim, tau, facies_class_number, train_well[i])
				if(not (new_vectors_train is None)):
					vectors_train = np.concatenate((vectors_train, new_vectors_train))

	"""___________________________________________"""


	return vectors_train




def create_test(testing_well_ts, dim, tau, curve_number):

	# from sklearn.preprocessing import LabelEncoder
	# label_enc = LabelEncoder()
	# testing_well_ts[:,0] = label_enc.fit_transform(testing_well_ts[:, 0])

	X = testing_well_ts[:, [0, curve_number]]
	y = testing_well_ts[:, -1].astype(int)
	#____________#
	# print('X_test: ', X)
	#____________#
	return extract_vector_test(X, dim, tau)


def train(training_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number, id_string):
	minMaxScaler = MinMaxScaler()
	#__debug___#
	# print(training_well_ts[:, [curve_number]].shape)
	#__________#
	training_well_ts[:, curve_number] = minMaxScaler.fit_transform(training_well_ts[:, [curve_number]]).ravel()




	vectors_train = create_train(training_well_ts, dim, tau, curve_number, facies_class_number)
	#__debug__#
	# print('vectors_train: ', vectors_train)
	# print('vectors_train.shape: ', vectors_train.shape)
	#_________#


	np.savez_compressed(FILE_PATH_MODEL+ id_string, train_data=vectors_train, dim=dim, tau=tau, epsilon=epsilon, lambd=lambd, percent=percent, curve_number=curve_number, facies_class_number=facies_class_number)



def create_sqrt_data(data, curve_number_index_to_sum_then_sqrt):
	square_root_data = np.zeros((data.shape[0], 2))

	for i in range(1, data.shape[1]):
		data[:, i] = np.square(data[:, i])
	new_data = np.sum(data[:, curve_number_index_to_sum_then_sqrt], axis=1)
	square_root_data[:, 0] = data[:, 0]
	square_root_data[:, 1] = np.sqrt(np.array(new_data, dtype=np.float32))

	return square_root_data

def train_each_facie_sqrt(data, facies_class, dim, tau, epsilon, lambd, percent, facies_class_number, id_string):
	vectors_train = create_train(np.concatenate((data, facies_class.reshape(-1, 1)), axis=1), dim, tau, curve_number=1, facies_class_number=facies_class_number)
	#__debug__#
	# print('vectors_train: ', vectors_train)
	# print('vectors_train.shape: ', vectors_train.shape)
	#_________#


	np.savez_compressed(os.path.join(FILE_PATH_MODEL,id_string), train_data=vectors_train, dim=dim, tau=tau, epsilon=epsilon, lambd=lambd, percent=percent, facies_class_number=facies_class_number)

def train_all_facies_sqrt(data, facies_class, dim, tau, epsilon, lambd, percent, curve_number_index_to_sum_then_sqrt, id_string):
	'''
	1st column is well encode
	other columns are feature of well
	'''

	y = list(set(facies_class.astype(int)))

	y_train = []
	for i in y:
		if i >= 0:
			y_train.append(i)



	#___________________#
	for i in range(1, data.shape[1]):
		minMaxScaler = MinMaxScaler()

		data[:, i] = minMaxScaler.fit_transform(data[:, i].reshape(-1, 1)).ravel()


	square_root_data = create_sqrt_data(data, curve_number_index_to_sum_then_sqrt)
	json.dump({"facies_train":np.array(y_train, dtype=np.int64).tolist(), "curve_number_index_to_sum_then_sqrt":curve_number_index_to_sum_then_sqrt}, open(os.path.join(FILE_PATH_MODEL,id_string + ".json"), "w"))
	for facies in y_train:
		train_each_facie_sqrt(square_root_data, facies_class, dim, tau, epsilon, lambd, percent, facies_class_number=facies, id_string=id_string+'facies_'+str(facies))



# def predict_each_facie_sqrt(test_data, facies_class_number, id_string):
# 	data = np.load(os.path.join(FILE_PATH_MODEL,id_string+'.npz'))
# 	vectors_train = data['train_data']
# 	dim = data['dim']
# 	tau = data['tau']
# 	percent = data['percent']
# 	# curve_number = data['curve_number']
# 	epsilon = data['epsilon']
# 	lambd = data['lambd']
# 	vectors_test, indx_vectors_timeseries_test = create_test(test_data, dim, tau, curve_number=1)
# 	# print(vectors_test.shape)
# 	r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1)

# 	# #_________________#
# 	# print('vectors_train.shape: ', vectors_train.shape)
# 	# print('vectors_test.shape: ', vectors_test.shape)
# 	# print('r_dist: ', r_dist.shape)
# 	# print('min_r_dist: ', r_dist.min())
# 	# print('r.max: ', r_dist.max())
# 	# #_________________#

# 	r = np.sum(r_dist < epsilon, axis=0)
# 	#__debug__#
# 	# print(r)
# 	#_________#

# # 	"""____________________________________________________"""
# 	# *********************************************

# 	predict_label = np.full((test_data[:, -1].shape[0], ), -1, dtype=int)
# 	indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
# 	# print('indx_vectors_timeseries_test.shape: ', indx_vectors_timeseries_test)

# 	index = indx_vectors_timeseries_test[r > lambd, :]
# 	index = index[:, 0]
# 	add_index = list(np.arange(0, dim*percent, dtype=int))

# #	index = index[:, :int(dim*percent)].ravel()
	# for i in add_index:
	# 	predict_label[(index+i).ravel()] = facies_class_number

	# return predict_label.ravel().tolist()
# def predict_all_facies_sqrt(test_data, id_string):

# 	file_param = json.load(open(os.path.join(FILE_PATH_MODEL,id_string + ".json"), "r"))
# 	curve_number_index_to_sum_then_sqrt = file_param["curve_number_index_to_sum_then_sqrt"]
# 	facies_train = file_param["facies_train"]

# 	# print("curve_number_index_to_sum_then_sqrt", curve_number_index_to_sum_then_sqrt)
# 		#___________________#
# 	for i in range(1, test_data.shape[1]):
# 		minMaxScaler = MinMaxScaler()

# 		test_data[:, i] = minMaxScaler.fit_transform(test_data[:, i].reshape(-1, 1)).ravel()

# 	sqrt_test_data = create_sqrt_data(test_data, curve_number_index_to_sum_then_sqrt)
	
# 	response_data = {}
# 	for y in facies_train:
# 		# print("label: ", str(y))
# 		preLabel = predict_each_facie_sqrt(sqrt_test_data, y, id_string=id_string+'facies_'+str(y))
# 		# print(preLabel)
# 		response_data[y]=preLabel

# 	return response_data


def predict(testing_well_ts, id_string):
	if sys.platform.startswith("win"):
		data = np.load(FILE_PATH_MODEL+id_string+'.npz')
	else:
		data = np.load(FILE_PATH_MODEL+"\\"+id_string+'.npz')


	vectors_train = data['train_data']

	dim = data['dim']
	tau = data['tau']
	percent = data['percent']
	curve_number = data['curve_number']
	epsilon = data['epsilon']
	lambd = data['lambd']
	minMaxScaler = MinMaxScaler()
	testing_well_ts[:, curve_number] = minMaxScaler.fit_transform(testing_well_ts[:, [curve_number]]).ravel()
	

	vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)
	r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1)
	

	#_________________#
	print('vectors_train.shape: ', vectors_train.shape)
	print('vectors_test.shape: ', vectors_test.shape)
	print('r_dist: ', r_dist.shape)
	print('min_r_dist: ', r_dist.min())
	print('r.max: ', r_dist.max())
	#_________________#

	r = np.sum(r_dist < epsilon, axis=0)
	# __debug__
	# print('r: ', r)
	# _________
	"""____________________________________________________"""


	predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)

	indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
	# print('indx_vectors_timeseries_test.shape: ', indx_vectors_timeseries_test)
	
	index = indx_vectors_timeseries_test[r > lambd, :]
	# print(index)

	index = index[:, 0]


	add_index = list(np.arange(0, dim*percent, dtype=int))

#	index = 	index[:, :int(dim*percent)].ravel()        
	for i in add_index: 
		predict_label[(index+i).ravel()] = 1
		# print(predict_label)
	return predict_label.ravel().tolist()

# def rqa(training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number):
# 	'''
# 	@Parameters:
# 	training_well_ts -- numpy array 2D:
# 		the 1st column of type string (name of well)
# 		the last column of type integer: facies_class
# 		the another column: features

# 		Example:
# 		array([['RD-1P'		2555.4432	2434.7698	108.8463	0.2312	2.5599	84.4916	0.6982	0.036	5],
# 				['RD-1P'	2555.5956	2434.9184	101.5264	0.2011	2.586	81.334	0.617	0.0333	5],
# 				['RD-1P'	2557.7292	2436.9983	74.2481		0.1072	2.5488	68.2637	0.3139	0		3]])


# 	testing_well_ts -- numpy array 2d like training_well_ts but containing only data of one well

# 	dim : the dimension -- type: integer, greater than or equal to 2
# 	tau : the step -- type: integer, greater than or equal to 1
# 	epsilon: type of float greater than 0
# 	lambd: the positive integer
# 	percent: the float number between 0 and 1
# 	curve_number: the positive integer is index of column feature in training_well_ts
# 	facies_class_number: the integer greater than or equal to 0-- the name of class to detect

# 	@Return:
# 	predict_label-- numpy array 1D of shape (the length of testing_well_ts, ) containg only 0, 1:
# 		0: predict not facies_class_number
# 		1: predict belong to facies_class_number
# 	'''
# 	if not (percent > 0 and percent <= 1.0):
# 		print('percent must > 0 and <= 1.0')
# 		raise AssertionError


# 	vectors_train = create_train(training_well_ts, dim, tau, curve_number, facies_class_number)


# 	vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)
# 	r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1)



# 	r = np.sum(r_dist < epsilon, axis=0)
# 	if __debug__:
# 		print(np.sum(r))

# # 	"""____________________________________________________"""

# 	predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)
# 	indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
# 	# print('indx_vectors_timeseries_test.shape: ', indx_vectors_timeseries_test)

# 	index = indx_vectors_timeseries_test[r > lambd, :]
# 	index = index[:, 0]
# 	add_index = list(np.arange(0, dim*percent, dtype=int))

# #	index = index[:, :int(dim*percent)].ravel()
# 	for i in add_index:
# 		predict_label[(index+i).ravel()] = 1

# 	return predict_label.ravel().tolist()


# def load_dataset(file_name='../data/data.csv'):
# 	# load dataset
# 	data = pd.read_csv(file_name)

# 	X = data.iloc[:, :].values


# 	# return X

# def get_data_from_json(data_json):
# 	trainset = data_json['data']['train']
# 	testset = data_json['data']['test']

# 	label_enc = LabelEncoder()
# 	well = label_enc.fit_transform(np.array(trainset['well']).reshape(-1, 1)).reshape(-1, 1)
# 	trainset['data'] = np.array(trainset['data']).T
# 	trainset['facies'] = np.array(trainset['facies']).reshape(-1, 1)
# 	training_well_ts = np.concatenate((well, trainset['data'], trainset['facies']), axis=1)

# 	label_enc = LabelEncoder()
# 	well = label_enc.fit_transform(np.array(testset['well']).reshape(-1, 1)).reshape(-1, 1)
# 	testset['data'] = np.array(testset['data']).T
# 	testing_well_ts = np.concatenate((well, testset['data']), axis=1)

# 	params = data_json['params']
# 	dim = params['dim']
# 	if dim == None:
# 		raise AssertionError

# 	tau = params['tau']
# 	if tau == None:
# 		raise AssertionError

# 	epsilon = params['epsilon']
# 	if epsilon == None:
# 		raise AssertionError

# 	lambd = params['lambd']
# 	if lambd == None:
# 		raise AssertionError

# 	percent = params['percent']
# 	if percent == None:
# 		raise AssertionError

# 	curve_number = params['curve_number']
# 	if curve_number == None:
# 		raise AssertionError

# 	facies_class_number = params['facies_class_number']
# 	if facies_class_number == None:
# 		raise AssertionError
# 	if (dim < 0 and (dim-1)*tau >= trainset['data'].shape[1] and (dim-1)*tau >= testset['data'].shape[1]):
# 		raise Exception('Dim must be > dimention of trainset and testset')
# 	if (facies_class_number not in trainset['facies']):
# 		raise Exception('Facies not exist in trainset')

# 	return training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number

def filter_sort(arr):
		x = []
		if (len(arr) > 0):
			x.append(arr[0])
			for t in arr:
				if x.count(t) == 0:
					x.append(t)
		return x

def state_phase(v, start, m, step):
	return [v[start + i*step] for i in range(0, m, 1)]

def split(indexSet, dataSet):
	outputArr = [[0]]
	i = 0
	while (i < len(indexSet)):
		a = [dataSet[indexSet[i]]];
		while (i < len(indexSet) -1 and (indexSet[i+1] - indexSet[i] == 1)):
			i+=1
			a.append(dataSet[indexSet[i]])
		if (len(a) > 1):
			outputArr.append(a)
		i+=1		
	return outputArr


def predict_diagonal(testing_well_ts, id_string):
	if sys.platform.startswith("win"):
		data = np.load(FILE_PATH_MODEL+id_string+'.npz')
	else:
		data = np.load(FILE_PATH_MODEL+"\\"+id_string+'.npz')
		

	# vectors_train = data['train_data']
	dim = data['dim']
	tau = data['tau']
	percent = data['percent']
	curve_number = data['curve_number']
	epsilon = data['epsilon']
	lambd = data['lambd']
	minMaxScaler = MinMaxScaler()
	# testing_well_ts[:, curve_number] = minMaxScaler.fit_transform(testing_well_ts[:, [curve_number]]).ravel()
	# vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)


	# trainSet = [75.807526,80.953362,90.760841,97.119331,97.439201,95.756302,93.359238,90.273407,86.506943,83.491196,81.683189,80.336388,79.212402,78.285912,77.54882,77.908432,79.654434,82.109657,85.057877,88.541397,92.573746,92.706635,87.519775,80.937218,74.211861,69.055145,66.01355,65.364136,67.195396,70.25882]
	trainSet_30 = np.array([0.325592096,0.486023159,0.791789665,0.990027486,1,0.947532507,0.872799552,0.776592998,0.659166436,0.565144882,0.508776881,0.466787841,0.431745475,0.402860416,0.379880194,0.39109177,0.445526649,0.522072819,0.613989093,0.722594346,0.848310369,0.852453433,0.690743369,0.48551984,0.275844254,0.115073986,0.020246572,0,0.057092832,0.152600806])
	trainSet_5_15 = np.array([1,0.947532507,0.872799552,0.776592998,0.659166436,0.565144882,0.508776881,0.466787841,0.431745475,0.402860416,0.379880194])
	trainSet_15_30 = np.array([0.379880194,0.39109177,0.445526649,0.522072819,0.613989093,0.722594346,0.848310369,0.852453433,0.690743369,0.48551984,0.275844254,0.115073986,0.020246572,0,0.057092832,0.152600806])
	
	vectors_train_1 = []
	for i in range(len(trainSet_15_30)-(dim-1)*tau):
		vectors_train_1.append(state_phase(trainSet_15_30, i, dim, tau)) 
	
	# testSet_300 = [78.869453,78.008301,75.581558,73.238037,71.504173,71.459229,73.4478,75.83062,78.084259,79.799255,80.844872,80.38166,78.141602,77.074661,78.122734,80.320938,83.361206,86.870644,90.730186,93.797852,95.70903,95.946472,94.345032,87.30777,73.686234,63.072395,58.528858,58.117874,61.220753,65.996002,71.855675,76.827644,80.282249,82.632019,84.008675,85.735664,88.235558,90.149986,91.045235,90.99617,90.026672,86.375076,79.478905,83.406456,86.22171,86.239113,84.986069,82.950256,81.365021,80.624146,79.560143,77.800262,76.190826,75.002068,76.785156,82.35466,86.284805,86.843201,85.263039,81.938042,79.117104,77.51825,75.644775,73.018806,69.979317,66.634514,65.481483,67.317505,69.82917,72.277832,74.679474,77.0392,79.818054,83.163239,85.634315,86.771347,86.554359,84.976982,83.008965,80.959938,80.552299,82.335991,86.705421,93.78653,101.03632,107.64285,110.9325,110.05161,103.46317,90.676407,82.472366,82.293304,83.275452,83.22728,69.273605,74.008728,80.14045,86.236588,91.839874,95.242744,95.899986,94.425423,91.015022,87.080711,83.07328,80.033913,78.295021,77.703857,78.211639,78.372162,77.723671,77.032516,87.913818,95.831078,100.99051,102.55498,98.15271,87.026421,79.942932,80.339981,83.314407,87.30069,90.476624,92.260429,91.926941,89.244629,86.293861,83.738884,83.713669,86.899574,88.476822,86.906509,84.239403,81.130264,80.111015,81.990051,78.647629,79.652397,81.467911,84.114616,85.045227,83.446411,81.450462,79.73819,79.068375,79.683273,78.841873,75.668983,73.200844,72.406868,75.982033,84.786797,91.092934,92.432922,91.579109,89.416687,87.31768,85.720169,84.082336,82.231163,80.330215,78.431694,76.790062,75.486565,75.809128,78.168968,81.460197,85.32972,87.628708,87.671066,88.407181,90.779076,94.245239,98.632774,102.90547,106.73246,109.2777,110.27426,109.98178,108.48315,105.65147,101.44622,97.148308,93.166679,89.476334,86.06926,83.157181,80.807693,80.283516,81.987823,86.038742,92.473991,94.059013,88.48391,83.401054,81.253754,83.004066,88.959152,91.617622,88.584389,84.583672,81.123878,79.110558,78.832848,78.010475,75.91539,73.772545,71.973061,71.062202,71.214066,71.354996,71.142174,71.044144,71.210503,72.07666,73.781639,76.6577,80.810944,85.114319,89.20797,91.840408,92.61203,91.772209,89.400574,86.769501,84.285255,84.164246,87.114159,87.864609,84.732819,80.002724,74.403549,73.13224,77.848122,81.369164,81.402214,79.241402,75.299911,73.18853,74.060158,76.883148,81.32814,86.797684,93.101021,94.486671,89.118248,82.595329,76.705795,71.418556,66.723686,67.981857,76.904678,84.520844,87.965904,86.965508,81.432045,77.496231,77.11554,77.261223,76.966248,75.957512,74.147804,75.636192,81.731483,86.364143,87.596245,83.281776,72.735535,65.888283,65.910789,68.002235,70.629753,72.498878,73.196304,75.030472,78.738419,81.877235,83.666939,81.7686,75.435417,72.189537,74.432693,78.766785,84.10685,88.00769,89.688576,90.065933,89.432381,87.61499,84.55854,78.519714,68.94191,66.056808,73.131294,84.57048,98.587967,102.39403,91.905029,83.155769,81.265976]
	
	testSet_100 = np.array([0.280524229,0.270739067,0.313555666,0.418441562,0.588419903,0.762447343,0.921033662,1,0.978854693,0.820702616,0.513763017,0.316829277,0.312530987,0.336106939,0.334950595,0,0.113664046,0.260852846,0.407187469,0.541691437,0.623375563,0.639152315,0.603756197,0.521891294,0.427450203,0.331253905,0.25829548,0.216554282,0.202363697,0.21455274,0.218406011,0.202839322,0.186248505,0.447448504,0.637498215,0.761347698,0.798901987,0.693227784,0.426147,0.256111529,0.265642484,0.337042034,0.432730685,0.50896733,0.551786641,0.543781435,0.47939393,0.408562278,0.347231383,0.34662611,0.423102104,0.460963122,0.423268575,0.359246082,0.284612816,0.260146274,0.305251556,0.225018453,0.249137387,0.292717857,0.356250635,0.378589468,0.340210721,0.292299003,0.251196803,0.235118242,0.249878549,0.22968118,0.153517605,0.094271211,0.07521223,0.161032207,0.37238599,0.523761531,0.555927246,0.535431909,0.483524093,0.433138521,0.3947911,0.35547577,0.311039324,0.265408056,0.219835046,0.180428523,0.149138755,0.156881712,0.213528445,0.292532686,0.385418575,0.440604586,0.441621368,0.459291426,0.516227533,0.599430974,0.704751471,0.807315317,0.899180223,0.960277386,0.984199291,0.977178461])
	testSet_300 = np.array([0.392913536,0.376608356,0.330660046,0.286287474,0.253458237,0.252607261,0.290259157,0.335375823,0.378046556,0.410518544,0.43031641,0.421545886,0.379132297,0.358930679,0.378775048,0.420396167,0.47796105,0.544409267,0.617486405,0.675570046,0.711756572,0.716252334,0.685930433,0.552685875,0.294773702,0.093809708,0.007781707,0,0.058750445,0.149165714,0.260113628,0.354253623,0.419663624,0.464154516,0.490220323,0.522919388,0.57025275,0.606500812,0.623451589,0.622522585,0.604165967,0.535026109,0.40445299,0.47881782,0.532122255,0.532451766,0.508726446,0.470180062,0.440164988,0.426137152,0.405991162,0.372669315,0.342196017,0.319687897,0.353449148,0.458902959,0.533316905,0.543889657,0.513970637,0.451014653,0.397602596,0.367329658,0.331857006,0.282136522,0.224586388,0.161255395,0.139423735,0.174187247,0.221743483,0.268106811,0.313579855,0.358259255,0.410874487,0.474212713,0.521000431,0.542529163,0.53842068,0.508554391,0.471291667,0.432495087,0.42477679,0.458549478,0.54128091,0.675355673,0.812624267,0.937713281,1,0.983321099,0.858574603,0.616468145,0.461131622,0.457741236,0.476337371,0.475425275,0.211224333,0.300879851,0.41697878,0.532403957,0.638497401,0.702927844,0.715372161,0.687452568,0.622879532,0.548386706,0.472509416,0.414961593,0.382037155,0.370843969,0.380458388,0.383497754,0.37121913,0.358132699,0.564160877,0.714067447,0.811756894,0.8413788,0.75802557,0.547358772,0.413238945,0.42075673,0.477074951,0.552551821,0.612685424,0.646460252,0.640145941,0.589358649,0.533488372,0.485112061,0.484634636,0.544957032,0.574820878,0.54508834,0.494588961,0.435720065,0.416421454,0.451999399,0.388713487,0.407737913,0.442113123,0.492226226,0.509846552,0.479574334,0.441782741,0.40936233,0.396679954,0.408322524,0.392391332,0.332315364,0.285583256,0.270549997,0.338242699,0.5049534,0.624354729,0.64972626,0.633560038,0.592616421,0.552873513,0.522626003,0.491615032,0.456564649,0.420571819,0.384624942,0.353542039,0.328861435,0.33496889,0.379650449,0.441967065,0.515233184,0.558762563,0.559564575,0.573502285,0.618412095,0.684040937,0.767115172,0.848015029,0.920475823,0.968667773,0.987536787,0.981998927,0.953623646,0.900008198,0.820385374,0.739008074,0.663619323,0.593745786,0.529235744,0.47409801,0.429612458,0.419687613,0.451957213,0.528657912,0.650503866,0.680514907,0.574955083,0.478715538,0.438058243,0.471198908,0.583953386,0.634289249,0.576857568,0.501107401,0.435599151,0.397478653,0.392220451,0.376649519,0.336980871,0.296407927,0.262336231,0.245089893,0.247965308,0.250633698,0.246604094,0.24474798,0.247897846,0.26429779,0.296580114,0.351035878,0.429674013,0.51115475,0.588664542,0.638507512,0.653117517,0.637216222,0.592311335,0.54249421,0.49545713,0.493165928,0.549020016,0.563229147,0.503931373,0.414371056,0.30835545,0.284284298,0.373575504,0.440243433,0.440869206,0.399956073,0.3253273,0.285350101,0.301853634,0.355304543,0.439466678,0.543027831,0.662376145,0.688612246,0.586965733,0.463459822,0.351946516,0.251837152,0.16294379,0.186766186,0.355712196,0.499917807,0.565147081,0.546205436,0.441434031,0.366912747,0.359704688,0.362463071,0.356877971,0.337778415,0.303513136,0.331694494,0.447103634,0.534819102,0.558147903,0.47645711,0.276773027,0.147126147,0.147552279,0.187152026,0.23690184,0.272292128,0.285497295,0.320225703,0.390432518,0.449863324,0.483749844,0.447806413,0.327892991,0.26643502,0.308907267,0.390969604,0.492079183,0.565938264,0.597764407,0.60490934,0.592913573,0.558502824,0.500631548,0.386291526,0.204943971,0.150317024,0.284266386,0.500857622,0.76626679,0.838331349,0.639731056,0.474071275,0.438289656])
	
	vectors_test_1 = []
	for i in range(len(testSet_300)-(dim-1)*tau):
		vectors_test_1.append(state_phase(testSet_300, i, dim, tau))

	vectors_train_1 = np.array(vectors_train_1)
	vectors_test_1 = np.array(vectors_test_1)
	# print("vectors_test_1", vectors_test_1)
	# print("vectors_train_1", vectors_train_1)

	r_dist = cdist(vectors_train_1, vectors_test_1, 'minkowski', p=1)
	
	# print("r_dist:",r_dist)

	print('vectors_train.shape: ', vectors_train_1.shape)
	print('vectors_test.shape: ', vectors_test_1.shape)
	print('r_dist.shape: ', r_dist.shape)
	print('rdist.min: ', r_dist.min())
	print('rdist.max: ', r_dist.max())
	print('epsilon: ', epsilon)
	print('lambd: ', lambd)

	# predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)

	# indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
	predict_label = np.zeros(len(testSet_300),dtype=int)
	
	r1 = np.array(r_dist < epsilon)
	len_r1 = int(np.size(r1[0]))
	high_r1 = int(np.size(r1)/len_r1)

	#x is an array whose each element is a diagonal of the crp matrix
	x = []
	for offset in range(-high_r1+1, len_r1):
		x.append(np.array(np.diagonal(r1, offset), dtype=int))
	# print(x)

	index_vecto = []

	for i_row in range(len(x)):
		lenn = np.size(x[i_row])
		i = 0
		while i<lenn:
			if (x[i_row][i] == 1):
				start = i
				while (i<lenn-1 and x[i_row][i+1] == 1 ):
					i+=1
				if (i-start+1 > lambd):
					for temp in range(i-start+1):
						if (i_row < high_r1):
							index_vecto.append(start+temp)
						else:
							index_vecto.append(i_row - high_r1 + start + 1 + temp)			
			i+=1


	index_vecto.sort()
	index_vecto = filter_sort(index_vecto)
	# print("index_vecto", index_vecto)

	index_timeseries_test = []
	for i in index_vecto:
		for j in range(((int(dim*percent))-1)*tau + 1):
			index_timeseries_test.append(i+j)
	index_timeseries_test.sort()
	index_timeseries_test = filter_sort(index_timeseries_test)
	# print("index_timeseries_test", index_timeseries_test)

	for i in index_timeseries_test:
		predict_label[i] = 1
	print('predict_label:', predict_label)
	print('num predict: ', np.sum(predict_label))

	testSetPr = split(index_timeseries_test, testSet_300)
	# print('testSetPr', testSetPr)
	# add_index = list(np.arange(0, dim*percent, dtype=int))  
	# print(add_index)     
	# for i in add_index: 
	# 	predict_label[(index+i).ravel()] = 1

	# xs = [i for i, _ in enumerate(trainSet_15_30)]
	colors = ['b:','g:','c:','m:','y:','k:','p:','b:','g:','c:','m:','y:','k:','p:']
	for i in range(1, np.size(testSetPr)):
		plt.plot(testSetPr[i], colors[i], label=i)
	plt.plot(trainSet_15_30, 'r-', label='train')
	plt.legend(loc=0)
	plt.xlabel('index')
	plt.ylabel('feature')
	plt.title('diagonal_sample15-30')
	plt.show()

	return predict_label.ravel().tolist()


def demo_predict_straight(testing_well_ts, id_string):
	if sys.platform.startswith("win"):
		data = np.load(FILE_PATH_MODEL+id_string+'.npz')
	else:
		data = np.load(FILE_PATH_MODEL+"\\"+id_string+'.npz')
	# vectors_train = data['train_data']
	dim = data['dim']
	tau = data['tau']
	percent = data['percent']
	curve_number = data['curve_number']
	epsilon = data['epsilon']
	lambd = data['lambd']
	minMaxScaler = MinMaxScaler()
	# testing_well_ts[:, curve_number] = minMaxScaler.fit_transform(testing_well_ts[:, [curve_number]]).ravel()
	# vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)

	trainSet_30 = np.array([0.325592096,0.486023159,0.791789665,0.990027486,1,0.947532507,0.872799552,0.776592998,0.659166436,0.565144882,0.508776881,0.466787841,0.431745475,0.402860416,0.379880194,0.39109177,0.445526649,0.522072819,0.613989093,0.722594346,0.848310369,0.852453433,0.690743369,0.48551984,0.275844254,0.115073986,0.020246572,0,0.057092832,0.152600806])
	trainSet_5_15 = np.array([1,0.947532507,0.872799552,0.776592998,0.659166436,0.565144882,0.508776881,0.466787841,0.431745475,0.402860416,0.379880194])
	trainSet_15_30 = np.array([0.379880194,0.39109177,0.445526649,0.522072819,0.613989093,0.722594346,0.848310369,0.852453433,0.690743369,0.48551984,0.275844254,0.115073986,0.020246572,0,0.057092832,0.152600806])
	
	vectors_train_1 = []
	for i in range(len(trainSet_15_30)-(dim-1)*tau):
		vectors_train_1.append(state_phase(trainSet_15_30, i, dim, tau)) 
	
	# testSet_300 = [78.869453,78.008301,75.581558,73.238037,71.504173,71.459229,73.4478,75.83062,78.084259,79.799255,80.844872,80.38166,78.141602,77.074661,78.122734,80.320938,83.361206,86.870644,90.730186,93.797852,95.70903,95.946472,94.345032,87.30777,73.686234,63.072395,58.528858,58.117874,61.220753,65.996002,71.855675,76.827644,80.282249,82.632019,84.008675,85.735664,88.235558,90.149986,91.045235,90.99617,90.026672,86.375076,79.478905,83.406456,86.22171,86.239113,84.986069,82.950256,81.365021,80.624146,79.560143,77.800262,76.190826,75.002068,76.785156,82.35466,86.284805,86.843201,85.263039,81.938042,79.117104,77.51825,75.644775,73.018806,69.979317,66.634514,65.481483,67.317505,69.82917,72.277832,74.679474,77.0392,79.818054,83.163239,85.634315,86.771347,86.554359,84.976982,83.008965,80.959938,80.552299,82.335991,86.705421,93.78653,101.03632,107.64285,110.9325,110.05161,103.46317,90.676407,82.472366,82.293304,83.275452,83.22728,69.273605,74.008728,80.14045,86.236588,91.839874,95.242744,95.899986,94.425423,91.015022,87.080711,83.07328,80.033913,78.295021,77.703857,78.211639,78.372162,77.723671,77.032516,87.913818,95.831078,100.99051,102.55498,98.15271,87.026421,79.942932,80.339981,83.314407,87.30069,90.476624,92.260429,91.926941,89.244629,86.293861,83.738884,83.713669,86.899574,88.476822,86.906509,84.239403,81.130264,80.111015,81.990051,78.647629,79.652397,81.467911,84.114616,85.045227,83.446411,81.450462,79.73819,79.068375,79.683273,78.841873,75.668983,73.200844,72.406868,75.982033,84.786797,91.092934,92.432922,91.579109,89.416687,87.31768,85.720169,84.082336,82.231163,80.330215,78.431694,76.790062,75.486565,75.809128,78.168968,81.460197,85.32972,87.628708,87.671066,88.407181,90.779076,94.245239,98.632774,102.90547,106.73246,109.2777,110.27426,109.98178,108.48315,105.65147,101.44622,97.148308,93.166679,89.476334,86.06926,83.157181,80.807693,80.283516,81.987823,86.038742,92.473991,94.059013,88.48391,83.401054,81.253754,83.004066,88.959152,91.617622,88.584389,84.583672,81.123878,79.110558,78.832848,78.010475,75.91539,73.772545,71.973061,71.062202,71.214066,71.354996,71.142174,71.044144,71.210503,72.07666,73.781639,76.6577,80.810944,85.114319,89.20797,91.840408,92.61203,91.772209,89.400574,86.769501,84.285255,84.164246,87.114159,87.864609,84.732819,80.002724,74.403549,73.13224,77.848122,81.369164,81.402214,79.241402,75.299911,73.18853,74.060158,76.883148,81.32814,86.797684,93.101021,94.486671,89.118248,82.595329,76.705795,71.418556,66.723686,67.981857,76.904678,84.520844,87.965904,86.965508,81.432045,77.496231,77.11554,77.261223,76.966248,75.957512,74.147804,75.636192,81.731483,86.364143,87.596245,83.281776,72.735535,65.888283,65.910789,68.002235,70.629753,72.498878,73.196304,75.030472,78.738419,81.877235,83.666939,81.7686,75.435417,72.189537,74.432693,78.766785,84.10685,88.00769,89.688576,90.065933,89.432381,87.61499,84.55854,78.519714,68.94191,66.056808,73.131294,84.57048,98.587967,102.39403,91.905029,83.155769,81.265976]
	
	testSet_100 = np.array([0.280524229,0.270739067,0.313555666,0.418441562,0.588419903,0.762447343,0.921033662,1,0.978854693,0.820702616,0.513763017,0.316829277,0.312530987,0.336106939,0.334950595,0,0.113664046,0.260852846,0.407187469,0.541691437,0.623375563,0.639152315,0.603756197,0.521891294,0.427450203,0.331253905,0.25829548,0.216554282,0.202363697,0.21455274,0.218406011,0.202839322,0.186248505,0.447448504,0.637498215,0.761347698,0.798901987,0.693227784,0.426147,0.256111529,0.265642484,0.337042034,0.432730685,0.50896733,0.551786641,0.543781435,0.47939393,0.408562278,0.347231383,0.34662611,0.423102104,0.460963122,0.423268575,0.359246082,0.284612816,0.260146274,0.305251556,0.225018453,0.249137387,0.292717857,0.356250635,0.378589468,0.340210721,0.292299003,0.251196803,0.235118242,0.249878549,0.22968118,0.153517605,0.094271211,0.07521223,0.161032207,0.37238599,0.523761531,0.555927246,0.535431909,0.483524093,0.433138521,0.3947911,0.35547577,0.311039324,0.265408056,0.219835046,0.180428523,0.149138755,0.156881712,0.213528445,0.292532686,0.385418575,0.440604586,0.441621368,0.459291426,0.516227533,0.599430974,0.704751471,0.807315317,0.899180223,0.960277386,0.984199291,0.977178461])
	testSet_300 = np.array([0.392913536,0.376608356,0.330660046,0.286287474,0.253458237,0.252607261,0.290259157,0.335375823,0.378046556,0.410518544,0.43031641,0.421545886,0.379132297,0.358930679,0.378775048,0.420396167,0.47796105,0.544409267,0.617486405,0.675570046,0.711756572,0.716252334,0.685930433,0.552685875,0.294773702,0.093809708,0.007781707,0,0.058750445,0.149165714,0.260113628,0.354253623,0.419663624,0.464154516,0.490220323,0.522919388,0.57025275,0.606500812,0.623451589,0.622522585,0.604165967,0.535026109,0.40445299,0.47881782,0.532122255,0.532451766,0.508726446,0.470180062,0.440164988,0.426137152,0.405991162,0.372669315,0.342196017,0.319687897,0.353449148,0.458902959,0.533316905,0.543889657,0.513970637,0.451014653,0.397602596,0.367329658,0.331857006,0.282136522,0.224586388,0.161255395,0.139423735,0.174187247,0.221743483,0.268106811,0.313579855,0.358259255,0.410874487,0.474212713,0.521000431,0.542529163,0.53842068,0.508554391,0.471291667,0.432495087,0.42477679,0.458549478,0.54128091,0.675355673,0.812624267,0.937713281,1,0.983321099,0.858574603,0.616468145,0.461131622,0.457741236,0.476337371,0.475425275,0.211224333,0.300879851,0.41697878,0.532403957,0.638497401,0.702927844,0.715372161,0.687452568,0.622879532,0.548386706,0.472509416,0.414961593,0.382037155,0.370843969,0.380458388,0.383497754,0.37121913,0.358132699,0.564160877,0.714067447,0.811756894,0.8413788,0.75802557,0.547358772,0.413238945,0.42075673,0.477074951,0.552551821,0.612685424,0.646460252,0.640145941,0.589358649,0.533488372,0.485112061,0.484634636,0.544957032,0.574820878,0.54508834,0.494588961,0.435720065,0.416421454,0.451999399,0.388713487,0.407737913,0.442113123,0.492226226,0.509846552,0.479574334,0.441782741,0.40936233,0.396679954,0.408322524,0.392391332,0.332315364,0.285583256,0.270549997,0.338242699,0.5049534,0.624354729,0.64972626,0.633560038,0.592616421,0.552873513,0.522626003,0.491615032,0.456564649,0.420571819,0.384624942,0.353542039,0.328861435,0.33496889,0.379650449,0.441967065,0.515233184,0.558762563,0.559564575,0.573502285,0.618412095,0.684040937,0.767115172,0.848015029,0.920475823,0.968667773,0.987536787,0.981998927,0.953623646,0.900008198,0.820385374,0.739008074,0.663619323,0.593745786,0.529235744,0.47409801,0.429612458,0.419687613,0.451957213,0.528657912,0.650503866,0.680514907,0.574955083,0.478715538,0.438058243,0.471198908,0.583953386,0.634289249,0.576857568,0.501107401,0.435599151,0.397478653,0.392220451,0.376649519,0.336980871,0.296407927,0.262336231,0.245089893,0.247965308,0.250633698,0.246604094,0.24474798,0.247897846,0.26429779,0.296580114,0.351035878,0.429674013,0.51115475,0.588664542,0.638507512,0.653117517,0.637216222,0.592311335,0.54249421,0.49545713,0.493165928,0.549020016,0.563229147,0.503931373,0.414371056,0.30835545,0.284284298,0.373575504,0.440243433,0.440869206,0.399956073,0.3253273,0.285350101,0.301853634,0.355304543,0.439466678,0.543027831,0.662376145,0.688612246,0.586965733,0.463459822,0.351946516,0.251837152,0.16294379,0.186766186,0.355712196,0.499917807,0.565147081,0.546205436,0.441434031,0.366912747,0.359704688,0.362463071,0.356877971,0.337778415,0.303513136,0.331694494,0.447103634,0.534819102,0.558147903,0.47645711,0.276773027,0.147126147,0.147552279,0.187152026,0.23690184,0.272292128,0.285497295,0.320225703,0.390432518,0.449863324,0.483749844,0.447806413,0.327892991,0.26643502,0.308907267,0.390969604,0.492079183,0.565938264,0.597764407,0.60490934,0.592913573,0.558502824,0.500631548,0.386291526,0.204943971,0.150317024,0.284266386,0.500857622,0.76626679,0.838331349,0.639731056,0.474071275,0.438289656])
	
	vectors_test_1 = []
	for i in range(len(testSet_300)-(dim-1)*tau):
		vectors_test_1.append(state_phase(testSet_300, i, dim, tau))

	vectors_train_1 = np.array(vectors_train_1)
	vectors_test_1 = np.array(vectors_test_1)

	r_dist = cdist(vectors_train_1, vectors_test_1, 'minkowski', p=1)

	print('vectors_train.shape: ', vectors_train_1.shape)
	print('vectors_test.shape: ', vectors_test_1.shape)
	print('r_dist: ', r_dist.shape)
	print('min_r_dist: ', r_dist.min())
	print('r.max: ', r_dist.max())
	print('epsilon: ', epsilon)
	print('lambd: ', lambd)

	r = np.sum(r_dist < epsilon, axis=0)
	# print('r',r)
	index_timeseries_test = []
	for i in range(len(testSet_300)-(dim-1)*tau):
		index_timeseries_test.append(state_phase(np.arange(len(testSet_300)),i,dim,tau))
	index_timeseries_test = np.array(index_timeseries_test)
	
	index = np.array(index_timeseries_test[r > lambd, :])
	index = index[:,0]
	# print(index)

	index_timeseries_test_pr = []
	for i in index:
		for j in range(((int(dim*percent))-1)*tau + 1):
			index_timeseries_test_pr.append(i+j)
	index_timeseries_test_pr.sort()
	index_timeseries_test_pr = filter_sort(index_timeseries_test_pr)
	# print("index_timeseries_test", index_timeseries_test_pr)

	predict_label = np.zeros(len(testSet_300),dtype=int)

	for i in index_timeseries_test_pr:
		# for j in range()
		predict_label[i] = 1
	print('predict_label: ', predict_label)
	print("num predict: ", np.sum(predict_label))

	testSetPr = split(index_timeseries_test_pr, testSet_300)
	# print('testSetPr', testSetPr)
	# add_index = list(np.arange(0, dim*percent, dtype=int))  
	# print(add_index)     
	# for i in add_index: 
	# 	predict_label[(index+i).ravel()] = 1

	# xs = [i for i, _ in enumerate(trainSet_15_30)]
	colors = ['b:','g:','c:','m:','y:','k:','p:','b:','g:','c:','m:','y:','k:','p:']
	for i in range(1,np.size(testSetPr)):
		plt.plot(testSetPr[i], colors[i], label=i)
	plt.plot(trainSet_15_30, 'r-', label='train')
	plt.legend(loc=0)
	plt.xlabel('index')
	plt.ylabel('feature')
	plt.title('straight_sample15-30')
	plt.show()
	# add_index = list(np.arange(0, dim*percent, dtype=int))  
	# print(add_index)     
	# for i in add_index: 
	# 	predict_label[(index+i).ravel()] = 1
	return predict_label.ravel().tolist()


# def predict_horizontal(testing_well_ts, id_string):
# 	if sys.platform.startswith("win"):
# 		data = np.load(FILE_PATH_MODEL+id_string+'.npz')
# 	else:
# 		data = np.load(FILE_PATH_MODEL+"\\"+id_string+'.npz')
# 	vectors_train = data['train_data']
# 	dim = data['dim']
# 	tau = data['tau']
# 	percent = data['percent']
# 	curve_number = data['curve_number']
# 	epsilon = data['epsilon']
# 	lambd = data['lambd']
# 	minMaxScaler = MinMaxScaler()
# 	testing_well_ts[:, curve_number] = minMaxScaler.fit_transform(testing_well_ts[:, [curve_number]]).ravel()

# 	vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)
# 	r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1)
	
# 	predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)
# 	indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))


# 	x = np.ravel(np.array(r_dist < epsilon))
# 	print(x)
# 	index = []
# 	count = 0
# 	for i_row in range(len(x)):
# 		lenn = np.size(x[i_row])
# 		i = 0
# 		while i<lenn:
# 			if (x[i_row][i] == 1):
# 				start = i
# 				while (x[i_row][i+1] == 1 and i<lenn):
# 					i+=1
# 				if (i-start+1 > lambd):
# 					count+=1
# 					index.append(i_row)
					
# 			i+=1

# 	print(index)
# 	print(filter_sort(x))

# 	add_index = list(np.arange(0, dim*percent, dtype=int))

# #	index = 	index[:, :int(dim*percent)].ravel()        
# 	for i in add_index: 
# 		predict_label[(index+i).ravel()] = 1

# 	return predict_label.ravel().tolist()

def load_dataset(file_name='../data/data.csv'):
	# load dataset
	data = pd.read_csv(file_name)

	X = data.iloc[:, :].values
	from sklearn.preprocessing import LabelEncoder
	label_enc = LabelEncoder()
	X[:, 0] = label_enc.fit_transform(X[:, 0])

	return X

# def load_new_data(file_name='../data/data.csv'):
# 	data = pd.read_csv(file_name).values[:, 1:]
# 	facies = create_truth_facies(data)
# 	from sklearn.preprocessing import LabelEncoder
# 	lb = LabelEncoder()
# 	data[:, 0] = lb.fit_transform(data[:, 0])
# 	data = data[:, :-2]()

def split_train_test(data, train_well=[0, 1, 2, 3, 7, 5, 6, 8, 9, 10, 11, 12, 13], test_well=4):

	train = data[data[:, 0] == train_well[0], :]
	if len(train_well) >= 2:
		for i in range(1, len(train_well)):
			train = np.concatenate((train, data[data[:, 0] == train_well[i], :]), axis = 0)
	#__debug__#
	# print('train: ', train)
	#_________#
	test = data[data[:, 0] == test_well, :]

	return train, test




def calculate_precision(predict_label, truth_label):
	"""
	predict_label: dictionary {"1": [binay label], "2": [], ...}
	truth_label: 1d numpy array contain all facies label
	"""

	for key, value in predict_label.items():
		new_y = np.array(truth_label, copy = True)
		new_y[new_y != key] = -1

		# print(value)
		print('accuracy: ', np.mean(value==new_y))
		print(classification_report(new_y, value))
		print(confusion_matrix(new_y, value))
		pass


# def create_truth_facies(data):
# 	facies = np.zeros((data.shape[0],), dtype=int)
# 	first_facies = data[:, -2].astype(int)
# 	second_facies = data[:, -1].astype(int)
	
# 	facies[np.where(second_facies == 0)[0]] = first_facies[np.where(second_facies == 0)[0]]
# 	facies[np.where(second_facies != 0)[0]] = second_facies[np.where(second_facies != 0)[0]]

# 	return facies

# def load_new_data(file_name='../data/data.csv'):
# 	data = pd.read_csv(file_name).values[:, 1:]
# 	facies = create_truth_facies(data)
# 	from sklearn.preprocessing import LabelEncoder
# 	lb = LabelEncoder()
# 	data[:, 0] = lb.fit_transform(data[:, 0])
# 	data = data[:, :-2]
# 	return data, facies



def load_new_data_csv(file_name):
	NULL_VALUE_DATA = -2
	NULL_VALUE_FACIES = -1
	df = pd.read_csv(file_name)
	print(df.head())
	# print(set(df.iloc[1:, [0]].values.ravel().tolist()))
	old_data = df.iloc[1:, [0, 2, 3]].values
	old_facies = df.iloc[1:, -1].values
	facies= np.zeros(old_facies.shape, dtype=np.int32)
	for i in range(old_facies.shape[0]):
		if old_facies[i] == '-9999' or np.isnan(old_facies[i]):
			facies[i]=NULL_VALUE_FACIES
		else:
			facies[i] = int(old_facies[i])

	data = np.zeros((old_data.shape[0], old_data.shape[1]-1), dtype=np.float32)
	labelEncoder = LabelEncoder()
	well_name_data = np.array(labelEncoder.fit_transform(old_data[:, 0]), dtype=np.int32)

	for i in range(old_data.shape[0]):
		for j in range(1, old_data.shape[1]):
			if isinstance(old_data[i,j], str):

				if old_data[i, j] == '-9999' or np.isnan(old_data[i, j]):
					data[i, j-1] = NULL_VALUE_DATA
				else:
					data[i, j-1] = float(old_data[i, j])
			else:
				if np.isnan(old_data[i, j]):
					data[i, j-1] = NULL_VALUE_DATA
				else: 
					data[i, j-1] = float(old_data[i, j])

	return np.concatenate((well_name_data.reshape(-1, 1), data), axis=1), facies



def test_crp(dim, tau, epsilon, lambd, percent, curve_number, facies_class_number):
	"""

	:param dim:
	:param tau:
	:param epsilon:
	:param lambd:
	:param percent:
	:param curve_number: index of column in data table
	:param facies_class_number: class to predict
	:return: predicted label
	"""


	# X = load_dataset()
	# load data from csv file to data, facice
	# data contain like:
	# [[ 0.         78.00830078  0.25607601]
 	# [ 0.         75.58155823  0.242126  ]
 	# [ 0.         73.23803711  0.238589  ]
 	# ...
 	# [13.         57.51266861  0.15773199]
 	# [13.         60.06167984  0.146786  ]
 	# [13.         62.60592651  0.126561  ]]
 	# 1st column is label encoder of well name
 	# 2st column is curve number 2 like in comment in main() function
 	# 3rd cloumn is curve number 3 like in comment in main() function

 	# facies is like: [0 0 0 ... 0 0 0]

	data, facies = load_new_data_csv('Data_PETREL_INPUT.csv')
	

	# convert label 0 to label 6, notice in main() function
	facies[facies == 0] = 6

	X = np.concatenate((data, facies.reshape(-1, 1)), axis = 1)
	# print("X: ",X)

	training_well_ts, testing_well_ts = split_train_test(X)
	#__debug__#
	# print('training_well_ts: ', training_well_ts)
	# print("testing_well_ts:", testing_well_ts)
	#_________#
	y_test = (testing_well_ts[:, -1] == facies_class_number).astype(int)



	# train() function -- save data of facies_class_number
	train(training_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number, id_string=str(facies_class_number))

	# predict() function -- get data of facies_class_number saved by train() then predict label
	target = predict_diagonal(testing_well_ts, id_string=str(facies_class_number))
	# target = demo_predict_straight(testing_well_ts, id_string=str(facies_class_number))
	target = np.array(target)
	#__debug__#
	# print('target.shape: ', target)
	# print('y_test.shape: ', y_test)
	#_________#


	# calculate_precision
	# print('num predict: ', np.sum(target))
	# print('accuracy: ', np.mean(target==y_test))
	# print(classification_report(y_test, target))
	# print(confusion_matrix(y_test, target))
	return target


def main():

	# data, facies = load_new_data_csv('Data_PETREL_INPUT.csv')


	dim = 5
	tau = 2
	# epsilon = [0, 1, 0.035, 0.035, 0.1, 0.02, 0.035, 0.05, 0.025]
	# epsilon = 0.5
	# epsilon = 0.5			#30
	epsilon = 0.7			#10
	lambd = 2
	percent = 0.6
	curve_number = 2
	facies_class_number = 6
	id_string = '192.168.1.1'



	"""
	Test crp

	Notice label 0 must convert to number > 0
	ex: in this case, label 0 convert to label 6 (now label 6 is label 0)
	in my experience:
	result in label 0(or in my case is label 6)


	dim = 5
	tau = 2
	# epsilon = [0, 1, 0.035, 0.035, 0.1, 0.02, 0.035, 0.05, 0.025]
	epsilon = 0.035
	lambd = 20
	percent = 0.5
	curve_number = 2
	facies_class_number = 6



	accuracy:  0.9221913457721318
             precision    recall  f1-score   support

          0       0.89      1.00      0.94      1596
          1       1.00      0.79      0.88       923

avg / total       0.93      0.92      0.92      2519

	1 in table is predicted true in binary vector
	"""

	test_crp(dim, tau, epsilon, lambd, percent, curve_number, facies_class_number)


	# notice2 : you can use this function to convert two timesiries with same mean and variant
	# before compute distance
	# 
	# using euclidien distance in line 339, function predict() ---- 
	# ---- r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1) -----
	# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
	# def timeseriesSaclerMeanVariance(ts, mu=0., std=1.):
	# 	scaler = TimeSeriesScalerMeanVariance(mu, std)  # Rescale time series
	# 	ts_scaler = scaler.fit_transform(ts.reshape(1, -1, 1))
	# 	return ts_scaler[0].ravel()

	#__________________________________________________


	# data, facies = load_new_data()
	# print(data)
	# print(data.shape)
	# facies[facies == 0] = -1
	# print(list(set(facies)))
	# training_well_ts, testing_well_ts = get_data()
	# print(training_well_ts)
	# print(testing_well_ts)
	# predict_vector = rqa(training_well_ts, testing_well_ts, dim=dim, tau=tau, epsilon=epsilon, lambd=lambd, percent=percent, curve_number=1, facies_class_number=5)
	# import json
	# data = json.load(open('data.json'))
	# training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number = get_data_from_json(data)
	# all_target = []
	# for curve_number in [1, 2]:
	# 	all_target.append(swap_all(dim, tau, epsilon[curve_number], lambd, percent, curve_number, facies_class_number))

	# train_all_facies_sqrt(data, facies, dim, tau, epsilon, lambd, percent, curve_number_index_to_sum_then_sqrt=[1, 2], id_string=id_string)
	# for w in list(set(data[:, 0])):
	# 		predict_label = predict_all_facies_sqrt(data[data[:, 0] == w, :], id_string)
	# 		# # print(type(predict_label))
	# 		# print(type(predict_label))
	# 		y = facies[data[:, 0] == w]
	# 		calculate_precision(predict_label, y)
	# 		# # print("save")
	# 		# # print("well" + str(w))
	# 		# with open("./result/"+str(w) + ".json", "w") as file:
	# 		# 	json.dump(predict_label, file)
	# 		# # print("well: ", str(w))
	# 		# # print(y)
	# 		# # print(len(predict_label))
	# 		# # for i, l in enumerate(predict_label):
	# 		# # 	# print(l)
	# 		# # 	# print(classification_report(y, l))
	# 		# # 	# print(confusion_matrix(y, l))
	# 		# # 	np.savez_compressed(".\\result\\well" + str(w)+"label" + str(i), predict_label=l)

if __name__ == '__main__':
	main()
