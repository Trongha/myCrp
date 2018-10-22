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
import myCrpFunctions
import random

def filter_sort(arr):
		x = []
		if (len(arr) > 0):
			x.append(arr[0])
			for t in arr:
				if x.count(t) == 0:
					x.append(t)
		return x

def state_phase(v, start, dim, tau):
	#v: vecto
	#dim là số chiều (số phần tử)	
	#tau là bước nhảy		
	return [v[start + i*tau] for i in range(0, dim, 1)]

def SyncPredictToTestArr(indexSet, dataSet):
	# Lấy các giá trị của dataSet tại các index liên tiếp trong indexSet
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


def predict_diagonal(trainSet, testSet, dim=5, tau=2, epsilon=0.7, lambd=3, percent=0.6, titleOfGraph = 'Pretty Girl'):

	# # vectors_train = data['train_data']
	# dim = 5 #data['dim']
	# tau = 2 #data['tau']
	# percent = 0.6 #data['percent']
	# # curve_number = data['curve_number']
	# epsilon = 0.7 #data['epsilon']
	# lambd = 2 #data['lambd']

	vectors_train_1 = []
	for i in range(len(trainSet)-(dim-1)*tau):
		vectors_train_1.append(state_phase(trainSet, i, dim, tau)) 

	#tách statephases
	vectors_test_1 = []
	for i in range(len(testSet)-(dim-1)*tau):
		vectors_test_1.append(state_phase(testSet, i, dim, tau))

	#ép kiểu về array
	vectors_train_1 = np.array(vectors_train_1)
	vectors_test_1 = np.array(vectors_test_1)

	print("train.shape: ", vectors_train_1.shape)

	# r_dist là ma trận khoảng cách
	# cdist là hàm trong numpy
	# minkowski là cách tính
	# p là norm
	# y là train: đánh số từ trên xuống dưới
	# x là test
	r_dist = cdist(vectors_train_1, vectors_test_1, 'minkowski', p=1)

	# epsilon = r_dist.min()
	# print("r_dist:",r_dist)

	print('vectors_train.shape: ', vectors_train_1.shape) #in ra shape
	print('vectors_test.shape: ', vectors_test_1.shape)
	print('r_dist.shape: ', r_dist.shape)
	print('r_dist min', r_dist.min())
	print('epsilon: ', epsilon)
	print('r_dist max', r_dist.max())
	print('lambd: ', lambd)

	# predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)

	# indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
	predict_label = np.zeros(len(testSet),dtype=int)
	
	#r1 là ma trận -2|-1 thay vì ma trận 01 như crp
	#-2 là false
	#-1 là true
	#r_dist < epsilon trả về ma trận 01, 1 tại điểm r_dist < epsilon
	#ma trận này trừ đi 2 thì thành ma trận -2|-1
	r1 = np.array((r_dist < epsilon)-2)
	# r1 = np.array(r1 - 2)
	
	print(r1)
	# f_crp = myCrpFunctions.crossRecurrencePlots("CRP", r1, dotSize = 0.2)

	len_r1 = int(np.size(r1[0]))
	high_r1 = int(np.size(r1)/len_r1)

	#x is an array whose each element is a diagonal of the crp matrix
	diagonalsMatrix = []

	for offset in range(-high_r1+1, len_r1):
		diagonalsMatrix.append(np.array(np.diagonal(r1, offset), dtype=int))
	# print(x)
	
	# f_crp = myCrpFunctions.crossRecurrencePlots("CRP2", diagonalsMatrix, dotSize = 0.2)
	
	#Mảng này gồm những hàng chứa index của statePhrase có dự đoán, nghĩa là giống với train
	#những đường ko có khoảng giống với train sẽ bị bỏ qua
	indexHavePredictMatrix = []					
	for i_row in range(len(diagonalsMatrix)):
		havePredict = 0
		lenn = np.size(diagonalsMatrix[i_row])
		i = 0
		while i<lenn:
			if (diagonalsMatrix[i_row][i] == -1):
				start = i
				while (i<lenn-1 and diagonalsMatrix[i_row][i+1] == -1 ):
					i+=1
				if (i-start+1 > lambd):
					havePredict = 1
					for j in range(start, i+1, 1):
						
						if (i_row < high_r1):
							diagonalsMatrix[i_row][j] = j
						else:
							diagonalsMatrix[i_row][j] = (i_row - high_r1 + 1 + j)			
			i+=1
		# Nếu hàng có dự đoán thì sẽ giữ lại
		if (havePredict == 1):
			indexHavePredictMatrix.append(diagonalsMatrix[i_row])

	#Mảng quy đổi về index của dữ liệu test từ index của statePhrase
	indexSampleOrigin = indexHavePredictMatrix

	for i_row in range(len(indexSampleOrigin)):
		for i_in_1_State in range(len(indexSampleOrigin[i_row])):
			if (indexSampleOrigin[i_row][i_in_1_State] >= 0):
				if ((i_in_1_State == len(indexSampleOrigin[i_row]) - 1) or (indexSampleOrigin[i_row][i_in_1_State+1] < 0)):
					for j in range(((int(dim*percent))-1)*tau):
						indexSampleOrigin[i_row] = np.insert(indexSampleOrigin[i_row], i_in_1_State+j+1, indexSampleOrigin[i_row][i_in_1_State] + j + 1)

	# Lấy giá trị từ index của sample trong testSet
	valueOfSample = []

	for i in range(len(indexSampleOrigin)):
		arr = []
		for j in range(len(indexSampleOrigin[i])):
			if (indexSampleOrigin[i][j] < 0): 
				arr.append(None)
			else:
				arr.append(testSet[indexSampleOrigin[i][j]])
		if (len(arr) == len(trainSet)):
			valueOfSample.append(arr)

	# print("----------------------------------------------------------------------------------------------------------------\n",
	# 	valueOfSample,
	# 	"\n------------------------------------------------------------------------------------------------------------------")


	# #khởi tạo label
	# for i in index_timeseries_test:
	# 	predict_label[i] = 1
	# print('predict_label:', predict_label)
	# print('num predict: ', np.sum(predict_label))

	# testSetPr = SyncPredictToTestArr(index_timeseries_test, testSet)

	f_line= plt.figure("line")
	for i in range(0, len(valueOfSample)):
		plt.plot(valueOfSample[i], ':', label=i)
	plt.plot(trainSet, 'r', label='train')
	plt.legend(loc=0)
	plt.xlabel('index')
	plt.ylabel('feature')

	titleOfGraph = titleOfGraph + "-epsi_" + str(epsilon) + "-lamb_" + str(lambd)
	plt.title(titleOfGraph)
	plt.ylim(ymin = min(trainSet) - 0.05)
	# plt.show()

	return predict_label.ravel().tolist()

###############################-----________MAIN________-----###############################

if (__name__ == "__main__"):
	dataTrain = myCrpFunctions.readCSVFile("data/15_1-SD-1X_LQC.csv")
	dataTest = myCrpFunctions.readCSVFile("data/15_1-SD-2X-DEV_LQC.csv")

	if (min(dataTrain) > min(dataTest)) :
		minOfNorm = min(dataTest) 
	else: 
		minOfNorm = min(dataTrain) 

	if max(dataTrain) < max(dataTest) : 
		maxOfNorm = max(dataTest) 
	else: 
		maxOfNorm = max(dataTrain) 
	
	trainSet = myCrpFunctions.ConvertSetNumber(dataTrain, minOfSet = minOfNorm, maxOfSet = maxOfNorm)
	testSet = myCrpFunctions.ConvertSetNumber(dataTest, minOfSet = minOfNorm, maxOfSet = maxOfNorm)

	# start = 1080
	# finish = start+16
	# s = str(start) + " - " + str(finish)
	# print(s)
	# print(trainSet[start:finish])
	# predict_diagonal(trainSet[start:finish], testSet[800:1000] ,
	# 					dim=5, tau=2, epsilon=0.2, lambd=5, percent=1, titleOfGraph = s)

	# plt.show()

	# for start in range(2345, 20099, 123):
		
	# 	finish = start+30
	# 	title = str(start) + " - " + str(finish)
	# 	print(title)
	# 	print(trainSet[start:finish])
	# 	predict_diagonal(trainSet[start:finish], testSet ,
	# 						dim=5, tau=2, epsilon=0.15, lambd=5, percent=1, titleOfGraph = title)

	# 	plt.show()
		

