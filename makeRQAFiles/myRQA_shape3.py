# -*- coding: utf-8 -*-

import myCrpFunctions

import matplotlib.pyplot as plt
import numpy as np 
import json

def convert2StatePhase(v, dim, tau, returnType='array'):
	#v: vecto
	#dim là số chiều (số phần tử)	
	#tau là bước nhảy
	#returnType: kiểu trả về
	#	array: trả về python array
	#	np.array: trả về mảng numpy
	
	if (returnType == 'np.array'):
		import numpy as np
		return np.array([v[start : start+(dim-1)*tau+1 : tau] for start in range(len(v)-(dim-1)*tau)])
	if (returnType == 'array'):
		return [v[start : start+(dim-1)*tau+1 : tau] for start in range(len(v)-(dim-1)*tau)]


def makeRPmatrix(TimeSeries, dim=5, tau=2, epsilon=0.09, distNorm = 1):
	
	#tách statephases
	StatePhase = convert2StatePhase(TimeSeries, dim, tau, 'np.array')

	from scipy.spatial.distance import cdist
	# r_dist là ma trận khoảng cách
	# cdist là hàm trong scipy.spatial.distance.cdist
	# minkowski là cách tính
	# p là norm
	# y là train: đánh số từ trên xuống dưới
	# x là test
	r_dist = cdist(StatePhase, StatePhase, 'minkowski', p=distNorm)

	import numpy as np
	r_Binary = np.array((r_dist < epsilon) + 0)

	return r_Binary

def getRQA(TimeSeries, numPointInterp = 200, dim=5, tau=2, epsilon=0.09, lambd = 2, distNorm = 1, interpolationKind = "" ,typeReturn = 'array', showCRP = 0):
	
	if (len(TimeSeries) > dim*tau+1):

		norm01TimeSeries = myCrpFunctions.ConvertSetNumber(TimeSeries)

		_, interpTimeSeries = myCrpFunctions.myInterpolation(norm01TimeSeries, numNew = numPointInterp, myKind = interpolationKind)

		r_Binary = makeRPmatrix(interpTimeSeries, dim = dim, tau = tau, epsilon = epsilon, distNorm = distNorm)

		return rqaCalculate(r_Binary, keyDot = 1, lambd = lambd, typeReturn = typeReturn, showCRP = showCRP)

	return None


def rqaCalculate(rpBinaryMatrix, keyDot = 1, lambd = 2, typeReturn = 'array', showCRP = 0):
	import math

	len_rpBinaryMatrix = int(np.size(rpBinaryMatrix[0]))
	N = high_rpBinaryMatrix = int(np.size(rpBinaryMatrix)/len_rpBinaryMatrix)

	content = []
	for y in range(high_rpBinaryMatrix):
		s = ";".join(str(i+2) for i in rpBinaryMatrix[y] )
		s += "\n"
		content.append(s)
	myCrpFunctions.writeContentToFile('rqaOut.csv', content)

	RR =0
	DET = 0
	LAM = 0
	RATIO = 0
	averageL = 0
	averageH = 0
	DIV = 0
	ENTR = 0
	TT = 0

	# Duyệt các đường cao
	Ph, Hmax, averageTime1, averageTime2 = getPverticalLengthDot(rpBinaryMatrix, high_rpBinaryMatrix, len_rpBinaryMatrix, keyDot = keyDot)

	#Đếm số đường chéo theo độ dài
	Pl, Lmax = getPdiagonalLengthDot(rpBinaryMatrix, high_rpBinaryMatrix, len_rpBinaryMatrix, keyDot = keyDot)

	num_L = 0
	num_H = 0
	sum_Well_L = 0
	sum_Well_H = 0
	num_Well_L = 0
	num_Well_H = 0

	sumDot = Ph[0]   # Đếm số điểm chấm theo đường thẳng, ph[0] là những đoạn chỉ có 1 chấm

	lmin = lambd - 1

	for i in range(1, N+1, 1):
		sumDot += (i+1)*Ph[i]

		num_L += Pl[i]
		num_H += Ph[i]

		if (i>=lmin):
			sum_Well_L += Pl[i]*i
			sum_Well_H += Ph[i]*i

			num_Well_L += Pl[i]
			num_Well_H += Ph[i]

	RR = sumDot/(N**2)

	if num_L*num_Well_L > 0:
		DET = num_Well_L/num_L
		averageL = sum_Well_L/num_Well_L 

	if (DET > 0):
		RATIO = DET/RR

	if num_H*num_Well_H > 0:
		LAM = num_Well_H/num_H
		averageH = sum_Well_H/num_Well_H

	for i in range(lmin, N, 1):
		if (Pl[i] > 0):
			pl = Pl[i]/num_Well_L
			ENTR -= pl*math.log(pl)
	if Lmax > 0:
		DIV = 1/Lmax

	if (showCRP == 1):
		import matplotlib.pyplot as plt
		x = myCrpFunctions.crossRecurrencePlots("test", rpBinaryMatrix, keyDot = keyDot, dotSize = 10 )
		plt.show()
		print(RR)
		print(DET)
		print(LAM)
		print("averageL, averageH", averageL, averageH)
		print(Lmax, Hmax)
		print(DIV)
		print(ENTR)
		print("averageTime1, averageTime2", averageTime1, averageTime2)

	if (typeReturn == 'array'):
		return [RR, DET, LAM, RATIO, averageL, averageH, Lmax, Hmax, DIV, ENTR, averageTime1, averageTime2]
	if (typeReturn == 'dict'):
		return {
			"RR" : RR,
			"DET" : DET,
			"LAM" : LAM,
			"RATIO" : RATIO,
			"averageL" : averageL,
			"averageH" : averageH,
			"Lmax" : Lmax,
			"Hmax" : Hmax,
			"DIV" : DIV,
			"ENTR" : ENTR,
			"averageTime1" : averageTime1,
			"averageTime2" : averageTime2
		}

def getPverticalLengthDot(rpBinaryMatrix, high_rpBinaryMatrix = None, len_rpBinaryMatrix = None, keyDot = 1):

	if(high_rpBinaryMatrix == None or len_rpBinaryMatrix == None):
		len_rpBinaryMatrix = int(np.size(rpBinaryMatrix[0]))
		high_rpBinaryMatrix = int(np.size(rpBinaryMatrix)/len_rpBinaryMatrix)
	N = high_rpBinaryMatrix

	Ph = [0]*(N+1)
	sumT1 = 0
	sumT2 = 0
	numT = 0
	Hmax = 0
	averageTime1 = 0
	averageTime2 = 0

	for x in range(len_rpBinaryMatrix):
		start=0
		length = 0
		numDot = 0
		y = 0
		while (y < high_rpBinaryMatrix):
			if (rpBinaryMatrix[y][x] == keyDot):
				if (y>0 and numDot>0):
					t2now = y-start
					sumT1 += (t2now - length)
					sumT2 += t2now
					numT += 1
				start = y
				while (y+1<high_rpBinaryMatrix and rpBinaryMatrix[y+1][x] == keyDot):
					y += 1

				numDot = y - start + 1
				length = numDot-1

				Ph[length] += 1

				if (length > Hmax):
					Hmax = length
			y+=1

	if (numT > 0):
		averageTime1 = sumT1/numT
		averageTime2 = sumT2/numT
		
	return Ph, Hmax, averageTime1, averageTime2


def getPdiagonalLengthDot(rpBinaryMatrix, high_rpBinaryMatrix = None, len_rpBinaryMatrix = None, keyDot = 1):
	if(high_rpBinaryMatrix == None or len_rpBinaryMatrix == None):
		len_rpBinaryMatrix = int(np.size(rpBinaryMatrix[0]))
		high_rpBinaryMatrix = int(np.size(rpBinaryMatrix)/len_rpBinaryMatrix)
	N = high_rpBinaryMatrix

	Pl = [0]*(N+1)
	Lmax = 0

	for index_diagonal in range(-(high_rpBinaryMatrix + 1), len_rpBinaryMatrix + 2, 1):
		offset = index_diagonal
		#---offset = x - y
		y = -offset if (index_diagonal < 0) else 0

		while (y < high_rpBinaryMatrix and y+offset < len_rpBinaryMatrix):
			if (rpBinaryMatrix[y][y+offset] == keyDot):
				start = y
				while (y+1<high_rpBinaryMatrix and y+1+offset < len_rpBinaryMatrix and rpBinaryMatrix[y+1][y+1+offset] == keyDot):
					y+=1
				numDot = y - start + 1
				if (numDot < N):
					length = numDot - 1
					Pl[length] += 1
					if (length > Lmax):
						Lmax = length
				else:
					print(numDot, "----------------------------------------------")
			y+=1

	return Pl, Lmax

def makeRQAcsvFiles(dataSets, pathFile, lenTimeSeries = 25, epsilon = 0.08, lambd = 2, interpolationKind = None):

	content = []
	title = "TimeSeries; || ;RR; DET; LAM; RATIO; averageL; averageH; Lmax; Hmax; DIV; ENTR; averageTime1; averageTime2\n"
	content.append(title)

	for dataSet in dataSets:
		for i in range(int(len(dataSet)/lenTimeSeries)):		
			start = i*lenTimeSeries
			end = (i+1)*lenTimeSeries
			rqa = getRQA(dataSet[start:end], epsilon = epsilon, lambd = lambd, interpolationKind = interpolationKind, typeReturn = 'array')
			
			print(rqa)

			line = str(dataSet[start:end]) + "; || "
			if (rqa!=None):
				for item in rqa:
					line += ";" + str(item)
			line += '\n'
			content.append(line)
			

		line = "---------; || ;; ; ; ; ; ; ; ; ; ; ; \n"
		content.append(line)


	myCrpFunctions.writeContentToFile(pathFile, content)

	return 0

if (__name__ == "__main__"):
	myLambd=2
	myDim=3
	distNorm = 2
	tau = 2
	formatSave = ".png"
	myEpsilon = 0.009
	pathFolder = "noSmoothingZoom/"
	# checkRecall(pathFolder)

	fileNames = ["RBA-3P_RBA-3P.csv",
			"RBA-6P_RBA-6P.csv",
			"RBA-12PST4_RBA-12PST4.csv",
			"RUBY-1X_RUBY-1X_1.csv",
			"RUBY-4X_RUBY-4X_1.csv",
			"TN-3X_TN-3X.csv"]

	path = [("data/" + name) for name in fileNames]

	shape = 3
	indexColOfShape = 3
	indexColFeature = 2

	pathFolder = "out23012018_csv/"

	interpolationKinds = ['linear', 'nearest', 'slinear', 'quadratic', 'cubic']

	# for i in range(6):
	# 	allTrainSetOrigin, minOfTrain, maxOfTrain = myCrpFunctions.readCSVFileByShape(path[i], shape, indexColFeature, indexColOfShape)
	
	for interpKind in interpolationKinds:
		for myEpsilon in [0.0085, 0.005, 0.08, 0.05]:
			for myLambd in [2, 3]:
				pathFolder1 = '{}interpKind_{}-Epsilon_{}-Lamb_{}/'.format(pathFolder, interpKind, str(myEpsilon), str(myLambd))
				myCrpFunctions.createFolder(pathFolder1)
				print("--------------------------------------{}--------------------------------------".format(pathFolder1))

				for i, inputFileName in enumerate(fileNames):
					csvName = '{}rqa_shape_{}-file_{}'.format(pathFolder1, shape, inputFileName)
					print("--------------------------------------{}--------------------------------------".format(csvName))

					allTrainSetOrigin, minOfTrain, maxOfTrain = myCrpFunctions.readCSVFileByShape(path[i], shape, indexColFeature, indexColOfShape)

					makeRQAcsvFiles(allTrainSetOrigin, csvName, epsilon = myEpsilon, lambd = myLambd, interpolationKind = interpKind)



	'''
	rqas = [getRQA(myset, typeReturn = 'array', showCRP=0) for myset in allTrainSetOrigin]

	content = []
	title = "TimeSeries; || ;RR; DET; LAM; RATIO; averageL; averageH; Lmax; Hmax; DIV; ENTR; averageTime1; averageTime2\n"
	content.append(title)

	for i, rqa in enumerate(rqas):
		line = str(allTrainSetOrigin[i]) + "; || "
		if (rqa!=None):
			# rqas[i]['TimeSeries'] = allTrainSetOrigin[i]
			for item in rqa:
				line += ";" + str(item)
		line += '\n'
		content.append(line)

	myCrpFunctions.writeContentToFile("testOut.csv", content)
	'''





	# for rqa in rqas:
	# 	print(json.dumps(rqa, indent = 4))


	# print(json.dumps(rqa, indent = 4))


	# plt.show()
