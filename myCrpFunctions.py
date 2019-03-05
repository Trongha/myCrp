# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import csv
import numpy as np 
import math
import os

def readCSVFile(path, indexOfCol = 0):
	col = []
	with open(path, 'r') as File:
		thisCSVFile = csv.reader(File)
		for hang in thisCSVFile:
			hang[indexOfCol] = float(hang[indexOfCol])
			# print([indexOfCol])
			if (hang[indexOfCol] > 0):
				col.append(hang[indexOfCol])
	return col

def readCSVFileForTest(path, indexOfCol = 0):
	col = []
	shape = []
	with open(path, 'r') as File:
		thisCSVFile = csv.reader(File)
		for hang in thisCSVFile:
			hang[indexOfCol] = float(hang[indexOfCol])
			# print([indexOfCol])
			if (hang[indexOfCol] > 0):
				col.append(hang[indexOfCol])
				shape.append(float(hang[indexOfCol+1]))
	return col, shape

def readCSVFileByShape(path, shape, indexColOfFeature = 0, indexColOfShape = 1):

	features = []
	maxOfSet = minOfSet = 0

	isShape = "false"
	feature = []

	with open(path, 'r') as File:
		thisCSVFile = csv.reader(File)
		for hang in thisCSVFile:
			if (float(hang[indexColOfShape]) == shape):
				if isShape == "false" :
					feature = []
					isShape = "true"

				hang[indexColOfFeature] = float(hang[indexColOfFeature])
				# print([indexColOfFeature])
				if (hang[indexColOfFeature] > 0):
					feature.append(hang[indexColOfFeature])
			else:
				if isShape == "true" :
					features.append(feature)
					minNow = min(feature)
					maxNow = max(feature)

					if (len(features)==1 or minOfSet > minNow):
						minOfSet = minNow
					if (len(features)==1 or maxOfSet < maxNow):
						maxOfSet = maxNow

				isShape = "false"

	return features, minOfSet, maxOfSet

def readCSVallShape(path, indexColOfFeature = 0, indexColOfShape = 1):
	features = []	#array of nparrays
	shapes = []

	shape = None
	feature = None
	import numpy as np 

	with open(path, 'r') as file:
		allData = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
		# f, delimiter=':', quoting=csv.QUOTE_NONE
		for i, row in enumerate(allData):
			print(i)
			if row[indexColOfShape] != shape:
				if (feature != None):
					
					feature = np.array([feature, [shape]*len(feature)])
					features.append(feature)
					print(feature)

				shape = row[indexColOfShape]

				if shape not in shapes:
					shapes.append(shape)
				feature = []

			feature.append(row[indexColOfFeature])

	return features, shapes

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def writeContentToFile(pathFile, content):
	file = open(pathFile, 'w')
	for line in content:
		file.write(line)
	file.close()

def smoothingMovingAverage(dataSet, sizeWindow = 2):
	if (len(dataSet) < 2*sizeWindow +1):
		return dataSet
	return [dataSet[i] for i in range(0,sizeWindow)] + [np.average(np.array(dataSet[i-sizeWindow:i+sizeWindow+1])) for i in range (sizeWindow, len(dataSet)-sizeWindow)] + [dataSet[i] for i in range(len(dataSet)-sizeWindow, len(dataSet))]

def smoothListGaussian(list,degree=5):  

    window=degree*2-1  

    weight=np.array([1.0]*window)  

    weightGauss=[]  

    for i in range(window):  

        i=i-degree+1  

        frac=i/float(window)  

        gauss=1/(np.exp((4*(frac))**2))  

        weightGauss.append(gauss)  

    weight=np.array(weightGauss)*weight  

    smoothed=[0.0]*(len(list)-window)  

    for i in range(len(smoothed)):  

        smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)  

    return list[0: degree] + smoothed + list[len(list)-degree+1: len(list)]

def smoothListTriangle(list,strippedXs=False,degree=5):  

    weight=[]  

    window=degree*2-1  

    smoothed=[0.0]*(len(list)-window)  

    for x in range(1,2*degree):weight.append(degree-abs(degree-x))  

    w=np.array(weight)  

    for i in range(len(smoothed)):  

        smoothed[i]=sum(np.array(list[i:i+window])*w)/float(sum(w))  

    return list[0: degree] + smoothed + list[len(list)-degree+1: len(list)]

def Savitzky_GolaySmoothing(dataSet, sizeWindow):
	return 0

def waveletsmoothing(set, type):
	import pywt
	cA,_ = pywt.dwt(set, type)
	return cA

def lineGraph(ySet):
	# print(ySet)
	plt.plot(ySet, label = 'trainSet')
	plt.title('lineGraph')
	plt.show()

def ConvertSetNumber(Set, minOfSet = 0, maxOfSet = 0, newMinOfSet = 0, newMaxOfSet = 1):
	if (minOfSet == 0):
		minOfSet = min(Set)
	if (maxOfSet == 0):
		maxOfSet = max(Set)

	print("min: ", minOfSet)
	print("max: ", maxOfSet)

	if (maxOfSet == minOfSet):
		ratio = 0
	else:
		ratio =(newMaxOfSet - newMinOfSet)/(maxOfSet - minOfSet)
	return [((x - minOfSet)*ratio + newMinOfSet) for x in Set]

#vẽ biểu đồ chấm từ mảng x và mảng y
def scatterGraph(windowTitle , dataX, dataY, dotSize = 0,myTitle = 'prettyGirl', labelX = 'xxxxx', labelY = 'yyyyy', pathSaveFigure = None):
	f = plt.figure(windowTitle, figsize = (12.8, 7.2), dpi = 100)
	plt.scatter(dataX, dataY, s = dotSize)
	plt.title(myTitle)
	plt.xlabel(labelX)
	plt.ylabel(labelY)
	# plt.show()
	
	if (pathSaveFigure != None):
		plt.savefig(pathSaveFigure, dpi = 200)

	return f
	# plt.show()

# vẽ biểu đồ crp từ ma trận 01
def crossRecurrencePlots(windowTitle, dataMatrixBinary, keyDot = 1, dotSize = 1, myTitle = 'prettyGirl', labelX = 'xxxxx', labelY = 'yyyyy', pathSaveFigure =None):
	dataX = []
	dataY = []
	hightOfData = len(dataMatrixBinary);

	print("crossRecurrencePlots()_len: ", hightOfData)
	for y in range(hightOfData):
		for x in range(len(dataMatrixBinary[y])):
			if (dataMatrixBinary[y][x] == keyDot):
				dataX.append(x)
				## append hight-y nếu muốn vẽ đồ thị đúng chiều như lưu trong ma trận
				dataY.append( hightOfData - y -1)
				## vẽ trục y từ dưới lên
				# dataY.append(y);

	return scatterGraph(windowTitle , dataX, dataY, dotSize, myTitle , labelX , labelY, pathSaveFigure = pathSaveFigure )

def myInterpolation(y, x = None, numNew = None, numInsert = None, myKind = None):

	import numpy as np
	from scipy import interpolate

	if(x == None):
		x = np.arange(0, len(y), 1)
	# print("x: ", x)

	if (numInsert != None):
		numNew = (len(y)-1)*(numInsert+1) + 1

	newStep = (x[len(x)-1] - x[0])/(numNew-1)
	print(x[len(x)-1])
	xnew = np.arange(x[0], x[len(x)-1] + newStep/2, newStep)

	if (myKind == None):
		f = interpolate.interp1d(x, y)
	else:
		f = interpolate.interp1d(x, y, kind = myKind)

	ynew = f(xnew)
	
	# print("xnew: ", len(xnew), xnew)
	
	print("lenTimeSeries = ", len(ynew))

	return xnew.tolist(), ynew.tolist()

def makeRPmatrix(TimeSeries, dim=5, tau=2, epsilon=0.7, lambd=3, typeReturn = 'array', distNorm = 1 ):
	return 0

def checkRecall(testShape, outputFolder = None, shape = 0 ):
	TP=TN=FP=FN=1
	floatShape = float(shape)

	for i in range(len(testShape)):
		if(floatShape == shapePredict[i]):
			if (floatShape == testShape[i]):
				TP += 1
			else:
				FP += 1
		else:
			if (floatShape != testShape[i]) :
				TN += 1 
			else:
				FN += 1

	print("TP: %5d, FP: %5d, FN: %5d, TN: %5d" % (TP, FP, FN, TN))
	pi = TP/(TP+FP)
	p = TP/(TP+FN)
	f1 = 2*pi*p/(pi+p)
	
	print("pi = %2.2f" % pi)
	print("p = %2.2f" % p)
	print("f1 = %2.2f" % f1)

	if (outputFolder != None):
		print(outputFolder)
		import time
		import datetime
		checkID = datetime.datetime.fromtimestamp(time.time()).strftime('%d/%m/%Y::%Hh%Mp%Ss')

		content = ["{}\n{}".format(checkID, outputFolder)]
		content.append('\n TP: {:5}\n FP: {:5}\n FN: {:5}\n TN: {:5}\n '.format(TP, FP, FN, TN))
		content.append('pi = {:2.2}\n '.format( pi))
		content.append('p = {:2.2}\n '.format( p))
		content.append('f1 = {:2.2}\n '.format( f1))
		pathOut = outputFolder + "readme.txt"
		myCrpFunctions.writeContentToFile(pathOut, content)



def rqaFromBinaryMatrix(dataMatrixBinary, keyDot = 1, lambd = 2, typeReturn = 'array'):

	len_dataMatrixBinary = int(np.size(dataMatrixBinary[0]))
	N = high_dataMatrixBinary = int(np.size(dataMatrixBinary)/len_dataMatrixBinary)

	content = []
	for y in range(high_dataMatrixBinary):
		s = ";".join(str(i+2) for i in dataMatrixBinary[y] )
		s += "\n"
		content.append(s)
	writeContentToFile('rqaOut.csv', content)

	x = crossRecurrencePlots("test", dataMatrixBinary, keyDot = keyDot, dotSize = 10 )

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
	Ph, Hmax, averageTime1, averageTime2 = getPverticalLengthDot(dataMatrixBinary, high_dataMatrixBinary, len_dataMatrixBinary, keyDot = keyDot)

	#Đếm số đường chéo theo độ dài
	Pl, Lmax = getPdiagonalLengthDot(dataMatrixBinary, high_dataMatrixBinary, len_dataMatrixBinary, keyDot = keyDot)

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
			print(pl)
			print(math.log(pl))
			ENTR -= pl*math.log(pl)

	# for i in range(len(Pl)):
	# 	print(i, ": ", Pl[i], Ph[i])

	if Lmax > 0:
		DIV = 1/Lmax

	print(RR)
	print(DET)
	print(LAM)
	print("averageL, averageH", averageL, averageH)
	print(Lmax, Hmax)
	print(DIV)
	print(ENTR)
	print("averageTime1, averageTime2", averageTime1, averageTime2)

	# plt.show()
	
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

def getPverticalLengthDot(dataMatrixBinary, high_dataMatrixBinary = None, len_dataMatrixBinary = None, keyDot = 1):

	if(high_dataMatrixBinary == None or len_dataMatrixBinary == None):
		len_dataMatrixBinary = int(np.size(dataMatrixBinary[0]))
		high_dataMatrixBinary = int(np.size(dataMatrixBinary)/len_dataMatrixBinary)
	N = high_dataMatrixBinary

	Ph = [0]*(N+1)
	sumT1 = 0
	sumT2 = 0
	numT = 0
	Hmax = 0

	for x in range(len_dataMatrixBinary):
		start=0
		length = 0
		numDot = 0
		y = 0
		while (y < high_dataMatrixBinary):
			if (dataMatrixBinary[y][x] == keyDot):
				if (y>0 and numDot>0):
					t2now = y-start
					sumT1 += (t2now - length)
					sumT2 += t2now
					numT += 1

					# print("t1 = %d___t2 = %d"%(t2now - length, t2now))

				start = y
				
				while (y+1<high_dataMatrixBinary and dataMatrixBinary[y+1][x] == keyDot):
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


def getPdiagonalLengthDot(dataMatrixBinary, high_dataMatrixBinary = None, len_dataMatrixBinary = None, keyDot = 1):
	if(high_dataMatrixBinary == None or len_dataMatrixBinary == None):
		len_dataMatrixBinary = int(np.size(dataMatrixBinary[0]))
		high_dataMatrixBinary = int(np.size(dataMatrixBinary)/len_dataMatrixBinary)
	N = high_dataMatrixBinary

	Pl = [0]*(N+1)
	Lmax = 0

	for index_diagonal in range(-(high_dataMatrixBinary + 1), len_dataMatrixBinary + 2, 1):
		offset = index_diagonal
		#---offset = x - y
		y = -offset if (index_diagonal < 0) else 0

		while (y < high_dataMatrixBinary and y+offset < len_dataMatrixBinary):
			if (dataMatrixBinary[y][y+offset] == keyDot):
				start = y
				while (y+1<high_dataMatrixBinary and y+1+offset < len_dataMatrixBinary and dataMatrixBinary[y+1][y+1+offset] == keyDot):
					y+=1
				numDot = y - start + 1
				length = numDot - 1
				Pl[length] += 1
				if (length > Lmax):
					Lmax = length
			y+=1

	return Pl, Lmax


if (__name__ == '__main__'):
	fileName = [
			"RBA-3P_RBA-3P.csv",
			"RBA-6P_RBA-6P.csv",
			"RBA-12PST4_RBA-12PST4.csv",
			"RUBY-1X_RUBY-1X_1.csv",
			"RUBY-4X_RUBY-4X_1.csv",
			"TN-3X_TN-3X.csv"]
	path = []

	for name in fileName:
		path.append("data/" + name)

	print(path[1])
	trainSetByShape, minSet, maxSet = readCSVFileByShape(path[1], 2, 2, 3)
	start = 0;

	A = [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
		 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
		 [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
		 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
		 [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
		 [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],  
		 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], 
		 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], 
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],  
		 [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
		 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
		 [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]
		]
	
	TimeSeries = [1, 2, 4, 5, 6, 7, 8, 9, 10, 20, 11, 22, 24, 35, 15, 15, 19, 30]

	plt.plot(TimeSeries, label="origin")

	kinds = ['linear', 'nearest', 'slinear', 'quadratic', 'cubic']

	for kind in kinds:
		f_line = plt.figure(kind, figsize = (12.8, 7.2), dpi = 100)
		x, y = myInterpolation(TimeSeries, myKind = kind, numNew = 50)
		plt.plot(x, y, '-', label=kind)






