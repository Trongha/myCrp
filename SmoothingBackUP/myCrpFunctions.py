import matplotlib.pyplot as plt
import csv
import numpy as np 

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

def smoothingMovingAverage(dataSet, sizeWindow = 2):
		return [np.average(np.array(dataSet[i-sizeWindow:i+sizeWindow+1])) for i in range (sizeWindow, len(dataSet)-sizeWindow)]


def waveletsmoothing(set):
	import pywt
	cA,_ = pywt.dwt(set, 'db1')
	return cA

def lineGraph(ySet):
	# print(ySet)
	plt.plot(ySet, label = 'trainSet')
	plt.title('lineGraph')
	plt.show()

def ConvertSetNumber(Set, lenOfSet = 0, minOfSet = 0, maxOfSet = 0, newMinOfSet = 0, newMaxOfSet = 1):
	if (lenOfSet == 0):
		lenOfSet = len(Set)
	if (minOfSet == 0):
		minOfSet = min(Set)
	if (maxOfSet == 0):
		maxOfSet = max(Set)

	print("min: ", minOfSet)
	print("max: ", maxOfSet)

	ratio =(newMaxOfSet - newMinOfSet)/(maxOfSet - minOfSet)
	return [((x - minOfSet)*ratio + newMinOfSet) for x in Set]

#vẽ biểu đồ chấm từ mảng x và mảng y
def scatterGraph(windowTitle , dataX, dataY, dotSize = 0,myTitle = 'prettyGirl', labelX = 'xxxxx', labelY = 'yyyyy'):
	f = plt.figure(windowTitle)
	plt.scatter(dataX, dataY, s = dotSize)
	plt.title(myTitle)
	plt.xlabel(labelX)
	plt.ylabel(labelY)
	# plt.show()
	return f
	# plt.show()

# vẽ biểu đồ crp từ ma trận 01
def crossRecurrencePlots(windowTitle, dataMatrixBinary, dotSize = 0, myTitle = 'prettyGirl', labelX = 'xxxxx', labelY = 'yyyyy'):
	dataX = []
	dataY = []
	hightOfData = len(dataMatrixBinary);
	for y in range(hightOfData):
		for x in range(len(dataMatrixBinary[y])):
			if (dataMatrixBinary[y][x] == 1):
				dataX.append(x)
				## append hight-y nếu muốn vẽ đồ thị đúng chiều như lưu trong ma trận
				dataY.append( hightOfData - y -1)
				## vẽ trục y từ dưới lên
				#dataY.append(y);

	return scatterGraph(windowTitle , dataX, dataY, dotSize, myTitle , labelX , labelY )

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
	trainSetByShape, minSet, maxSet = readCSVFileByShape(path[1], '1', 2, 3)
	start = 0;

	print(minSet, maxSet)
	print(len(trainSetByShape))	

	print(trainSetByShape)		




