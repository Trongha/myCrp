import matplotlib.pyplot as plt
import csv
import numpy as np 

def readCSVFile(path, indexOfCol = 0):
	col = []
	with open(path, 'r') as File:
		thisCSVFile = csv.reader(File)
		for hang in thisCSVFile:
			hang[indexOfCol] = float(hang[indexOfCol])
			if (hang[indexOfCol] > 0):
				col.append(hang[indexOfCol])
	return col
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
	dataSet = readCSVFile("data/15_1-SD-2X-DEV_LQC.csv")
	start = 0;

	#### START-1X = 4840
	#### START-2X = 8800/24037
	#### START-3X = 8768/29721
	#### START-5X = 0/9960
	#### START-4X = 0/13060
	#
	# for i in range(0,len(dataSet)):
	# 	if (dataSet[i] > 0):
	# 		start = i;
	# 		break
	# 		
	print(len(dataSet))			
	# print("start: ", start)
	# for start in range(100, 1000, 8):
	# 	start = 132
	# 	finish = start+16
	# 	s = str(start) + " - " + str(finish)
	# 	print(s)
	# 	lineGraph(dataSet[start:finish])

	a = np.random.randint(2, size = (18, 5));
	print(a)

	f3 = crossRecurrencePlots('crpTest', a, dotSize = 10)

	plt.show()


