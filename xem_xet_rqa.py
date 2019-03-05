import matplotlib.pyplot as plt 
import csv
import os
import numpy as np 

pathFolder = "out17022019_rqa/interpKind_linear-num_200-Epsilon_0.005-Lamb_3/"
pathCsv = pathFolder + "rqa.csv"
colName = ['shape', 'RR', 'DET', 'LAM', 'averageL', 'averageH', 'Lmax', 'Hmax', 'ENTR', 't1', 't2', 
			'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'average']


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def makePlot(dataSets, labels, figureName = "GirlLikeYou", pathSaveFigure = None):
	f_line = plt.figure(figureName, figsize = (12.8, 7.2), dpi = 100)

	plt.plot(dataSets[:, 0], ':', label = colName[int(labels[0])])

	for col in range(1, len(dataSets[0])):
		print(dataSets[:, col])
		plt.plot(dataSets[:, col], ':', label= colName[int(labels[col])])

	# plt.plot(trainSet, 'r', label='train')
	plt.legend(loc=0)
	plt.xlabel('index')
	plt.ylabel('feature')

	# titleOfGraph = titleOfGraph + " - lamb_" + str(lambd) + ' - minkowski_' + str(distNorm) + " - numPredict_" + str(len(valueOfSample))
	# plt.title(titleOfGraph)

	if (pathSaveFigure != None):
		plt.savefig(pathSaveFigure, dpi = 200)

def readCSV(pathCsv, indexCols):
	data = [indexCols]
	with open(pathCsv, 'r') as file:
		thisCSV = csv.reader(file, delimiter = ';')
		for i, row in enumerate(thisCSV):
			if (i>0):
				thisRow = []
				# s = ''
				# for indexCol in indexCols:
				# 	s = s + ';' + row[indexCol]
				# print(s)
				for col in (indexCols):
					if (col == 21):
						s = 0
						for j in range(11, 21, 1):
							s += float(row[j]) 
							print(row[j])
						thisRow.append(s/10)
					else:
						thisRow.append(float(row[col]))
				data.append(thisRow)
	return np.array(data)

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


createFolder(pathFolder)
for indexRQA in [1, 4, 6, 9, 10]:

	listIndex = [0,14]
	listIndex.append(indexRQA)
	# data[0] là indexCol của file csv
	allData = readCSV(pathCsv, listIndex)

	for i in range(len(listIndex)):
		if i == 0:
			allData[1:][:, i] = ConvertSetNumber(allData[1:][:, i], minOfSet=0, maxOfSet=5, newMinOfSet = 0, newMaxOfSet = 1)
		else:
			allData[1:][:, i] = ConvertSetNumber(allData[1:][:, i], newMinOfSet = 1, newMaxOfSet = 2)

	windowLen = 90
	listColName = ""
	for col in allData[0]:
		listColName += colName[int(col)] + '.' 

	for i in range(1, len(allData)-windowLen-1, 456):
		start = i 
		end = start + windowLen
		pathSave = "{}{}-start_{}.png".format(pathFolder, listColName, i)
		figureName = listColName + str(i)
		makePlot(allData[start:end], labels = allData[0], pathSaveFigure = pathSave, figureName = figureName)
		# plt.show()



# plt.show()