import matplotlib.pyplot as plt
import csv
import numpy as np 

def readFile(path):
	col6 = []
	with open(path, 'r') as File:
		readFile = csv.reader(File)
		for hang in readFile:
			hang[5] = float(hang[5])
			if (hang[5] < 0):
				col6.append(0)
			else:
				col6.append(hang[5])
	return col6
def myGraph(ySet):
	print(ySet)
	plt.plot(ySet, label = 'trainSet')
	plt.title('myGraph')
	plt.show()

if (__name__ == '__main__'):
	dataSet = readFile("data/15_1-SD-1X_LQC.csv")
	start = 0;
	for i in range(3,len(dataSet)):
		if (dataSet[i] > 0):
			start = i;
			break

	print("start: ", start)
	myGraph(dataSet[5000:7500])


