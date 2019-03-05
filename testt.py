
import csv


path = "data/GR-Emerald-3X_GR-Emerald-3X.csv"

indexColOfFeature = 1
indexColOfShape = 2

features = []
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



for feature in features:
	print(feature[1])

print(shapes)
		