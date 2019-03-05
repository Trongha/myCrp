import matplotlib.pyplot as plt 
import myCrpFunctions
import myRP
import numpy as np 

def makePlot(dataSet, myLabel, figureName = "GirlLikeYou", pathSaveFigure = None):
	f_line = plt.figure(figureName, figsize = (12.8, 7.2), dpi = 100)

	plt.plot(dataSet, ':', label= myLabel)

	# plt.plot(trainSet, 'r', label='train')
	plt.legend(loc=0)
	plt.xlabel('index')
	plt.ylabel('feature')

	# titleOfGraph = titleOfGraph + " - lamb_" + str(lambd) + ' - minkowski_' + str(distNorm) + " - numPredict_" + str(len(valueOfSample))
	# plt.title(titleOfGraph)

	if (pathSaveFigure != None):
		plt.savefig(pathSaveFigure, dpi = 200)


def predict_diagonal(trainSet, testSet, dim=3, tau=2, epsilon=0.007, lambd=3, percent=1, distNorm = 2,
	titleOfGraph = 'Pretty Girl', figureName = 'GrilLikeYou', pathSaveFigure = None):

	# # vectors_train = data['train_data']
	# dim = 5 #data['dim']
	# tau = 2 #data['tau']
	# percent = 0.6 #data['percent']
	# # curve_number = data['curve_number']
	# epsilon = 0.7 #data['epsilon']
	# lambd = 2 #data['lambd']

	vectors_train_1 = myRP.convert2StatePhase(trainSet, dim, tau, returnType = 'np.array')

	#tách statephases
	vectors_test_1 = myRP.convert2StatePhase(testSet, dim, tau, returnType = 'np.array')

	# print("train.shape: ", vectors_train_1.shape)

	# r_dist là ma trận khoảng cách
	# cdist là hàm trong scipy.spatial.distance.cdist
	# minkowski là cách tính
	# p là norm
	# y là train: đánh số từ trên xuống dưới
	# x là test
	# 
	from scipy.spatial.distance import cdist
	r_dist = cdist(vectors_train_1, vectors_test_1, 'minkowski', p=distNorm)

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
	# predict_label = np.zeros(len(testSet),dtype=int)
	
	#r1 là ma trận -2|-1 thay vì ma trận 01 như crp
	#-2 là false
	#-1 là true
	#r_dist < epsilon trả về ma trận 01, 1 tại điểm r_dist < epsilon
	#ma trận này trừ đi 2 thì thành ma trận -2|-1
	r1 = np.array((r_dist < epsilon)-2)

	# r1 = np.array(r1 - 2)
	
	# f_crp = myCrpFunctions.crossRecurrencePlots("CRP", r1, dotSize = 0.2)
	

	# aaa = myCrpFunctions.rqaFromBinaryMatrix(r1, keyDot = -1)
	# print(aaa)

	len_r1 = int(np.size(r1[0]))
	high_r1 = int(np.size(r1)/len_r1)

	#x is an array whose each element is a diagonal of the crp matrix
	
	#Đoạn này lấy ra những đoạn đường chéo có số lượng predict liền nhau lớn hơn lamdb
	#Mỗi phần tử của diagonals_have_predict là một mảng chứa đúng 1 đoạn trùng với train tương ứng đúng vị trí
	diagonals_have_predict = []
	for index_diagonal in range(-(high_r1 - lambd + 1), len_r1 - lambd + 2, 1):
		offset = index_diagonal
		#---offset = x - y
		y = -offset if (index_diagonal < 0) else 0

		while (y < high_r1 and y+offset < len_r1):
			if (r1[y][y+offset] == -1):
				start = y
				while ( y+1<high_r1 and y+1+offset < len_r1 and r1[y+1][y+1+offset] == -1):
					y+=1
				if (y - start + 1 >= lambd):
					predicts = np.full(y+1, -2)

					for index in range(start, y+1, 1):
						predicts[index] = index + offset
					diagonals_have_predict.append(predicts)
			y+=1
	
	'''
	#Mảng quy đổi về index của dữ liệu test từ index của statePhrase
	#

	indexSampleOrigin = diagonals_have_predict

	for i_row in range(len(indexSampleOrigin)):
		for i_in_1_State in range(len(indexSampleOrigin[i_row])):
			if (indexSampleOrigin[i_row][i_in_1_State] >= 0):
				#Chọn số index state cuối cùng
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
				try:
					shapePredict[indexSampleOrigin[i][j]] = shape
				except (IndexError):
					print("^v^v^v^v^v^v^v^v^v^v^v^v^v^v^indexError: {} ^v^v^v^v^v^v^v^v^v^v^v^v^v^v^" % (indexSampleOrigin[i][j]))

		valueOfSample.append(arr)

	# print(valueOfSample)

	# testSetPr = SyncPredictToTestArr(index_timeseries_test, testSet)
	f_line = plt.figure(figureName, figsize = (12.8, 7.2), dpi = 100)

	print("_________num diagonals_have_predict: ", len(diagonals_have_predict))

	if (len(valueOfSample) > 0):
		# f_line.set_size_inches(1080, 720)
		for i in range(0, len(valueOfSample)):
			plt.plot( valueOfSample[i], ':', label=i)

		plt.plot(trainSet, 'r', label='train')
		plt.legend(loc=0)
		plt.xlabel('index')
		plt.ylabel('feature')

		titleOfGraph = titleOfGraph + " - lamb_" + str(lambd) + ' - minkowski_' + str(distNorm) + " - numPredict_" + str(len(valueOfSample))
		plt.title(titleOfGraph)

		if (pathSaveFigure != None):
			plt.savefig(pathSaveFigure, dpi = 200)
		# plt.ylim(ymin = min(trainSet) - 0.09)
		# plt.show()
	'''

	return len(diagonals_have_predict)

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

pathData = "data/GR-Emerald-3X_GR-Emerald-3X.csv"

# dataShape1, _, _ = myCrpFunctions.readCSVFileByShape(pathData, 2, 1, 2)

dataByShape = [[],[],[],[],[]]

for i in range(5):
	allTimeseries, _, _ = myCrpFunctions.readCSVFileByShape(pathData, i+1, 1, 2)
	for timeseries in allTimeseries:
		dataByShape[i] = allTimeseries

data = dataByShape[4][2][10:15]
print(len(data))

folderOut = "out24032019/RP/"
myCrpFunctions.createFolder(folderOut)

nPredict = []


# a = [1, 5, 3, 6, 5]
# b = [1, 2 ,3 ,4 ,5 ]

# for item in a:
# 	print(item in b)

# print(data)
# plt.plot(data)
# plt.plot(data, 'x')
# 
numInsert = 3
xnew, ynew = myInterpolation(data, numInsert = numInsert, myKind = 'cubic')

print(xnew)

for x, y in enumerate(data):
	print(x , xnew[x*(numInsert+1)])
# plt.plot(xnew, ynew, '.')
# plt.axvline(x=1, linewidth=1, color='k')

# plt.show()

'''
for i in [0, 1,2 , 3,4 ]:
	nums = []
	for j, data in enumerate(dataByShape[i]):
		print(len(data))
		_, dataInter = myCrpFunctions.myInterpolation(data, numNew=200)
		r_Binary = myRP.makeRPmatrix(dataInter, epsilon=1, dim = 5, tau=4)
		pathSave = folderOut + "RP_" + str(i+1) + "_" + str(j) + ".png"
		x = myCrpFunctions.crossRecurrencePlots("{}_{}".format(i+1, j), r_Binary, keyDot = 1, dotSize = 10, pathSaveFigure = pathSave)
		
		# nums.append(predict_diagonal(dataInter, dataInter, epsilon = 0.9))

	nPredict.append(nums)
		# plt.show()
'''

# plt.show()
# for range in RangeDataShape:

# for i, arr in enumerate(nPredict):
# 	print(arr)
# 	plt.plot(arr, label = i+1)

# plt.legend(loc=0)
# plt.show()
# 
# 

