# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import myCrpFunctions
import os
import pywt

testShape = []
shapePredict = []
shape = 0


def rqa(training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number):
    '''
    @Parameters:
    training_well_ts -- numpy array 2D:
        the 1st column of type string (name of well)
        the last column of type integer: facies_class
        the another column: features

        Example:
        array([['RD-1P'     2555.4432   2434.7698   108.8463    0.2312  2.5599  84.4916 0.6982  0.036   5],
                ['RD-1P'    2555.5956   2434.9184   101.5264    0.2011  2.586   81.334  0.617   0.0333  5],
                ['RD-1P'    2557.7292   2436.9983   74.2481     0.1072  2.5488  68.2637 0.3139  0       3]])


    testing_well_ts -- numpy array 2d like training_well_ts but containing only data of one well

    dim : the dimension -- type: integer, greater than or equal to 2
    tau : the step -- type: integer, greater than or equal to 1
    epsilon: type of float greater than 0
    lambd: the positive integer
    percent: the float number between 0 and 1
    curve_number: the positive integer is index of column feature in training_well_ts
    facies_class_number: the integer greater than or equal to 0-- the name of class to detect

    @Return:
    predict_label-- numpy array 1D of shape (the length of testing_well_ts, ) containg only 0, 1:
        0: predict not facies_class_number
        1: predict belong to facies_class_number
    '''
    if not (percent > 0 and percent <= 1.0):
        print('percent must > 0 and <= 1.0')
        raise AssertionError

    vectors_train = create_train(training_well_ts, dim, tau, curve_number, facies_class_number)

    vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)
    r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1)

    r = np.sum(r_dist < epsilon, axis=0)
    if __debug__:
        print(np.sum(r))

#   """____________________________________________________"""

    predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)
    indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
    # print('indx_vectors_timeseries_test.shape: ', indx_vectors_timeseries_test)

    index = indx_vectors_timeseries_test[r > lambd, :]
    index = index[:, 0]
    add_index = list(np.arange(0, dim*percent, dtype=int))

#   index = index[:, :int(dim*percent)].ravel()
    for i in add_index:
        predict_label[(index+i).ravel()] = 1

    return predict_label.ravel().tolist()

#Make Folder
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

def state_phase(v, start, dim, tau):
    #v: vecto
    #dim là số chiều (số phần tử)   
    #tau là bước nhảy       
    return [v[start + i*tau] for i in range(0, dim, 1)]

def checkRecall(outputFolder = None):
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
        writeContentToFile(pathOut, content)


    

def refreshPredict():
    for i in range(len(shapePredict)):
        shapePredict[i] = 0

# def SyncPredictToTestArr(indexSet, dataSet):
#   # Lấy các giá trị của dataSet tại các index liên tiếp trong indexSet
#   outputArr = [[0]]
#   i = 0
#   while (i < len(indexSet)):
#       a = [dataSet[indexSet[i]]];
#       while (i < len(indexSet) -1 and (indexSet[i+1] - indexSet[i] == 1)):
#           i+=1
#           a.append(dataSet[indexSet[i]])
#       if (len(a) > 1):
#           outputArr.append(a)
#       i+=1        
#   return outputArr


def predict_diagonal(trainSet, testSet, dim=5, tau=2, epsilon=0.7, lambd=3, percent=0.6, distNorm = 1,
    titleOfGraph = 'Pretty Girl', figureName = 'GrilLikeYou', pathSaveFigure = None):

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

    # print("train.shape: ", vectors_train_1.shape)

    # r_dist là ma trận khoảng cách
    # cdist là hàm trong numpy
    # minkowski là cách tính
    # p là norm
    # y là train: đánh số từ trên xuống dưới
    # x là test
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

    print("_________num predict: ", len(valueOfSample))

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

    return f_line

def makeTestFeature(trainSet, testSet, dim=3, tau=2, epsilon=0.0055, lambd=3, percent=1, distNorm = 1, pathFolder = "result/", formatSave = ".png", trainSetID = None):
    
    createFolder(pathFolder)
    if (trainSetID == None):
        import time
        import datetime
        trainSetID = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%Hh%Mp%S')

    numSample = len(trainSet)
    title = "IDtrain_" + str(trainSetID) + "_" + str(numSample) + " - dim_" + str(dim) + " - epsil_" + str(epsilon) + " - lambd_" + str(lambd)
    print("\n------------------------------------------------", title,"------------------------------------------------\n")

    pathSave = pathFolder + title + formatSave
    print("---------------------------", title, "---------------------------\n")

    f2 = predict_diagonal(trainSet, testSet,
                            dim=dim, tau=tau, epsilon=epsilon, lambd=lambd, percent=percent, distNorm=distNorm,
                            titleOfGraph = title,  figureName = title, pathSaveFigure = pathSave)

def makeTestFeatureZoom(trainSet, testSet, dim=3, tau=2, epsilon=0.0055, lambd=3, percent=1, distNorm = 1, pathFolder = "result/", formatSave = ".png", trainSetID = None):
    
    createFolder(pathFolder)
    if (trainSetID == None):
        import time
        import datetime
        trainSetID = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%Hh%Mp%S')

    numSample = 45
    for i in range (0, len(trainSet), 50):
        title = "IDtrain_" + str(trainSetID) + "_" + str(i) + " - dim_" + str(dim) + " - epsil_" + str(epsilon) + " - lambd_" + str(lambd)
        print("\n------------------------------------------------", title,"------------------------------------------------\n")

        pathSave = pathFolder + title + formatSave

        trainSetZoom = trainSet[i:i+numSample]
        if (len(trainSetZoom) > dim*tau):
            f2 = predict_diagonal(trainSetZoom, testSet,
                                    dim=dim, tau=tau, epsilon=epsilon, lambd=lambd, percent=percent, distNorm=distNorm,
                                    titleOfGraph = title,  figureName = title, pathSaveFigure = pathSave)

def veDoThiTatCaTrainShape(dataSet, titleOfGraph = "Autumn in my heart"):
    for set in dataSet:
        f_line = plt.figure(titleOfGraph, figsize = (12.8, 7.2), dpi = 100)
        plt.plot(set)
        plt.title(titleOfGraph)
    
    return f_line
    
###############################-----________MAIN________-----###############################

if (__name__ == "__main__"):

    myLambd=2
    myDim=3
    distNorm = 2
    tau = 2
    formatSave = ".png"
    epsilon = 0.009
    pathFolder = "noSmoothingZoom/"
    # checkRecall(pathFolder)

    fileName = ["RBA-3P_RBA-3P.csv",
            "RBA-6P_RBA-6P.csv",
            "RBA-12PST4_RBA-12PST4.csv",
            "RUBY-1X_RUBY-1X_1.csv",
            "RUBY-4X_RUBY-4X_1.csv",
            "TN-3X_TN-3X.csv"]

    path = [("data/" + name) for name in fileName]

    shape = 2
    indexColOfShape = 3

    indexColFeature = 2

    allTrainSet, minOfTrain, maxOfTrain = myCrpFunctions.readCSVFileByShape(path[0], shape, indexColFeature, indexColOfShape)

    for i in range(1, 5):
        allTrainSetI, minOfTrainI, maxOfTrainI = myCrpFunctions.readCSVFileByShape(path[i], shape, indexColFeature, indexColOfShape)
        allTrainSet += allTrainSetI
        if (minOfTrainI < minOfTrain) :
            minOfTrain = minOfTrainI
        if (maxOfTrainI > maxOfTrain) :
            maxOfTrain = maxOfTrainI
    
    # dataTest = myCrpFunctions.readCSVFile(path[5], indexColFeature)

    dataTest, testShape = myCrpFunctions.readCSVFileForTest(path[5], indexColFeature)
    shapePredict = [0 for i in range(len(testShape))]

    print("data: ", len(dataTest), " Pre: ", len(shapePredict), " shape: ")

    if (minOfTrain > min(dataTest)) :
        minOfNorm = min(dataTest)
    else: 
        minOfNorm = minOfTrain

    if maxOfTrain < max(dataTest) : 
        maxOfNorm = max(dataTest) 
    else: 
        maxOfNorm = maxOfTrain 

    for i in range(len(allTrainSet)):
        allTrainSet[i] = myCrpFunctions.ConvertSetNumber(allTrainSet[i], minOfSet = minOfNorm, maxOfSet = maxOfNorm)
    testSet = myCrpFunctions.ConvertSetNumber(dataTest, minOfSet = minOfNorm, maxOfSet = maxOfNorm)

    print("len(trainSet): ", len(allTrainSet))
    print("len(testSet): ", len(testSet))

    

'''
    for i in range(0, len(allTrainSet)):
        if (len(allTrainSet[i]) > myDim*tau):
            makeTestFeatureZoom(allTrainSet[i], testSet, 
                            dim= myDim, tau= tau, epsilon= epsilon, lambd= myLambd, percent=1, distNorm = distNorm, 
                            pathFolder = "noSmoothingZoom/", formatSave = ".png", trainSetID = i)
    checkRecall(pathFolder)
'''