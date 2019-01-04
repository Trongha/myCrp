import numpy as np
import myCrpFunctions
import matplotlib.pyplot as plt
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


createFolder("testMakeFolder")








# r1 =np.array      ([[-1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -1, -1, -1, -1, -1, -2],
# 					[-1, -2, -1, -1, -1, -1, -2, -2, -2, -2, -1, -1, -2, -1, -1, -1, -2],
# 					[-1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1],
# 					[-1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1],
# 					[-2, -1, -1, -2, -2, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -2],
# 					[-2, -2, -2, -2, -1, -1, -1, -1, -1, -2, -1, -2, -1, -1, -2, -1, -2],
# 					[-2, -1, -1, -1, -2, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -2],
# 					[-1, -2, -1, -1, -1, -1, -1, -2, -2, -2, -1, -1, -2, -1, -1, -1, -2],
# 					[-1, -2, -1, -1, -1, -1, -2, -2, -2, -2, -1, -1, -2, -1, -1, -1, -2]
# 					])
r1 =np.array      ([[-1, -2, -2, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1],
					[-1, -1, -2, -2, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1],
					[-2, -1, -1, -2, -2, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
					[-2, -2, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
					[-2, -2, -2, -2, -2, -2, -2, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2],
					[-2, -2, -2, -2, -1, -2, -2, -2, -1, -2, -2, -2, -2, -2, -1, -2, -2],
					[-1, -2, -2, -2, -2, -1, -2, -2, -2, -1, -2, -2, -2, -2, -2, -1, -2],
					[-2, -1, -2, -2, -2, -2, -1, -2, -2, -2, -1, -2, -2, -2, -2, -2, -1],
					[-1, -2, -1, -2, -2, -2, -2, -2, -2, -2, -2, -1, -2, -2, -2, -2, -2]
					])

print(r1)

r2 = np.array([])

len_r1 = int(np.size(r1[0]))
high_r1 = int(np.size(r1)/len_r1)

print("high_r1: ", high_r1)
print("len_r1: ", len_r1)
#x is an array whose each element is a diagonal of the crp matrix
diagonals_have_predict = []
lambd = 3

for yStart in range(high_r1 - lambd + 1, 0, -1):
	offset = -yStart		# x = y + offset
	y = yStart
	while y < high_r1 :
		if (r1[y][y+offset] == -1):
			start = y
			while ( y+1 < high_r1 and r1[y+1][y+1+offset] == -1):
				y+=1
			if (y - start + 1 >= lambd):
				predicts = np.full(y+1, -2)

				for index in range(start, y+1, 1):
					predicts[index] = index + offset
				diagonals_have_predict.append(predicts)
		y+=1

for xStart in range(0, len_r1 - lambd + 2):
	offset = xStart		# y = x - offset
	y = 0
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
print("diagonals_have_predict: \n")
for row in diagonals_have_predict:
	print (row)	


diagonals_have_predict2 = []
for index_diagonal in range(-(high_r1 - lambd + 1), len_r1 - lambd + 2, 1):
	offset = index_diagonal
	#---offset = x - y
	print(offset)
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
				diagonals_have_predict2.append(predicts)
		y+=1

print("diagonals_have_predict2: \n")
for row in diagonals_have_predict2:
	print (row)





# testNone = [1, 2, None, None];
# print(testNone)

# x = np.array([0, 1, 2,3 ,4 , 5])
# x = np.insert(x, 6, 100)
# print(x)
# plt.show()