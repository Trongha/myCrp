import numpy as np

diagonalsMatrix = np.array([[-1, -1, -1, -1, -2, -2],
					[-2, -2, -2, -2, -2, -1],
					[-1, -1, -1, -1, -2, -2],
					[-2, -1, -1, -1, -1, -1],
					[-2, -2, -2, -2, -2, -1]
					])

x = np.array([0, 1, 2,3 ,4 , 5])
x = np.insert(x, 6, 100)
print(x)


diagonalsMatrix2 = []
lambd = 3
i_row = 0
print(np.size(diagonalsMatrix))
while (i_row < np.size(diagonalsMatrix)):
	print(i_row)
	havePredict = 0
	lenn = np.size(diagonalsMatrix[i_row])
	i = 0
	while i<lenn:
		if (diagonalsMatrix[i_row][i] == -1):
			start = i
			while (i<lenn-1 and diagonalsMatrix[i_row][i+1] == -1 ):
				i+=1
			if (i-start+1 > lambd):
				havePredict = 1			
		i+=1
	if (havePredict == 1):
		diagonalsMatrix2.append(diagonalsMatrix[i_row])
	i_row += 1

print(diagonalsMatrix2)

for test in diagonalsMatrix2:
	for i in range(len(test)):
		if (test[i] == -1  ):
			if (i == len(test)-1 or test[i+1] != -1):
				for j in range(5):
					np.insert(test, i+j, 10)
					# test.insert(i+j+1, 10)

print(diagonalsMatrix2)