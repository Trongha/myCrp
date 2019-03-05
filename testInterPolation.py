import numpy as np 

def myInterpolation(y, x = None, numNew = 100):
	import numpy as np
	from scipy import interpolate

	if(x == None):
		x = np.arange(0, len(y), 1)
	# print("x: ", x)
	xnew = np.arange(x[0], x[len(x)-1], len(x)/numNew)
	f = interpolate.interp1d(x, y)
	ynew = f(xnew)
	
	# print("xnew: ", xnew)

	return xnew.tolist(), ynew.tolist()


x = [1,2,3,4,5]

y = [1,2,5,4,5]

import matplotlib.pyplot as plt 
from scipy import interpolate

xnew, ynew = myInterpolation(y, x, kind = 'linear')

print(xnew)
plt.plot(x, y)
plt.plot(xnew, ynew, 'o')

plt.show()
