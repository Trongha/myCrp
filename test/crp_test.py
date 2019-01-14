import numpy
import matplotlib.pyplot as plt

import myCrpFunctions


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

dataTrain, minOfTrain, maxOfTrain = myCrpFunctions.readCSVFileByShape(path[1], shape, indexColFeature, indexColOfShape)



def smooth(x,window_len=10,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError
        # , "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError
        # , "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError
        # , "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')

    return y[int(window_len/2-1):-int(window_len/2)]

from numpy import *
from pylab import *

def testSmooth1():
    # smooth_demo()
    
    # print(dataTrain)
    print("len: ", len(dataTrain[3]))
    a = numpy.array(dataTrain[3])
    # print(smooth(a))
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    s = "len1: " + str(a.size)

    for w in windows:
        plt.plot(smooth(a, window_len = 9,  window=w))
        s += "\nlen_" + w + " : " + str(smooth(a, window=w).size)

    print(s )
    show()

def testSmoothSavitzky_Golay():
    from scipy.signal import savgol_filter
    import scipy.fftpack

    y = dataTrain[3]
    y = myCrpFunctions.ConvertSetNumber(y)
    yhat = savgol_filter(y, 11, 3)
    print('len: {} - {} '.format(len(y), len(yhat)))
    print(yhat)

    plt.plot(y)
    plt.plot(yhat)
    show()

def fftpackSmoothing():
    import numpy as np
    import scipy.fftpack

    N = 100
    x = np.linspace(0,2*np.pi,N)
    y = dataTrain[3]

    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
    spectrum = w**2

    cutoff_idx = spectrum < (spectrum.max()/5)
    w2 = w.copy()
    w2[cutoff_idx] = 0

    y2 = scipy.fftpack.irfft(w2)
    plt.plot(w)
    plt.plot(y)
    plt.plot(y2)

    show()

def smoothListGaussian(list,degree=5):  

    window=degree*2-1  

    weight=numpy.array([1.0]*window)  

    weightGauss=[]  

    for i in range(window):  

        i=i-degree+1  

        frac=i/float(window)  

        gauss=1/(numpy.exp((4*(frac))**2))  

        weightGauss.append(gauss)  

    weight=numpy.array(weightGauss)*weight  

    smoothed=[0.0]*(len(list)-window)  

    for i in range(len(smoothed)):  

        smoothed[i]=sum(numpy.array(list[i:i+window])*weight)/sum(weight)  

    return list[0: degree] + smoothed + list[len(list)-degree+1: len(list)]

def testGaussianSmoothing():
    size = 5
    y = dataTrain[3]
    
    y1 = smoothListGaussian(y)
    print("len1{} len2 {}".format(len(y), len(y1)))


    plt.plot(y)
    plt.plot(y1)
    show()

def smoothListTriangle(list,strippedXs=False,degree=5):  

    weight=[]  

    window=degree*2-1  

    smoothed=[0.0]*(len(list)-window)  

    for x in range(1,2*degree):weight.append(degree-abs(degree-x))  

    w=numpy.array(weight)  

    for i in range(len(smoothed)):  

        smoothed[i]=sum(numpy.array(list[i:i+window])*w)/float(sum(w))  

    return list[0: degree] + smoothed + list[len(list)-degree+1: len(list)]

def testTriangle():
    size = 5
    y = dataTrain[3]
    
    y1 = smoothListTriangle(y)
    print("len1{} len2 {}".format(len(y), len(y1)))

    plt.plot(y)
    plt.plot(y1)
    show()

def testWaveletsSmoothing():
    modes = ['zpd', 'cpd', 'sym', 'ppd', 'sp1', 'per']
    import pywt
    f = plt.figure('girlLikeYou')

    y = dataTrain[3]
    plt.plot(y, label='origin')

    types = ['db1', 'db5',  'db11', 'db15']
    print(len(y))

    for type in types:

        y1,_ = pywt.dwt(y, type, mode='per')
        print(type, len(y1))

        plt.plot(y1, label=type)



if __name__=='__main__':
    # testSmoothSavitzky_Golay()
    # fftpackSmoothing()
    testWaveletsSmoothing()
    plt.legend(loc=0)
    title('title')
    plt.show()
