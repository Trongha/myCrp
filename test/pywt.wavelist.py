['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
['haar']
['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38']
['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
['coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17']
['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']
['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8']
['dmey']
['gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8']
['mexh']
['morl']
['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8']
['shan']
['fbsp']
['cmor']

bsub "python3 crpWaveletsSmoothing1.py 1>stdWavelets1.out 2>stdWavelets1.err"
bsub "python3 crpWaveletsSmoothing2.py 1>stdWavelets2.out 2>stdWavelets2.err"
bsub "python3 crpWaveletsSmoothing3.py 1>stdWavelets3.out 2>stdWavelets3.err"
bsub "python3 crpWaveletsSmoothing4.py 1>stdWavelets4.out 2>stdWavelets4.err"
bsub "python3 crpWaveletsSmoothing5.py 1>stdWavelets5.out 2>stdWavelets5.err"
bsub "python3 crpWaveletsSmoothing6.py 1>stdWavelets6.out 2>stdWavelets6.err"

watch bjobs

TP:   116, FP:  1877, FN:   160, TN:  5405
pi = 0.05820
p = 0.42029
f1 = 0.10225
1401_waveletSmoothing/Type_coif14-Epsilon_0.0085/
