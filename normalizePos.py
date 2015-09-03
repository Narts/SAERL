import numpy
from numpy import genfromtxt


RLData = genfromtxt('Seperate action RLDataBlk noRw.csv', delimiter=',', skip_header=0)
colLen, rowLen = RLData.shape

zerosArray = numpy.zeros((colLen, rowLen))
print colLen, rowLen


for i in range(0, colLen):
    for j in range(0, rowLen):
        zerosArray[i, j] = RLData[i, j]/10
        #print zerosArray[i, j]

numpy.savetxt("Seperate action RLDataBlk noRw normalized.csv", zerosArray, delimiter=",")
