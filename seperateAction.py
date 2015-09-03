import numpy
from numpy import genfromtxt


RLData = genfromtxt('RLDataB.csv', delimiter=',', skip_header=1)
colLen, rowLen = RLData.shape

print colLen, rowLen
zerosArray = numpy.zeros((colLen, 8))

for i in range(0, colLen):
    if (RLData[i, 0] == 0):
        zerosArray[i, 0] = 1
    elif (RLData[i, 0] == 1):
        zerosArray[i, 1] = 1
    elif (RLData[i, 0] == 2):
        zerosArray[i, 2] = 1
    elif (RLData[i, 0] == 3):
        zerosArray[i, 3] = 1

for j in range(2, 6):
    zerosArray[:, [j+2]] = RLData[:, [j]]

numpy.savetxt("Seperate action RLDataBlk noRwB.csv", zerosArray, delimiter=",")


#SprtActRLData = genfromtxt('The seperate action RLData.csv', delimiter=',')
#shapeSprtActRLData = SprtActRLData.shape
"""
zerosSAArray = numpy.zeros((colLen, 8))

for k in range(0, 9):
    if (k == 4):
        pass
    elif(k == 0 or k == 1 or k == 2 or k == 3):
        zerosSAArray[:, [k]] = zerosArray[:, [k]]
    elif(k == 5 or k == 6 or k == 7 or k == 8):
        zerosSAArray[:, [k-1]] = zerosArray[:, [k]]

numpy.savetxt("Seperate action RLDataNoBlk 8Col noReward.csv", zerosSAArray, delimiter=",")
"""
