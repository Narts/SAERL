import numpy
from numpy import genfromtxt


RLData = genfromtxt('RLData.csv', delimiter=';', skip_header=1)
colLen, rowLen = RLData.shape

rptCount = 0
blkCount = 0

for i in range(0, colLen):
    if (RLData[i, 2] == RLData[i, 4] and RLData[i, 3] == RLData[i, 5]):
        print "block ", i+2, " ", RLData[i, 2:4], " ", RLData[i, 4:6]
        blkCount = blkCount+1

    elif (RLData[i, 4] == RLData[i+1, 2] and RLData[i, 5] == RLData[i+1, 3]):
        pass

    else:
        print "repeat ", i+2, " ", RLData[i, 4:6], " ", RLData[i+1, 2:4]
        rptCount = rptCount+1

print "rptCount: ", rptCount, "blkCount: ", blkCount
