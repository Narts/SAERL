from pylearn2.utils import serial
import theano
from numpy import genfromtxt
import numpy


l1_path = 'dae_l1.pkl'
l1 = serial.load(l1_path)
print l1

"""encode"""
#"""
#layer 1
l1Input = l1.get_input_space().make_theano_batch()
l1Encode = l1.encode(l1Input)
l1Decode = l1.decode(l1Encode)
l1EncodeFunction = theano.function([l1Input], l1Encode)
l1DecodeFunction = theano.function([l1Encode], l1Decode)

#Seperate action RLDataNoBlk 8Col noReward
RLData = genfromtxt('Seperate action RLDataBlk noRw.csv', delimiter=',', skip_header=0)
shapeRLData = RLData.shape
print RLData
#"""

"""
zerosArray = numpy.zeros(shapeRLData)
zerosArray[:, [5]] = RLData[:, [5]]
print "The "+str(5)+"th RLData inputted"
zerosArrayFlt = numpy.around(zerosArray, decimals=2)
print zerosArrayFlt

l1encode1 = l1EncodeFunction(zerosArray)
l1decode1 = l1DecodeFunction(l1encode1)

print "The "+str(5)+"th RLData decode"
l1decodeFlt = numpy.around(l1decode1, decimals=2)

print l1decodeFlt
numpy.savetxt("The AE tanh MSE "+str(5)+"th RLData Decode.csv", l1decodeFlt, fmt="%10.2f", delimiter=" , ")
"""

"""
#activate one hidden node
for i in range(0, 7):
    zerosArray = numpy.zeros(shapeRLData)
    zerosArray[:, [i]] = RLData[:, [i]]
    print "The "+str(i)+"th RLData inputted"
    zerosArrayFlt = numpy.around(zerosArray, decimals=2)
    print zerosArrayFlt

    l1encode1 = l1EncodeFunction(zerosArray)
    l1decode1 = l1DecodeFunction(l1encode1)

    print "The "+str(i)+"th RLData decode"
    l1decodeFlt = numpy.around(l1decode1, decimals=2)

    print l1decodeFlt
    numpy.savetxt("The test "+str(i)+"th RLData Decode.csv", l1decodeFlt, delimiter=" , ")


#"""
#"""
l1encode = l1EncodeFunction(RLData)
l1encodeFlt = numpy.around(l1encode, decimals=2)

print "l1 encode"
print l1encodeFlt

numpy.savetxt("moreDataDAE 7-7 8ColNoRwBLK softPluslinear l1encode.csv", l1encode, fmt="%10.2f", delimiter=",")
#numpy.savetxt("moreDataDAE 7-6 8ColNoRwBLK softPluslinear l1encodeFlt0.csv", l1encode, fmt="%10.0f", delimiter=",")

l11decode = l1DecodeFunction(l1encode)
l11decodeFlt = numpy.around(l11decode, decimals=2)

numpy.savetxt("moreDataDAE 7-7 8ColNoRwBLK softPluslinear l11decode.csv", l11decode, fmt="%10.2f", delimiter=",")

print "l11 decode"
print l11decodeFlt
#"""

"""
#Separate layer 2
l2Sep_path = 'daeSep_l2.pkl'
l2Sep = serial.load(l2Sep_path)
print l2Sep

l2SepInput = l2Sep.get_input_space().make_theano_batch()
l2SepEncode = l2Sep.encode(l2SepInput)
l2SepDecode = l2Sep.decode(l2SepEncode)
l2SepEncodeFunction = theano.function([l2SepInput], l2SepEncode)
l2SepDecodeFunction = theano.function([l2SepEncode], l2SepDecode)

#Seperate action RLDataNoBlk 8Col noReward
RLData = genfromtxt("moreDataDAE 7-6 8ColNoRwBLK softPluslinear l1encodeFlt0.csv", delimiter=',', skip_header=0)
shapeRLData = RLData.shape
print RLData

l2Sepencode = l2SepEncodeFunction(RLData)
l2SepencodeFlt = numpy.around(l2Sepencode, decimals=2)

print "l2Sep encode"
print l2SepencodeFlt

numpy.savetxt("moreDataDAE 7-6 8ColNoRwBLK softPluslinear l2Sepencode.csv", l2Sepencode, fmt="%10.2f", delimiter=",")

l2Sepdecode = l2SepDecodeFunction(l2Sepencode)
l2SepdecodeFlt = numpy.around(l2Sepdecode, decimals=2)

numpy.savetxt("moreDataDAE 7-6 8ColNoRwBLK softPluslinear l2Sepdecode.csv", l2Sepdecode, fmt="%10.2f", delimiter=",")

print "l2Sep decode"
print l2SepdecodeFlt
#"""

#"""
#layer 2
l2_path = 'dae_l2.pkl'
l2 = serial.load(l2_path)
print l2

l2Input = l2.get_input_space().make_theano_batch()
l2Encode = l2.encode(l2Input)
l2EncodeFunction = theano.function([l2Input], l2Encode)

l2encode = l2EncodeFunction(l1encode)
l2encodeFlt = numpy.around(l2encode, decimals=2)

print "l2 encode"
print l2encodeFlt
numpy.savetxt("moreDataDAE 7-7 8ColNoRwBLK softPluslinear l2encode.csv", l2encode, fmt="%10.2f", delimiter=",")
#for item in l2encode:
    #print item


#decode
l2Decode = l2.decode(l2Encode)
l2DecodeFunction = theano.function([l2Encode], l2Decode)

l2decode = l2DecodeFunction(l2encode)
l2decodeFlt = numpy.around(l2decode, decimals=2)

print "l2 decode"
print l2decodeFlt
numpy.savetxt("moreDataDAE 7-7 8ColNoRwBLK softPluslinear l2decode.csv", l2decode, fmt="%10.2f", delimiter=",")

#active one hidden node each time
#shapeL2encode = l2encode.shape
#print shapeL2encode
#print zerosArray.shape

l1decode = l1DecodeFunction(l2decode)
l1decodeFlt = numpy.around(l1decode, decimals=2)
print "l1 decode"
print l1decodeFlt
numpy.savetxt("moreDataDAE 7-7 8ColNoRwBLK softPluslinear l1decode.csv", l1decode, fmt="%10.2f", delimiter=",")

#"""
"""
#activate one hidden node
for i in range(0, 4):
    zerosArray = numpy.zeros(shapeL2encode)
    #extractedColumn = l2encode[:, [i]]
    zerosArray[:, [i]] = l2encode[:, [i]]
    print "The "+str(i)+"th hidden node activation"
    zerosArrayFlt = numpy.around(zerosArray, decimals=2)
    print zerosArrayFlt
    l2decode1 = l2DecodeFunction(zerosArray)

    print "test1", l2decode1

    l1decode = l1DecodeFunction(l2decode1)
    print "The "+str(i)+"th hidden node activation decode"
    l1decodeFlt = numpy.around(l1decode, decimals=2)

    print l1decodeFlt
    numpy.savetxt("The"+str(i)+"thActive.csv", l1decodeFlt, delimiter=" , ")

#activate two hidden nodes
for i in range(0, 4):
    for j in range(i+1, 4):
        zerosArray = numpy.zeros(shapeL2encode)
        #extractedColumn = l2decode[:, [i]]
        zerosArray[:, [i]] = l2encode[:, [i]]
        zerosArray[:, [j]] = l2encode[:, [j]]
        print "The "+str(i)+"th, " + str(j)+"th hidden node activation"
        zerosArrayFlt = numpy.around(zerosArray, decimals=2)
        print zerosArrayFlt
        l2decode2 = l2DecodeFunction(zerosArray)

        print "test2", l2decode2

        l1decode = l1DecodeFunction(l2decode2)
        print "The "+str(i)+"th, " + str(j)+"th hidden node activation decode"
        l1decodeFlt = numpy.around(l1decode, decimals=2)

        print l1decodeFlt
        numpy.savetxt("The"+str(i)+"th, " + str(j)+"th Active.csv", l1decodeFlt, delimiter=" , ")
"""
