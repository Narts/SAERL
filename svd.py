from __future__ import division
from pylearn2.utils import serial
import theano
from numpy import genfromtxt
import numpy
import numpy as np


l1_path = 'dae_l1_best_bck.pkl'
l1 = serial.load(l1_path)
print l1

"""encode"""
#layer 1
l1Input = l1.get_input_space().make_theano_batch()
l1Encode = l1.encode(l1Input)
l1Decode = l1.decode(l1Encode)
l1EncodeFunction = theano.function([l1Input], l1Encode)
l1DecodeFunction = theano.function([l1Encode], l1Decode)

#Seperate action RLDataNoBlk 8Col noReward
RLData = genfromtxt('/Users/shengtaoran/Desktop/pylearn2Test/stackedAutoencoderRL/Data/trainingData/Seperate action RLDataBlk noRw.csv', delimiter=',', skip_header=0)
row, col = RLData.shape
print row, col

l1encode = l1EncodeFunction(RLData)
l1encodeFlt = numpy.around(l1encode, decimals=2)
print "l1 encode"
print l1encodeFlt

l11decode = l1DecodeFunction(l1encode)
l11decodeFlt = numpy.around(l11decode, decimals=2)
print "l11 decode"
print l11decodeFlt


#layer 2
l2_path = 'dae_l2_best_bck.pkl'
l2 = serial.load(l2_path)
print l2

l2Input = l2.get_input_space().make_theano_batch()
l2Encode = l2.encode(l2Input)
l2Decode = l2.decode(l2Encode)
l2EncodeFunction = theano.function([l2Input], l2Encode)
l2DecodeFunction = theano.function([l2Encode], l2Decode)

l2encode = l2EncodeFunction(l1encode)
l2encodeFlt = numpy.around(l2encode, decimals=2)
print "l2 encode"
print l2encodeFlt

#decode
l2decode = l2DecodeFunction(l2encode)
l2decodeFlt = numpy.around(l2decode, decimals=2)
print "l2 decode"
print l2decodeFlt

l1decode = l1DecodeFunction(l2decode)
l1decodeFlt = numpy.around(l1decode, decimals=2)
print "l1 decode"
print l1decodeFlt
numpy.savetxt("/Users/shengtaoran/Desktop/pylearn2Test/stackedAutoencoderRL/Data/resultData/svd l21decode.csv", l1decode, fmt="%10.2f", delimiter=",")

print "SVD"
U, D, Vt = np.linalg.svd(l2encode, full_matrices=False)
print U.shape, D.shape, Vt.shape, D
l2encode_r = np.dot(np.dot(U, np.diag(D)), Vt)
print(np.std(l2encode), np.std(l2encode_r), np.std(l2encode - l2encode_r))

#D_a = np.zeros(6)
#for i in xrange(D.size-1):
#    D_a[i] = D[i]
#l2encode_r = np.dot(np.dot(U, np.diag(D_a)), Vt)
#print l2encode
#print l2encode_r
#print(np.std(l2encode), np.std(l2encode_r), np.std(l2encode - l2encode_r))
print Vt
V = Vt.transpose()
print V
#l2encode_r = np.dot(np.diag(D), V)
#print l2encode_r

#basis decode
maxV = V.max()
minV = V.min()
maxv = 58.39  # rescale to maxv
minv = 2.64   # rescale to minv

print maxV, minV, maxv, minv

#rescaledV = (maxv-minv)*((V-minV)/(maxV-minV)) + minv
rescaledV = V*11
print "------ Rescale from (", minV, ",", maxV, ") to (", minv, ",", maxv, ") -------"
print rescaledV


V2decode = l2DecodeFunction(rescaledV)
V2decodeFlt = numpy.around(V2decode, decimals=2)
print "V2 decode"
print V2decodeFlt

V1decode = l1DecodeFunction(V2decode)
V1decodeFlt = numpy.around(V1decode, decimals=2)
print "---------------------- V1 decode ----------------------"
print V1decodeFlt
print "-------------------------------------------------------"
