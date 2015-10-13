#All code based on NLTK documentation

# K-Means clusterer
# example from figure 14.9, page 517, Manning and Schutze
from __future__ import division
from pylearn2.utils import serial
import theano
from numpy import genfromtxt
import numpy
from nltk import cluster
from nltk.cluster import euclidean_distance
from numpy import array

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


vectors = []
for r in range(100):
    vectors.append(array(l2encode[r, :]))
#print vectors
print "K-Means clustering"
#vectors = [array(f) for f in [[2, 1], [1, 3], [4, 7], [6, 7]]]
#means = [[4, 3], [5, 5]]

#clusterer = cluster.KMeansClusterer(2, euclidean_distance, initial_means=means)
#clusters = clusterer.cluster(vectors, True, trace=True)

#print 'Clustered:', vectors
#print 'As:', clusters
#print 'Means:', clusterer.means()
#print

#vectors = []
#vectors = [array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

# test k-means using the euclidean distance metric, 2 means and repeat
# clustering 10 times with random seeds

clusterer = cluster.KMeansClusterer(2, euclidean_distance, avoid_empty_clusters=True)
clusters = clusterer.cluster(vectors, True)

#print 'Clustered:', vectors
print 'As:'  # clusters
i = 2
for clst in clusters:
    print i, clst
    i = i+1
print 'Means:', clusterer.means()
#print vectors

# classify a new vector
#vector = array([3, 3])
#print 'classify(%s):' % vector,
#print clusterer.classify(vector)
#print

#"""
print "GAAC Clustering"
# use a set of tokens with 2D indices
#vectors = [array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

# test the GAAC clusterer with 4 clusters
clusterer = cluster.GAAClusterer(2)
clusters = clusterer.cluster(vectors, True)

print 'Clusterer:', clusterer
#print 'Clustered:', vectors
print 'As:'  # clusters
i = 2
for clst in clusters:
    print i, clst
    i = i+1
# show the dendrogram
#print "The dendogram"
#clusterer.dendrogram().show()

# classify a new vector
#print "Classify the vector [3,3]"
#vector = array([3, 3])
#print 'classify(%s):' % vector,
#print clusterer.classify(vector)
#"""
