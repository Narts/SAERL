from pylearn2.utils import serial
import theano
from numpy import genfromtxt
import numpy

"""dae_mlp predict"""

"""
dae_mlp_path = 'dae_mlp.pkl'
dae_mlp = serial.load(dae_mlp_path)
print dae_mlp


MLPInput = dae_mlp.get_input_space().make_theano_batch()
MLPOutput = dae_mlp.fprop(MLPInput)

MLPPredictFunc = theano.function([MLPInput], MLPOutput)

RLData = genfromtxt('Seperate action RLDataBlk noRw.csv', delimiter=',', skip_header=0)

mlpPredict = MLPPredictFunc(RLData)
mlpPredictFlt = numpy.around(mlpPredict, decimals=2)

print "mlpPredict"
print mlpPredictFlt

numpy.savetxt("AE OverCp 8ColNoRw linear MLP.csv", mlpPredictFlt, fmt="%10.2f", delimiter=",")
"""

"""deepAE predict"""

deepAE_path = 'deepAE.pkl'
deepAE = serial.load(deepAE_path)
print deepAE

deepAE_Input = deepAE.get_input_space().make_theano_batch()
deepAE_encode = deepAE.encode(deepAE_Input)
deepAE_decode = deepAE.decode(deepAE_encode)

deepAE_EncodeFunc = theano.function([deepAE_Input], deepAE_encode)
deepAE_DecodeFunc = theano.function([deepAE_encode], deepAE_decode)

RLData = genfromtxt('Seperate action RLDataBlk noRw.csv', delimiter=',', skip_header=0)
print "RLData"
print RLData

deepAEEncode = deepAE_EncodeFunc(RLData)
deepAEEncode_Flt = numpy.around(deepAEEncode, decimals=2)

print "deepAEEncode"
print deepAEEncode_Flt

numpy.savetxt("moreDataDAE OverCp 8ColNoRwBLK softPluslinearEncode.csv", deepAEEncode, fmt="%10.2f", delimiter=",")

deepAEDecode = deepAE_DecodeFunc(deepAEEncode)
deepAEDecode_Flt = numpy.around(deepAEDecode, decimals=2)

print "deepAEDecode"
print deepAEDecode_Flt

numpy.savetxt("moreDataDAE OverCp 8ColNoRwBLK softPluslinearDecode.csv", deepAEDecode, fmt="%10.2f", delimiter=",")
