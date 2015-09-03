#import cPickle
from pylearn2.utils import serial

model = serial.load('./dae_l2.pkl')
params = model.get_params()
for param in params:
    print param
    print param.get_value()
