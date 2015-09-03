from pylearn2.config import yaml_parse
import numpy
from numpy import genfromtxt

"""
RLData = genfromtxt('RLData.csv', delimiter=';', skip_header=1)
print RLData


shapeRLData = RLData.shape
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

    numpy.savetxt("The test "+str(i)+"th Column RLData.csv", zerosArray, delimiter=" , ")
"""

#"""
############ layer 1
layer1_yaml = open('dae_l1.yaml', 'r').read()
hyper_params_l1 = {'train_stop': 500000,
                   'batch_size': 10,
                   'monitoring_batches': 5,
                   'nhid': 7,  # 20, # 12,
                   'max_epochs': 50,
                   'save_path': '.'}
layer1_yaml = layer1_yaml % (hyper_params_l1)
#print layer1_yaml

#train = yaml_parse.load(layer1_yaml)
#train.main_loop()

#"""
#"""
######### layer 2
layer2_yaml = open('dae_l2.yaml', 'r').read()
hyper_params_l2 = {'train_stop': 500000,
                   'batch_size': 500000,
                   'monitoring_batches': 5,
                   'nvis': hyper_params_l1['nhid'],
                   'nhid': 14,  # 20,  # 6,  # 15,
                   'save_path': '.'}
layer2_yaml = layer2_yaml % (hyper_params_l2)
print layer2_yaml

train = yaml_parse.load(layer2_yaml)
train.main_loop()
#"""

"""
############ Separate layer 2
layer2Sep_yaml = open('daeSep_l2.yaml', 'r').read()
hyper_params_l2Sep = {'train_stop': 500000,
                      'batch_size': 1000,
                      'monitoring_batches': 5,
                      'nhid': 6,  # 20, # 12,
                      'max_epochs': 3,
                      'save_path': '.'}
layer2Sep_yaml = layer2Sep_yaml % (hyper_params_l2Sep)
print layer2Sep_yaml

train = yaml_parse.load(layer2Sep_yaml)
train.main_loop()
#"""
"""
######### layer 3
layer3_yaml = open('dae_l3.yaml', 'r').read()
hyper_params_l3 = {'train_stop': 50000,
                   'batch_size': 8,
                   'monitoring_batches': 5,
                   'nvis': hyper_params_l2['nhid'],
                   'nhid': 8,  # 6,  # 15,
                   'max_epochs': 20,
                   'save_path': '.'}
layer3_yaml = layer3_yaml % (hyper_params_l3)
print layer3_yaml

train = yaml_parse.load(layer3_yaml)
train.main_loop()
#"""

"""
######### mlp
mlp_yaml = open('dae_mlp.yaml', 'r').read()
hyper_params_mlp = {'train_stop': 50000,
                    #'valid_stop' : 60000,
                    'batch_size': 100,
                    'max_epochs': 50,
                    'save_path': '.'}
mlp_yaml = mlp_yaml % (hyper_params_mlp)
print mlp_yaml

train = yaml_parse.load(mlp_yaml)
train.main_loop()
"""
