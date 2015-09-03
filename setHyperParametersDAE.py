from pylearn2.config import yaml_parse
from pylearn2.utils import serial
import pickle

#dae_list = []
#"""
#l1_path = './dae_l1.pkl'
l1_path = './dae_l1.pkl'
l1 = serial.load(l1_path)
#dae_list.append(l1)

#fp = open(l1_path)
#l1 = pickle.load(fp)
#fp.close()


l2_path = './dae_l2.pkl'
l2 = serial.load(l2_path)
#dae_list.append(l2)

#fp = open(l2_path)
#l2 = pickle.load(fp)
#fp.close()
#"""
"""
############ layer 1
layer1_yaml = open('dae_l1.yaml', 'r').read()
hyper_params_l1 = {'train_stop': 50000,
                   'batch_size': 100,
                   'monitoring_batches': 5,
                   'nhid': 20,  # 7,  # 12,
                   'max_epochs': 50,
                   'save_path': '.'}
layer1_yaml = layer1_yaml % (hyper_params_l1)
l1 = yaml_parse.load(layer1_yaml)



######### layer 2
layer2_yaml = open('dae_l2.yaml', 'r').read()
hyper_params_l2 = {'train_stop': 50000,
                   'batch_size': 100,
                   'monitoring_batches': 5,
                   'nvis': hyper_params_l1['nhid'],
                   'nhid': 8,  # 8,  # 6,  # 15,
                   'max_epochs': 50,
                   'save_path': '.'}
layer2_yaml = layer2_yaml % (hyper_params_l2)
l2 = yaml_parse.load(layer2_yaml)
"""

dae_list = [l1_path, l2_path]
#autoencoders = (dae_list)


deepAE_yaml = open('deepAE.yaml', 'r').read()
hyper_params_deepAE = {'train_stop': 500000,
                       'autoencoders': dae_list,
                       'batch_size': 10,
                       'monitoring_batches': 5,
                       'max_epochs': 50,
                       'save_path': '.'}
deepAE_yaml = deepAE_yaml % (hyper_params_deepAE)
print deepAE_yaml

train = yaml_parse.load(deepAE_yaml)
train.main_loop()
