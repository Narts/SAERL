from pylearn2.config import yaml_parse

mlp_yaml = open('dae_mlp.yaml', 'r').read()
hyper_params_mlp = {'train_stop': 50000,
                    #'valid_stop': 60000,
                    'batch_size': 100,
                    'max_epochs': 50,
                    'save_path': '.'}
mlp_yaml = mlp_yaml % (hyper_params_mlp)
print mlp_yaml

train = yaml_parse.load(mlp_yaml)
train.main_loop()
