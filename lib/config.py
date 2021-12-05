"""
Config File from : https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi

"""
import os
import yaml
import argparse
import copy
import numpy as nps
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
import sys
import ref
from datetime import datetime
import utils.logger as logger
from tensorboardX import SummaryWriter
from pprint import pprint
import numpy as np
def get_default_config_pytorch():
    config = edict()
    config.exp_id = ' '          # Experiment ID
    config.task = ''             # Specify the Task 'rot | trans | trans_rot'
    config.flag= True            # Set Flag to True for training
    config.gpu = -1              # -1| no GPU , 0| GPU In USe
    config.threads_num = 12         # number of threads used for dataloader
    config.load_model = ''          # path to a previously trained model
    config.bbox='yolo'
    return config

def get_default_dataset_config():
    """
    Default configuration for Dataset
    """
    config = edict()            
    config.name = 'YCB'  # YCB|Linemode
    return config

def get_default_dataiter_config():
    """
    
    """
    config = edict()
    config.out_res = 64
    config.tp = 'gaussian' #|gaussian|uniform
    return config

def get_default_augment_config():
    """
    Default Dynamic Zoom-In Settings
    """
    config = edict()
    config.pad_ratio = 1.5
    config.scale_ratio = 0.25
    config.shift_ratio = 0.25
    return config

def get_default_train_config():
    config = edict()
    config.begin_epoch = 1
    config.end_epoch = 160
    config.test_interval = 10
    config.train_batch_size = 6
    config.lr_backbone = 1e-4
    config.lr_rot_head = 1e-4
    config.lr_trans_head = 1e-4
    config.lr_epoch_step = [50, 100, 150]
    config.lr_factor = 0.1
    config.optimizer_name = 'RMSProp'  # 'Adam' | 'Sgd' | 'Moment' | 'RMSProp'
    config.momentum = 0.0
    config.weightDecay = 0.0
    config.alpha = 0.99
    config.epsilon = 1e-8
    config.rot_rep='allo' # ego| allo egocentric or allocentric rotation representation
    config.disp_interval= 200
    return config

def get_default_loss_config():
    config = edict()
    config.rot_loss_type = 'quatloss'
    config.rot_loss_weight = 1
    config.trans_loss_type = 'L2'
    config.trans_loss_weight = 1
    return config

def get_default_network_config():
    config = edict()
    # ------ backbone -------- #
    config.arch = 'resnet'
    config.back_freeze = False
    config.back_input_channel = 3 # # channels of backbone's input
    config.back_layers_num = 34   # 18 | 34 | 50 | 101 | 152
    config.back_filters_num = 256  # number of filters for each layer
    # ------ rotation head -------- #
    config.rot_representation='quat'
    config.rot_head_freeze = False
    config.rot_layers_num = 3
    config.rot_filters_num = 256  # number of filters for each layer
    config.rot_conv_kernel_size = 3  # kernel size for hidden layers
    config.rot_output_conv_kernel_size = 1  # kernel size for output layer
    config.rot_output_channels = 4  # # channels of output, 3-channels coordinates map and 1-channel for confidence map
    # ------ translation head -------- #
    config.trans_head_freeze = False
    config.trans_layers_num = 3
    config.trans_filters_num = 256
    config.trans_conv_kernel_size = 3
    config.trans_output_channels = 3
    return config



def get_base_config():
    base_config = edict()
    base_config.pytorch = get_default_config_pytorch()
    base_config.dataset = get_default_dataset_config()
    base_config.dataiter = get_default_dataiter_config()
    base_config.train = get_default_train_config()
    base_config.augment = get_default_augment_config()
    base_config.network = get_default_network_config()
    base_config.loss = get_default_loss_config()
    return base_config

def update_config_from_file(_config, config_file, check_necessity=True):
    config = copy.deepcopy(_config)
    with open(config_file) as f:
        #
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        #exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            if isinstance(vv, list) and not isinstance(vv[0], str):
                                config[k][vk] = np.asarray(vv)
                            else:
                                config[k][vk] = vv
                        else:
                            if check_necessity:
                                raise ValueError("{}.{} not exist in config".format(k, vk))
                else:
                    raise ValueError("{} is not dict type".format(v))
            else:
                if check_necessity:
                    raise ValueError("{} not exist in config".format(k))
    return config

class config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='pose experiment')
        self.parser.add_argument('--cfg', type=str,default='../Config-Files/6D-Head/config_rot.yaml', help='path/to/configure_file')
        self.parser.add_argument('--load_model', type=str, default='',help='path/to/model, requird when resume/test')
        self.parser.add_argument('--debug', action='store_true', help='')
        self.parser.add_argument('--test', action='store_true', help='')

    def parse(self):
        config = get_base_config()                  # get default arguments
        args, rest = self.parser.parse_known_args() # get arguments from command line
        for k, v in vars(args).items():
            config.pytorch[k] = v 
        config_file = config.pytorch.cfg
        config = update_config_from_file(config, config_file, check_necessity=False) # update arguments from config file
        # complement config regarding dataset
        # automatically correct config
        if config.network.back_freeze == True:
            config.loss.backbone_loss_weight = 0
        if config.network.rot_head_freeze == True:
            config.loss.rot_loss_weight = 0
        if config.network.trans_head_freeze == True:
            config.loss.trans_loss_weight = 0

        if not config.pytorch.flag:
            config.pytorch.exp_id = config.pytorch.exp_id + 'TEST'

        # complement config regarding paths
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        # save path
        config.pytorch['save_path'] = os.path.join(ref.exp_dir, config.pytorch.exp_id, now)
        if not os.path.exists(config.pytorch.save_path):
            os.makedirs(config.pytorch.save_path, exist_ok=True)
        # debug path
        # tensorboard path
        config.pytorch['tensorboard'] = os.path.join(config.pytorch.save_path, 'tensorboard')
        if not os.path.exists(config.pytorch.tensorboard):
            os.makedirs(config.pytorch.tensorboard)
        config.writer = SummaryWriter(config.pytorch.tensorboard)
        # logger path
        logger.set_logger_dir(config.pytorch.save_path, action='k')

        pprint(config)
        # copy and save current config file
        os.system('cp {} {}'.format(config_file, os.path.join(config.pytorch.save_path, 'config_copy.yaml')))
        # save all config infos
        args = dict((name, getattr(config, name)) for name in dir(config) if not name.startswith('_'))
        refs = dict((name, getattr(ref, name)) for name in dir(ref) if not name.startswith('_'))
        file_name = os.path.join(config.pytorch.save_path, 'config.txt')
        with open(file_name, 'wt') as cfg_file:
            cfg_file.write('==> Cmd:\n')
            cfg_file.write(str(sys.argv))
            cfg_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                cfg_file.write('  %s: %s\n' % (str(k), str(v)))
            cfg_file.write('==> Ref:\n')
            for k, v in sorted(refs.items()):
                cfg_file.write('  %s: %s\n' % (str(k), str(v)))

        return config
