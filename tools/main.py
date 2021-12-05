# encoding: utf-8
"""
6DoF pose estimation Framework based on CDPN: https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
Initial Code motivated by https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi and modified by Mohamed 
"""
import os
import numpy as np
import random
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import cv2
cv2.ocl.setUseOpenCL(False)
import _init_paths
import utils.logger as logger
from model import build_model, save_model
from datasets.data import DATASET
from train import train
from test import test
from config import config
from tqdm import tqdm 
from Loss import QuatLoss, FancyQuatLoss,GEodistance
import ref
from torchinfo import summary
classes_YCB = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
classes_Line = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
classes={'YCB':classes_YCB,
         'Linemode':classes_Line}
def model_info(points):
    """
    load model info for Linemode Dataset
    """
    infos = {}
    extent= 2 * np.max(np.absolute(points), axis=0)
    infos['diameter'] = np.sqrt(np.sum(extent * extent))
    infos['min_x'],infos['min_y'],infos['min_z']=np.min(points,axis=0)
    infos['max_x'],infos['max_y'],infos['max_z']=np.min(points,axis=0)
    return infos
def main():
    arg = config().parse() # Load Configurations
    network, optimizer = build_model(arg) # Build Model 
    criterions = {'L2': torch.nn.MSELoss(),
                  'quatloss': QuatLoss(),
                  'fancyquatloss':FancyQuatLoss(),
                  'acos':GEodistance()}
    batch_size = 6
    #summary(network, input_size=(1, 3, 240, 240))

    if arg.pytorch.gpu > -1:
        """
        If GPU is used pass Network to cuda() and the loss function
        """
        logger.info('GPU{} is used'.format(arg.pytorch.gpu))
        network = network.cuda(arg.pytorch.gpu)
        for k in criterions.keys():
            criterions[k] = criterions[k].cuda(arg.pytorch.gpu)
    else: 
        logger.info('GPU Is not in Use{}'.format(arg.pytorch.gpu))
        network = network
        for k in criterions.keys():
            criterions[k] = criterions[k]

    def _worker_init_fn():
        """
        Configuration for Dataloader worker
        """
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
    # Initilize Test Dataloader
    test_loader = torch.utils.data.DataLoader(
        DATASET(arg, 'test',arg.network.rot_representation),
        batch_size=arg.train.train_batch_size,
        shuffle=False,
        num_workers=int(arg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn()
    )
    # Initilize Train Dataloader
    train_loader = torch.utils.data.DataLoader(
        DATASET(arg, 'train',arg.network.rot_representation),
        batch_size=arg.train.train_batch_size,
        shuffle=True,
        num_workers=int(arg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn()
    )

    for epoch in range(arg.train.begin_epoch, arg.train.end_epoch + 1):
        if arg.pytorch.flag==True: # True for training
            log_dict_train, _ = train(epoch, arg, train_loader, network, criterions, optimizer)
            for k, v in log_dict_train.items():
                logger.info('{} {:8f} | '.format(k, v)) 
            if epoch % arg.train.test_interval == 0:
                save_model(os.path.join(arg.pytorch.save_path, 'model_{}.checkpoint'.format(epoch)), network)  # optimizer
                test(epoch, arg, test_loader, network, criterions)
                logger.info('\n')
            if epoch in arg.train.lr_epoch_step:
                """
                Change Learning rate at predifiened epochs
                """
                if optimizer is not None:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= arg.train.lr_factor
                        logger.info("drop lr to {}".format(param_group['lr']))

            #torch.save(network.cpu(), os.path.join(arg.pytorch.save_path, 'model_cpu.pth'))
        else:  # testing 
            obj_vtx = {}
            model_info_={}
            logger.info('load 3d object models...')
            for obj in tqdm(classes[arg.dataset.name]):
                obj_vtx[obj] = np.loadtxt(os.path.join(ref.model_dir,obj,'points.xyz')) # load 3D models 
                model_info_[obj] = model_info(obj_vtx[obj])
            test(epoch, arg, test_loader, network, classes[arg.dataset.name],obj_vtx,model_info_)
if __name__ == '__main__':
    main()
