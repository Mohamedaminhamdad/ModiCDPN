import os
from easydict import EasyDict as edict

import torch
import torch.nn as nn

class modi_CDPN(nn.Module):
    def __init__(self, backbone, rot_head_net, trans_head_net):
        super(modi_CDPN, self).__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net
        self.trans_head_net = trans_head_net

    def forward(self, x):                     
        features = self.backbone(x)           
        out_poses = self.rot_head_net(features) 
        trans = self.trans_head_net(features)
        return  out_poses, trans
