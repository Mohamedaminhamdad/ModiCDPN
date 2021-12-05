import torch.nn as nn
import torch
import torch.nn.functional as f
import numpy as np
def R_from_6d(Output_network):
    m1 = Output_network[:,0:3] # Network gives an Output of 6 degree of freedom. three of them are useless. 
    m2 = Output_network[:,3:6]
    """
    IN order to recover the rotaiton matric from the 6dof representation the Gram-Schmit process is used: normalization followed
    by cross products
    See: https://arxiv.org/abs/1812.07035
    """    
    x = norm(m1)
    z = cross_product(x,m2) 
    z = norm(z)
    y = cross_product(z,x)   
    matrix = torch.cat((x.view(-1,3,1),y.view(-1,3,1),z.view(-1,3,1)), 2) # Rotation Matrix lying in the SO(3) 
    return matrix
def norm(x):
    x_abs = torch.sqrt(x.pow(2).sum(1))
    if torch.cuda.is_available():
        x_abs = torch.max(x_abs, torch.autograd.Variable(torch.FloatTensor([1e-8])).cuda())
    else: 
        x_abs = torch.max(x_abs, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    x_abs = x_abs.view(x.shape[0],1).expand(x.shape[0],x.shape[1])
    x_norm = x/x_abs
    return x_norm

def cross_product( x, y):
    p1 = x[:,1]*y[:,2] - x[:,2]*y[:,1]
    p2 = x[:,2]*y[:,0] - x[:,0]*y[:,2]
    p3 = x[:,0]*y[:,1] - x[:,1]*y[:,0]
    cross= torch.cat((p1.view(x.shape[0],1), p2.view(x.shape[0],1), p3.view(x.shape[0],1)),1)    
    return cross

class RotHeadNet(nn.Module):
    def __init__(self, in_channels, num_layers=3, num_filters=256, kernel_size=3, output_dim=4, freeze=False,
                 with_bias_end=True,rotation_rep='quat'):
        super(RotHeadNet, self).__init__()

        self.freeze = freeze
        padding = 1
        self.representation=rotation_rep
        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.LeakyReLU(inplace=True))

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(256 * 8 * 8, 4096))
        self.linears.append(nn.LeakyReLU(inplace=True))
        self.linears.append(nn.Linear(4096, 4096))
        self.linears.append(nn.LeakyReLU(inplace=True))
        if self.representation=='quat':
            self.linears.append(nn.Linear(4096, 4))
        else: 
            self.linears.append(nn.Linear(4096, 6))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end and (m.bias is not None):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        if self.representation=='quat': 
            if self.freeze:
                with torch.no_grad():
                    for i, l in enumerate(self.features):
                        x = l(x)
                    x = x.view(-1, 256*8*8)
                    for i, l in enumerate(self.linears):
                        x = l(x)
                    x= f.normalize(x,dim=1,p=2)
                    return x.detach()
            else:
                for i, l in enumerate(self.features):
                    x = l(x)
                x = x.view(-1, 256*8*8)
                for i, l in enumerate(self.linears):
                    x = l(x)
                x= f.normalize(x,dim=1,p=2) # normalization layer to output normalized quaternions
                return  x
        else: 
            if self.freeze:
                with torch.no_grad():
                    for i, l in enumerate(self.features):
                        x = l(x)
                    x = x.view(-1, 256*8*8)
                    for i, l in enumerate(self.linears):
                        x = l(x)
                    x= f.normalize(x,dim=1,p=2)
                    out_rotation_matrix=R_from_6d(x) # 6D to R
                    return out_rotation_matrix.detach()
            else:
                for i, l in enumerate(self.features):
                    x = l(x)
                x = x.view(-1, 256*8*8)
                for i, l in enumerate(self.linears):
                    x = l(x)
                out_rotation_matrix=R_from_6d(x) # 6D to R
                return  out_rotation_matrix
