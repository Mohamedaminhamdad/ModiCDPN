import torch 
import torch.nn as nn

class QuatLoss(nn.Module):
    def __init__(self):
        """
        Quatloss implemented and motivated from Silhonet: https://github.com/gidobot/SilhoNet
        """
        super(QuatLoss,self).__init__()
    def forward(self,predictions,labels,batch_size):
        product = torch.mul(predictions, labels)
        internal_dot_products = torch.sum(product, [1])
        product=torch.abs(internal_dot_products)
        logcost = torch.log(1e-4+1 - product)
        return torch.sum(logcost)/batch_size
class FancyQuatLoss(nn.Module):
    def __init__(self):
        """
        Our own Implementation of Loss Function
        """
        super(FancyQuatLoss,self).__init__()
    def forward(self,predictions,labels,batch_size):
        product = torch.mul(predictions, labels)
        internal_dot_products = torch.sum(product, [1])
        product= torch.abs(internal_dot_products)
        prod_sq=torch.square(product)
        logcost = -torch.log(1e-4  + prod_sq)
        return torch.sum(logcost)/batch_size
class GEodistance(nn.Module):
    def __init__(self):
        """
        Implementation of geodistance Loss as mentioned in: https://arxiv.org/pdf/2001.08942v1.pdf
        """
        self.eps = 1e-6
        super(GEodistance,self).__init__()
    def forward(self,predictions,labels,batch_size):
        product = torch.bmm(predictions, labels.transpose(1,2))
        Rt = torch.sum(product[:,torch.eye(3).byte()],1)
        theta = torch.clamp(0.5*(Rt-1), -1+self.eps, 1-self.eps)
        return torch.mean(torch.acos(theta))
