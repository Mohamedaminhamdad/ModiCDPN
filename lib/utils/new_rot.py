def R_from_6d(Output_network):
    m1 = Output_network[:,0:3] # Network gives an Output of 6 degree of freedom. three of them are useless. 
    m2 = Output_network[:,3:6]
    """
    IN order to recover the rotaiton matric from the 6dof representation the Gram-Schmit process is used normalization follwed
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
    x_abs = torch.max(x_abs, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    x_abs = x_abs.view(x.shape[0],1).expand(x.shape[0],v.shape[1])
    x_norm = x/x_abs
    return x_norm

def cross_product( x, y):
    p1 = x[:,1]*y[:,2] - x[:,2]*y[:,1]
    p2 = x[:,2]*y[:,0] - x[:,0]*y[:,2]
    p3 = x[:,0]*y[:,1] - x[:,1]*y[:,0]
    cross= torch.cat((p1.view(u.shape[0],1), p2.view(u.shape[0],1), p3.view(u.shape[0],1)),1)    
    return cross