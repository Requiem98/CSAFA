from libraries import *
import utilities as ut



class GeneralizedMeanPooling(nn.Module):

    def __init__(self, kernel_size, stride, padding=0, eps=1e-6, norm = False):
        super().__init__()
        
        self.p = nn.Parameter(torch.ones(1))
        self.eps = eps
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        self.norm = norm

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = self.avg_pool(x).pow(1. / self.p)
        
        if(self.norm):
            x = f.normalize(x, p=2, dim=1)
        
        return x
    
    
class AdaptiveGeneralizedMeanPooling(nn.Module):

    def __init__(self, output_size=1, eps=1e-6, norm = False):
        super().__init__()
        
        self.p = nn.Parameter(torch.ones(1))
        self.output_size = output_size
        self.eps = eps
        self.norm = norm

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = f.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p).squeeze(3).squeeze(2)
        
        if(self.norm):
            x = f.normalize(x, p=2, dim=1)
            
        return x
    
    
    
class ChannelGeneralizedMeanPooling(nn.Module):

    def __init__(self, eps=1e-6, dimension:int = 8, dim_first:bool = True):
        super().__init__()
        
        self.p = nn.Parameter(torch.ones(dimension))
        self.eps = eps
        self.dim_first = dim_first
        self.dimension = dimension
        
    def broadcast_p(self, A, B, C, D, E):
        if(self.dim_first):
            return self.p.broadcast_to((A, B)).unsqueeze(2).broadcast_to((A,B,C)).unsqueeze(3).broadcast_to((A,B,C,D)).unsqueeze(4).broadcast_to((A,B,C,D,E))
        else:
            return self.p.broadcast_to((D, E)).unsqueeze(0).broadcast_to((C,D,E)).unsqueeze(0).broadcast_to((B,C,D,E)).unsqueeze(0).broadcast_to((A,B,C,D,E))

    def forward(self, x):
        
        if(self.dim_first):
            x = x.unsqueeze(1).repeat(1,self.dimension,1,1,1)
            x = x.clamp(min=self.eps).pow(self.broadcast_p(*x.shape))
            x = torch.mean(x, axis=2).reshape(x.shape[0],self.dimension, -1) #(B, D, H1 X H2)
            x = x.pow(1. / self.p.broadcast_to((x.shape[0], self.dimension)).unsqueeze(2).broadcast_to((x.shape[0],self.dimension,x.shape[2])))
        else:
            x = x.unsqueeze(-1).repeat(1,1,1,1,self.dimension)  
            x = x.clamp(min=self.eps).pow(self.broadcast_p(*x.shape))
            x = torch.mean(x, axis=1).reshape(x.shape[0], -1, self.dimension) #(B, H1 X H2, D)
            x = x.pow(1. / self.p.broadcast_to((x.shape[1], self.dimension)).unsqueeze(0).broadcast_to((x.shape[0],x.shape[1], self.dimension)))
        
        
        return x
    
    
