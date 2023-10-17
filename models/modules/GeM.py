from libraries import *
import utilities as ut



class GeneralizedMeanPooling(nn.Module):

    def __init__(self, kernel_size, stride, padding=0, eps=1e-6, norm = False):
        super(GeneralizedMeanPooling, self).__init__()
        
        self.p = Parameter(torch.ones(1))
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
        
        self.p = Parameter(torch.ones(1))
        self.output_size = output_size
        self.eps = eps
        self.norm = norm

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = f.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p).squeeze(3).squeeze(2)
        
        if(self.norm):
            x = f.normalize(x, p=2, dim=1)
            
        return x