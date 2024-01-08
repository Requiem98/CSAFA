from libraries import *
import utilities as ut
from models.modules.PCA import *
from models.modules.SAFA import *

class VGG16_base(nn.Module):
    def __init__(self, out_dim : int = 512, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Linear(512, out_dim)
         
    def forward(self, x):
        x = self.maxpool(self.cnn(x)).reshape(-1, 512) #(B , channels)
        x = self.linear(x)
        
        return f.normalize(x, p=2, dim=1)
    
    
class VGG16_SAFA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
         
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)

        return f.normalize(x, p=2, dim=1)

        
class VGG16_SAFA_Linear(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.linear = nn.Linear(512*dimension, out_dim) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.linear(x)
        return f.normalize(x, p=2, dim=1)
    
    
class VGG16_CSAFA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, norm:bool, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = SpatialAware_v3((img_size[0] // 32) * (img_size[1] // 32), dimension, norm=norm) #2*8 if downsize = 2, 4*16 if downsize = 1
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)

        return f.normalize(x, p=2, dim=1)





