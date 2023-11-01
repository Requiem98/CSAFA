from libraries import *
import utilities as ut
from models.modules.GeM import AdaptiveGeneralizedMeanPooling
from models.modules.PCA import LearnablePCA
from models.modules.SAFA import SpatialAware


class VGG16_GEM(nn.Module):
    def __init__(self, num_comp : int = 512, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.gem = AdaptiveGeneralizedMeanPooling(norm=True)
        self.pca = LearnablePCA(num_comp)
         
    def forward(self, x):
        x = self.gem(self.cnn(x)) #(B , channels)
        x = self.pca(x)
        
        return f.normalize(x, p=2, dim=1)
    
    
    
class VGG16_SAFA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
         
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)

        return f.normalize(x, p=2, dim=1)