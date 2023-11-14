from libraries import *
import utilities as ut
from models.modules.GeM import AdaptiveGeneralizedMeanPooling
from models.modules.PCA import LearnablePCA
from models.modules.SAFA import SpatialAware


class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(*list(resnet101(weights=ResNet101_Weights.DEFAULT).children())[:-2])

    def forward(self, x):
        return self.resnet(x) #(B, 2048, H/32, W/32)


class ResNet101_GEM(nn.Module):
    def __init__(self, num_comp : int = 1024, *args, **kargs):
        super().__init__()
        
        self.cnn = ResNet101()
        self.gem = AdaptiveGeneralizedMeanPooling(norm=True)
        self.pca = LearnablePCA(2048, num_comp)
         
    def forward(self, x):
        x = self.gem(self.cnn(x)) #(B , channels)
        x = self.pca(x)
        
        return f.normalize(x, p=2, dim=1)


class ResNet101_GEM_wo_PCA(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()
        
        self.cnn = ResNet101()
        self.gem = AdaptiveGeneralizedMeanPooling(norm=True)
         
    def forward(self, x):
        x = self.gem(self.cnn(x)) #(B , channels)
        
        return f.normalize(x, p=2, dim=1)
    
    
class ResNet101_SAFA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, *args, **kargs):
        super().__init__()
        
        self.cnn = ResNet101()
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
         
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)

        return f.normalize(x, p=2, dim=1)
