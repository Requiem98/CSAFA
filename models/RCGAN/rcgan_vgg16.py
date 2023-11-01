from libraries import *
import utilities as ut
from models.GANs.networks import UnetGeneratorSkip
from models.modules.SAFA import SpatialAware


class feature_extractor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UnetGeneratorSkip()
        self.model.load_state_dict(torch.load("cache/checkpoints/unet_generator_skip.pth"))
        
    def forward(self, x):
        
        return self.model(x)[1]
    
    

class RCGAN_VGG16_safa(nn.Module):
    def __init__(self, img_size:tuple, dimension: int, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        
        self.final_cnn = nn.Sequential(nn.Conv2d(512,256,kernel_size=1),nn.ReLU(True))
        nn.init.trunc_normal_(self.final_cnn[0].weight , mean=0.0, std=0.005)
        
        self.fe = feature_extractor()
        self.fe.freeze()
        
        self.sa1 = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension)
        self.sa2 = SpatialAware((img_size[0] // 8) * (img_size[1] // 8), dimension)
         

    def forward(self, input: tuple):
        
        x, y = input
        
        
        x = self.final_cnn (self.cnn(x)) 
        x = f.normalize(self.sa1(x), p=2, dim=1)
        
        y = self.fe(y)
        y = f.normalize(self.sa2(y), p=2, dim=1)
        
        
        return x,y




