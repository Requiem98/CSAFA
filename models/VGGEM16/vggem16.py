from libraries import *
import utilities as ut
from models.modules.GeM import GeneralizedMeanPooling, AdaptiveGeneralizedMeanPooling
from models.modules.PCA import LearnablePCA
from models.modules.cbam import CBAM
from models.modules.SAFA import SpatialAware

class N_conv(nn.Module):

    def __init__(self,in_channels,out_channels,N = 2, apply_gem = True):
        super().__init__()
        
        model = []
        model.append(nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding=(1,1)))
        model.append(nn.ReLU(True))
        for i in range(N-1):
            model.append(nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=(1,1)))
            model.append(nn.ReLU(True))
        
        if(apply_gem):
            model.append(GeneralizedMeanPooling(kernel_size=(2,2),stride=(2,2)))
        self.conv = nn.Sequential(*model)
    def forward(self,x):
        return self.conv(x)
    
    

class VGGEM16(nn.Module):

    def __init__(self,in_channels=3, init_weights=True):
        super().__init__()
        self.conv1 = N_conv(3,64)
        self.conv2 = N_conv(64,128)
        self.conv3 = N_conv(128,256,N=3)
        self.conv4 = N_conv(256,512,N=3)
        self.conv5 = N_conv(512,512,N=3, apply_gem = False)
        
        if init_weights:
            self._initialize_weights()
            
    def _initialize_weights(self):
        ckp = torch.load("cache/checkpoints/vgg16-397923af.pth")
        idx = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        i = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
            
                state_dict = OrderedDict({"weight" : ckp[f"features.{idx[i]}.weight"], "bias" : ckp[f"features.{idx[i]}.bias"]})
                m.load_state_dict(state_dict)
                i +=1
                
                
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x
    
    

class VGGEM16_GEM(nn.Module):

    def __init__(self,in_channels=3, init_weights=True, *args, **kargs):
        super().__init__()
        self.cnn = VGGEM16(in_channels, init_weights)
        
        self.final_gem = AdaptiveGeneralizedMeanPooling(norm = True)
        self.pca = LearnablePCA(512)
                
                
    def forward(self,x):
        x = self.cnn(x)

        x = self.pca(self.final_gem(x))
        
        return f.normalize(x, p=2, dim=1)
    
    
class VGGEM16_SAFA(nn.Module):

    def __init__(self,in_channels=3, init_weights=True, *args, **kargs):
        super().__init__()
        self.cnn = VGGEM16(in_channels, init_weights)
        
        self.safa = SpatialAware(4*16)
        self.pca = LearnablePCA(512)
                
                
    def forward(self,x):
        x = self.cnn(x)

        x = self.pca(self.safa(x))
        
        return f.normalize(x, p=2, dim=1)
