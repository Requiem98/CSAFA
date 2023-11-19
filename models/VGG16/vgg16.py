from libraries import *
import utilities as ut
from models.modules.GeM import AdaptiveGeneralizedMeanPooling
from models.modules.PCA import LearnablePCA
from models.modules.SAFA import SpatialAware, GemSpatialAware

class N_conv(nn.Module):

    def __init__(self,in_channels,out_channels,N = 2, padding:tuple = (1,1)):
        super().__init__()
        
        model = []
        model.append(nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding=padding, padding_mode = "circular"))
        model.append(nn.ReLU(True))
        for i in range(N-1):
            model.append(nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=padding, padding_mode = "circular"))
            model.append(nn.ReLU(True))
        
        
        model.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.conv = nn.Sequential(*model)
    def forward(self,x):
        return self.conv(x)
    
    

class VGG16_cir(nn.Module):

    def __init__(self,in_channels=3, padding:tuple = (1,1), init_weights=True):
        super().__init__()
        self.conv1 = N_conv(3,64, padding=padding)
        self.conv2 = N_conv(64,128, padding=padding)
        self.conv3 = N_conv(128,256,N=3, padding=padding)
        self.conv4 = N_conv(256,512,N=3, padding=padding)
        self.conv5 = N_conv(512,512,N=3, padding=padding)
        
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

class VGG16_GEM(nn.Module):
    def __init__(self, num_comp : int = 512, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.gem = AdaptiveGeneralizedMeanPooling(norm=True)
        self.pca = LearnablePCA(512, num_comp)
         
    def forward(self, x):
        x = self.gem(self.cnn(x)) #(B , channels)
        x = self.pca(x)
        
        return f.normalize(x, p=2, dim=1)


class VGG16_GEM_wo_PCA(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.gem = AdaptiveGeneralizedMeanPooling(norm=True)
         
    def forward(self, x):
        x = self.gem(self.cnn(x)) #(B , channels)
        
        return f.normalize(x, p=2, dim=1)
    
    
class VGG16_SAFA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
         
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)

        return f.normalize(x, p=2, dim=1)

class VGG16_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)
        return f.normalize(x, p=2, dim=1)
    
    
class VGG16_GEM_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = GemSpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)

        return f.normalize(x, p=2, dim=1)

class VGG16_cir_GEM_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, padding:tuple = (1,1), *args, **kargs):
        super().__init__()
        
        self.cnn = VGG16_cir(in_channels=3, padding=padding, init_weights=True)
        self.sa = GemSpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)

        return f.normalize(x, p=2, dim=1)
