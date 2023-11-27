from libraries import *
import utilities as ut
from models.modules.GeM import AdaptiveGeneralizedMeanPooling
from models.modules.PCA import LearnablePCA
from models.modules.SAFA import SpatialAware, MaxSpatialAware, GemSpatialAware

class CircConv2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple, padding:tuple = (1,1)): 
        super().__init__()
        
        self.pad = (padding[0],padding[1],0,0)
        self.topBottom_pad = nn.ReplicationPad2d((0,0,padding[0],padding[1]))
        self.cnn = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)
        
    def forward(self,x):
        x = f.pad(x, self.pad, "circular")
        x = self.topBottom_pad(x)
        return self.cnn(x)


class N_cir_conv(nn.Module):

    def __init__(self,in_channels,out_channels,N = 2, padding:tuple = (1,1), pool:bool = True):
        super().__init__()
        
        model = []
        model.append(CircConv2d(in_channels,out_channels,kernel_size=(3,3), padding))
        model.append(nn.ReLU(True))
        for i in range(N-1):
            model.append(CircConv2d(out_channels,out_channels,kernel_size=(3,3), padding))
            model.append(nn.ReLU(True))
        
        if(pool):
            model.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.conv = nn.Sequential(*model)
    def forward(self,x):
        return self.conv(x)
    
    
    
    

class VGG16_cir_shi(nn.Module):

    def __init__(self,in_channels=3, padding:tuple = (1,1), init_weights=True):
        super().__init__()
        self.conv1 = N_cir_conv(3,64)
        self.conv2 = N_cir_conv(64,128)
        self.conv3 = N_cir_conv(128,256,N=3)
        self.conv4 = N_cir_conv(256,512,N=3, pool=False)
        
        model = []
        for i in range(3):
            model.append(nn.ReplicationPad2d((0,0,1,1)))
            model.append(CircConv2d(512,512,kernel_size=(3,3)))
            model.append(nn.ReLU(True))
            
        self.conv5 = nn.Sequential(*model)
        
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
        

class VGG16_cir(nn.Module):

    def __init__(self,in_channels=3, padding:tuple = (1,1), init_weights=True):
        super().__init__()
        self.conv1 = N_cir_conv(3,64)
        self.conv2 = N_cir_conv(64,128)
        self.conv3 = N_cir_conv(128,256,N=3)
        self.conv4 = N_cir_conv(256,512,N=3)
        self.conv5 = N_cir_conv(512,512,N=3)
        
        
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
    
    
class VGG16_SAFA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
         
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)

        return f.normalize(x, p=2, dim=1)

class VGG16_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, norm:bool, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim, norm) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)
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
        

class VGG16_cir_shi_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, norm:bool, *args, **kargs):
        super().__init__()
        
        self.cnn = VGG16_cir_shi(in_channels=3, init_weights=True)
        self.sa = SpatialAware(((img_size[0] // 32)+1) * (img_size[1] // 8), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim, norm) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)
        return f.normalize(x, p=2, dim=1)
        
class VGG16_cir_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, norm:bool, *args, **kargs):
        super().__init__()
        
        self.cnn = VGG16_cir(in_channels=3, init_weights=True)
        self.sa = SpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim, norm) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)
        return f.normalize(x, p=2, dim=1)
    
    
class VGG16_GEM_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, norm:bool, *args, **kargs):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.sa = GemSpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim, norm) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)

        return f.normalize(x, p=2, dim=1)

class VGG16_cir_shi_GEM_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, norm:bool, padding:tuple = (1,1), *args, **kargs):
        super().__init__()
        
        self.cnn = VGG16_cir(in_channels=3, padding=padding, init_weights=True)
        self.sa = GemSpatialAware(((img_size[0] // 32)+1) * (img_size[1] // 8), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim, norm) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)

        return f.normalize(x, p=2, dim=1)
        
        
class VGG16_cir_GEM_SAFA_PCA(nn.Module):
    def __init__(self, img_size:tuple, dimension : int, out_dim:int, norm:bool, padding:tuple = (1,1), *args, **kargs):
        super().__init__()
        
        self.cnn = VGG16_cir(in_channels=3, padding=padding, init_weights=True)
        self.sa = GemSpatialAware((img_size[0] // 32) * (img_size[1] // 32), dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        self.pca = LearnablePCA(512*dimension, out_dim, norm) 
        
    def forward(self, x):
        x = self.sa(self.cnn(x)) #(B , channels)
        x = self.pca(x)

        return f.normalize(x, p=2, dim=1)