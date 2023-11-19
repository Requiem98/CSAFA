from libraries import *
import utilities as ut
from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis

from models.Siamese_Autoencoder.Siamese_AE import Siamese_AE_v0_l, Siamese_AE, Encoder

from models.Siamese_CNN_baseline.Siamese_CNN_base import Siamese_CNN_base_v0_l
from models.Siamese_CNN_baseline.Siamese_CNN_base import Siamese_CNN_base_v1_l
from models.Siamese_CNN_baseline.Siamese_CNN_base import Siamese_CNN_base_v2_l

from models.Siamese_Autoencoder.Siamese_AE import Siamese_AE_v0_l
from models.Siamese_Autoencoder.Siamese_AE import Siamese_AE_v1_l
from models.Siamese_Autoencoder.Siamese_AE import Siamese_AE_v2_l

from models.Siamese_CNN_SAFA.Siamese_CNN_safa import Siamese_CNN_safa_v0_l

from models.Siamese_VGG16_RMAC.Siamese_VGG16_rmac import Siamese_VGG16_rmac_v0_l

from models.Siamese_VGG16_GeM.Siamese_VGG16_gem import Siamese_VGG16_gem_v0_l


def compute_parameters(k, filters_pre, filter_post):
    return (((k*filters_pre)+1)*filter_post)/1000000



def compute_GFLOPS(model, input):
    
    flops = FlopCountAnalysis(model, input)

    return flops.total()/1e9

class Lightning_model(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        
        self.example_input_array = torch.Tensor(1, 3, 224, 224)
        
        self.model = model
        
        
        
    def forward(self, x):
        
        return self.model(x)
    



vgg = Lightning_model(vgg16())
resnet = Lightning_model(resnet101())
resnet2 = Lightning_model(resnet34())
mymodel = Siamese_AE_v0_l({"pano":[3,224,224]}, 3, 128, 1000)
mymodel2 = Lightning_model(Encoder((224,224), 3, 128, 1000))

Siamese_CNN_base = Siamese_CNN_base_v0_l({"pano":[3,224,224]}, 3, 128, 1000)
Siamese_VGG16_rmac = Siamese_VGG16_rmac_v0_l({"pano":[3,224,224]})

pl.utilities.model_summary.summarize(vgg)
pl.utilities.model_summary.summarize(resnet)
pl.utilities.model_summary.summarize(resnet2)
pl.utilities.model_summary.summarize(mymodel)
pl.utilities.model_summary.summarize(mymodel2)
pl.utilities.model_summary.summarize(Siamese_CNN_base)
pl.utilities.model_summary.summarize(Siamese_VGG16_rmac)


compute_GFLOPS(mymodel, (torch.rand([1,3,224,224]), torch.rand([1,3,224,224])))
compute_GFLOPS(mymodel2, torch.rand([1,3,224,224]))
compute_GFLOPS(vgg, torch.rand([1,3,224,224]))
compute_GFLOPS(resnet, torch.rand([1,3,224,224]))
compute_GFLOPS(resnet2, torch.rand([1,3,224,224]))




