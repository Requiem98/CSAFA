from libraries import *
import utilities as ut
from collections import OrderedDict

from torchvision.models import vgg16, resnet101

from models.Siamese_Autoencoder.Siamese_AE import Siamese_AE_v0_l, Siamese_AE, Encoder

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
mymodel = Siamese_AE_v0_l({"pano":[3,224,224]}, 3, 128, 1000)
mymodel2 = Lightning_model(Encoder((224,224), 3, 128, 1000))


pl.utilities.model_summary.summarize(vgg)
pl.utilities.model_summary.summarize(resnet)
pl.utilities.model_summary.summarize(mymodel)
pl.utilities.model_summary.summarize(mymodel2)


compute_GFLOPS(mymodel, (torch.rand([1,3,224,224]), torch.rand([1,3,224,224])))
compute_GFLOPS(mymodel2, torch.rand([1,3,224,224]))
compute_GFLOPS(vgg, torch.rand([1,3,224,224]))
compute_GFLOPS(resnet, torch.rand([1,3,224,224]))





