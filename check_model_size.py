from libraries import *
import utilities as ut
from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis

from models.VGG16.vgg16 import *



def compute_parameters(k, filters_pre, filter_post):
    return (((k*filters_pre)+1)*filter_post)/1000000



def compute_GFLOPS(model, input):
    
    flops = FlopCountAnalysis(model, input)

    return flops.total()/1e9

class Lightning_model(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        
        self.example_input_array = torch.Tensor(2, 3, 128, 512)
        
        self.model = model
        
        
        
    def forward(self, x):
        
        return self.model(x)
    


vit = Lightning_model(ViT(
        img_size=(128,512), patch_size=16, embed_dim=384, num_classes=1000, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)))




vgg = Lightning_model(vgg16())
safa = Lightning_model(VGG16_SAFA((128,512), 8))


pl.utilities.model_summary.summarize(vit)
pl.utilities.model_summary.summarize(safa)



compute_GFLOPS(vit, torch.rand([2,3,128,512]))
compute_GFLOPS(safa, torch.rand([2,3,128,512]))

