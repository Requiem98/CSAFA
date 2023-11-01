from libraries import *
import utilities as ut
from models.modules.pretrained_ViT import vit_base_patch16
from models.modules.SAM import SAM
from models.modules.PCA import LearnablePCA


class VIT_base_16(nn.Module):
    def __init__(self, img_size:tuple, out_dim:int = 1000, *args, **kargs):
        super().__init__()
        
        self.vit = vit_base_patch16(img_size=img_size, num_classes = out_dim, attn_drop_rate = 0.1)
    
    def forward(self, x):
        x = self.vit(x)
        
        return f.normalize(x, dim=1)




        







