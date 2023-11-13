from libraries import *
import utilities as ut
from models.GANs.networks import *
from models.modules.PCA import LearnablePCA
from models.modules.pretrained_ViT import vit_base_patch16

#(batch, h*w, features)

class RetrivialTransformer(nn.Module):
    def __init__(self, img_size:list, out_dim:int = 768):
        super().__init__()
        
        self.vit = vit_base_patch16(img_size=tuple(img_size), num_classes = 0, attn_drop_rate = 0.1)
        self.pca1 = LearnablePCA(768, out_dim)
        self.pca2 = LearnablePCA(768, out_dim)
        
    def forward(self, x, cls_token_gen):
        
        x = self.vit(x)
        
        x = f.normalize(x, dim=1)
        
        output1 = self.pca1(x)
        
        cls_token_gen = f.normalize(cls_token_gen, dim=1)
        
        output2 = self.pca2(cls_token_gen)
        
        return f.normalize(output1, dim=1), f.normalize(output2, dim=1)


