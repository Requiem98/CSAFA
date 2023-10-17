from libraries import *
import utilities as ut
from models.GANs.networks import UnetGeneratorSkip


class feature_extractor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UnetGeneratorSkip()
        self.model.load_state_dict(torch.load("cache/checkpoints/unet_generator_skip.pth"))
        
    def forward(self, x):
        
        return self.model(x)[1]
