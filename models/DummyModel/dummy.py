from libraries import *


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, padding : int):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, 1.e-3)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        return self.relu(self.bn(self.conv(x)))
    
    
class DeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, padding : int):
        super().__init__()
        
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, 1.e-3)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        return self.relu(self.bn(self.conv(self.pd(self.up(x)))))

class dummy_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(3, 4, 4, 2, 1)
        self.conv2 = DeConvBlock(4, 3, 3, 1, 0)
    
    def forward(self, x):
        return self.conv2(self.conv(x))
    
    
    
class dummy_l(pl.LightningModule):
    def __init__(self, *args):
        super().__init__()
        self.model = dummy_module()
        self.flat = nn.Flatten()
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["pano"]
        
        pred = self.model(x)
        
        
        loss = nn.functional.mse_loss(self.flat(pred), self.flat(x))
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_epoch = True, on_step = True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        
        scheduler1 = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False),
                "interval": "epoch",
                "monitor": "train_loss",
                "name": 'Scheduler'
            }
        
        scheduler2 = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4], gamma=10.0, verbose=False),
                "interval": "epoch",
                "name": 'WarmUp Scheduler'
            }
        
        return [optimizer], [scheduler1, scheduler2]
        
        
        


    def on_train_epoch_end(self):
        pass
    
    
    

    
  







