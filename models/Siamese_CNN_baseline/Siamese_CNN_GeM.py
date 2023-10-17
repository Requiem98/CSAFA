from libraries import *
import utilities as ut
from models.modules.GeM import AdaptiveGeneralizedMeanPooling
from models.modules.SAFA import SpatialAware
from models.modules.PCA import LearnablePCA


class CNN_GeM(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, img_size:tuple):
        super().__init__()
    
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        img_H, img_W = img_size

        # encoder
        self.e1 = nn.Conv2d(self.in_channels, self.hidden_dim, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.leakyrelu1 = nn.LeakyReLU(0.2)

        self.e2 = nn.Conv2d(self.hidden_dim, self.hidden_dim*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim*2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        
        self.e3 = nn.Conv2d(self.hidden_dim*2, self.hidden_dim*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.hidden_dim*4)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        
        self.e4 = nn.Conv2d(self.hidden_dim*4, self.hidden_dim*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(self.hidden_dim*8)
        self.leakyrelu4 = nn.LeakyReLU(0.2)

        self.e5 = nn.Conv2d(self.hidden_dim*8, self.hidden_dim*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(self.hidden_dim*8)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        
        self.final_gem = SpatialAware(img_H*img_W//1024)
        
        
    def forward(self, x):
        x = self.leakyrelu1(self.bn1(self.e1(x)))
        
        x = self.leakyrelu2(self.bn2(self.e2(x)))
        
        x = self.leakyrelu3(self.bn3(self.e3(x)))
        
        x = self.leakyrelu4(self.bn4(self.e4(x)))
        
        x = self.leakyrelu5(self.bn5(self.e5(x)))
        
        x = self.final_gem(x)
        
        return f.normalize(x, p=2, dim=1)



class Siamese_CNN_GeM(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        
        self.cnn = CNN_GeM(in_channels, hidden_dim)
        

    def forward(self, input: tuple):
        
        x, y = input
        
        output1 = self.cnn(x)
        output2 = self.cnn(y)
        
        return output1, output2
    
    
class Semi_Siamese_CNN_GeM(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, img_size:tuple):
        super().__init__()
        
        self.cnn_A = CNN_GeM(in_channels, hidden_dim, img_size)
        self.cnn_B = CNN_GeM(in_channels, hidden_dim, img_size)
        

    def forward(self, input: tuple):
        
        x, y = input
        
        output_A = self.cnn_A(x)
        output_B = self.cnn_B(y)
        
        return output_A, output_B
   

#### Siamese Generated-Pano, Pano

class Siamese_CNN_GeM_v0_l(pl.LightningModule):
    def __init__(self, img_size:dict, in_channels: int, hidden_dim: int, **kargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Siamese_CNN_GeM(in_channels, hidden_dim)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.Y_ge = torch.zeros((1, hidden_dim*8)).to("cuda")
        self.gt_Y = torch.zeros((1, hidden_dim*8)).to("cuda")
        
        
    def forward(self, x, y):
        
        return self.model((x, y))
    
 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        query = batch["pano"]
        gt = batch["generated_pano"]
        
       
        latent_variable, gt_latent_variable = self.model((query, gt))
        
        loss = self.loss(latent_variable, gt_latent_variable)
        
        
        self.log("triplet_loss", loss, prog_bar=True, on_epoch = True, on_step = False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["pano"]
        gt = batch["generated_pano"]
        
        generated_code, gt_generated_code = self.model((x, gt))
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
    def on_validation_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        self.log("R@1", rk1, prog_bar=False, on_epoch = True, on_step = False)
        self.log("R@5", rk5, prog_bar=False, on_epoch = True, on_step = False)
        self.log("R@10", rk10, prog_bar=False, on_epoch = True, on_step = False)
        self.log("R@1%", rk_1_percent, prog_bar=False, on_epoch = True, on_step = False)
        
        self.Y_ge = torch.zeros((1, self.Y_ge.shape[1])).type_as(self.Y_ge)
        self.gt_Y = torch.zeros((1, self.gt_Y.shape[1])).type_as(self.gt_Y)
        
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False, min_lr = 1e-6),
                "interval": "epoch",
                "monitor": "triplet_loss",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["pano"]
        gt = batch["generated_pano"]
        
        generated_code, gt_generated_code = self.model((x, gt))
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
    def on_predict_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        ut.save_object(self.Y_ge, f"./Data/Y_ge_{self.__class__.__name__[:-2]}.pkl")
        ut.save_object(self.gt_Y, f"./Data/gt_Y_{self.__class__.__name__[:-2]}.pkl")
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        print(f"""
              
              ##########Test Retrieval##########
              
              R@1 = {rk1}
              
              R@5 = {rk5}
              
              R@10 = {rk10}
              
              R@1% = {rk_1_percent}
              
              """)
        
       
        
        self.Y_ge = torch.zeros((1, self.Y_ge.shape[1])).type_as(self.Y_ge)
        self.gt_Y = torch.zeros((1, self.gt_Y.shape[1])).type_as(self.gt_Y)
        
    def compute_recalls(self): 
        
        out_euclidean = torch.cdist(self.Y_ge, self.gt_Y)
        
        out_euclidean_ordered = out_euclidean.sort(descending=False)
        
        rk1 = ut.R_K(out_euclidean_ordered[1], k = 1)
        rk5 = ut.R_K(out_euclidean_ordered[1], k = 5)
        rk10 = ut.R_K(out_euclidean_ordered[1], k = 10)
        rk_1_percent = ut.R_K_percent(out_euclidean_ordered[1], k = 1)
        
        return rk1, rk5, rk10, rk_1_percent
    
    
    
#### Semi-Siamese Polar, Pano


class Siamese_CNN_GeM_v1_l(pl.LightningModule):
    def __init__(self, img_size:dict, in_channels: int, hidden_dim: int, **kargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Semi_Siamese_CNN_GeM(in_channels, hidden_dim, (img_H, img_W))
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.Y_ge = torch.zeros((1, hidden_dim*8*8)).to("cuda")
        self.gt_Y = torch.zeros((1, hidden_dim*8*8)).to("cuda")
        
        
    def forward(self, x, y):
        
        return self.model((x, y))
    
 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        query = batch["pano"]
        gt = batch["polar"]
        
       
        latent_variable, gt_latent_variable = self.model((query, gt))
        
        loss = self.loss(latent_variable, gt_latent_variable)
        
        
        self.log("triplet_loss", loss, prog_bar=True, on_epoch = True, on_step = False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["pano"]
        gt = batch["polar"]
        
        generated_code, gt_generated_code = self.model((x, gt))
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
    def on_validation_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        self.log("R@1", rk1, prog_bar=False, on_epoch = True, on_step = False)
        self.log("R@5", rk5, prog_bar=False, on_epoch = True, on_step = False)
        self.log("R@10", rk10, prog_bar=False, on_epoch = True, on_step = False)
        self.log("R@1%", rk_1_percent, prog_bar=False, on_epoch = True, on_step = False)
        
        self.Y_ge = torch.zeros((1, self.Y_ge.shape[1])).type_as(self.Y_ge)
        self.gt_Y = torch.zeros((1, self.gt_Y.shape[1])).type_as(self.gt_Y)
        
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False, min_lr = 1e-6),
                "interval": "epoch",
                "monitor": "triplet_loss",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["pano"]
        gt = batch["polar"]
        
        generated_code, gt_generated_code = self.model((x, gt))
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
    def on_predict_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        ut.save_object(self.Y_ge, f"./Data/Y_ge_{self.__class__.__name__[:-2]}.pkl")
        ut.save_object(self.gt_Y, f"./Data/gt_Y_{self.__class__.__name__[:-2]}.pkl")
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        print(f"""
              
              ##########Test Retrieval##########
              
              R@1 = {rk1}
              
              R@5 = {rk5}
              
              R@10 = {rk10}
              
              R@1% = {rk_1_percent}
              
              """)
        
        self.Y_ge = torch.zeros((1, self.Y_ge.shape[1])).type_as(self.Y_ge)
        self.gt_Y = torch.zeros((1, self.gt_Y.shape[1])).type_as(self.gt_Y)
        
    def compute_recalls(self): 
        
        out_euclidean = torch.cdist(self.Y_ge, self.gt_Y)
        
        out_euclidean_ordered = out_euclidean.sort(descending=False)
        
        rk1 = ut.R_K(out_euclidean_ordered[1], k = 1)
        rk5 = ut.R_K(out_euclidean_ordered[1], k = 5)
        rk10 = ut.R_K(out_euclidean_ordered[1], k = 10)
        rk_1_percent = ut.R_K_percent(out_euclidean_ordered[1], k = 1)
        
        return rk1, rk5, rk10, rk_1_percent