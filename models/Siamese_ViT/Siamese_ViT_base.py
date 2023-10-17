from libraries import *
import utilities as ut
from models.modules.pretrained_ViT import vit_base_patch16
from models.modules.SAM import SAM

class Semi_Siamese_ViT(nn.Module):
    def __init__(self, img_size:tuple, out_dim:int = 1000, *args, **kwargs):
        super().__init__()
        
        self.vit_q = vit_base_patch16(img_size=img_size, drop_rate = 0.1, attn_drop_rate = 0.1, num_classes = out_dim)   #query
        self.vit_gen = vit_base_patch16(img_size=img_size, drop_rate = 0.1, attn_drop_rate = 0.1, num_classes = out_dim) #generated pano

           
            
    def forward(self, q, gen):
        q_embedd = self.vit_q(q)
        gen_embedd = self.vit_gen(gen)
        
        return f.normalize(q_embedd, dim=1), f.normalize(gen_embedd, dim=1)
        
        



class Siamese_ViT_v0_l(pl.LightningModule):
    def __init__(self, img_size:dict, out_dim:int = 1000, *args, **kargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Semi_Siamese_ViT(img_size = (img_H, img_W), out_dim = out_dim, *args, **kargs)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.Y_ge = torch.zeros((1, out_dim)).to("cuda")
        self.gt_Y = torch.zeros((1, out_dim)).to("cuda")
        
        
        
    def forward(self, x, y):
        
        return self.model(x, y)
    
 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        query = batch["pano"]
        gen = batch["generated_pano"]
        optimizer = self.optimizers()
        
        latent_variable, gt_latent_variable = self.model(query, gen)
        
        loss = self.loss(latent_variable, gt_latent_variable)
        
        self.log("triplet_loss", loss, prog_bar=True, on_epoch = True, on_step = False)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        query = batch["pano"]
        gen = batch["generated_pano"]
        
        latent_variable, gt_latent_variable = self.model(query, gen)
        
        self.Y_ge = torch.vstack([self.Y_ge, latent_variable])
        self.gt_Y = torch.vstack([self.gt_Y, gt_latent_variable])
        
    
    def on_validation_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        self.log("R@1", rk1, prog_bar=True, on_epoch = True, on_step = False)
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
        
        query = batch["pano"]
        gen = batch["generated_pano"]
        
       
        latent_variable, gt_latent_variable = self.model(query, gen)
        
        self.Y_ge = torch.vstack([self.Y_ge, latent_variable])
        self.gt_Y = torch.vstack([self.gt_Y, gt_latent_variable])
        
    
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
    
    
    
class Siamese_ViT_v1_l(pl.LightningModule):
    def __init__(self, img_size:dict, out_dim:int = 1000, *args, **kargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Semi_Siamese_ViT(img_size = (img_H, img_W), out_dim = out_dim, *args, **kargs)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.Y_ge = torch.zeros((1, out_dim)).to("cuda")
        self.gt_Y = torch.zeros((1, out_dim)).to("cuda")
        
        self.accumulated_loss = torchmetrics.aggregation.MeanMetric()
        
        
    def forward(self, x, y):
        
        return self.model(x, y)
    
 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        query = batch["pano"]
        gen = batch["generated_pano"]
        optimizer = self.optimizers()
        
        
        ###FIRST STEP####
        latent_variable, gt_latent_variable = self.model(query, gen)
        
        loss = self.loss(latent_variable, gt_latent_variable)
        
        self.manual_backward(loss)
        optimizer.first_step(zero_grad=True)
        
        ###SECOND STEP###
        latent_variable, gt_latent_variable = self.model(query, gen)
        
        loss_2 = self.loss(latent_variable, gt_latent_variable)
        
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=True)
        
        
        self.log("triplet_loss", loss, prog_bar=True, on_epoch = True, on_step = False)
        self.accumulated_loss(loss)
        
        return loss
    
    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step(self.accumulated_loss.compute())
        self.accumulated_loss.reset()
    
    def validation_step(self, batch, batch_idx):
        query = batch["pano"]
        gen = batch["generated_pano"]
        
        latent_variable, gt_latent_variable = self.model(query, gen)
        
        self.Y_ge = torch.vstack([self.Y_ge, latent_variable])
        self.gt_Y = torch.vstack([self.gt_Y, gt_latent_variable])
        
    
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
        optimizer = SAM(self.parameters(), torch.optim.Adam, lr=1e-4)
        
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False, min_lr = 1e-6),
                "interval": "epoch",
                "monitor": "triplet_loss",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    
    def predict_step(self, batch, batch_idx):
        
        query = batch["pano"]
        gen = batch["generated_pano"]
        
       
        latent_variable, gt_latent_variable = self.model(query, gen)
        
        self.Y_ge = torch.vstack([self.Y_ge, latent_variable])
        self.gt_Y = torch.vstack([self.gt_Y, gt_latent_variable])
        
    
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
        
        
        
        
        
        
        
        

        
        
        