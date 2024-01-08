from libraries import *
import utilities as ut

from models.VGG16.vgg16 import *


class Siamese_model(nn.Module):
    def __init__(self, module : str, module_dsm:bool = False, *args, **kargs):
        super().__init__()
        
        self.module = eval(module)(*args, **kargs)
        self.module_dsm = module_dsm

    def forward(self, input: list):
        
        x, y = input
        
        output1 = self.module(x)
        output2 = self.module(y)
        
        return output1, output2
    
    
class Semi_Siamese_model(nn.Module):
    def __init__(self, module : str, module_dsm:bool = False, *args, **kargs):
        super().__init__()
        
        self.module_A = eval(module)(*args, **kargs)
        self.module_B = eval(module)(*args, **kargs)
        self.module_dsm = module_dsm

    def forward(self, input: list):
        
        x, y = input
        
        output1 = self.module_A(x)
        output2 = self.module_B(y)
        
        if(self.module_dsm):
            if(self.training):
                output2, output1 = DSM(output2, output1)
            else:
                output1 = output1.reshape(output1.shape[0], -1)
                output2 = output2.reshape(output2.shape[0], -1)
        
        return output1, output2
    



class ModelWrapper(pl.LightningModule):
    def __init__(self, model : str, 
                 model_type:str, 
                 final_dim:int, 
                 data_to_include : list, 
                 img_size : dict, 
                 optim_lr: float,
                 optim_patience:int,
                 *args, **kargs):
        
        super().__init__()
        
        self.save_hyperparameters(ignore=["model"])
        
        self.data_to_include = data_to_include
        self.lr = optim_lr
        self.patience = optim_patience
        
        assert model_type in ["Base", "Siamese", "Semi_Siamese"], "Unrecognized aggregation type. The available aggregations are 'Base', 'Siamese', or 'Semi_Siamese'."
        
        img_H, img_W = img_size["pano"][1:]
        
        if(model_type == "Siamese"):
            self.example_input_array = [torch.Tensor(2, 3, img_H, img_W)] * 2
            self.model = Siamese_model(model, img_size = (img_H, img_W), *args, **kargs)
            
        elif(model_type == "Semi_Siamese"):
            self.example_input_array = [torch.Tensor(2, 3, img_H, img_W)] * 2
            self.model = Semi_Siamese_model(model, img_size = (img_H, img_W), *args, **kargs)
            
        elif(model_type == "Base"):
            self.example_input_array = [torch.Tensor(2, 3, img_H, img_W)] * 2
            self.model = eval(model)(img_size = (img_H, img_W), *args, **kargs)
        
        self.loss = ut.triplet_loss(alpha=10.0, *args, **kargs)
        
        
        self.register_buffer("Y_ge", torch.zeros((1, final_dim)))
        self.register_buffer("gt_Y", torch.zeros((1, final_dim)))
        #self.Y_ge = torch.zeros((1, final_dim)).to("cuda")
        #self.gt_Y = torch.zeros((1, final_dim)).to("cuda")
        
        
    def forward(self, *input):
        
        return self.model(input)
    
 
    def training_step(self, batch, batch_idx):
        
        x = [batch[el] for el in self.data_to_include]
        
       
        latent_variable, gt_latent_variable = self.model(x)
        
        loss = self.loss(latent_variable, gt_latent_variable)
        
        
        self.log("triplet_loss", loss, prog_bar=True, on_epoch = True, on_step = False)
        
        return loss
    
    def on_validation_start(self):
        self.loss.set_valid_mode()
        
    
    def validation_step(self, batch, batch_idx):
        x = [batch[el] for el in self.data_to_include]
        
        generated_code, gt_generated_code = self.model(x)
        
        loss = self.loss(generated_code, gt_generated_code)
        
        self.log("val_triplet_loss", loss, prog_bar=True, on_epoch = True, on_step = False)
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
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
        
        self.loss.set_train_mode()
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience, verbose=False),
                "interval": "epoch",
                "monitor": "triplet_loss",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    
    def predict_step(self, batch, batch_idx):
        
        x = [batch[el] for el in self.data_to_include]
        
        generated_code, gt_generated_code = self.model(x)
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
    def on_predict_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        ut.save_object(self.Y_ge, f"./Data/Y_ge_{self.__class__.__name__[:-2]}.pkl")
        ut.save_object(self.gt_Y, f"./Data/gt_Y_{self.__class__.__name__[:-2]}.pkl")
        
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
    
    
