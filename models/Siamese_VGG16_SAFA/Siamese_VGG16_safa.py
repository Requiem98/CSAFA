from libraries import *
import utilities as ut
    


class SpatialAware(nn.Module):
    def __init__(self, in_shape, dimension = 8):
        super().__init__() 
        
        hidden = in_shape//2
        
        self.weight1 = nn.init.trunc_normal_(Parameter(torch.zeros(( in_shape, hidden, dimension  ))), mean=0.0, std=0.005)
        self.bias1 = nn.init.constant_(Parameter(torch.zeros((   1, hidden, dimension   ))), 0.1)
        
        self.weight2 = nn.init.trunc_normal_(Parameter(torch.zeros((  hidden, in_shape, dimension   ))), mean=0.0, std=0.005)
        self.bias2 = nn.init.constant_(Parameter(torch.zeros((   1, in_shape, dimension ))), 0.1)
        
    def forward(self, x):
        
        w = torch.mean(x, axis=1).reshape(x.shape[0], -1) #(B, H1 X H2)
        
        w = torch.einsum('bi, ijd -> bjd', w, self.weight1) + self.bias1
        w = torch.einsum('bjd, jid -> bid', w, self.weight2) + self.bias2
        
        x = x.reshape(x.shape[0], x.shape[1], -1) #(B, CHANNELS, HIDDEN1, HIDDEN2) -> (B, CHANNELS, HIDDEN)
        
        x = torch.einsum('bci, bid -> bcd', x, w)  #(B ,CHANNELS, DIMENSION)
        
        x = x.reshape(x.shape[0], -1)
        
        return x


class Siamese_VGG16_safa(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        
        self.cnn = vgg16(weights=VGG16_Weights).features
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.sa = SpatialAware(4*16//4, dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
         
    
    def forward_once(self, x):
        
        x = self.maxpool(self.cnn(x)) #(B , C, H1, H2)
        
        x = self.sa(x)
        
        return f.normalize(x, p=2, dim=1)

    def forward(self, input: tuple):
        
        x, y = input
        
        output1 = self.forward_once(x)
        output2 = self.forward_once(y)
        
        return output1, output2
    
    
class Semi_Siamese_VGG16_safa(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        
        self.cnn_A = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.maxpool_A = nn.MaxPool2d(2, 2)
        
        self.sa_A = SpatialAware(2*8//4, dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
        
        
        
        
        self.cnn_B = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.maxpool_B = nn.MaxPool2d(2, 2)
        
        self.sa_B = SpatialAware(2*8//4, dimension) #2*8 if downsize = 2, 4*16 if downsize = 1
         
    
    def forward_A(self, x):
        
        x = self.maxpool_A(self.cnn_A(x)) #(B , C, H1, H2)
        
        x = self.sa_A(x)
        
        return f.normalize(x, p=2, dim=1)
    
    def forward_B(self, x):
        
        x = self.maxpool_B(self.cnn_B(x)) #(B , C, H1, H2)
        
        x = self.sa_B(x)
        
        return f.normalize(x, p=2, dim=1)

    def forward(self, input: tuple):
        
        x, y = input
        
        output1 = self.forward_A(x)
        output2 = self.forward_B(y)
        
        return output1, output2
   

#### Siamese Generated-Pano, Pano

class Siamese_VGG16_safa_v0_l(pl.LightningModule):
    
    
    def __init__(self, img_size:dict, dimension: int, *args):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Siamese_VGG16_safa(dimension)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.gen_query_img = None
        self.query_img = None
        self.gen_gt_img = None
        self.gt_img = None
        
        self.Y_ge = torch.zeros((1, 512*dimension)).to("cuda")
        self.gt_Y = torch.zeros((1, 512*dimension)).to("cuda")
        
        
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
        
        generated_code = self.model.forward_once(x)
        gt_generated_code = self.model.forward_once(gt)
        
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
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False),
                "interval": "epoch",
                "monitor": "triplet_loss",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["pano"]
        gt = batch["generated_pano"]
        
        generated_code = self.model.forward_once(x)
        gt_generated_code = self.model.forward_once(gt)
        
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
    
    
#### Semi-Siamese polar, Pano


class Siamese_VGG16_safa_v1_l(pl.LightningModule):
    
    
    def __init__(self, img_size:dict, dimension: int, *args):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Semi_Siamese_VGG16_safa(dimension)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.gen_query_img = None
        self.query_img = None
        self.gen_gt_img = None
        self.gt_img = None
        
        self.Y_ge = torch.zeros((1, 512*dimension)).to("cuda")
        self.gt_Y = torch.zeros((1, 512*dimension)).to("cuda")
        
        
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
        
        generated_code = self.model.forward_A(x)
        gt_generated_code = self.model.forward_B(gt)
        
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
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False),
                "interval": "epoch",
                "monitor": "triplet_loss",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["pano"]
        gt = batch["polar"]
        
        generated_code = self.model.forward_A(x)
        gt_generated_code = self.model.forward_B(gt)
        
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
    
    
    
class Siamese_VGG16_safa_v2_l(pl.LightningModule):
    
    
    def __init__(self, img_size:dict, dimension: int, *args):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Semi_Siamese_VGG16_safa(dimension)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.gen_query_img = None
        self.query_img = None
        self.gen_gt_img = None
        self.gt_img = None
        
        self.Y_ge = torch.zeros((1, 512*dimension)).to("cuda")
        self.gt_Y = torch.zeros((1, 512*dimension)).to("cuda")
        
        
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
        
        generated_code = self.model.forward_A(x)
        gt_generated_code = self.model.forward_B(gt)
        
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
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False),
                "interval": "epoch",
                "monitor": "triplet_loss",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["generated_pano"]
        gt = batch["pano"]
        
        generated_code = self.model.forward_A(x)
        gt_generated_code = self.model.forward_B(gt)
        
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