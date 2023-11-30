from libraries import *
import utilities as ut

from models.VGG16.vgg16 import *
from models.ResNet101.resnet101 import ResNet101_SAFA, ResNet101_GEM, ResNet101_GEM_wo_PCA
from models.VGGEM16.vggem16 import VGGEM16_GEM, VGGEM16_SAFA
from models.CBAM_VGGEM16.cbam_vggem16 import CBAM_VGGEM16_GEM, CBAM_VGGEM16_SAFA
from models.CBAM_VGG16.cbam_vgg16 import CBAM_VGG16_GEM, CBAM_VGG16_SAFA
from models.RCGAN.rcgan_vgg16 import RCGAN_VGG16_safa
from models.ViT.ViT_base import VIT_base_16
from models.RT_CGAN.rt_cgan import RetrivialTransformer
from models.GANs.networks import *
from models.modules.SAM import SAM


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
        
        return output1, output2
    
    
    
class Triple_Semi_Siamese_model(nn.Module):
    def __init__(self, module : str, aggr_type:str = "concat", module_dsm:bool = False, *args, **kargs):
        super().__init__()
        
        assert aggr_type in ["sum", "concat", "wsum"], "Unrecognized aggregation type. The available aggregations are 'sum', 'concat' or 'wsum'."
        
        self.aggr_type = aggr_type
        
        self.module_A = eval(module)(*args, **kargs)
        self.module_B = eval(module)(*args, **kargs)
        self.module_C = eval(module)(*args, **kargs)
        self.module_dsm = module_dsm
        
        if(self.aggr_type == "wsum"):
            self.a = nn.Parameter(torch.tensor(0.5))
            
    def forward(self, input: list):
        
        x,y,z = input
        
        output1 = self.module_A(x)
        output2 = self.module_B(y)
        output3 = self.module_C(z)

        if(self.aggr_type == "concat"):
            output1 = f.normalize(torch.hstack([output1, output1]), dim=1)
            output2 = torch.hstack([output2, output3])
        elif(self.aggr_type == "sum"):
            output2 = output2 + output3
        elif(self.aggr_type == "wsum"):
            output2 = self.a*output2 + (torch.tensor(1.0) - self.a)*output3
        
        return output1, f.normalize(output2, dim=1)


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
        
        assert model_type in ["Base", "Siamese", "Semi_Siamese", "Triple_Semi_Siamese"], "Unrecognized aggregation type. The available aggregations are 'Base', 'Siamese', 'Semi_Siamese', or 'Triple_Semi_Siamese'."
        
        img_H, img_W = img_size["pano"][1:]
        
        if(model_type == "Siamese"):
            self.example_input_array = [torch.Tensor(2, 3, img_H, img_W)] * 2
            self.model = Siamese_model(model, img_size = (img_H, img_W), *args, **kargs)
            
        elif(model_type == "Semi_Siamese"):
            self.example_input_array = [torch.Tensor(2, 3, img_H, img_W)] * 2
            self.model = Semi_Siamese_model(model, img_size = (img_H, img_W), *args, **kargs)
        
        elif(model_type == "Triple_Semi_Siamese"):
            self.example_input_array = [torch.Tensor(2, 3, img_H, img_W)] * 3
            self.model = Triple_Semi_Siamese_model(model, img_size = (img_H, img_W), *args, **kargs)
            
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
    




class SAM_Wrapper(ModelWrapper):
    def __init__(self, *args, **kargs):
        
        super().__init__(*args, **kargs)
        
        self.automatic_optimization = False
        self.accumulated_loss = torchmetrics.aggregation.MeanMetric()
        
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = [batch[el] for el in self.data_to_include]
        
        optimizer = self.optimizers()
        
        
        ###FIRST STEP####
        latent_variable, gt_latent_variable = self.model(x)
        
        loss = self.loss(latent_variable, gt_latent_variable)
        
        self.manual_backward(loss)
        optimizer.first_step(zero_grad=True)
        
        ###SECOND STEP###
        latent_variable, gt_latent_variable = self.model(x)
        
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
        
        
    def configure_optimizers(self):
        optimizer = SAM(self.parameters(), torch.optim.Adam, lr=self.lr, weight_decay=0.0005)
        
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience, verbose=False),
                "interval": "epoch",
                "monitor": "triplet_loss",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    
    
class RT_CGAN_Wrapper(pl.LightningModule):
    def __init__(self, generator:nn.Module, discriminator:nn.Module, retrivial:nn.Module, final_dim:int, img_size : dict):
        super().__init__()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = torch.Tensor(2, 3, img_H, img_W)
        self.automatic_optimization = False
        
        self.G = generator
        self.D = discriminator
        self.R = retrivial
        
        
        self.triplet_l = ut.triplet_loss(alpha=10.0)
        

        
        self.register_buffer("Y_ge", torch.zeros((1, final_dim)))
        self.register_buffer("gt_Y", torch.zeros((1, final_dim)))
        #self.Y_ge = torch.zeros((1, final_dim)).to("cuda")
        #self.gt_Y = torch.zeros((1, final_dim)).to("cuda")
        
        
        self.accumulated_loss_G = torchmetrics.aggregation.MeanMetric()
        self.accumulated_loss_D = torchmetrics.aggregation.MeanMetric()
        
        self.sample_generated_image = None
        self.sample_image_gt_pano = None
        self.sample_image_gt_aerial = None
        
        self.last_loss_G = 0.5
        self.last_loss_D = 0.5
        self.D_is_training = 0.0
        
    def forward(self, x):
        return self.G(x)
    
    
    def train_R(self, x, y, generated_features, optim):
        # train generator
        # generate images
        self.toggle_optimizer(optim)
        
        latent_variable, gt_latent_variable = self.R(y, generated_features.detach())
        
        r_loss = self.triplet_l(latent_variable, gt_latent_variable)
        
        
        self.log("r_loss", r_loss, prog_bar=False, on_epoch = True, on_step = False)
        
        self.manual_backward(r_loss)
        optim.step()
        optim.zero_grad()
        self.untoggle_optimizer(optim)
        
        return r_loss
        
        
    def train_G(self, x, y, generated_imgs, generated_features, valid, optim):
        # train generator
        # generate images
        self.toggle_optimizer(optim)
        xg = torch.cat([x, generated_imgs], axis=1)
        
        g_loss_bce = bce_l(self.D(xg), valid)
        
        g_loss_l1 = mae(generated_imgs, y)*100
        
        latent_variable, gt_latent_variable = self.R(y, generated_features)
        
        r_loss = self.triplet_l(latent_variable, gt_latent_variable)*1000
        
        g_loss = g_loss_l1 + g_loss_bce + r_loss
        
        self.log("r_loss_1000", r_loss, prog_bar=True, on_epoch = True, on_step = False)
        self.log("g_loss_bce", g_loss_bce, prog_bar=False, on_epoch = True, on_step = False)
        self.log("g_loss_l1", g_loss_l1, prog_bar=False, on_epoch = True, on_step = False)
        self.log("g_loss", g_loss, prog_bar=True, on_epoch = True, on_step = False)
        
        
        self.manual_backward(g_loss)
        optim.step()
        optim.zero_grad()
        self.untoggle_optimizer(optim)
        
        return g_loss
        
    def train_D(self, x, y, generated_imgs, valid, fake, optim, update:bool):        
        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optim)

        xy = torch.cat([x,y], axis=1)
        xg = torch.cat([x, generated_imgs], axis=1)
        
        real_loss = bce_l(self.D(xy), valid)
        
        fake_loss = bce_l(self.D(xg), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        
        self.log("d_loss", d_loss, prog_bar=True, on_epoch = True, on_step = False)
        
        if(update):
            self.manual_backward(d_loss)
            optim.step()
        optim.zero_grad()
        self.untoggle_optimizer(optim)
        
        return d_loss
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        
        optimizer_G, optimizer_D, optimizer_R = self.optimizers()
        
        y = batch["pano"]
        x = batch["polar"]
        generated_imgs, generated_features = self.G(x)
        
        
        
        valid = Uniform(0.9,1.1).sample([y.size(0), 1, 14, 62]).type_as(y)
        #valid = torch.ones(y.size(0), 1, y.size(2)//8-2, y.size(3)//8-2).type_as(y)
        fake = Uniform(0.0,0.2).sample([y.size(0), 1, 14, 62]).type_as(y)
        #fake = torch.zeros(y.size(0), 1, y.size(2)//8-2, y.size(3)//8-2).type_as(y)
        
        
        if(self.last_loss_D<=0.3):
            g_loss = self.train_G(x, y, generated_imgs, generated_features, valid, optimizer_G)
            d_loss = self.train_D(x, y, generated_imgs.detach(), valid, fake, optimizer_D, False)
            r_loss = self.train_R(x, y, generated_features, optimizer_R)
            self.D_is_training = 0.0
        elif(self.last_loss_D>=0.7):
            g_loss = self.last_loss_G
            d_loss = self.train_D(x, y, generated_imgs.detach(), valid, fake, optimizer_D, True)
            r_loss = self.train_R(x, y, generated_features, optimizer_R)
            self.D_is_training = 1.0
        else:
            g_loss = self.train_G(x, y, generated_imgs, generated_features, valid, optimizer_G)
            d_loss = self.train_D(x, y, generated_imgs.detach(), valid, fake, optimizer_D, True)
            r_loss = self.train_R(x, y, generated_features, optimizer_R)
            self.D_is_training = 1.0
        
        
        self.accumulated_loss_D(d_loss)
        self.accumulated_loss_G(g_loss)
        
        #save images
        sample_imgs = generated_imgs
        grid = make_grid(sample_imgs, nrow=2)
        self.sample_generated_image = grid
        
        sample_imgs = y
        grid = make_grid(sample_imgs, nrow=2)
        self.sample_image_gt_pano = grid
        
        sample_imgs = x
        grid = make_grid(sample_imgs, nrow=2)
        self.sample_image_gt_aerial = grid
        
        self.log("D training", self.D_is_training, prog_bar=False, on_epoch = False, on_step = True)
    
    def on_train_epoch_end(self):
        
        self.last_loss_D = self.accumulated_loss_D.compute()
        self.last_loss_G = self.accumulated_loss_G.compute()
        
        self.accumulated_loss_D.reset()
        self.accumulated_loss_G.reset()
        
        #scheduler_G, scheduler_D = self.lr_schedulers()
        
        #scheduler_G.step(self.last_loss_G)
        #scheduler_D.step(self.last_loss_D)
        
        if(self.current_epoch%10 == 0):
            self.logger.experiment.add_image("Ground Truth pano image", ut.revert_norm_img(self.sample_image_gt_pano), self.current_epoch)
            self.logger.experiment.add_image("Ground Truth aerial image", ut.revert_norm_img(self.sample_image_gt_aerial), self.current_epoch)
            self.logger.experiment.add_image("Generated image", ut.revert_norm_img(self.sample_generated_image), self.current_epoch)
        

    
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.parameters(), lr=1e-5, betas=(0.5,0.999))
        optimizer_D = torch.optim.Adam(self.parameters(), lr=1e-5, betas=(0.5,0.999))
        optimizer_R = torch.optim.Adam(self.parameters(), lr=1e-5, betas=(0.5,0.999))
        
        """
        scheduler_D = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min', patience=20, verbose=False),
                "name": 'Scheduler_D'
        }
        
        scheduler_G = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', patience=20, verbose=False),
                "name": 'Scheduler_G'
        }
        """
        
        return [optimizer_G, optimizer_D, optimizer_R]#, [scheduler_G, scheduler_D]
    
    
    def validation_step(self, batch, batch_idx):

        y = batch["pano"]
        x = batch["polar"]
        
        generated_imgs, generated_features = self.G(x)
        
        generated_code, gt_generated_code = self.R(y, generated_features)
        
        loss = self.triplet_l(generated_code, gt_generated_code)
        l1_loss = mae(generated_imgs, y)*100
        
        self.log("val_triplet_loss", loss, prog_bar=True, on_epoch = True, on_step = False)
        self.log("val_l1_loss", l1_loss, prog_bar=True, on_epoch = True, on_step = False)
        
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
        
    
    def compute_recalls(self): 
        
        out_euclidean = torch.cdist(self.Y_ge, self.gt_Y)
        
        out_euclidean_ordered = out_euclidean.sort(descending=False)
        
        rk1 = ut.R_K(out_euclidean_ordered[1], k = 1)
        rk5 = ut.R_K(out_euclidean_ordered[1], k = 5)
        rk10 = ut.R_K(out_euclidean_ordered[1], k = 10)
        rk_1_percent = ut.R_K_percent(out_euclidean_ordered[1], k = 1)
        
        return rk1, rk5, rk10, rk_1_percent
    
    
    
   
class GAN_Wrapper(pl.LightningModule):
    def __init__(self, generator:nn.Module, discriminator:nn.Module, img_size : dict):
        super().__init__()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = torch.Tensor(2, 3, img_H, img_W)
        self.automatic_optimization = False
        
        self.G = generator
        self.D = discriminator
        
        
        
        self.sample_generated_image = None
        self.sample_image_gt_pano = None
        self.sample_image_gt_aerial = None
        
        self.last_loss_G = 0.0
        self.last_loss_D = 0.0
        self.D_is_training = 0.0
        
    def forward(self, x):
        return self.G(x)
        
        
    def train_G(self, x, y, generated_imgs, valid, optim):
        # train generator
        # generate images
        self.toggle_optimizer(optim)
        xg = torch.cat([x, generated_imgs], axis=1)
        
        g_loss_bce = bce_l(self.D(xg), valid)
        
        g_loss_l1 = mae(generated_imgs, y)*100
        
        g_loss = g_loss_l1 + g_loss_bce
        
        self.log("g_loss_bce", g_loss_bce, prog_bar=False, on_epoch = True, on_step = False)
        self.log("g_loss_l1", g_loss_l1, prog_bar=False, on_epoch = True, on_step = False)
        self.log("g_loss", g_loss, prog_bar=True, on_epoch = True, on_step = False)
        
        
        self.manual_backward(g_loss)
        optim.step()
        optim.zero_grad()
        self.untoggle_optimizer(optim)
        
        return g_loss
        
    def train_D(self, x, y, generated_imgs, valid, fake, optim, update:bool):        
        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optim)

        xy = torch.cat([x,y], axis=1)
        xg = torch.cat([x, generated_imgs], axis=1)
        
        real_loss = bce_l(self.D(xy), valid)
        
        fake_loss = bce_l(self.D(xg), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        
        self.log("d_loss", d_loss, prog_bar=True, on_epoch = True, on_step = False)
        
        if(update):
            self.manual_backward(d_loss)
            optim.step()
        optim.zero_grad()
        self.untoggle_optimizer(optim)
        
        return d_loss
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        
        optimizer_G, optimizer_D = self.optimizers()
        
        y = batch["pano"]
        x = batch["polar"]
        generated_imgs,_ = self.G(x)
        
        
        
        valid = Uniform(0.9,1.1).sample([y.size(0), 1, 6, 30]).type_as(y)
        #valid = torch.ones(y.size(0), 1, y.size(2)//8-2, y.size(3)//8-2).type_as(y)
        fake = Uniform(0.0,0.2).sample([y.size(0), 1, 6, 30]).type_as(y)
        #fake = torch.zeros(y.size(0), 1, y.size(2)//8-2, y.size(3)//8-2).type_as(y)
        
        
        if(self.last_loss_D<=0.3):
            g_loss = self.train_G(x, y, generated_imgs, valid, optimizer_G)
            d_loss = self.train_D(x, y, generated_imgs.detach(), valid, fake, optimizer_D, False)
            self.D_is_training = 0.0
        elif(self.last_loss_D>=0.7):
            g_loss = self.last_loss_G
            d_loss = self.train_D(x, y, generated_imgs.detach(), valid, fake, optimizer_D, True)
            self.D_is_training = 1.0
        else:
            g_loss = self.train_G(x, y, generated_imgs, valid, optimizer_G)
            d_loss = self.train_D(x, y, generated_imgs.detach(), valid, fake, optimizer_D, True)
            self.D_is_training = 1.0
        
        
        self.last_loss_D = d_loss
        self.last_loss_G = g_loss
        
        
        #save images
        sample_imgs = generated_imgs
        grid = make_grid(sample_imgs, nrow=2)
        self.sample_generated_image = grid
        
        sample_imgs = y
        grid = make_grid(sample_imgs, nrow=2)
        self.sample_image_gt_pano = grid
        
        sample_imgs = x
        grid = make_grid(sample_imgs, nrow=2)
        self.sample_image_gt_aerial = grid
        
        self.log("D training", self.D_is_training, prog_bar=False, on_epoch = False, on_step = True)
    
    def on_train_epoch_end(self):
        
        #scheduler_G, scheduler_D = self.lr_schedulers()
        
        #scheduler_G.step(self.last_loss_G)
        #scheduler_D.step(self.last_loss_D)
        
        if(self.current_epoch%10 == 0):
            self.logger.experiment.add_image("Ground Truth pano image", ut.revert_norm_img(self.sample_image_gt_pano), self.current_epoch)
            self.logger.experiment.add_image("Ground Truth aerial image", ut.revert_norm_img(self.sample_image_gt_aerial), self.current_epoch)
            self.logger.experiment.add_image("Generated image", ut.revert_norm_img(self.sample_generated_image), self.current_epoch)
       
        
    def validation_step(self, batch, batch_idx):

        y = batch["pano"]
        x = batch["polar"]
        generated_imgs,_ = self.G(x)
            
        generated_imgs, _ = self.G(x)
        
        g_val_loss_l1 = mae(generated_imgs, y)*100
        
        self.log("val_l1_loss", g_val_loss_l1, prog_bar=True, on_epoch = True, on_step = False)
        
        return g_val_loss_l1
    
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.parameters(), lr=1e-5, betas=(0.5,0.999))
        optimizer_D = torch.optim.Adam(self.parameters(), lr=1e-5, betas=(0.5,0.999))
        
        """
        scheduler_D = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min', patience=20, verbose=False, min_lr = 1e-8),
                "name": 'Scheduler_D'
        }
        
        scheduler_G = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', patience=20, verbose=False, min_lr = 1e-8),
                "name": 'Scheduler_G'
        }
        """
        
        return [optimizer_G, optimizer_D]#, [scheduler_G, scheduler_D]
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["pano"]
        y = batch["filecode"]
        generated_imgs,_ = self.G(x)
        
        pred = [ut.revert_norm_img(generated_imgs), y]
        
        return pred
   
    
    
