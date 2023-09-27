from libraries import *
import utilities as ut



class GAN_l(pl.LightningModule):
    def __init__(self, generator:nn.Module, discriminator:nn.Module):
        super().__init__()
        
        self.example_input_array = torch.Tensor(2, 3, 128, 512)
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
        
        self.log("g_loss_bce", g_loss_bce, prog_bar=False, on_epoch = True, on_step = True)
        self.log("g_loss_l1", g_loss_l1, prog_bar=False, on_epoch = True, on_step = True)
        self.log("g_loss", g_loss, prog_bar=True, on_epoch = True, on_step = True)
        
        
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
        
        self.log("d_loss", d_loss, prog_bar=True, on_epoch = True, on_step = True)
        
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
        
        y = batch["pano_img"]
        x = batch["aerial_img"]
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
        
        self.logger.experiment.add_image("Ground Truth pano image", ut.revert_norm_img(self.sample_image_gt_pano), self.current_epoch)
        self.logger.experiment.add_image("Ground Truth aerial image", ut.revert_norm_img(self.sample_image_gt_aerial), self.current_epoch)
        self.logger.experiment.add_image("Generated image", ut.revert_norm_img(self.sample_generated_image), self.current_epoch)
        

    
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
        
        x = batch["aerial_img"]
        y = batch["filecode"]
        generated_imgs,_ = self.G(x)
        
        pred = [ut.revert_norm_img(generated_imgs), y]
        
        return pred




