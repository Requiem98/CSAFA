from libraries import *
import utilities as ut


class Encoder(nn.Module):
    def __init__(self, img_size:tuple, in_channels: int, hidden_dim: int, latent_variable_size: int):
        super().__init__()
        self.W = img_size[1]
        self.H = img_size[0]
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_variable_size = latent_variable_size
        self.intermediate_results = {}
        
        self.final_conv_dim = int(self.hidden_dim*8*(self.W/32)*(self.H/32))

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
        
        self.fc1 = nn.Linear(self.final_conv_dim, latent_variable_size)
        
        
    def forward(self, x):
        x = self.leakyrelu1(self.bn1(self.e1(x)))
        
        self.intermediate_results["layer1_e"] = x
        
        x = self.leakyrelu2(self.bn2(self.e2(x)))
        
        self.intermediate_results["layer2_e"] = x
        
        x = self.leakyrelu3(self.bn3(self.e3(x)))
        
        self.intermediate_results["layer3_e"] = x
        
        x = self.leakyrelu4(self.bn4(self.e4(x)))
        
        self.intermediate_results["layer4_e"] = x
        
        x = self.leakyrelu5(self.bn5(self.e5(x)))
        
        self.intermediate_results["layer5_e"] = x
        
        x = x.view(-1, self.final_conv_dim)

        return f.normalize(self.fc1(x), p=2, dim=1)
    
    
class Decoder(nn.Module):
    def __init__(self, img_size:tuple, in_channels: int, hidden_dim: int, latent_variable_size: int):
        super().__init__()
        self.W = img_size[1]
        self.H = img_size[0]
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_variable_size = latent_variable_size
        self.intermediate_results = {}
        
        self.final_conv_dim = int(self.hidden_dim*8*(self.W/32)*(self.H/32))
        
        
        
        self.d1 = nn.Linear(latent_variable_size, self.final_conv_dim)
        self.relu = nn.ReLU()

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(self.hidden_dim*8, self.hidden_dim*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(self.hidden_dim*8, 1.e-3)
        self.leakyrelu6 = nn.LeakyReLU(0.2)
        
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(self.hidden_dim*8, self.hidden_dim*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(self.hidden_dim*4, 1.e-3)
        self.leakyrelu7 = nn.LeakyReLU(0.2)
        
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(self.hidden_dim*4, self.hidden_dim*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(self.hidden_dim*2, 1.e-3)
        self.leakyrelu8 = nn.LeakyReLU(0.2)
        
        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(self.hidden_dim*2, self.hidden_dim, 3, 1)
        self.bn9 = nn.BatchNorm2d(self.hidden_dim, 1.e-3)
        self.leakyrelu9 = nn.LeakyReLU(0.2)
        
        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(self.hidden_dim, self.in_channels, 3, 1)
        
    def forward(self, x):
        
        x = self.relu(self.d1(x))
        x = x.view(-1, self.hidden_dim*8, int(self.H/32), int(self.W/32))
        
        self.intermediate_results["layer1_d"] = x
        
        
        x = self.leakyrelu6(self.bn6(self.d2(self.pd1(self.up1(x)))))
        
        self.intermediate_results["layer2_d"] = x
        
        x = self.leakyrelu7(self.bn7(self.d3(self.pd2(self.up2(x)))))
        
        self.intermediate_results["layer3_d"] = x
        
        x = self.leakyrelu8(self.bn8(self.d4(self.pd3(self.up3(x)))))
        
        self.intermediate_results["layer4_d"] = x
        
        x = self.leakyrelu9(self.bn9(self.d5(self.pd4(self.up4(x)))))
        
        self.intermediate_results["layer5_d"] = x

        return self.d6(self.pd5(self.up5(x)))


class Siamese_AE(nn.Module):
    def __init__(self, img_size:tuple, in_channels: int, hidden_dim: int, latent_variable_size: int):
        super().__init__()
        
        self.encoder = Encoder(img_size, in_channels, hidden_dim, latent_variable_size)
        self.decoder = Decoder(img_size, in_channels, hidden_dim, latent_variable_size)
        
    
    def forward_once(self, x):
        latent_variables = self.encoder(x)

        x = self.decoder(latent_variables)
        return x, self.encoder.intermediate_results, self.decoder.intermediate_results, latent_variables

    def forward(self, input: tuple):
        
        x, y = input
        
        output1 = self.forward_once(x)
        output2 = self.forward_once(y)
        
        return output1, output2
    
    
class Semi_Siamese_AE(nn.Module):
    def __init__(self, img_size:tuple, in_channels: int, hidden_dim: int, latent_variable_size: int):
        super().__init__()
        
        self.encoder_A = Encoder(img_size, in_channels, hidden_dim, latent_variable_size)
        self.decoder_A = Decoder(img_size, in_channels, hidden_dim, latent_variable_size)
        
        self.encoder_B = Encoder(img_size, in_channels, hidden_dim, latent_variable_size)
        self.decoder_B = Decoder(img_size, in_channels, hidden_dim, latent_variable_size)
        
    
    def forward_A(self, x):
        latent_variables = self.encoder_A(x)

        x = self.decoder_A(latent_variables)
        return x, self.encoder_A.intermediate_results, self.decoder_A.intermediate_results, latent_variables
    
    def forward_B(self, x):
        latent_variables = self.encoder_B(x)

        x = self.decoder_B(latent_variables)
        return x, self.encoder_B.intermediate_results, self.decoder_B.intermediate_results, latent_variables

    def forward(self, input: tuple):
        
        x, y = input
        
        output_A = self.forward_A(x)
        output_B = self.forward_B(y)
        
        return output_A, output_B


#### Siamese Generated-Pano, Pano

class Siamese_AE_v0_l(pl.LightningModule):
    def __init__(self, img_size:dict, in_channels: int, hidden_dim: int, latent_variable_size: int, **kargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Siamese_AE((img_H, img_W), in_channels, hidden_dim, latent_variable_size)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.gen_query_img = None
        self.query_img = None
        self.gen_gt_img = None
        self.gt_img = None
        
        self.Y_ge = torch.zeros((1, latent_variable_size)).to("cuda")
        self.gt_Y = torch.zeros((1, latent_variable_size)).to("cuda")
        
        
    def forward(self, x, y):
        
        return self.model((x, y))
    
 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        query = batch["pano"]
        gt = batch["generated_pano"]
        
       
        out1, out2 = self.model((query, gt))
        
        pred, enc_out, dec_out, latent_variable = out1
        gt_pred, gt_enc_out, gt_dec_out, gt_latent_variable = out2
       
        
        loss1 = self.AE_loss(pred, query, enc_out, dec_out)
        loss2 = self.AE_loss(gt_pred, gt, gt_enc_out, gt_dec_out)
        
        loss3 = self.loss(latent_variable, gt_latent_variable)
        
        loss = loss1 + loss2 + loss3
        
        #self.accuracy(positive_dis, negative_dis)
        
        #save images
        sample_imgs = pred
        grid = make_grid(sample_imgs, nrow=4)
        self.gen_query_img = grid
        
        sample_imgs = query
        grid = make_grid(sample_imgs, nrow=4)
        self.query_img = grid
        
        sample_imgs = gt_pred
        grid = make_grid(sample_imgs, nrow=4)
        self.gen_gt_img = grid
        
        sample_imgs = gt
        grid = make_grid(sample_imgs, nrow=4)
        self.gt_img = grid
        
        
        self.log("loss_tot", loss, prog_bar=True, on_epoch = True, on_step = False)
        self.log("loss_query", loss1, prog_bar=True, on_epoch = True, on_step = False)
        self.log("loss_gt", loss2, prog_bar=True, on_epoch = True, on_step = False)
        self.log("triplet_loss", loss3, prog_bar=True, on_epoch = True, on_step = False)
    
        #self.log("Triplets_accuracy", self.accuracy, prog_bar=True, on_epoch = True, on_step = False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["pano"]
        gt = batch["generated_pano"]
        
        generated_code = self.model.encoder(x)
        gt_generated_code = self.model.encoder(gt)
        
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
        
        
    
    def on_train_epoch_end(self):
        
        self.logger.experiment.add_image("gen_query_img", ut.revert_norm_img(self.gen_query_img), self.current_epoch)
        self.logger.experiment.add_image("query_img", ut.revert_norm_img(self.query_img), self.current_epoch)
        self.logger.experiment.add_image("gen_gt_img", ut.revert_norm_img(self.gen_gt_img), self.current_epoch)
        self.logger.experiment.add_image("gt_img", ut.revert_norm_img(self.gt_img), self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False),
                "interval": "epoch",
                "monitor": "loss_tot",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    def AE_loss(self, pred, target, enc_out, dec_out):
        
        L1 = mse(self.flat(dec_out["layer1_d"]), self.flat(enc_out["layer5_e"]))
        L2 = mse(self.flat(dec_out["layer2_d"]), self.flat(enc_out["layer4_e"]))
        L3 = mse(self.flat(dec_out["layer3_d"]), self.flat(enc_out["layer3_e"]))
        L4 = mse(self.flat(dec_out["layer4_d"]), self.flat(enc_out["layer2_e"]))
        L5 = mse(self.flat(dec_out["layer5_d"]), self.flat(enc_out["layer1_e"]))
        
        
        L_last = mse(self.flat(pred), self.flat(target))
        
        L_mid = (L1 + L2 + L3 + L4 + L5)*0.01
        
        loss = L_last + L_mid
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["generated_pano"]
        gt = batch["pano"]
        
        generated_code = self.model.encoder(x)
        gt_generated_code = self.model.encoder(gt)
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
    def on_predict_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        print(f"""
              
              ##########Test Retrieval##########
              
              R@1 = {rk1}
              
              R@5 = {rk5}
              
              R@10 = {rk10}
              
              R@1% = {rk_1_percent}
              
              """)
        
        ut.save_object(self.Y_ge, "./Data/Y_ge_{self.__class__.__name__[:-2]}.pkl")
        ut.save_object(self.gt_Y, "./Data/gt_Y_{self.__class__.__name__[:-2]}.pkl")
        
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

#### Siamese Polar, Pano
    
class Siamese_AE_v1_l(pl.LightningModule):
    def __init__(self, img_size:dict, in_channels: int, hidden_dim: int, latent_variable_size: int, **kargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Siamese_AE((img_H, img_W), in_channels, hidden_dim, latent_variable_size)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.gen_query_img = None
        self.query_img = None
        self.gen_gt_img = None
        self.gt_img = None
        
        self.Y_ge = torch.zeros((1, latent_variable_size)).to("cuda")
        self.gt_Y = torch.zeros((1, latent_variable_size)).to("cuda")
        
        
    def forward(self, x, y):
        
        return self.model((x, y))
    
 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        query = batch["pano"]
        gt = batch["polar"]
        
       
        out1, out2 = self.model((query, gt))
        
        pred, enc_out, dec_out, latent_variable = out1
        gt_pred, gt_enc_out, gt_dec_out, gt_latent_variable = out2
       
        
        loss1 = self.AE_loss(pred, query, enc_out, dec_out)
        loss2 = self.AE_loss(gt_pred, gt, gt_enc_out, gt_dec_out)
        
        loss3 = self.loss(latent_variable, gt_latent_variable)
        
        loss = loss1 + loss2 + loss3
        
        
        #save images
        sample_imgs = pred
        grid = make_grid(sample_imgs, nrow=4)
        self.gen_query_img = grid
        
        sample_imgs = query
        grid = make_grid(sample_imgs, nrow=4)
        self.query_img = grid
        
        sample_imgs = gt_pred
        grid = make_grid(sample_imgs, nrow=4)
        self.gen_gt_img = grid
        
        sample_imgs = gt
        grid = make_grid(sample_imgs, nrow=4)
        self.gt_img = grid
        
        
        self.log("loss_tot", loss, prog_bar=True, on_epoch = True, on_step = False)
        self.log("loss_query", loss1, prog_bar=True, on_epoch = True, on_step = False)
        self.log("loss_gt", loss2, prog_bar=True, on_epoch = True, on_step = False)
        self.log("triplet_loss", loss3, prog_bar=True, on_epoch = True, on_step = False)
    
        #self.log("Triplets_accuracy", self.accuracy, prog_bar=True, on_epoch = True, on_step = False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        query = batch["pano"]
        gt = batch["polar"]
        
        generated_code = self.model.encoder(query)
        gt_generated_code = self.model.encoder(gt)
        
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
        
        
    
    def on_train_epoch_end(self):
        
        self.logger.experiment.add_image("gen_query_img", ut.revert_norm_img(self.gen_query_img), self.current_epoch)
        self.logger.experiment.add_image("query_img", ut.revert_norm_img(self.query_img), self.current_epoch)
        self.logger.experiment.add_image("gen_gt_img", ut.revert_norm_img(self.gen_gt_img), self.current_epoch)
        self.logger.experiment.add_image("gt_img", ut.revert_norm_img(self.gt_img), self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False),
                "interval": "epoch",
                "monitor": "loss_tot",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    def AE_loss(self, pred, target, enc_out, dec_out):
        
        L1 = mse(self.flat(dec_out["layer1_d"]), self.flat(enc_out["layer5_e"]))
        L2 = mse(self.flat(dec_out["layer2_d"]), self.flat(enc_out["layer4_e"]))
        L3 = mse(self.flat(dec_out["layer3_d"]), self.flat(enc_out["layer3_e"]))
        L4 = mse(self.flat(dec_out["layer4_d"]), self.flat(enc_out["layer2_e"]))
        L5 = mse(self.flat(dec_out["layer5_d"]), self.flat(enc_out["layer1_e"]))
        
        
        L_last = mse(self.flat(pred), self.flat(target))
        
        L_mid = (L1 + L2 + L3 + L4 + L5)*0.01
        
        loss = L_last + L_mid
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["pano"]
        gt = batch["polar"]
        
        generated_code = self.model.encoder(x)
        gt_generated_code = self.model.encoder(gt)
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
    def on_predict_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        print(f"""
              
              ##########Test Retrieval##########
              
              R@1 = {rk1}
              
              R@5 = {rk5}
              
              R@10 = {rk10}
              
              R@1% = {rk_1_percent}
              
              """)
        
        ut.save_object(self.Y_ge, "./Data/Y_ge_{self.__class__.__name__[:-2]}.pkl")
        ut.save_object(self.gt_Y, "./Data/gt_Y_{self.__class__.__name__[:-2]}.pkl")
        
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


class Siamese_AE_v2_l(pl.LightningModule):
    def __init__(self, img_size:dict, in_channels: int, hidden_dim: int, latent_variable_size: int, **kargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Semi_Siamese_AE((img_H, img_W), in_channels, hidden_dim, latent_variable_size)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.gen_query_img = None
        self.query_img = None
        self.gen_gt_img = None
        self.gt_img = None
        
        self.Y_ge = torch.zeros((1, latent_variable_size)).to("cuda")
        self.gt_Y = torch.zeros((1, latent_variable_size)).to("cuda")
        
        
    def forward(self, x, y):
        
        return self.model((x, y))
    
 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        query = batch["pano"]
        gt = batch["polar"]
        
       
        out1, out2 = self.model((query, gt))
        
        pred, enc_out, dec_out, latent_variable = out1
        gt_pred, gt_enc_out, gt_dec_out, gt_latent_variable = out2
       
        
        loss1 = self.AE_loss(pred, query, enc_out, dec_out)
        loss2 = self.AE_loss(gt_pred, gt, gt_enc_out, gt_dec_out)
        
        loss3 = self.loss(latent_variable, gt_latent_variable)
        
        loss = loss1 + loss2 + loss3
        
        
        #save images
        sample_imgs = pred
        grid = make_grid(sample_imgs, nrow=4)
        self.gen_query_img = grid
        
        sample_imgs = query
        grid = make_grid(sample_imgs, nrow=4)
        self.query_img = grid
        
        sample_imgs = gt_pred
        grid = make_grid(sample_imgs, nrow=4)
        self.gen_gt_img = grid
        
        sample_imgs = gt
        grid = make_grid(sample_imgs, nrow=4)
        self.gt_img = grid
        
        
        self.log("loss_tot", loss, prog_bar=True, on_epoch = True, on_step = False)
        self.log("loss_query", loss1, prog_bar=True, on_epoch = True, on_step = False)
        self.log("loss_gt", loss2, prog_bar=True, on_epoch = True, on_step = False)
        self.log("triplet_loss", loss3, prog_bar=True, on_epoch = True, on_step = False)
    
        #self.log("Triplets_accuracy", self.accuracy, prog_bar=True, on_epoch = True, on_step = False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        query = batch["pano"]
        gt = batch["polar"]
        
        generated_code = self.model.encoder_A(query)
        gt_generated_code = self.model.encoder_B(gt)
        
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
        
        
    
    def on_train_epoch_end(self):
        
        self.logger.experiment.add_image("gen_query_img", ut.revert_norm_img(self.gen_query_img), self.current_epoch)
        self.logger.experiment.add_image("query_img", ut.revert_norm_img(self.query_img), self.current_epoch)
        self.logger.experiment.add_image("gen_gt_img", ut.revert_norm_img(self.gen_gt_img), self.current_epoch)
        self.logger.experiment.add_image("gt_img", ut.revert_norm_img(self.gt_img), self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False),
                "interval": "epoch",
                "monitor": "loss_tot",
                "name": 'Scheduler'
            }
        
        return [optimizer], [scheduler]  
    
    def AE_loss(self, pred, target, enc_out, dec_out):
        
        L1 = mse(self.flat(dec_out["layer1_d"]), self.flat(enc_out["layer5_e"]))
        L2 = mse(self.flat(dec_out["layer2_d"]), self.flat(enc_out["layer4_e"]))
        L3 = mse(self.flat(dec_out["layer3_d"]), self.flat(enc_out["layer3_e"]))
        L4 = mse(self.flat(dec_out["layer4_d"]), self.flat(enc_out["layer2_e"]))
        L5 = mse(self.flat(dec_out["layer5_d"]), self.flat(enc_out["layer1_e"]))
        
        
        L_last = mse(self.flat(pred), self.flat(target))
        
        L_mid = (L1 + L2 + L3 + L4 + L5)*0.01
        
        loss = L_last + L_mid
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        
        x = batch["pano"]
        gt = batch["polar"]
        
        generated_code = self.model.encoder_A(x)
        gt_generated_code = self.model.encoder_B(gt)
        
        self.Y_ge = torch.vstack([self.Y_ge, generated_code])
        self.gt_Y = torch.vstack([self.gt_Y, gt_generated_code])
        
    
    def on_predict_epoch_end(self):
        
        self.Y_ge = self.Y_ge[1:, :]
        self.gt_Y = self.gt_Y[1:, :]
        
        rk1, rk5, rk10, rk_1_percent = self.compute_recalls()
        
        print(f"""
              
              ##########Test Retrieval##########
              
              R@1 = {rk1}
              
              R@5 = {rk5}
              
              R@10 = {rk10}
              
              R@1% = {rk_1_percent}
              
              """)
        
        ut.save_object(self.Y_ge, "./Data/Y_ge_{self.__class__.__name__[:-2]}.pkl")
        ut.save_object(self.gt_Y, "./Data/gt_Y_{self.__class__.__name__[:-2]}.pkl")
        
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