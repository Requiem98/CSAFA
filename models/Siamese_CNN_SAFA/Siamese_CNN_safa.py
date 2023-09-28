from libraries import *
import utilities as ut


class CNN_base(nn.Module):
    def __init__(self, img_size:tuple, in_channels: int, hidden_dim: int):
        super().__init__()
        self.W = img_size[1]
        self.H = img_size[0]
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
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
        
        
    def forward(self, x):
        x = self.leakyrelu1(self.bn1(self.e1(x)))
        
        x = self.leakyrelu2(self.bn2(self.e2(x)))
        
        x = self.leakyrelu3(self.bn3(self.e3(x)))
        
        x = self.leakyrelu4(self.bn4(self.e4(x)))
        
        x = self.leakyrelu5(self.bn5(self.e5(x)))

        return x #f.normalize(self.fc1(x), p=2, dim=1)
    
    
class SpatialAware(nn.Module):
    def __init__(self, in_shape, dimension = 8):
        super().__init__() 
        
        self.hidden1 = in_shape
        hidden = in_shape//2
        
        self.weight1 = nn.init.trunc_normal_(Parameter(torch.zeros(( in_shape, hidden, dimension  ))), mean=0.0, std=0.005)
        self.bias1 = nn.init.constant_(Parameter(torch.zeros((   1, hidden, dimension   ))), 0.1)
        
        self.weight2 = nn.init.trunc_normal_(Parameter(torch.zeros((  hidden, in_shape, dimension   ))), mean=0.0, std=0.005)
        self.bias2 = nn.init.constant_(Parameter(torch.zeros((   1, in_shape, dimension ))), 0.1)
        
    def forward(self, x):
        
        x = torch.einsum('bi, ijd -> bjd', x, self.weight1) + self.bias1
        x = torch.einsum('bjd, jid -> bid', x, self.weight2) + self.bias2
        
        return x


class Siamese_CNN_safa(nn.Module):
    def __init__(self, img_size:tuple, in_channels: int, hidden_dim: int, dimension: int):
        super().__init__()
        
        self.cnn = CNN_base(img_size, in_channels, hidden_dim)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.sa = SpatialAware(img_size[0]*img_size[1]//4096, dimension)
         
    
    def forward_once(self, x):
        
        x = self.maxpool(self.cnn(x)) #(B , C, H1, H2)
        
        x_reduced = torch.mean(x, axis=1).reshape(x.shape[0], -1) #(B, H1 X H2)
        
        x_w = self.sa(x_reduced)  #(B, HIDDEN, DIMENSION)
        
        x = x.reshape(x.shape[0], x.shape[1], -1) #(B, CHANNELS, HIDDEN)
        
        x = torch.einsum('bci, bid -> bcd', x, x_w)  #(B ,CHANNELS, DIMENSION)
        
        x = x.reshape(x.shape[0], -1)
        
        return f.normalize(x, p=2, dim=1)

    def forward(self, input: tuple):
        
        x, y = input
        
        output1 = self.forward_once(x)
        output2 = self.forward_once(y)
        
        return output1, output2
   

#### Siamese Generated-Pano, Pano

class Siamese_CNN_safa_v0_l(pl.LightningModule):
    def __init__(self, img_size:dict, in_channels: int, hidden_dim: int, dimension: int, *args):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Siamese_CNN_safa((img_H, img_W), in_channels, hidden_dim, dimension)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        self.gen_query_img = None
        self.query_img = None
        self.gen_gt_img = None
        self.gt_img = None
        
        self.Y_ge = torch.zeros((1, hidden_dim*8*dimension)).to("cuda")
        self.gt_Y = torch.zeros((1, hidden_dim*8*dimension)).to("cuda")
        
        
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
        
        x = batch["generated_pano"]
        gt = batch["pano"]
        
        generated_code = self.model.forward_once(x)
        gt_generated_code = self.model.forward_once(gt)
        
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
        
        ut.save_object(self.Y_ge, f"./Data/Y_ge_{self.__class__.__name__[:-2]}.pkl")
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