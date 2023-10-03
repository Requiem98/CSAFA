from libraries import *
import utilities as ut



class LearnablePCA(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
        self.linear = nn.LazyLinear(k)


    @staticmethod
    def Center(x):
        #Convert to torch Tensor and keep the number of rows and columns
        
        if(x.dim() !=3 and x.dim() !=2):
            raise IndexError("PCA support just tensor of size: [B, N, M] and [B, N]")

        if(x.dim() == 3):
            
            mean = torch.mean(x, dim = 1).reshape(-1, 1, x.shape[2])
            sd = torch.std(x, dim=1).reshape(-1, 1, x.shape[2])
        
            x = (x - mean)/sd
            
        elif(x.dim() == 2):
            
            mean = torch.mean(x, dim = 0)
            sd = torch.std(x, dim=0)
            
            x = (x - mean)/sd
        
        return x



    def forward(self, x):
 
        x = self.Center(x)

        x = self.linear(x)
        
        return x


class GeneralizedMeanPooling(nn.Module):

    def __init__(self, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        
        self.p = Parameter(torch.ones(1))
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = f.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p).squeeze(3).squeeze(2)
        
        return f.normalize(x, p=2, dim=1)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'
            


class Siamese_VGG16_gem(nn.Module):
    def __init__(self, num_comp : int = 512):
        super().__init__()
        
        self.cnn = vgg16().features
        self.gem = GeneralizedMeanPooling()
        self.pca = LearnablePCA(num_comp)
         
    
    def forward_once(self, x):
        
        x = self.gem(self.cnn(x)) #(B , channels)
        
        return f.normalize(self.pca(x), p=2, dim=1)

    def forward(self, input: tuple):
        
        x, y = input
        
        output1 = self.forward_once(x)
        output2 = self.forward_once(y)
        
        return output1, output2
   

#### Siamese Generated-Pano, Pano

class Siamese_VGG16_gem_v0_l(pl.LightningModule):
    def __init__(self, img_size : dict, num_comp : int = 512, *args):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Siamese_VGG16_gem(num_comp)
        
        self.flat = nn.Flatten()
        
        self.loss = ut.triplet_loss(alpha=10.0)
        
        if(num_comp == None):
            self.Y_ge = torch.zeros((1, 512)).to("cuda")
            self.gt_Y = torch.zeros((1, 512)).to("cuda")
        else:
            self.Y_ge = torch.zeros((1, num_comp)).to("cuda")
            self.gt_Y = torch.zeros((1, num_comp)).to("cuda")
        
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