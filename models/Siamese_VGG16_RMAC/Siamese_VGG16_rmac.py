from libraries import *
import utilities as ut

    
class PCA():
    def __init__(self, k):
        self.k = k

    def __repr__(self):
        return f'PCA({self.data})'

    @staticmethod
    def Center(data):
        #Convert to torch Tensor and keep the number of rows and columns
        
        if(data.dim() !=3):
            raise IndexError("PCA support just tensor of size: [B, N, M]")

        
        for_subtraction = torch.mean(data, dim = 1)
        
        X = data - for_subtraction.reshape(-1, 1, data.shape[2])
        
        return X
    
    def fit_transform(self, data):
        #Center the Data using the static method within the class
        X = self.Center(data)
        U,S,V = torch.linalg.svd(X, full_matrices=False)

        S = torch.einsum("bi, bij -> bij", S, torch.eye(S.shape[1]).unsqueeze(0).type_as(S))
        
        y = torch.matmul(S, V)
        
        return y
    """
    def fit_transform(self, data):
        #Center the Data using the static method within the class
        X = self.Center(data)
        U,S,V = torch.svd(X)

        S = S[:, :self.k]
        U = U[:,:,:self.k]
        
        y=torch.mul(U, S.unsqueeze(1))
        
        return y
    """
    
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



class R_MAC(nn.Module):
    def __init__(self, apply_pca=True, num_comp = None):
        super().__init__()
        self.apply_pca = apply_pca
        
        if(self.apply_pca):
            self.pca = LearnablePCA(512)
        
    def extract_rois(self, x, kernel):
        
        stride = torch.max(torch.tensor(kernel//2), torch.tensor(1)).item()
        # Extract patches
        patches = x.unfold(1, x.shape[1], x.shape[1]).unfold(2, kernel, stride).unfold(3, kernel, stride)
        
        return patches.squeeze(0).reshape(-1, patches.shape[2]*patches.shape[3], patches.shape[4], patches.shape[5], patches.shape[6])
    
    def compute_regions_width(self, l, m):
        return (2*(m/(l+1))).int().item()
     
    def forward(self, x):
        
        b, c, _, _ = x.shape
        
        m = torch.min(torch.tensor(x.size()[2:]))
        
        l1 = torch.amax(self.extract_rois(x, self.compute_regions_width(1, m)), dim = (3,4)) #(B, num_rois, channels)
        l2 = torch.amax(self.extract_rois(x, self.compute_regions_width(2, m)), dim = (3,4)) #(B, num_rois, channels)
        l3 = torch.amax(self.extract_rois(x, self.compute_regions_width(3, m)), dim = (3,4)) #(B, num_rois, channels)
        
        l = torch.cat([l1, l2, l3], dim=1)#(B, num_rois_TOT, channels)
        
        l = f.normalize(l, p=2, dim=2)
        
        if(self.apply_pca):
            l = f.normalize(self.pca(l), p=2, dim=2) #(B, num_rois_TOT, num_comp)
            
        return f.normalize(torch.sum(l, dim = 1), p=2, dim=1) #(B, num_comp)
            


class Siamese_VGG16_rmac(nn.Module):
    def __init__(self, apply_pca=True, num_comp = None):
        super().__init__()
        
        self.cnn = vgg16().features
        self.r_mac = R_MAC(apply_pca=True, num_comp = None)
         
    
    def forward_once(self, x):
        
        x = self.r_mac(self.cnn(x)) #(B , num_comp)
        
        return x

    def forward(self, input: tuple):
        
        x, y = input
        
        output1 = self.forward_once(x)
        output2 = self.forward_once(y)
        
        return output1, output2
   

#### Siamese Generated-Pano, Pano

class Siamese_VGG16_rmac_v0_l(pl.LightningModule):
    def __init__(self, img_size : dict, apply_pca=True, num_comp = None, *args):
        super().__init__()
        
        self.save_hyperparameters()
        
        img_H, img_W = img_size["pano"][1:]
        
        self.example_input_array = (torch.Tensor(1, 3, img_H, img_W), torch.Tensor(1, 3, img_H, img_W))
        
        self.model = Siamese_VGG16_rmac(apply_pca=True, num_comp = None)
        
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