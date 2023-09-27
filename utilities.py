from libraries import *


###############################################################################
###########################   UTILS  ##########################################
###############################################################################

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def read_object(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def get_conv_out_dim(h, w, k, s, p=0):
    h2 = (h-k+2*p)/s + 1
    w2 = (w-k+2*p)/s + 1
    
    print("H: ", h2)
    print("W: ", w2)

def get_train_dataframe():
    return pd.read_csv("Data/small_CVUSA/splits/train-19zl.csv", names=["aerial_filename", "pano_filename", "annotation_filename"])

def get_test_dataframe():
    return pd.read_csv("Data/small_CVUSA/splits/val-19zl.csv", names=["aerial_filename", "pano_filename", "annotation_filename"])
    
def extract_image_patches(x, patch_size = 16):
    
    b, c, h, w = x.shape
    
    grid = (int(h//patch_size), int(w//patch_size))
    
    num_patches = int(h//patch_size)*int(w//patch_size)
    
    patches = x.unfold(1, c, c).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    return patches.reshape(b,num_patches,c,patch_size,patch_size), grid
        

def patches_to_image(x, grid_size=(2, 2)):
    # x shape is batch_size x num_patches x c x p_h x p_w
    batch_size, num_patches, c, p_h, p_w = x.size()
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.view(batch_size, grid_size[0], grid_size[1], c, p_h, p_w)
    output_h = grid_size[0] * p_h
    output_w = grid_size[1] * p_w
    x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_image = x_image.view(batch_size, c, output_h, output_w)
    return x_image


def revert_norm_img(batch):
    return((batch * 0.5) + 0.5)


###############################################################################
#######################  Scores and losses  ###################################
###############################################################################

def R_K(m, k=10):
    true_pos = 0
    
    for i, raw in enumerate(m):
        if(i in raw[:k]):
            true_pos += 1
            
    return true_pos/len(m)


def R_K_percent(m, k=1):
    true_pos = 0
    
    k = m.shape[0]//100
    
    for i, raw in enumerate(m):
        if(i in raw[:k]):
            true_pos += 1
            
    return true_pos/len(m) 
    
class triplet_loss(nn.Module):
    def __init__(self, alpha = 10.0):
        super().__init__()
        
        self.alpha = alpha
    
    def forward(self, grd_global, sat_global):
        
        dist_array = torch.cdist(sat_global, grd_global)
        
        pos_dist = torch.diag(dist_array)
        pair_n = grd_global.shape[0] * (grd_global.shape[0] - 1.0)

        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = torch.sum(torch.log(1.0 + torch.exp(triplet_dist_g2s * self.alpha)))/pair_n
        
        triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
        loss_s2g = torch.sum(torch.log(1.0 + torch.exp(triplet_dist_s2g * self.alpha)))/pair_n
        loss = (loss_g2s + loss_s2g) / 2.0
        
        
        return loss
    

###############################################################################
#############################  DATA  ##########################################
###############################################################################
    
class CVUSA(Dataset):

    def __init__(self, dataframe, downscale_factor = 0, data_to_include = ["satellite", "pano", "polar", "generated_pano"], test = False):

        self.data = dataframe
        self.downscale_factor = 2**downscale_factor
        self.data_to_include = data_to_include
        self.test = test
        self.tanh_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    
    def __len__(self):
        return len(self.data)
    
    
    def get_satellite(self, idx):
        
        satellite_img_name = "Data/small_CVUSA/"+self.data['aerial_filename'].iloc[idx]
        
        return self.preprocessing(read_image(satellite_img_name), "satellite")
    
    
    def get_polar(self, idx):
        if(self.test):
            polar_img_name = "Data/small_CVUSA/"+self.data['aerial_filename'].iloc[idx].replace('bingmap', 'bingmap_transformed_test').replace(".jpg", ".png")
        else:
            polar_img_name = "Data/small_CVUSA/"+self.data['aerial_filename'].iloc[idx].replace('bingmap', 'bingmap_transformed').replace(".jpg", ".png")
        
        return self.preprocessing(read_image(polar_img_name), "polar")
    
    
    def get_pano(self, idx):
        
        pano_img_name = "Data/small_CVUSA/"+ self.data['pano_filename'].iloc[idx]
        
        return self.preprocessing(read_image(pano_img_name), "pano")
    
    def get_generated_pano(self, idx):
        if(self.test):
            generated_pano_img_name = "Data/small_CVUSA/"+self.data['pano_filename'].iloc[idx].replace("streetview/panos/", "generated_img_test/").replace(".jpg", ".png")
        else:
            generated_pano_img_name = "Data/small_CVUSA/"+self.data['pano_filename'].iloc[idx].replace("streetview/panos/", "generated_img/").replace(".jpg", ".png")
        
        return self.preprocessing(read_image(generated_pano_img_name), "generated_pano")

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {}

        if("satellite" in self.data_to_include):
            
            satellite_img = self.get_satellite(idx)
            sample["satellite"] = satellite_img
            
        if("pano" in self.data_to_include):
            
            pano_img = self.get_pano(idx)
            sample["pano"] = pano_img
            
        if("polar" in self.data_to_include):
            
            polar_img = self.get_polar(idx)
            sample["polar"] = polar_img
            
        if("generated_pano" in self.data_to_include):
            
            generated_pano_img = self.get_generated_pano(idx)
            sample["generated_pano"] = generated_pano_img
            
        #filecode = torch.Tensor([int(filename.split("/")[-1].split(".")[0])])
        
        
        
        return sample
        

    
    
    def preprocessing(self, img: torch.Tensor, data_type: str):
        
        if(data_type in ["pano", "generated_pano", "polar"]):
            img = F.to_pil_image(img)
            img = F.resize(img, (256//self.downscale_factor, 1024//self.downscale_factor))
            img = F.to_tensor(img)
            img = self.tanh_norm(img)
        else:
            img = F.to_pil_image(img)
            img = F.resize(img, (256//self.downscale_factor, 256//self.downscale_factor))
            img = F.to_tensor(img)
            img = self.tanh_norm(img)
        
        return img
    
    
    
    
class CVUSA_DataModule(pl.LightningDataModule):
    def __init__(self, data_to_include : list, downscale_factor : int, batch_size : int, num_workers : int):
        super().__init__()
       
        data_available = ["satellite", "pano", "polar", "generated_pano"]
        assert all(element in data_available for element in data_to_include), "One or more data are unavailable"
        
        self.data_to_include = data_to_include
        
        if(downscale_factor == None):
            downscale_factor = 0    
        self.downscale_factor = downscale_factor
        
        assert batch_size > 0
        self.batch_size = batch_size
        
        assert num_workers > 0
        self.num_workers = num_workers
         
        
        self.train_data = CVUSA(get_train_dataframe(), self.downscale_factor, self.data_to_include)
        self.val_data = CVUSA(get_test_dataframe(), self.downscale_factor, self.data_to_include, test=True)
         
        self.images_info = {k:v.shape for k, v in self.train_data.__getitem__(0).items()}
        
        self.num_step_per_epoch = (len(self.train_data)//self.batch_size)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          drop_last = True,
                          num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers = self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers = self.num_workers)  
    
    
    
###############################################################################
##############################  CLI  ##########################################
###############################################################################    
    
    
class CLI(LightningCLI):
    

    def add_arguments_to_parser(self, parser):
        
        default_model = {
            "class_path": "models.DummyModel.dummy.dummy_l",
        }
        
        parser.set_defaults({"model": default_model})
        
        
        #Define TensorBoard Logger
        default_logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            
            "init_args" : { 
                "save_dir": CKP_DIR,
                "name": "Dummy",
                "version": 0
            }
        }
        
        parser.set_defaults({"trainer.logger": default_logger})
        
        #Define CallBacks
        default_callBack = {
            "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
            
            "init_args" : { 
                "logging_interval": "epoch"
            }
        }
        
        parser.set_defaults({"trainer.callbacks": default_callBack})
        
        
        parser.set_defaults({"trainer.accelerator": "gpu"})
        parser.set_defaults({"trainer.max_epochs": 1})
        parser.set_defaults({"trainer.check_val_every_n_epoch": 1})
        parser.set_defaults({"trainer.num_sanity_val_steps": 0})
        
        
        parser.link_arguments("data.images_info", "model.init_args.img_size", apply_on="instantiate")
        parser.link_arguments("data.num_step_per_epoch", "trainer.log_every_n_steps", apply_on="instantiate")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    