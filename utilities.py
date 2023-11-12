from libraries import *


###############################################################################
###########################   UTILS  ##########################################
###############################################################################
def get_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print("Total memory:", t/1000/1000/1000)
    print("Reserved memory:", r/1000/1000/1000)
    print("Allocated memory:", a/1000/1000/1000)
    print("Free memory:", f/1000/1000/1000)

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

def get_cvusa_train_dataframe():
    return pd.read_csv("./Data/small_CVUSA/splits/train-19zl.csv", names=["aerial_filename", "pano_filename", "annotation_filename"])

def get_cvusa_test_dataframe():
    return pd.read_csv("./Data/small_CVUSA/splits/val-19zl.csv", names=["aerial_filename", "pano_filename", "annotation_filename"])

def get_university_train_dataframe():
    return pd.read_csv("./Data/University-Release/train/train.csv", index_col=["index_1", "index_2"], dtype = {"index_1" : str})

def get_university_tests_dataframe(mode:str):
    
    modes = ["UAV->Satellite", "Satellite->UAV"]
    
    if(mode == modes[0]):
        gallery = pd.read_csv("./Data/University-Release/test/gallery.csv", index_col=["index_1", "index_2"], dtype = {"index_1" : str}, usecols = ["index_1", "index_2", "satellite"])
        query = pd.read_csv("./Data/University-Release/test/query.csv", index_col=["index_1", "index_2"], dtype = {"index_1" : str}, usecols = ["index_1", "index_2", "drone"])   
    elif(mode == modes[1]):
        gallery = pd.read_csv("./Data/University-Release/test/gallery.csv", index_col=["index_1", "index_2"], dtype = {"index_1" : str}, usecols = ["index_1", "index_2", "drone"])
        query = pd.read_csv("./Data/University-Release/test/query.csv", index_col=["index_1", "index_2"], dtype = {"index_1" : str}, usecols = ["index_1", "index_2", "satellite"])  
    else:
        raise KeyError(f"Unrecognized mode. The available modes are:  {modes}")
    
    return query, gallery

    
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
        
        dist_array = 2.0 - 2.0 * torch.matmul(sat_global, grd_global.T)
        

        pos_dist = torch.diag(dist_array)
            
        pair_n = grd_global.shape[0] * (grd_global.shape[0] - 1.0)

        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = torch.sum(torch.nan_to_num(torch.log(1.0 + torch.exp(triplet_dist_g2s * self.alpha)), posinf=1e10))/pair_n
        
        triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
        loss_s2g = torch.sum(torch.nan_to_num(torch.log(1.0 + torch.exp(triplet_dist_s2g * self.alpha)), posinf=1e10))/pair_n
        loss = (loss_g2s + loss_s2g) / 2.0
        
        
        return torch.min(torch.tensor(1e10), torch.nan_to_num(loss, nan=1e10))
    

###############################################################################
#############################  DATA  ##########################################
###############################################################################

    
class CVUSA(Dataset):

    def __init__(self, dataframe, downscale_factor = 0, data_to_include = ["satellite", "pano", "polar", "generated_pano"], test = False, tanh = False):

        self.data = dataframe
        self.downscale_factor = 2**downscale_factor
        self.data_to_include = data_to_include
        self.test = test
        
        self.tanh = tanh
        self.tanh_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    
    
    def __len__(self):
        return len(self.data)
    
    
    def get_satellite(self, idx):
        
        satellite_img_name = "./Data/small_CVUSA/"+self.data['aerial_filename'].iloc[idx]
        
        return self.preprocessing(read_image(satellite_img_name), "satellite")
    
    
    def get_polar(self, idx):
        if(self.test):
            polar_img_name = "./Data/small_CVUSA/"+self.data['aerial_filename'].iloc[idx].replace('bingmap', 'bingmap_transformed_test').replace(".jpg", ".png")
        else:
            polar_img_name = "./Data/small_CVUSA/"+self.data['aerial_filename'].iloc[idx].replace('bingmap', 'bingmap_transformed').replace(".jpg", ".png")
        
        return self.preprocessing(read_image(polar_img_name), "polar")
    
    
    def get_pano(self, idx):
        
        pano_img_name = "./Data/small_CVUSA/"+ self.data['pano_filename'].iloc[idx]
        
        return self.preprocessing(read_image(pano_img_name), "pano")
    
    def get_generated_pano(self, idx):
        if(self.test):
            generated_pano_img_name = "./Data/small_CVUSA/"+self.data['pano_filename'].iloc[idx].replace("streetview/panos/", "generated_img_test/").replace(".jpg", ".png")
        else:
            generated_pano_img_name = "./Data/small_CVUSA/"+self.data['pano_filename'].iloc[idx].replace("streetview/panos/", "generated_img/").replace(".jpg", ".png")
        
        return self.preprocessing(read_image(generated_pano_img_name), "generated_pano")

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        self.rnd = torch.rand(1)
        
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
        
        if(not self.test):
            if(self.rnd < 0.5):
                img = F.hflip(img)
        
        if(data_type in ["polar", "pano"]):
            img = F.to_pil_image(img)
            img = F.resize(img, (256//self.downscale_factor, 1024//self.downscale_factor))
            img = F.to_tensor(img)
            
            if(self.tanh == True):
                img = self.tanh_norm(img)
            else:
                img = self.norm(img)
        
        elif(data_type in ["generated_pano"]):
            img = F.to_pil_image(img)
            img = F.resize(img, (256//self.downscale_factor, 1024//self.downscale_factor))
            img = F.to_tensor(img)
            img = self.norm(img)
        else:
            img = F.to_pil_image(img)
            img = F.resize(img, (256//self.downscale_factor, 256//self.downscale_factor))
            img = F.to_tensor(img)
            img = self.norm(img)
        
        return img
    
    
    
    
class CVUSA_DataModule(pl.LightningDataModule):
    def __init__(self, data_to_include : list, downscale_factor : int, batch_size : int, num_workers : int, tanh = False):
        super().__init__()
        
        data_available = ["satellite", "pano", "polar", "generated_pano"]
        assert all(element in data_available for element in data_to_include), "One or more data are unavailable"
        
        self.data_to_include = data_to_include
        
        if(downscale_factor == None):
            downscale_factor = 0    
        self.downscale_factor = downscale_factor
        
        assert batch_size > 0
        self.batch_size = batch_size
        
        assert num_workers >= 0
        self.num_workers = num_workers
         
        
        self.train_data = CVUSA(get_cvusa_train_dataframe(), self.downscale_factor, self.data_to_include,tanh=tanh)
        self.val_data = CVUSA(get_cvusa_test_dataframe(), self.downscale_factor, self.data_to_include, test=True, tanh=tanh)
         
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
    



class UNI(Dataset):

    def __init__(self, dataframe, downscale_factor = 0, test = False):

        self.downscale_factor = 2**downscale_factor
        self.data = dataframe
        self.tanh_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.test = test
    
    
    def __len__(self):
        return len(self.data)
    
    
    def _get_img(self, idx, col):
        
        img_path = self.data.iloc[idx, col]
        
        img_name = Path(img_path).parent.parent.name
        
        return img_name, self.preprocessing(read_image(img_path))

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {}
        
        if(self.test):
            img_name, img = self._get_img(idx, 0)
            sample[img_name] = img
        else:
            img_name, img = self._get_img(idx, 0)
            sample[img_name] = img
            
            img_name, img = self._get_img(idx, 1)
            sample[img_name] = img
        
        return sample

    
    def preprocessing(self, img: torch.Tensor):
        
        img = F.to_pil_image(img)
        img = F.resize(img, (512//self.downscale_factor, 512//self.downscale_factor))
        img = F.to_tensor(img)
        img = self.tanh_norm(img)
        return img
    
    
    
    
class UNI_DataModule(pl.LightningDataModule):
    def __init__(self, mode : str, downscale_factor : int, batch_size : int, num_workers : int):
        super().__init__()
       
        
        self.mode = mode
        
        if(downscale_factor == None):
            downscale_factor = 0    
        self.downscale_factor = downscale_factor
        
        assert batch_size > 0
        self.batch_size = batch_size
        
        assert num_workers >= 0
        self.num_workers = num_workers
         
        
        self.train_data = UNI(get_university_train_dataframe(), self.downscale_factor)
        self.val_query_data = UNI(get_university_tests_dataframe(self.mode)[0], self.downscale_factor, test=True)
        self.val_gallery_data = UNI(get_university_tests_dataframe(self.mode)[1], self.downscale_factor, test=True)
         
        self.images_info = {k:v.shape for k, v in self.train_data.__getitem__(0).items()}
        
        self.num_step_per_epoch = (len(self.train_data)//self.batch_size)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          drop_last = True,
                          num_workers = self.num_workers)

    def val_dataloader(self):
        query_loader = DataLoader(self.val_query_data, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers = self.num_workers)
        
        gallery_loader = DataLoader(self.val_gallery_data, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers = self.num_workers)
        
        return [query_loader, gallery_loader]
    
    def predict_dataloader(self):
        query_loader = DataLoader(self.val_query_data, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers = self.num_workers)
        
        gallery_loader = DataLoader(self.val_gallery_data, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers = self.num_workers)
        
        return [query_loader, gallery_loader]


    
###############################################################################
##############################  CLI  ##########################################
###############################################################################    
    
    
class CLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        
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
        lr_callBack = {
            "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
            
            "init_args" : { 
                "logging_interval": "epoch"
            }
        }
        
        checkpoint_callback = {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            
            "init_args" : { 
                "monitor" : "R@1",
                "save_last" : "True",
                "mode" : "max",
                "auto_insert_metric_name" : "True",
                "save_on_train_epoch_end" : "False"
            }
        }
        
        parser.set_defaults({"trainer.callbacks": [lr_callBack, checkpoint_callback]})
        
        
        parser.set_defaults({"trainer.accelerator": "gpu"})
        parser.set_defaults({"trainer.max_epochs": 1})
        parser.set_defaults({"trainer.check_val_every_n_epoch": 1})
        parser.set_defaults({"trainer.num_sanity_val_steps": 0})
        
        
        parser.link_arguments("data.images_info", "model.init_args.img_size", apply_on="instantiate")
        parser.link_arguments("data.data_to_include", "model.init_args.data_to_include", apply_on="instantiate")
        #parser.link_arguments("data.num_step_per_epoch", "trainer.log_every_n_steps", apply_on="instantiate")
        parser.set_defaults({"trainer.log_every_n_steps" : 1})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
