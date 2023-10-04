from libraries import *
import utilities as ut

#Models
from models.Siamese_CNN_baseline.Siamese_CNN_base import Siamese_CNN_base_v0_l
from models.Siamese_CNN_baseline.Siamese_CNN_base import Siamese_CNN_base_v1_l
from models.Siamese_CNN_baseline.Siamese_CNN_base import Siamese_CNN_base_v2_l

from models.Siamese_Autoencoder.Siamese_AE import Siamese_AE_v0_l
from models.Siamese_Autoencoder.Siamese_AE import Siamese_AE_v1_l
from models.Siamese_Autoencoder.Siamese_AE import Siamese_AE_v2_l

from models.Siamese_CNN_SAFA.Siamese_CNN_safa import Siamese_CNN_safa_v0_l

from models.Siamese_VGG16_RMAC.Siamese_VGG16_rmac import Siamese_VGG16_rmac_v0_l

from models.Siamese_VGG16_GeM.Siamese_VGG16_gem import Siamese_VGG16_gem_v0_l

#Train Dummy (Default model)
Dummy_args = ["fit", "--data.data_to_include", '["pano"]', "--data.downscale_factor", "2", "--data.batch_size", "64", "--data.num_workers", "2"]

###############################################################################
##########################  CNN Baseline  #####################################
###############################################################################
#Train Siamese_CNN_base_v0_l
Siamese_CNN_base_v0_args = ["fit", "--model", "Siamese_CNN_base_v0_l", 
                        "--model.init_args.in_channels", "3",
                        "--model.init_args.hidden_dim", "128",
                        "--model.init_args.latent_variable_size", "1000",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", "1", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "-1",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "Siamese_CNN_base"]
#Train Siamese_CNN_base_v1_l
Siamese_CNN_base_v1_args = ["fit", "--model", "Siamese_CNN_base_v1_l", 
                        "--model.init_args.in_channels", "3",
                        "--model.init_args.hidden_dim", "128",
                        "--model.init_args.latent_variable_size", "1000",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", "1", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "Siamese_CNN_base"]
#Train Siamese_CNN_base_v2_l
Siamese_CNN_base_v2_args = ["fit", "--model", "Siamese_CNN_base_v2_l", 
                        "--model.init_args.in_channels", "3",
                        "--model.init_args.hidden_dim", "128",
                        "--model.init_args.latent_variable_size", "1000",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", "1", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "2",
                        "--trainer.logger.name", "Siamese_CNN_base"]

###############################################################################
###########################  AutoEncoder  #####################################
###############################################################################

#Train Siamese_AE_v0_l
Siamese_AE_v0_args = ["fit", "--model", "Siamese_AE_v0_l", 
                        "--model.init_args.in_channels", "3",
                        "--model.init_args.hidden_dim", "128",
                        "--model.init_args.latent_variable_size", "1000",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", "1", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "Siamese_AE"]
#Train Siamese_AE_v1_l
Siamese_AE_v1_args = ["fit", "--model", "Siamese_AE_v1_l", 
                        "--model.init_args.in_channels", "3",
                        "--model.init_args.hidden_dim", "128",
                        "--model.init_args.latent_variable_size", "1000",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", "1", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "Siamese_AE"]
#Train Siamese_AE_v2_l
Siamese_AE_v2_args = ["fit", "--model", "Siamese_AE_v2_l", 
                        "--model.init_args.in_channels", "3",
                        "--model.init_args.hidden_dim", "128",
                        "--model.init_args.latent_variable_size", "1000",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", "1", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "2",
                        "--trainer.logger.name", "Siamese_AE"]

###############################################################################
##############################  CNN SAFA  #####################################
###############################################################################

#Train Siamese_CNN_safa_v0_l
Siamese_CNN_safa_v0_args = ["fit", "--model", "Siamese_CNN_safa_v0_l", 
                        "--model.init_args.in_channels", "3",
                        "--model.init_args.hidden_dim", "128",
                        "--model.init_args.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", "1", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "Siamese_CNN_safa"]

###############################################################################
###########################  VGG16 R-MAC  #####################################
###############################################################################

#Train Siamese_CNN_safa_v0_l
Siamese_VGG16_rmac_v0_args = ["fit", "--model", "Siamese_VGG16_rmac_v0_l",
                        "--model.init_args.apply_pca", "True",
                        #"--model.init_args.num_comp", "None",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", "2", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "-1",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "Siamese_VGG16_rmac"]

###############################################################################
#############################  VGG16 GeM  #####################################
###############################################################################

#Train Siamese_CNN_safa_v0_l
Siamese_VGG16_gem_v0_args = ["fit", "--model", "Siamese_VGG16_gem_v0_l",
                        "--model.init_args.num_comp", "512",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", "1", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "Siamese_VGG16_gem"]

#Predict Siamese_AE_v0_l
#Siamese_AE_v0_l_ckp = glob.glob(CKP_DIR+f"Siamese_AE/version_0/checkpoints/*")[0]
#Siamese_AE_v0_l_pred_args = ["predict", "--config", "Data/Models/lightning_logs/Siamese_AE/version_0/config.yaml", "--ckpt_path", Siamese_AE_v0_l_ckp]

available_models = {"Dummy" : Dummy_args,
                    
                    "Siamese_CNN_base_v0" : Siamese_CNN_base_v0_args,
                    "Siamese_CNN_base_v1" : Siamese_CNN_base_v1_args,
                    "Siamese_CNN_base_v2" : Siamese_CNN_base_v2_args,
                    
                    "Siamese_AE_v0" : Siamese_AE_v0_args,
                    "Siamese_AE_v1" : Siamese_AE_v1_args,
                    "Siamese_AE_v2" : Siamese_AE_v2_args,
                    
                    "Siamese_CNN_safa_v0" : Siamese_CNN_safa_v0_args,
                    
                    "Siamese_VGG16_rmac_v0" : Siamese_VGG16_rmac_v0_args,
                    
                    "Siamese_VGG16_gem_v0" : Siamese_VGG16_gem_v0_args}

















