from libraries import *
import utilities as ut
from models.DummyModel.dummy import dummy
from models.modules.ModelWrappers import *



downscale = 1



#Train Dummy (Default model)
Dummy_args = ["fit", "--model", "dummy", "--data.data_to_include", '["pano"]', "--data.downscale_factor", f"{downscale}", "--data.batch_size", "64", "--data.num_workers", "2"]


###############################################################################
##############################  VGG16 base  ###################################
###############################################################################

VGG16_base_v0_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_base",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "VGG16_base"]


###############################################################################
##############################  VGG16 SAFA  ###################################
###############################################################################


VGG16_safa_v0_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "4096",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "VGG16_safa"]

# safa + linear                        
VGG16_safa_v1_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_Linear",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "VGG16_safa"]

#safa_v4
VGG16_safa_v2_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_v4_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.norm", "True",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "2",
                        "--trainer.logger.name", "VGG16_safa"]



#Predict
#predict_ckp = "output/lightning_logs/VGG16_safa/version_14/checkpoints/epoch=202-step=450660.ckpt"
#predict_ckp_args = ["predict", "--config", "output/lightning_logs/VGG16_safa/version_14/config_predict.yaml", "--ckpt_path", predict_ckp]




available_models = {"Dummy" : Dummy_args,
                    
                    "VGG16_base_v0" : VGG16_base_v0_args,
                    "VGG16_safa_v0" : VGG16_safa_v0_args,
                    "VGG16_safa_v1" : VGG16_safa_v1_args,
                    "VGG16_safa_v2" : VGG16_safa_v2_args}








