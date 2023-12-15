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
                        "--data.batch_size", "16",
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
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "VGG16_safa"]

VGG16_safa_v1_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA",
                      "--model.init_args.model_type", "Triple_Semi_Siamese",
                      "--model.init_args.final_dim", "4096",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.aggr_type", "sum",
                        
                        "--data.data_to_include", '["pano", "polar", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "8",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "VGG16_safa"]

VGG16_safa_v2_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA",
                      "--model.init_args.model_type", "Triple_Semi_Siamese",
                      "--model.init_args.final_dim", "4096",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.aggr_type", "wsum",
                        
                        "--data.data_to_include", '["pano", "polar", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "8",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "2",
                        "--trainer.logger.name", "VGG16_safa"]

#safa + pca
VGG16_safa_v3_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        "--model.dict_kwargs.norm", "False",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "3",
                        "--trainer.logger.name", "VGG16_safa"]

#safa + pca + cirCNN
VGG16_safa_v4_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_cir_SAFA_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        "--model.dict_kwargs.norm", "False",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "4",
                        "--trainer.logger.name", "VGG16_safa"]

#safa + pca + cirCNN + normalization
VGG16_safa_v5_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_cir_SAFA_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        "--model.dict_kwargs.norm", "True",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "5",
                        "--trainer.logger.name", "VGG16_safa"]

#safa + pca + shiCirCNN + normalization          
VGG16_safa_v6_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_cir_shi_SAFA_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        "--model.dict_kwargs.norm", "True",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "6",
                        "--trainer.logger.name", "VGG16_safa"]
                        
#safa + pca + normalization                     
VGG16_safa_v7_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        "--model.dict_kwargs.norm", "True",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "7",
                        "--trainer.logger.name", "VGG16_safa"]
                        
# safa + linear                        
VGG16_safa_v8_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_Linear",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "8",
                        "--trainer.logger.name", "VGG16_safa"]
 

#safa_v2
VGG16_safa_v9_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_v2",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "4096",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "9",
                        "--trainer.logger.name", "VGG16_safa"]
                        
                        
#safa_v3
VGG16_safa_v10_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_v3",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "4096",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "10",
                        "--trainer.logger.name", "VGG16_safa"]

 
#safa_v2 + pca
VGG16_safa_v11_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_v2_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        "--model.dict_kwargs.norm", "False",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "11",
                        "--trainer.logger.name", "VGG16_safa"]
                        
                        
#safa_v2 + pca
VGG16_safa_v12_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_v3_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        "--model.dict_kwargs.norm", "False",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "12",
                        "--trainer.logger.name", "VGG16_safa"]
                        

#safa_v2 + pca
VGG16_safa_v13_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_SAFA_v3_PCA_v2",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "8",
                        "--model.dict_kwargs.out_dim", "512",
                        "--model.dict_kwargs.norm", "False",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "13",
                        "--trainer.logger.name", "VGG16_safa"]

###############################################################################
#############################  VGG16 DSM  #####################################
###############################################################################

VGG16_DSM_v0_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_cir_shi",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "4096",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                      "--model.dict_kwargs.module_dsm", "True",
                      "--model.dict_kwargs.loss_dsm", "True",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "VGG16_DSM"]

#DSM + PCA
VGG16_DSM_v1_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_cir_shi",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "4096",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                      "--model.init_args.static_pca", "True",
                      
                      "--model.dict_kwargs.module_dsm", "True",
                      "--model.dict_kwargs.loss_dsm", "True",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "VGG16_DSM"]

###############################################################################
##########################  VGG16 gem + safa  #################################
###############################################################################


VGG16_gem_safa_v0_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_GEM_SAFA_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                      "--model.dict_kwargs.dimension", "8",
                      "--model.dict_kwargs.out_dim", "512",
                  
                      "--data.data_to_include", '["pano", "polar"]', 
                      "--data.downscale_factor", f"{downscale}", 
                      "--data.batch_size", "16",
                      "--data.num_workers", "16",
                    
                      "--trainer.max_epochs", "300",
                      "--trainer.logger.version", "0",
                      "--trainer.logger.name", "VGG16_gem_safa"]

VGG16_gem_safa_v1_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_cir_GEM_SAFA_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                      "--model.dict_kwargs.dimension", "8",
                      "--model.dict_kwargs.out_dim", "512",
                  
                      "--data.data_to_include", '["pano", "polar"]', 
                      "--data.downscale_factor", f"{downscale}", 
                      "--data.batch_size", "16",
                      "--data.num_workers", "16",
                    
                      "--trainer.max_epochs", "300",
                      "--trainer.logger.version", "1",
                      "--trainer.logger.name", "VGG16_gem_safa"]
                      
VGG16_gem_safa_v2_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_cir_GEM_SAFA_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                      "--model.dict_kwargs.dimension", "8",
                      "--model.dict_kwargs.out_dim", "512",
                  
                      "--data.data_to_include", '["pano", "polar"]', 
                      "--data.downscale_factor", f"{downscale}", 
                      "--data.batch_size", "16",
                      "--data.num_workers", "16",
                    
                      "--trainer.max_epochs", "300",
                      "--trainer.logger.version", "2",
                      "--trainer.logger.name", "VGG16_gem_safa"]



###############################################################################
#############################  VGG16 GeM  #####################################
###############################################################################

VGG16_gem_v0_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_GEM",
                      "--model.init_args.model_type", "Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.num_comp", "512",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "VGG16_gem"]


VGG16_gem_v1_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_GEM",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.num_comp", "512",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "VGG16_gem"]


VGG16_gem_v2_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_GEM",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.num_comp", "512",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "32",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "2",
                        "--trainer.logger.name", "VGG16_gem"]


VGG16_gem_v3_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_GEM",
                      "--model.init_args.model_type", "Triple_Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.num_comp", "512",
                        "--model.dict_kwargs.aggr_type", "sum",
                        
                        "--data.data_to_include", '["pano", "polar", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "8",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "3",
                        "--trainer.logger.name", "VGG16_gem"]


VGG16_gem_v4_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_GEM",
                      "--model.init_args.model_type", "Triple_Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.num_comp", "512",
                        "--model.dict_kwargs.aggr_type", "wsum",
                        
                        "--data.data_to_include", '["pano", "polar", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "8",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "4",
                        "--trainer.logger.name", "VGG16_gem"]


VGG16_gem_v5_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "VGG16_GEM_wo_PCA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "512",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                     
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "5",
                        "--trainer.logger.name", "VGG16_gem"]

###############################################################################
###########################  CBAM_VGG16  ######################################
###############################################################################

CBAM_VGG16_safa_v0_args = ["fit", "--model", "ModelWrapper",
                             
                        "--model.init_args.model", "CBAM_VGG16_SAFA",
                        "--model.init_args.model_type", "Siamese",
                        "--model.init_args.final_dim", "512",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "10",
                        
                        
                        "--model.dict_kwargs.in_channels", "3",
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "CBAM_VGG16_safa"]

CBAM_VGG16_safa_v1_args = ["fit", "--model", "ModelWrapper",
                             
                        "--model.init_args.model", "CBAM_VGG16_SAFA",
                        "--model.init_args.model_type", "Semi_Siamese",
                        "--model.init_args.final_dim", "512",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "10",
                        
                        
                        "--model.dict_kwargs.in_channels", "3",
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "CBAM_VGG16_safa"]

CBAM_VGG16_safa_v2_args = ["fit", "--model", "ModelWrapper",
                             
                        "--model.init_args.model", "CBAM_VGG16_SAFA",
                        "--model.init_args.model_type", "Semi_Siamese",
                        "--model.init_args.final_dim", "512",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "10",
                        
                        
                        "--model.dict_kwargs.in_channels", "3",
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "2",
                        "--trainer.logger.name", "CBAM_VGG16_safa"]


CBAM_VGG16_safa_v3_args = ["fit", "--model", "ModelWrapper",
                             
                        "--model.init_args.model", "CBAM_VGG16_SAFA",
                        "--model.init_args.model_type", "Triple_Semi_Siamese",
                        "--model.init_args.final_dim", "1024",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "10",
                        
                        
                        "--model.dict_kwargs.in_channels", "3",
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "polar", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "3",
                        "--trainer.logger.name", "CBAM_VGG16_safa"]



###############################################################################
###########################  CBAM_VGGEM16  ####################################
###############################################################################

CBAM_VGGEM16_safa_v0_args = ["fit", "--model", "ModelWrapper",
                             
                        "--model.init_args.model", "CBAM_VGGEM16_SAFA",
                        "--model.init_args.model_type", "Siamese",
                        "--model.init_args.final_dim", "512",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "5",
                        
                        
                        "--model.dict_kwargs.in_channels", "3",
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "CBAM_VGGEM16_safa"]


CBAM_VGGEM16_safa_v1_args = ["fit", "--model", "ModelWrapper",
                             
                        "--model.init_args.model", "CBAM_VGGEM16_SAFA",
                        "--model.init_args.model_type", "Semi_Siamese",
                        "--model.init_args.final_dim", "512",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "5",
                        
                        
                        "--model.dict_kwargs.in_channels", "3",
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "CBAM_VGGEM16_safa"]

CBAM_VGGEM16_safa_v2_args = ["fit", "--model", "ModelWrapper",
                             
                        "--model.init_args.model", "CBAM_VGGEM16_SAFA",
                        "--model.init_args.model_type", "Semi_Siamese",
                        "--model.init_args.final_dim", "512",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "5",
                        
                        
                        "--model.dict_kwargs.in_channels", "3",
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "2",
                        "--trainer.logger.name", "CBAM_VGGEM16_safa"]

###############################################################################
###############################  ResNet101  ###################################
###############################################################################

ResNet101_gem_v0_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "ResNet101_GEM",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "1024",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.num_comp", "1024",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "ResNet101_gem"]


ResNet101_safa_v0_args = ["fit", "--model", "ModelWrapper", 
                      
                      "--model.init_args.model", "ResNet101_SAFA",
                      "--model.init_args.model_type", "Semi_Siamese",
                      "--model.init_args.final_dim", "8192",
                      "--model.init_args.optim_lr", "1e-5",
                      "--model.init_args.optim_patience", "5",
                      
                        "--model.dict_kwargs.dimension", "4",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "150",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "ResNet101_safa"]






###############################################################################
###############################  ViT  #########################################
###############################################################################

ViT_v0_args = ["fit", "--model", "ModelWrapper",
               
                        "--model.init_args.model", "VIT_base_16",
                        "--model.init_args.model_type", "Semi_Siamese",
                        "--model.init_args.final_dim", "1000",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "5",
                           
                        "--model.dict_kwargs.out_dim", "1000",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "ViT"]

ViT_v1_args = ["fit", "--model", "SAM_Wrapper",
               
                        "--model.init_args.model", "VIT_base_16",
                        "--model.init_args.model_type", "Semi_Siamese",
                        "--model.init_args.final_dim", "1000",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "5",
                        
                        "--model.dict_kwargs.out_dim", "1000",
                        
                        "--data.data_to_include", '["pano", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "ViT"]


ViT_v2_args = ["fit", "--model", "SAM_Wrapper",
               
                        "--model.init_args.model", "VIT_base_16",
                        "--model.init_args.model_type", "Triple_Semi_Siamese",
                        "--model.init_args.final_dim", "2000",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "5",
                        
                        "--model.dict_kwargs.out_dim", "1000",
                        "--model.dict_kwargs.aggr_type", "concat",
                        
                        "--data.data_to_include", '["pano", "polar", "generated_pano"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "2",
                        "--trainer.logger.name", "ViT"]



###############################################################################
########################  RCGAN VGG16 safa  ###################################
###############################################################################

RCGAN_VGG16_safa_v0_args = ["fit", "--model", "ModelWrapper",
                            
                        "--model.init_args.model", "RCGAN_VGG16_safa",
                        "--model.init_args.model_type", "Base",
                        "--model.init_args.final_dim", f"{256*8}",
                        "--model.init_args.optim_lr", "1e-5",
                        "--model.init_args.optim_patience", "10",
                            
                        "--model.dict_kwargs.dimension", "8",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "12",
                        "--data.tanh", "True",
                        
                        "--trainer.max_epochs", "300",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "RCGAN_VGG16_safa"]


###############################################################################
############################  TransGan  #######################################
###############################################################################

TransGan_v0_args = ["fit", "--model", "GAN_Wrapper",
                        "--model.init_args.generator", "UnetGeneratorViT",
                        "--model.init_args.discriminator", "NLayerDiscriminator",
                        "--model.init_args.discriminator.init_args.input_nc", "6",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "16",
                        "--data.tanh", "True",
                        
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "TransGan",
                        
                        "--trainer.callbacks.init_args.monitor", "val_l1_loss",
                        "--trainer.callbacks.init_args.mode", "min"]


###############################################################################
############################  TransGan  #######################################
###############################################################################

RT_CGAN_v0_args = ["fit", "--model", "RT_CGAN_Wrapper",
                        "--model.init_args.final_dim", "768",
                        
                        "--model.init_args.generator", "UnetGeneratorViT",
                        
                        "--model.init_args.discriminator", "NLayerDiscriminator",
                        "--model.init_args.discriminator.init_args.input_nc", "6",
                        
                        "--model.init_args.retrivial", "RetrivialTransformer",
                        "--model.init_args.retrivial.init_args.out_dim", "768",
                        "--model.init_args.retrivial.init_args.img_size", f"{[128//downscale, 512//downscale]}",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "16",
                        "--data.num_workers", "8",
                        "--data.tanh", "True",

                        "--trainer.devices", "-1",
                        "--trainer.strategy", "ddp_spawn",
                   
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "0",
                        "--trainer.logger.name", "RT_CGAN"]
                        
                        
RT_CGAN_v1_args = ["fit", "--model", "RT_CGAN_Wrapper",
                        "--model.init_args.final_dim", "768",
                        
                        "--model.init_args.generator", "UnetGeneratorViT",
                        
                        "--model.init_args.discriminator", "NLayerDiscriminator",
                        "--model.init_args.discriminator.init_args.input_nc", "6",
                        
                        "--model.init_args.retrivial", "RetrivialTransformer",
                        "--model.init_args.retrivial.init_args.out_dim", "768",
                        "--model.init_args.retrivial.init_args.img_size", f"{[128//downscale, 512//downscale]}",
                        
                        "--data.data_to_include", '["pano", "polar"]', 
                        "--data.downscale_factor", f"{downscale}", 
                        "--data.batch_size", "8",
                        "--data.num_workers", "16",
                        "--data.tanh", "True",
                   
                        "--trainer.max_epochs", "400",
                        "--trainer.logger.version", "1",
                        "--trainer.logger.name", "RT_CGAN"]



#Predict
predict_ckp = "output/lightning_logs/Siamese_VGG16_gem/version_1/checkpoints/epoch=121-step=270840.ckpt"
predict_ckp_args = ["predict", "--config", "output/lightning_logs/Siamese_VGG16_gem/version_1/config.yaml", "--ckpt_path", predict_ckp]




available_models = {"Dummy" : Dummy_args,
                    
                    "VGG16_base_v0" : VGG16_base_v0_args,
                    
                    "VGG16_safa_v0" : VGG16_safa_v0_args,
                    "VGG16_safa_v1" : VGG16_safa_v1_args,
                    "VGG16_safa_v2" : VGG16_safa_v2_args,
                    "VGG16_safa_v3" : VGG16_safa_v3_args,
                    "VGG16_safa_v4" : VGG16_safa_v4_args,
                    "VGG16_safa_v5" : VGG16_safa_v5_args,
                    "VGG16_safa_v6" : VGG16_safa_v6_args,
                    "VGG16_safa_v7" : VGG16_safa_v7_args,
                    "VGG16_safa_v8" : VGG16_safa_v8_args,
                    "VGG16_safa_v9" : VGG16_safa_v9_args,
                    "VGG16_safa_v10" : VGG16_safa_v10_args,
                    "VGG16_safa_v11" : VGG16_safa_v11_args,
                    
                    "VGG16_DSM_v0" : VGG16_DSM_v0_args,
                    "VGG16_DSM_v1" : VGG16_DSM_v0_args,
                    
                    "VGG16_gem_v0" : VGG16_gem_v0_args,
                    "VGG16_gem_v1" : VGG16_gem_v1_args,
                    "VGG16_gem_v2" : VGG16_gem_v2_args,
                    "VGG16_gem_v3" : VGG16_gem_v3_args,
                    "VGG16_gem_v4" : VGG16_gem_v4_args,
                    "VGG16_gem_v5" : VGG16_gem_v5_args,
                    
                    "CBAM_VGGEM16_safa_v0" : CBAM_VGGEM16_safa_v0_args,
                    "CBAM_VGGEM16_safa_v1" : CBAM_VGGEM16_safa_v1_args,
                    "CBAM_VGGEM16_safa_v2" : CBAM_VGGEM16_safa_v2_args,
                    
                    "ResNet101_gem_v0" : ResNet101_gem_v0_args,
                    "ResNet101_safa_v0" : ResNet101_safa_v0_args,
                    
                    "VGG16_gem_safa_v0" : VGG16_gem_safa_v0_args,
                    "VGG16_gem_safa_v1" : VGG16_gem_safa_v1_args,
                    "VGG16_gem_safa_v2" : VGG16_gem_safa_v2_args,
                    
                    "ViT_v0" : ViT_v0_args,
                    "ViT_v1" : ViT_v1_args,
                    "ViT_v2" : ViT_v2_args,
                    
                    
                    "RCGAN_VGG16_safa_v0" : RCGAN_VGG16_safa_v0_args,
                    
                    "TransGan_v0" : TransGan_v0_args,
                    
                    "RT_CGAN_v0" : RT_CGAN_v0_args,
                    "RT_CGAN_v1" : RT_CGAN_v1_args}








