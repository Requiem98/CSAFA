from libraries import *
import utilities as ut
import models_config as mconfig

local_home = Path(__file__).parent.resolve()
torch.hub.set_dir(f"{local_home}/cache")

def cli_main(args = None):
    cli = ut.CLI(datamodule_class = ut.CVUSA_DataModule, args=args, save_config_kwargs={"overwrite": True}) 
    

    
if __name__ == "__main__":
    
    
    if(len(sys.argv)>1):
        selected_model = sys.argv[1]
        
        try:
            config_arg = mconfig.available_models[selected_model]
        except KeyError as e:
            raise KeyError("Unrecognized model. The available models are:  "
                           f"{[k for k in mconfig.available_models.keys()]}") from e
            
        cli_main(config_arg)
    else:
        #config_arg = mconfig.Dummy_args
        config_arg = mconfig.Siamese_VGG16_gem_v0_args
        cli_main(config_arg)

    
