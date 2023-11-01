from libraries import *
import utilities as ut
import models_config as mconfig

local_home = Path(__file__).parent.resolve()
os.chdir(local_home)
torch.hub.set_dir(f"{local_home}/cache")

def cli_main(args = None):
    cli = ut.CLI(datamodule_class = ut.CVUSA_DataModule, args=args, save_config_kwargs={"overwrite": True}) 
    
def launch_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', CKP_DIR, '--port', '8088'])
    url = tb.launch()
    

    
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
        
        #launch_tensorboard()
        
        #config_arg = mconfig.Dummy_args
        
        config_arg = mconfig.ViT_v2_args
        #config_arg.extend(["--ckpt_path", "output/lightning_logs/TransGan/version_0/checkpoints/epoch=181-step=1481396.ckpt"])
        cli_main(config_arg)

