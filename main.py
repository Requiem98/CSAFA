from libraries import *
import utilities as ut
import models_config as mconfig


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
        print("Run code using terminal, not IDE.")

