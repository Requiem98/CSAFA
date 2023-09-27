from libraries import *
import utilities as ut
from models.GANs.GAN_l import GAN_l
from models.GANs.networks import UnetAFL_v5 as G
from models.GANs.networks import NLayerDiscriminator as D
#from models.GANs.networks import Discriminator_PatchGAN_Feedback as D


if __name__ == '__main__':
    
    
    #MODEL_NAME = "GANs/panoGAN+NLayerD/"
    MODEL_NAME = "GANs/UnetAFL_v5+4LayerD/"
    #MODEL_NAME = "GANs/UnetAFL_v5_128+4LayerD/"
    #MODEL_NAME = "GANs/UnetAFL_v5+Discriminator_PatchGAN/"
    
    version = 0
    
    generator = G(3,3, use_dropout=True, ngf=64)
    discriminator = D(6, n_layers=4)
    
    
    
    if not os.path.exists(CKP_DIR):
        os.makedirs(CKP_DIR)
        
    
    down_f = 1
    
    
    train_dataset = ut.CVUSA_Dataset_small(ut.get_train_dataframe_small(), downscale_factor = down_f)
    
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=8, 
                            shuffle=True,
                            drop_last = True,
                            num_workers=12)
    
    

    model = GAN_l(generator, discriminator)
      


    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', CKP_DIR, '--port', '8088'])
    url = tb.launch()
        
    logger = TensorBoardLogger(CKP_DIR, name=MODEL_NAME, version=version)
    
    print('\n\n')
    # train model  
    trainer = pl.Trainer(accelerator = 'gpu', 
                         max_epochs = -1,
                         logger = logger,
                         #limit_train_batches = 1,
                         #callbacks=[clb.LearningRateMonitor(logging_interval='epoch')]
                         )
    
    print('\n\n')
    
    ckp = glob.glob(CKP_DIR+MODEL_NAME+f"version_{version}/checkpoints/*")
    
    if(len(ckp)==1):
        ckp = ckp[0]
    else:
        ckp = None
    
    trainer.fit(model=model, train_dataloaders=train_dl, ckpt_path=ckp)
    
    

