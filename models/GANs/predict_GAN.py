from libraries import *
import utilities as ut
from models.GANs.GAN_l import GAN_l
from models.GANs.networks import UnetGeneratorSkip as G
#from models.GANs.networks import UnetAFL_v5 as G
#from models.GANs.networks import FeaturePyramidDiscriminator as D
from models.GANs.networks import NLayerDiscriminator as D
#from models.GANs.networks import Discriminator_PatchGAN_Feedback as D

if __name__ == '__main__':
    
    save_on_disk = True
    
    #MODEL_NAME = "GANs/UnetAFL_v5+Discriminator_PatchGAN/"
    MODEL_NAME = "GANs/UnetGeneratorSkip/"
    version = 0
    
    ckp = glob.glob(CKP_DIR+MODEL_NAME+f"version_{version}/checkpoints/*")
    
    if(len(ckp)==1):
        ckp = ckp[0]
    else:
        ckp = None
    
    #dataset = ut.CVUSA_Dataset_small(ut.get_test_dataframe_small(), downscale_factor = 1, test=True)
    #dataset = ut.CVUSA_Dataset_small(ut.get_train_dataframe_small(), downscale_factor = 1, test=False)
    
    dataloader = DataLoader(dataset, 
                         batch_size=16, 
                         shuffle=False,
                         num_workers=12)
    
    if(MODEL_NAME == "GANs/UnetGeneratorSkip/"):
        
        generator = G()
        
        ckp = torch.load(CKP_DIR+MODEL_NAME+f"version_{version}/checkpoints/rgan_best_ckpt.pth")['generator_model_dict']

        new_ckp = {}
        for k, v in ckp.items():
            new_ckp[k.replace("module.", "")] = v
        
        generator.load_state_dict(new_ckp)
        
        discriminator = D(6)
        
        model = GAN_l(generator=generator, discriminator=discriminator)
    else:
        generator = G(3,3, use_dropout=True)
        discriminator = D(6)
        model = GAN_l.load_from_checkpoint(ckp, generator=generator, discriminator=discriminator)
    
    
    
    trainer = pl.Trainer(accelerator = 'gpu')
    predictions = trainer.predict(model, dataloader)
    
    
    try:
        shutil.rmtree("lightning_logs")
        print("Directory removed successfully")
    except OSError as o:
        print(f"Error, {o.strerror}: {path}")
    
    
    if(save_on_disk):
        for b in tqdm(predictions):
            for img, code in zip(b[0], b[1]):
                
                str_code =str(int(code.item()))
                
                img_path = "Data/small_CVUSA/generated_img/" + "0"*(7-len(str_code)) + str_code +".png"
                img = F.to_pil_image(img)
                img.save(img_path)
        
        
    
    
    
