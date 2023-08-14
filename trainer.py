from modules.retarget import retarget 
import pytorch_lightning as pl
from options import get_parser
import os
                
if __name__ == '__main__':
    opt = get_parser()
    mode = opt.mode
    opt.device = f"cuda:{opt.gpu[0]}"
    #set seed
    pl.seed_everything(opt.seed, workers=True)

    result_dir = f'./results/'
    
    wandb_logger = None
    model = retarget(opt)
        

    #load checkpoint
    module_path = opt.model_path
    print("MODULE PATH: ",module_path)        
           
    model.to(opt.device)
    print("INFERENCE")
    model.inference()
    
