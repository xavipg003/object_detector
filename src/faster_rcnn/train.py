from pt_lightning.classes import CustomModel, MyDataModule
import lightning as L
import torch
import gc
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
import argparse
import os
import wandb
from pt_lightning.utils import get_end_config, make_callbacks, build_model
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detector")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    config_path = args.config

    config = OmegaConf.load(config_path)
    config, name= get_end_config(config)

    wandb_logger = False
    if not config['training']['debug']:
        wandb.login(key=os.getenv("API_KEY"))  
        wandb_logger = WandbLogger(project='detector_objetos_xavi', name=name)

    datamodule=MyDataModule(config)
    datamodule.setup()
    print(len(datamodule.train_dataset)+len(datamodule.val_dataset))
    
    device='cuda'

    model=build_model(config)
    Lmodel=CustomModel(config,model)
    torch.set_float32_matmul_precision('high')

    callbacks=make_callbacks(name) if wandb_logger else None
    
    trainer = L.Trainer(
        enable_progress_bar=True,
        max_epochs=config['training']['epochs'],
        devices='auto',
        accelerator='gpu',
        default_root_dir=config['paths']['checkpoints'],
        logger=wandb_logger,
        callbacks=callbacks,
        check_val_every_n_epoch=1, val_check_interval=None,
        accumulate_grad_batches=config['training']['accum'],
    )

    trainer.fit(Lmodel, datamodule=datamodule)
    trainer.test(Lmodel, datamodule=datamodule)

    gc.collect()
    torch.cuda.empty_cache()

    print("Training completed.")
