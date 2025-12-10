from src.faster_rcnn.pt_lightning.utils import build_model, gethyperparameters
from src.faster_rcnn.pt_lightning.classes import MyDataModule, CustomModel
import torch
import lightning as L

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def test(config):
    config=gethyperparameters(config, None, from_name=True)
    inf_name=config['inf_name']

    datamodule=MyDataModule(config)
    datamodule.setup()

    model=build_model(config)
    Lmodel=CustomModel(config,model,None)
    state_dict = torch.load(f"{config['paths']['model_path']}/{inf_name}")['state_dict']
    Lmodel.load_state_dict(state_dict)

    torch.set_float32_matmul_precision('high')

    trainer = L.Trainer(
            enable_progress_bar=True,
            max_epochs=config['training']['epochs'],
            devices='auto',
            accelerator='gpu',
            default_root_dir=config['paths']['checkpoints'],
            logger=None
        )
    trainer.test(Lmodel, datamodule=datamodule)
