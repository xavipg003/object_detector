from src.faster_rcnn.pt_lightning.utils import build_model, gethyperparameters, save_image
from src.faster_rcnn.pt_lightning.classes import MyDataModule, CustomModel
import random
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def inference(config):
    config=gethyperparameters(config, None, from_name=True)
    inf_name=config['inf_name']
    output_dir=config['paths']['output_images']
    model_path=config['paths']['model_path']

    datamodule=MyDataModule(config)
    datamodule.setup()

    model=build_model(config)
    Lmodel=CustomModel(config,model,None)
    state_dict = torch.load(f'{model_path}/{inf_name}')['state_dict']
    Lmodel.load_state_dict(state_dict)

    test_dataset = datamodule.test_dataset
    i=random.randint(0, len(test_dataset)-1)
    inputs=test_dataset[i]
    Lmodel.eval()

    with torch.no_grad():
        prediction = Lmodel.model([inputs[0]])
        save_image(inputs[0].permute(1, 2, 0).numpy(), f"{output_dir}/output.png", 
                   ground_truth=inputs[1]['boxes'].numpy(),
                    prediction=prediction[0]['boxes'].numpy(),
                    scores=prediction[0]['scores'].numpy(),
                    threshold=0.75)
    
