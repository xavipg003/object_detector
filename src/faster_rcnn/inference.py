from pt_lightning.utils import make_transforms, build_model, gethyperparameters, save_image
from pt_lightning.classes import MyDataModule, CustomModel
import argparse
from omegaconf import OmegaConf
import random
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detector")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    config_path = args.config

    config = OmegaConf.load(config_path)
    config=gethyperparameters(config, None, from_name=True)
    inf_name=config['inf_name']

    datamodule=MyDataModule(config)
    datamodule.setup()

    transform_train, transform_test = make_transforms(config)

    model=build_model(config)
    Lmodel=CustomModel(config,model,None)
    state_dict = torch.load(f'../../models/fasterrcnn/{inf_name}')['state_dict']
    Lmodel.load_state_dict(state_dict)

    test_dataset = datamodule.test_dataset
    i=random.randint(0, len(test_dataset)-1)
    inputs=test_dataset[i]
    Lmodel.eval()

    with torch.no_grad():
        prediction = Lmodel.model([inputs[0]])
        save_image(inputs[0].permute(1, 2, 0).numpy(), '../../output_imgs/output.png', 
                   ground_truth=inputs[1]['boxes'].numpy(),
                    prediction=prediction[0]['boxes'].numpy(),
                    scores=prediction[0]['scores'].numpy(),
                    threshold=0.75)
