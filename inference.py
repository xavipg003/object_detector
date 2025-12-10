from omegaconf import OmegaConf
import argparse
from src.faster_rcnn.inference import inference as inference_fasterrcnn
from src.detr.inference import inference as inference_detr

parser = argparse.ArgumentParser(description="Inference with object detector")
parser.add_argument('model_type', type=str, help="Type of model to make inference on test dataset, 'fasterrcnn' or 'detr'")
args = parser.parse_args()
model_type = args.model_type

if model_type == 'fasterrcnn':
    config_path = "config/config_faster.yaml"
elif model_type == 'detr':
    config_path = "config/config_detr.yaml"
else:
    raise ValueError("Invalid model type. Choose 'fasterrcnn' or 'detr'.")

config = OmegaConf.load(config_path)

if model_type == 'fasterrcnn':
    inference_fasterrcnn(config)
elif model_type == 'detr':
    inference_detr(config)