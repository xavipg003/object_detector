from src.faster_rcnn.train_optuna import exec_optuna
from omegaconf import OmegaConf

config_path = "config/config_faster.yaml"
config= OmegaConf.load(config_path)
exec_optuna(config)