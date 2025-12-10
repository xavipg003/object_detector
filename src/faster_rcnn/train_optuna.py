import torch
import gc
from src.faster_rcnn.train import train
import optuna


def exec_optuna(config):
    def objective(trial):
        map=train(config, trial=trial)
        return map

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize",
                                 sampler=sampler)
    study.optimize(objective, n_trials=50)

    print("\nDetalles del mejor Trial:")
    print(study.best_trial)

    gc.collect()
    torch.cuda.empty_cache()

    print("Training completed.")