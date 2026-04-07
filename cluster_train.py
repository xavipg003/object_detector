import optuna

storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage("optuna_journal.log")
)

optuna.create_study(
    study_name="faster_rcnn_optuna",
    storage=storage,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
)
print("Estudio creado.")