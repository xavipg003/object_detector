from datasets import load_from_disk
from utils import transform, collate_fn, make_transforms, get_end_config
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
import argparse
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detector")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    config_path = args.config

    config = OmegaConf.load(config_path)
    name= get_end_config(config)

    debug = config['training']['debug']


    dataset = load_from_disk(config['paths']['data'])

    transform_train, transform_test = make_transforms(config)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",do_resize=True,do_pad=True)
    train_dataset = dataset["train"].with_transform(lambda x: transform(x, processor, aug_transform=transform_train))
    val_dataset   = dataset["validation"].with_transform(lambda x: transform(x, processor, aug_transform=transform_test))

    #print_sample(train_dataset[0])

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        ignore_mismatched_sizes=True,
        num_labels=1,
    )

    print(f"Debug mode: {debug}")

    training_args = TrainingArguments(
        output_dir=config['paths']['checkpoints'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        num_train_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=1e-4,
        save_strategy="epoch",
        logging_dir="./logs",
        remove_unused_columns=False,
        eval_strategy="epoch",
        run_name=name,
        report_to="all" if not debug else "none",
        logging_strategy="epoch" if not debug else "no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    trainer.train()