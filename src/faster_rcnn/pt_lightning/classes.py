import optuna
import torch
import lightning as L
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import random_split

from torch.utils.data import Dataset

import os
import sys
from PIL import Image

import json

import numpy as np
import torch

from PIL import ImageFile

from pathlib import Path
from .utils import make_transforms, convert_to_8bit
from transformers import DetrImageProcessor


class CustomModel(L.LightningModule):
    def __init__(self, config, model, trial):
        super(CustomModel, self).__init__()
        self.model = model
        self.learning_rate = config['training']['learning_rate']
        self.map = MeanAveragePrecision(iou_type="bbox", 
                                        extended_summary=True, 
                                        class_metrics=False, 
                                        iou_thresholds=[0.5],
                                        average='macro',
                                        )
        self.map_test = self.map.clone()

        self.batch_size = config['training']['batch_size']
        self.model_type = config['model']['model_type']

        self.config = config
        self.trial = trial
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        images = list(image for image in images)
        images=torch.stack(images)
        
        
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        
        loss_dict = self.model(images,targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log('train_loss', losses, on_epoch=True, prog_bar=True, logger=True, on_step=False, batch_size=self.batch_size)
        
        return losses
       
    def validation_step(self, batch, batch_idx):
        images, targets = batch

        images = list(image for image in images)
        images=torch.stack(images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        predictions = self.model(images)

        self.map.update(predictions, targets)

    def on_validation_epoch_end(self):
        metrica= self.map.compute()
        
        precision = metrica['precision'][0, 70, 0, 0, 1]
        recall = metrica['recall'][0, 0, 0, 1]

        self.log("val_map", metrica["map"], prog_bar=True, batch_size=1)
        self.log("val_precision", precision, prog_bar=True, batch_size=1)
        self.log("val_recall", recall, prog_bar=True, batch_size=1)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        self.log("val_f1", f1_score, prog_bar=True, batch_size=self.batch_size)

        if self.trial is not None:
            self.trial.report(metrica["map"], self.current_epoch)

            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            self.map.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch

        images = list(image for image in images)
        images=torch.stack(images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        predictions = self.model(images)

        self.map_test.update(predictions, targets)

    def on_test_epoch_end(self):
        metrica= self.map_test.compute()
        
        precision = metrica['precision'][0, 70, 0, 0, 1]
        recall = metrica['recall'][0, 0, 0, 1]

        self.log("test_map", metrica["map"], prog_bar=True, batch_size=1)
        self.log("test_precision", precision, prog_bar=True, batch_size=1)
        self.log("test_recall", recall, prog_bar=True, batch_size=1)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        self.log("test_f1", f1_score, prog_bar=True, batch_size=self.batch_size)
        self.map_test.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4, momentum=0.9)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-4),
            'monitor': 'train_loss',
            'interval': 'epoch' 
        }

        return [optimizer], [scheduler]
    
class AlbumentationsImageDataset(Dataset):
    def __init__(self, config, img_dir, file, transform=None, bbox_mode=0):
        self.img_dir = img_dir
        self.transform = transform
        self.file = file
        self.bbox_mode = bbox_mode
        self.model_type = config['model']['model_type']
        self.config = config

        with open(self.file, 'r') as f:
            self.data = json.load(f)

        #ImageFile.LOAD_TRUNCATED_IMAGES = True
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        data=self.data
        
        img_name=os.listdir(self.img_dir)[idx]

        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = np.array(Image.open(img_path))
        except:
            print("TRUNCADOS")
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            image = np.array(Image.open(img_path))
        
        image = convert_to_8bit(image)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

            image = np.repeat(image, 3, axis=-1)

        img_id = [img['id'] for img in data['images'] if img['file_name'] == img_name][0]
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        label_list=[]
        boxes=[]

        for ann in annotations:
            bbox = ann['bbox']

            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h

            boxes.append([x_min, y_min, x_max, y_max])

            label_list.append(ann['category_id']+1)
            
            labels = label_list

        if len(annotations) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        
        
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
                
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            if len(boxes)==0:
                boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return image, target

class MyDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.train_image_dir=Path(config['paths']['train'])
        self.test_image_dir=Path(config['paths']['test'])
        self.validation_image_dir=Path(config['paths']['validation'])
        self.label_file=Path(config['paths']['labels'])
        self.transform_train, self.transform_test = make_transforms(config)
        self.batch_size = config['training']['batch_size']
        self.model_type = config['model']['model_type']
        self.config=config

        if os.getenv("OMP_NUM_THREADS") is not None:
            self.nworkers=int(os.getenv("OMP_NUM_THREADS"))
            print(f"Usando numworkers de entorno: {self.nworkers}")
        else:
            self.nworkers=31

    def setup(self, stage=None):
        self.train_dataset=AlbumentationsImageDataset(self.config, img_dir=self.train_image_dir, file=self.label_file, transform=self.transform_train)
        self.val_dataset=AlbumentationsImageDataset(self.config, img_dir=self.validation_image_dir, file=self.label_file, transform=self.transform_test)
        self.test_dataset=AlbumentationsImageDataset(self.config, img_dir=self.test_image_dir, file=self.label_file, transform=self.transform_test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=self.nworkers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=self.nworkers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=self.nworkers)