import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from swin_utils.build import make_swin
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights, vgg16, VGG16_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

def save_image(image, path, ground_truth=[], prediction=[], scores=[], threshold=0.5):
    fig, ax = plt.subplots(1)
    ax.axis('off')
    ax.imshow(image)

    for i,box in enumerate(prediction):
        score = scores[i].item()
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = box

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    for box in ground_truth:
        xmin, ymin, xmax, ymax = box

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='green',
            facecolor='none'
        )
        ax.add_patch(rect)
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=150)
    
    plt.close(fig)

def gethyperparameters(config, trial, from_name=False):
    if not from_name:
        config['model']['model_type'] = trial.suggest_categorical('model_type', ["swin", "fasterrcnn"])
        if config['model']['model_type']=="swin":
            config['model']['backbone_name'] = trial.suggest_categorical('backbone_name',                                                                               ["swin_base_patch4_window7_224", "swin_tiny_patch4_window7_224"])
            config['model']['lora'] = trial.suggest_categorical('lora', [True, False])
            config['model']['fpn'] = trial.suggest_categorical('fpn', [True, False])
        config['training']['batch_size'] = trial.suggest_categorical('batch_size', [1, 2, 4])
        config['training']['learning_rate'] = trial.suggest_float('learning_rate',
                                                                            1e-6, 1e-3, log=True)
        config['training']['accum'] = trial.suggest_categorical('accum', [1, 2, 4, 8])
    else:
        name=config['inf_name']
        parts=name.split('_')
        config['model']['model_type']=parts[0]
        if config['model']['model_type']=="swin":
            config['model']['backbone_name']="_".join(parts[1:6]).split('-')[1]
            config['model']['fpn']=parts[6].split('-')[1]=="True"
            config['model']['lora']=parts[7].split('-')[1]=="True"
        else:
            config['model']['lora']=parts[1].split('-')[1]=="True"
    return config

def build_model(config):
    if config['model']['model_type']=="swin":
        model = make_swin(config)
    elif config['model']['model_type']=="fasterrcnn":
        model = fasterrcnn_resnet50_fpn(num_classes=config['model']['nclasses'],
                                        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
                                        trainable_backbone_layers=5,
                                        max_size=config['size'], min_size=config['size'])
    elif config['model']['model_type']=="custom":
        backbone= vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        backbone.out_channels = 512

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )

        model = FasterRCNN(
            backbone=backbone,
            num_classes=config['model']['nclasses'], 
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['model_type']}. Choose 'transformer' or 'fasterrcnn' or 'custom'.")
    return model

def showImg(image, true_labels, predicted_labels=[], scores=[], threshold=0.5, all=False):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())

    for label in true_labels:
        
        x_min, y_min, x_max, y_max = label

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            label='Faster R-CNN'
        )
        ax.add_patch(rect)

    for i, label in enumerate(predicted_labels):
        
        x_min, y_min, x_max, y_max = label

        if scores[i] < threshold:
            if all:
                edgecolor = 'blue'
            else:
                edgecolor = 'none'
        else:
            edgecolor = 'yellow'
        # if i < 3:
        #     edgecolor = 'green'
        # else:
        #     edgecolor = 'none'

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor=edgecolor,
            facecolor='none',
            label='Faster R-CNN'
        )
        ax.add_patch(rect)

    plt.savefig('../inference_imgs/inf.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def increase_contrast_brightness(image, alpha=1.1, beta0=40, **kwargs):
    brillo_medio=np.mean(image)
    beta=beta0-brillo_medio

    image = adjust_gamma(image, gamma=1.2)

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def build_transforms(transform_cfg):
    transforms = []
    for t in transform_cfg:
        t = dict(t)
        name = t.pop('name')
        cls = getattr(A, name, None)
        if cls is None and name == 'ToTensorV2':
            cls = ToTensorV2
        if cls:
            transforms.append(cls(**t))
    return transforms

def get_name(config):
    if config['model']['model_type'] == "swin":
        name = (
            f"{config['model']['model_type']}"
            f"_backbone-{config['model']['backbone_name']}"
            f"_fpn-{config['model']['fpn']}"
            f"_lora-{config['model']['lora']}"
            f"_bs-{config['training']['batch_size']}"
            f"_lr-{config['training']['learning_rate']}"
            f"_accum-{config['training']['accum']}"
            f"_size-{config['size']}"
        )
    else:
        name = (
            f"{config['model']['model_type']}"
            f"_lora-{config['model']['lora']}"
            f"_bs-{config['training']['batch_size']}"
            f"_lr-{config['training']['learning_rate']}"
            f"_accum-{config['training']['accum']}"
            f"_size-{config['size']}"
        )
    return name

def make_transforms(config):
    train_cfg = config.get('train_transforms', {})
    test_cfg = config.get('test_transforms', {})

    # Construir transforms
    transform_train = A.Compose(
        build_transforms(train_cfg),
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )
    transform_test = A.Compose(
        build_transforms(test_cfg),
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )
    return transform_train, transform_test

def make_callbacks(config,name):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_map',
        dirpath=config['paths']['checkpoints'],
        filename=name+'-{val_map:.2f}',
        save_top_k=1,
        mode='max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [lr_monitor,
                 checkpoint_callback
                 ]
    return callbacks

def convert_to_8bit(image_16bit):
    """Convierte una imagen de 16 bits (NumPy array) a 8 bits."""
    min_16bit = image_16bit.min()
    max_16bit = image_16bit.max()

    if max_16bit > min_16bit:
        # Escalar al rango 0-1
        image_scaled = (image_16bit - min_16bit) / (max_16bit - min_16bit)

        # Escalar al rango 0-255 y convertir a uint8
        image_8bit = (image_scaled * 255).astype(np.uint8)
    else:
        # Si todos los valores son iguales, crear una imagen de 8 bits con ese valor (o 0)
        image_8bit = np.full_like(image_16bit, 0, dtype=np.uint8)

    return image_8bit
