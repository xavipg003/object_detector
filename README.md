# Object Detection Pipeline

This repository contains a modular pipeline for training object detection models. It currently supports **Faster R-CNN** and includes an implementation of **DETR**, allowing for easy switching between architectures and configurations.

## Features

  * **Supported Models:**
      * Faster R-CNN (ResNet, MobileNet, etc.)
      * DETR (Transformer-based detection)
  * **Configuration Management:** All training parameters are controlled via centralized **YAML** files.
  * **Hyperparameter Optimization:** Integrated support for **Optuna**.
  * **Data Format:** COCO Style (JSON).

## Directory Structure

The project is organized to separate the source code (`src/`) from the configuration (`config/`) and the execution scripts (located at the root).

```text
.
├── config/             # YAML configuration files
├── data/               # Dataset (Images and labels.json)
├── src/                # Library source code (Models, Engine, Utils)
├── train.py            # Standard training script
├── train_optuna.py     # Hyperparameter optimization script, only for Faster-RCNN
├── eval.py             # Evaluation script (mAP metrics)
├── inference.py        # Inference script for new images
└── requirements.txt
```

## Dataset Structure

The project expects data to be organized in a `data/` folder. The `labels.json` file must follow the **COCO Object Detection** standard.

```text
data/
├── train/
├── test/
├── validation/
└── labels.json
```
Below is an example of the expected JSON structure:

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "0566_1176902707_01_WRI-L2_M015.png",
            "width": 652,
            "height": 1072
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [186, 468, 188, 77],
            "area": 14476,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "object"
        }
    ]
}
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage & Execution

All execution scripts are located at the **root** of the repository.

### 1\. Standard Training (`train.py`)

The main entry point for training. It loads the configuration, initializes the model and dataset, and runs the training loop with logging and checkpointing.

```bash
# Train Faster R-CNN
python train.py fasterrcnn 

# Train DETR
python train.py detr 
```

### 2\. Hyperparameter Optimization (`train_optuna.py`)

A dedicated script that uses **Optuna** to find the best hyperparameters (e.g., learning rate, weight decay). It runs multiple trials based on the search space defined in the config.

```bash
python train_optuna.py
```

### 3\. Evaluation (`eval.py`)

Evaluates a trained model against the test set. It calculates standard metrics such as **mAP** (Mean Average Precision) following COCO standards.

```bash
python eval.py fasterrcnn(or DETR)
```

### 4\. Inference (`inference.py`)

Used to run object detection on new, unseen images. It generates visual results with bounding boxes drawn over the detected objects.

```bash
python inference.py fasterrcnn(or DETR)
```

## Configuration (YAML)

Modify the files in `config/` to adjust your experiments:

  * `config_faster.yaml`: Standard configuration for Faster R-CNN.
  * `config_detr.yaml`: Base configuration for DETR.
