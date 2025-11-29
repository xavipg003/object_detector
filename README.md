# Object Detection Pipeline

This repository contains a modular pipeline for training object detection models. It currently supports **Faster R-CNN** with various backbones and includes an implementation of **DETR** (Work in Progress).

## Features

* **Supported Models:**
    * Faster R-CNN (ResNet, MobileNet, etc.)
    * DETR (Work in Progress)
* **Data Format:** COCO Style (JSON).
* **Modular Design:** Easy to swap backbones and configuration.

## Dataset Structure

To train a model, the project expects the data to be organized in a `data/` folder at the project root. The structure must strictly follow this hierarchy:

```text
.
├── data/
│   ├── train/          # Training images
│   ├── test/           # Testing images
│   ├── validation/     # Validation images
│   └── labels.json     # COCO style annotations
├── src/
└── ...

```

The labels.json file must follow the COCO Object Detection standard, including images, annotations, and categories keys.

Below is an example of the expected JSON structure:

```text

{
    "images": [
        {
            "id": 1,
            "file_name": "0566_1176902707_01_WRI-L2_M015.png",
            "width": 652,
            "height": 1072
        },
        {
            "id": 2,
            "file_name": "4313_0451538895_01_WRI-L2_M012.png",
            "width": 508,
            "height": 964
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [
                186,
                468,
                188,
                77
            ],
            "area": 14476,
            "iscrowd": 0,
            "label_file": "../../data/annotations/yolov5/labels/file_name.txt",
            "original_labels": [
                0.431748,
                0.472948,
                0.289877,
                0.072761
            ]
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "object"
        }
    ]
}
