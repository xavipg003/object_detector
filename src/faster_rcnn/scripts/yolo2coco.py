import os
import json
import numpy as np
from PIL import Image
import time

img_id=1
images = []
annotations = []
ann_id = 1
yolomode="xcycwh" #xcycwh or xnyn

output_json = os.path.expanduser('../../data/labels.json')

images_dir = os.path.expanduser('../../data/images/')
labels_dir = os.path.expanduser('../../data/annotations/yolov5/labels/')


for img_file in os.listdir(images_dir):
    if not img_file.endswith(".png"):
        continue

    print(img_file)

    img_path = os.path.join(images_dir, img_file)

    image = np.array(Image.open(img_path).convert('RGB'))
    width, height = image.shape[1], image.shape[0]

    label_file = os.path.join(labels_dir, img_file.replace(".png", ".txt"))
    if not os.path.exists(label_file):
        continue

    with open(label_file, "r") as f:
        images.append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        for line in f.readlines():
            clase= int(line.strip().split()[0])

            if clase ==3:
                x_center, y_center, w, h = list(map(float, line.strip().split()[1:]))
                original_labels= [x_center, y_center, w, h]

                xmin = max(0,int((x_center - w / 2) * width))
                ymin = max(0,int((y_center - h / 2) * height))

                w = int(w * width)
                h = int(h * height)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [xmin, ymin, w, h],
                    "label_file": label_file,
                    "original_labels": original_labels,
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
        img_id += 1

categories = [{"id": 0, "name": "objeto"}]

coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_json, "w") as f:
    json.dump(coco_format, f, indent=4)
