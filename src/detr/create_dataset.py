import json
import os
from PIL import Image
from datasets import Dataset, DatasetDict


def create_split(split_name):
    split_dir = os.path.join(root_dir, split_name)
    images, objects = [], []

    for img_info in coco_data["images"]:
        img_path = os.path.join(split_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        anns = image_to_anns[img_info["id"]]
        if not anns:
            continue  # saltar imágenes sin anotaciones

        boxes = [a["bbox"] for a in anns]  # formato [x, y, w, h]
        categories = [a["category_id"] for a in anns]

        images.append(Image.open(img_path).convert("RGB"))
        objects.append({
            "bbox": boxes,
            "category": categories,
            "image_id": img_info["id"],
            "file_name": img_info["file_name"]
        })

    return Dataset.from_dict({"image": images, "objects": objects})


if __name__ == "__main__":
    root_dir = "../../data/"

    with open(os.path.join(root_dir, "labels.json"), "r") as f:
        coco_data = json.load(f)
    # Crear diccionario imagen_id → info de imagen
    id_to_image = {img["id"]: img for img in coco_data["images"]}

    # Crear diccionario imagen_id → lista de anotaciones
    from collections import defaultdict
    image_to_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_to_anns[ann["image_id"]].append(ann)

    train_dataset = create_split("train")
    val_dataset   = create_split("validation")
    test_dataset  = create_split("test")

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    dataset.save_to_disk("dataset")

