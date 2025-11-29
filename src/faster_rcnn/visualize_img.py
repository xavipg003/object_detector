from pt_lightning.classes import MyDataModule
from pt_lightning.utils import make_transforms
from omegaconf import OmegaConf
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

config_path = "config.yaml"

config = OmegaConf.load(config_path)

transform_train, transform_test = make_transforms(config)

datamodule=MyDataModule(config)
datamodule.setup()

dataset = datamodule.train_dataset

idx = random.randint(0, len(dataset) - 1)

img, target = dataset[idx]

if hasattr(img, 'permute'):
    img_np = img.permute(1, 2, 0).cpu().numpy()
else:
    img_np = np.array(img)

img_np = img_np.astype(np.uint8)

plt.imshow(img_np)
plt.title(f"Random image idx: {idx}")
plt.axis('off')

fig, ax = plt.subplots()
ax.imshow(img_np)
if "boxes" in target:
    boxes = target["boxes"]
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
ax.axis('off')

output_path = "../output_imgs/visualized_image.png"

fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f"Image saved to {output_path}")
