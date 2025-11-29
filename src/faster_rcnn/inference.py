from pt_lightning.classes import MyDataModule, CustomModel
from build_model import build_model
from pt_lightning.transforms_callbacks import make_transforms
from pt_lightning.utils import showImg
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

size=640
transform_train, transform_test = make_transforms(OmegaConf.load('config.yaml'))
nclasses=2


train_image_dir=Path('../data_helmet/train/')
test_image_dir=Path('../data_helmet/test/')
label_file=Path('../data_helmet/labels.json')

datamodule=MyDataModule(train_image_dir, test_image_dir, label_file, transform_train=transform_train, transform_test=transform_test, bit8=False)
datamodule.setup()
len(datamodule.test_dataset)

auto_name=True
checkpoint=True
model_type='fasterrcnn' #transformer, fasterrcnn, custom
backbone_name="swin_base_patch4_window12_384"
fpn=False

if checkpoint:
    dir='../checkpoints/'
else:
    dir='../models/'

if auto_name:
    model_name = sorted(Path(dir).iterdir())[0].name
else:
    model_name = 'transformer1408swin.ckpt'

model_name=f"{dir}{model_name}"

model = build_model(model_type, nclasses, backbone_name, fpn=fpn, size=size)

Lmodel=CustomModel.load_from_checkpoint(model_name, model=model)
Lmodel.to('cpu')
Lmodel.eval();
print(len(datamodule.test_dataset))
print(len(datamodule.train_dataset))

while True:
    random_idx=np.random.randint(0, len(datamodule.test_dataset), 1)[0]
    img, target=datamodule.test_dataset[random_idx]
    if target['boxes'].shape[0] == 0:
        print(random_idx)
    break
thresh=0.6

img, target=datamodule.test_dataset[random_idx]

output=Lmodel([img])
print(output[0]['scores'])

print(target)
showImg(img, target['boxes'].detach().numpy(), output[0]['boxes'].detach().numpy(), output[0]['scores'].detach().numpy(), threshold=thresh, 
        all=False)