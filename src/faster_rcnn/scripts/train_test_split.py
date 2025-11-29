from pathlib import Path
import random

def split_folder(folder_path, split_ratio=0.5):
    folder = Path(folder_path)
    files = list(folder.glob('*'))
    random.shuffle(files)
    split_point = int(len(files) * split_ratio)
    return files[:split_point], files[split_point:]


train_files, test_files = split_folder('/home/xavi/datos_proyecto/datos_munyeca_raw/images', split_ratio=0.8)
train_folder = Path('/home/xavi/datos_proyecto/datos_munyeca_raw/images/train')
test_folder = Path('/home/xavi/datos_proyecto/datos_munyeca_raw/images/test')

train_folder.mkdir(parents=True, exist_ok=True)
test_folder.mkdir(parents=True, exist_ok=True)

for file in train_files:
    file.rename(train_folder / file.name)

for file in test_files:
    file.rename(test_folder / file.name)