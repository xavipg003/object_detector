from datasets import load_from_disk
from utils import transform, make_transforms, get_name
from transformers import DetrImageProcessor, DetrForObjectDetection
import argparse
from omegaconf import OmegaConf
import random
import torch
from draw_utils import save_image_with_boxes

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detector")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    config_path = args.config

    config = OmegaConf.load(config_path)
    name= get_name(config)

    debug = config['training']['debug']


    dataset = load_from_disk(config['paths']['data'])

    transform_train, transform_test = make_transforms(config)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",do_resize=True,do_pad=True)
    model = DetrForObjectDetection.from_pretrained("../../models/detr",ignore_mismatched_sizes=True,
        num_labels=1,)
    test_dataset   = dataset["test"].with_transform(lambda x: transform(x, processor, aug_transform=transform_test))
    
    i=random.randint(0, len(test_dataset)-1)
    i=0
    inputs=test_dataset[i]
    print(dataset['test'][i])
    print(inputs)

    image=inputs['pixel_values']

    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'].unsqueeze(0), pixel_mask=inputs['pixel_mask'].unsqueeze(0))
    
    outputs = processor.post_process_object_detection(outputs, threshold=0.8)
    print(outputs)
    save_image_with_boxes(image, outputs[0]["boxes"].numpy(), 
                             inputs['labels']['boxes'].numpy(),
                             "../../output_imgs/output_detr.png")



