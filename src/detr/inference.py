from datasets import load_from_disk
from src.detr.utils import transform, make_transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
import random
import torch
from src.detr.draw_utils import save_image_with_boxes

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def inference(config):
    dataset = load_from_disk(config['paths']['data'])

    transform_train, transform_test = make_transforms(config)
    output_dir = config['paths']['output_images']

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",do_resize=True,do_pad=True)
    model = DetrForObjectDetection.from_pretrained(config['paths']['model_path'],ignore_mismatched_sizes=True,
        num_labels=1,)
    test_dataset   = dataset["test"].with_transform(lambda x: transform(x, processor, aug_transform=transform_test))
    
    i=random.randint(0, len(test_dataset)-1)
    inputs=test_dataset[i]

    image=inputs['pixel_values']

    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'].unsqueeze(0), pixel_mask=inputs['pixel_mask'].unsqueeze(0))
    
    outputs = processor.post_process_object_detection(outputs, threshold=0.8)
    save_image_with_boxes(image, outputs[0]["boxes"].numpy(), 
                             inputs['labels']['boxes'].numpy(),
                             f"{output_dir}/output_detr.png")




