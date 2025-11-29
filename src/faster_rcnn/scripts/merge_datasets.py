import json

def merge_json_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Ensure unique image_id and annotation_id
    max_image_id = max(image['id'] for image in data1['images'])
    max_annotation_id = max(ann['id'] for ann in data1['annotations'])

    # Update image_id and annotation_id in the second dataset
    for image in data2['images']:
        image['id'] += max_image_id

    for annotation in data2['annotations']:
        annotation['id'] += max_annotation_id
        annotation['image_id'] += max_image_id

    # Merge datasets
    merged_data = {
        'images': data1['images'] + data2['images'],
        'annotations': data1['annotations'] + data2['annotations'],
        'categories': data1['categories']  # Assuming categories are the same
    }

    # Write merged data to output file
    with open(output_file, 'x') as out_file:
        json.dump(merged_data, out_file, indent=4)

if __name__ == "__main__":
    file1 = '/home/xavi/datos_proyecto/datos/Annotations/coco/COCO_fracture_masks.json'
    file2 = '/home/xavi/datos_proyecto/datos_kaggle/labels.json'
    output_file = '/home/xavi/datos_proyecto/datos_merged/labels.json'

    merge_json_files(file1, file2, output_file)
    print(f"Merged JSON saved to {output_file}")