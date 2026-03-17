import os
import shutil
import json
from sklearn.model_selection import train_test_split

# Define paths
dataset_path = 'dataset/train'
images_path = os.path.join(dataset_path, 'images')
annotations_path = os.path.join(dataset_path, 'annotations')

# Create directories
os.makedirs(os.path.join(images_path, 'train/fruit'), exist_ok=True)
os.makedirs(os.path.join(images_path, 'train/bomb'), exist_ok=True)
os.makedirs(os.path.join(images_path, 'val/fruit'), exist_ok=True)
os.makedirs(os.path.join(images_path, 'val/bomb'), exist_ok=True)
os.makedirs(os.path.join(images_path, 'test/fruit'), exist_ok=True)
os.makedirs(os.path.join(images_path, 'test/bomb'), exist_ok=True)

# Load annotations
with open('annotations.json', 'r') as f:
    annotations = json.load(f)

# Split data
train_annotations, temp_annotations = train_test_split(annotations, test_size=0.3, random_state=42)
val_annotations, test_annotations = train_test_split(temp_annotations, test_size=0.5, random_state=42)

# Save annotations
with open(os.path.join(annotations_path, 'train.json'), 'w') as f:
    json.dump(train_annotations, f)
with open(os.path.join(annotations_path, 'val.json'), 'w') as f:
    json.dump(val_annotations, f)
with open(os.path.join(annotations_path, 'test.json'), 'w') as f:
    json.dump(test_annotations, f)

# Move images to respective directories
for split, data in zip(['train', 'val', 'test'], [train_annotations, val_annotations, test_annotations]):
    for anno in data:
        image_path = anno['image']
        label = anno['label']
        destination_path = os.path.join(images_path, split, label, os.path.basename(image_path))
        shutil.copy(image_path, destination_path)
