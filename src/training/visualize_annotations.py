import cv2
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt

def draw_yolo_boxes(image_path, label_path, class_names):
    """
    Reads an image and its YOLO .txt label, draws the bounding boxes, and shows it.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
        
    h, w, _ = image.shape
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        cls_id = int(parts[0])
        # YOLO coordinates are normalized [x_center, y_center, width, height]
        x_center, y_center, bbox_w, bbox_h = map(float, parts[1:])
        
        # Convert normalized coordinates to absolute pixels
        x_center *= w
        y_center *= h
        bbox_w *= w
        bbox_h *= h
        
        # Calculate top-left and bottom-right coords
        x1 = int(x_center - (bbox_w / 2))
        y1 = int(y_center - (bbox_h / 2))
        x2 = int(x_center + (bbox_w / 2))
        y2 = int(y_center + (bbox_h / 2))
        
        # Determine color (green for Fruit, red for Bomb - assuming from dataset.yaml)
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255) 
        class_name = class_names.get(cls_id, f"Class {cls_id}")
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    # Convert BGR (OpenCV) to RGB (Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

if __name__ == "__main__":
    # Settings based on your recent auto annotate run
    IMAGE_DIR = Path("c:/Fruit-Ninja/dataset2/images/train")
    LABEL_DIR = Path("c:/Fruit-Ninja/dataset2/labels/train")
    
    # Class names from dataset.yaml
    CLASSES = {0: "Fruit", 1: "Bomb"}
    
    # Number of random samples to visualize
    NUM_SAMPLES = 5
    
    # Get all overlapping pairs
    valid_pairs = []
    
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        print(f"Error: Could not find image or label directory.")
        exit(1)
        
    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith(".txt"):
            continue
            
        # Try to find corresponding image (.png, .jpg)
        base_name = os.path.splitext(label_file)[0]
        
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = IMAGE_DIR / (base_name + ext)
            if img_path.exists():
                valid_pairs.append((img_path, LABEL_DIR / label_file))
                break
                
    if not valid_pairs:
        print("No image/label pairs found!")
        exit(1)
        
    print(f"Found {len(valid_pairs)} annotated images. Visualizing {NUM_SAMPLES} random samples...")
    
    # Select random samples
    if len(valid_pairs) > NUM_SAMPLES:
        samples = random.sample(valid_pairs, NUM_SAMPLES)
    else:
        samples = valid_pairs
        
    # Set up matplotlib figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Auto-Annotation Verification", fontsize=16)
    
    # Visualize each sample
    for i, (img_path, label_path) in enumerate(samples):
        print(f"Processing: {img_path.name}")
        annotated_img = draw_yolo_boxes(img_path, label_path, CLASSES)
        
        if annotated_img is not None:
            # Add subplot
            ax = fig.add_subplot(1, len(samples), i + 1)
            ax.imshow(annotated_img)
            ax.set_title(img_path.name)
            ax.axis('off')
            
    print("Opening visualization window...")
    plt.tight_layout()
    plt.show()    
    print("Done visualizing.")
