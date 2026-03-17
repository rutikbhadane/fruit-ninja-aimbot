import os
from pathlib import Path
from ultralytics import YOLO

def auto_annotate(image_dir, model_path="best.pt", conf_threshold=0.5):
    """
    Automatically annotates images in a given directory using a trained YOLOv8 model.
    Only annotates images that do not already have a corresponding label file.
    Existing images are never modified.
    """
    
    label_dir = None
    
    # Try to infer YOLO standard structure: images/xxx -> labels/xxx
    p = Path(image_dir)
    if "images" in p.parts:
        parts = list(p.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        inferred_label_dir = Path(*parts)
        if inferred_label_dir.exists() or inferred_label_dir.parent.exists():
            label_dir = inferred_label_dir
            label_dir.mkdir(parents=True, exist_ok=True)
            print(f"Recognized YOLO structure. Using label directory: {label_dir}")
    
    if label_dir is None:
        # Default to placing labels alongside images if no standard structure
        label_dir = Path(image_dir)
    
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Supported image formats
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images_to_process = []
    
    print(f"\nScanning directory: {image_dir}")
    for file_name in os.listdir(image_dir):
        file_path = p / file_name
        
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            # Add to list regardless of existing label
            images_to_process.append(str(file_path))
            
            # Optional: Tell the user we are overwriting
            expected_label_file = label_dir / (file_path.stem + ".txt")
            if expected_label_file.exists():
                print(f"  [Overwriting] Existing label will be replaced for: {file_name}")
                
    if not images_to_process:
        print("\nNo new images found to annotate!")
        return
        
    print(f"\nFound {len(images_to_process)} unannotated images. Starting inference...")
    
    # YOLO predict automatically saves .txt files if save_txt=True.
    # However, if we pass a directory, it creates a new 'runs/detect/predict/labels' folder
    # which we'd have to copy over. To safely place them in our specific labels_dir 
    # without touching Images and managing exact paths, we'll process them in a loop or explicitly move them.
    # The safest approach is predicting in memory and writing our own cleanly formatted .txt files.
    
    for img_path in images_to_process:
        print(f"  Annotating: {Path(img_path).name}")
        results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
        
        # Get predictions for the first image (since we passed one image)
        result = results[0]
        
        if len(result.boxes) == 0:
            continue  # No objects found, do not create an empty file (optional, but standard usually)
            
        txt_path = label_dir / (Path(img_path).stem + ".txt")
        
        with open(txt_path, "w") as f:
            # Each box in result.boxes contains cls, conf, xywhn (normalized coordinates)
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                # Extract normalized x_center, y_center, width, height
                x_center, y_center, width, height = box.xywhn[0].tolist()
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
    print(f"\nSuccessfully auto-annotated {len(images_to_process)} images!")
    print(f"Annotations saved to: {label_dir}")

if __name__ == "__main__":
    # Example Usage: Replace with the actual paths!
    
    # Path to the unannotated images folder
    # e.g. "c:/Fruit-Ninja/dataset/images/train"
    TARGET_IMAGE_DIR = "c:/Fruit-Ninja/dataset2/images/train" 
    
    # Path to the .pt file you downloaded from Colab
    MODEL_PATH = "best.pt" 
    
    auto_annotate(image_dir=TARGET_IMAGE_DIR, model_path=MODEL_PATH)
