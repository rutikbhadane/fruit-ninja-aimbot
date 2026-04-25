import os
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"           # Your current trained YOLOv8 model
RAW_IMAGES_DIR = "raw_images"    # Folder containing new, unlabelled gameplay screenshots
CONF_THRESH = 0.40               # Keep this slightly high so it only auto-labels confident detections

def auto_annotate():
    print(f"Loading YOLOv8 model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    img_dir = Path(RAW_IMAGES_DIR)
    if not img_dir.exists():
        img_dir.mkdir(parents=True)
        print(f"\n[INFO] Created '{RAW_IMAGES_DIR}' folder.")
        print("Please place your unlabelled screenshots in this folder and run the script again!")
        return

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"\n[INFO] No images found in {RAW_IMAGES_DIR}.")
        return
        
    print(f"\nFound {len(images)} images to auto-annotate.")
    
    annotated_count = 0
    for img_path in images:
        # Run inference
        results = model.predict(source=str(img_path), conf=CONF_THRESH, verbose=False)
        result = results[0]
        
        if len(result.boxes) == 0:
            continue # No detections, skip creating a label file
            
        # Create YOLO format .txt file alongside the image
        txt_path = img_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for box in result.boxes:
                # YOLO format: class_id center_x center_y width height (normalized 0-1)
                cls_id = int(box.cls[0])
                cx, cy, w, h = box.xywhn[0].tolist()
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                
        annotated_count += 1
        print(f"Annotated: {img_path.name} -> Found {len(result.boxes)} objects")
        
    print(f"\n✅ Done! Successfully generated YOLO labels for {annotated_count} images.")
    print("Next step: Open these in your labeling tool (like Roboflow or CVAT) to fix any mistakes, then train YOLOv11!")

if __name__ == "__main__":
    auto_annotate()
