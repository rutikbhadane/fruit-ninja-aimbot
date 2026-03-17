import os
import cv2
import random
from ultralytics import YOLO

def main():
    model_path = r"c:\Fruit-Ninja\runs\detect\fruit_ninja_aimbot6\weights\best.pt"
    val_images_dir = r"c:\Fruit-Ninja\dataset\images\val"
    output_dir = r"c:\tmp\inference_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Pick random validation images
    val_images = [f for f in os.listdir(val_images_dir) if f.endswith('.png')]
    random.seed(42)
    sample_images = random.sample(val_images, min(10, len(val_images)))
    
    print(f"Running inference on {len(sample_images)} images with LOW confidence threshold...")
    
    for img_name in sample_images:
        img_path = os.path.join(val_images_dir, img_name)
        
        # Run inference with very low confidence to see if model learned anything
        results = model(img_path, conf=0.05)
        
        for r in results:
            boxes = r.boxes
            num_detections = len(boxes)
            print(f"  {img_name}: {num_detections} detections")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = r.names[cls_id]
                print(f"    -> {cls_name} (conf={conf:.3f})")
            
            im_array = r.plot()
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, im_array)

if __name__ == "__main__":
    main()
