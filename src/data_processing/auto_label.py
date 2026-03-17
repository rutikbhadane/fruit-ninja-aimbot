"""
Auto-labeling script for Fruit Ninja game screenshots (v5 - Hybrid).

Strategy:
1. FRUITS: Use pretrained YOLOv8n (COCO) to detect fruits (apple, orange, banana, sports ball).
2. BOMBS: Use shape-filtered color detection (v4 logic).
3. INTEGRATION: Merge detections, apply game area mask, and save as YOLO labels.
"""

import cv2
import numpy as np
import os
import random
from ultralytics import YOLO

# Classes from COCO that might match game fruits
FRUIT_CLASSES = {
    46: 'banana',
    47: 'apple',
    49: 'orange',
    32: 'sports ball', # For melons/round fruits
    54: 'donut'        # Sometimes matches sliced round fruits
}

def detect_layout(img):
    h, w = img.shape[:2]
    top_strip = img[0:40, :, :]
    top_brightness = np.mean(top_strip)
    if top_brightness > 100:
        return int(h * 0.12), int(h * 0.88), int(w * 0.15), int(w * 0.85)
    else:
        return int(h * 0.05), int(h * 0.78), int(w * 0.08), int(w * 0.82)

def get_game_area_mask(img):
    h, w = img.shape[:2]
    gt, gb, gl, gr = detect_layout(img)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[gt:gb, gl:gr] = 255
    return mask

def detect_bombs_color(hsv, game_mask, img_h, img_w):
    """
    Detect bombs by focusing on the Red X pattern and verifying a dark core.
    """
    # 1. Strict Red X Mask (very saturated red)
    red1 = cv2.inRange(hsv, np.array([0, 180, 100]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 180, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    red_mask = cv2.bitwise_and(red_mask, game_mask)
    
    # 2. Dark Mask (lenient but only inside red marking vicinity)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))

    # Cleanup red mask to find distinct 'X' clusters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(red_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bomb_boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80: continue # Red X is small but distinct
        
        x, y, w, h = cv2.boundingRect(cnt)
        # Bombs have a dark body around/behind the red X
        # Expand slightly to find the dark bomb body
        pad = int(max(w, h) * 1.5)
        bx1, by1 = max(0, x-pad), max(0, y-pad)
        bx2, by2 = min(img_w, x+w+pad), min(img_h, y+h+pad)
        
        dark_crop = dark_mask[by1:by2, bx1:bx2]
        dark_pixels = np.count_nonzero(dark_crop)
        
        # If there is a significant dark core near the red X, it's a bomb
        if dark_pixels > 400:
            # The bomb box should encompass both
            # Find the actual dark core bounding box in this vicinity
            dark_contours, _ = cv2.findContours(dark_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if dark_contours:
                best_cnt = max(dark_contours, key=cv2.contourArea)
                dx, dy, dw, dh = cv2.boundingRect(best_cnt)
                # Map back to global coords
                final_x, final_y = bx1 + dx, by1 + dy
                bomb_boxes.append((
                    (final_x + dw/2) / img_w,
                    (final_y + dh/2) / img_h,
                    dw / img_w,
                    dh / img_h
                ))
    
    # Non-maximum suppression for bombs (avoid multiple boxes on one bomb)
    # Simple deduplication by center proximity
    final_bombs = []
    for b in bomb_boxes:
        if not any(abs(b[0]-fb[0]) < 0.05 and abs(b[1]-fb[1]) < 0.05 for fb in final_bombs):
            final_bombs.append(b)
            
    return final_bombs

def main():
    print("Initializing Hybrid Auto-Labeler v5...")
    model = YOLO('yolov8n.pt')
    
    labels_dir = r"c:\Fruit-Ninja\dataset\auto_labels"
    preview_dir = r"c:\tmp\auto_label_previews_v5"
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    
    image_dirs = [r"c:\Fruit-Ninja\dataset\images\train", r"c:\Fruit-Ninja\dataset\images\val"]
    all_images = []
    for d in image_dirs:
        if os.path.exists(d):
            all_images.extend([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg'))])
    
    print(f"Processing {len(all_images)} images...")
    
    for i, img_path in enumerate(sorted(all_images)):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        game_mask = get_game_area_mask(img)
        gt, gb, gl, gr = detect_layout(img)

        # 1. FRUITS via YOLO
        fruit_boxes = []
        results = model(img, conf=0.05, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in FRUIT_CLASSES:
                    # Convert xyxy [px] to YOLO [normalized]
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    # Only keep if central point is in game mask
                    if gl/w <= cx <= gr/w and gt/h <= cy <= gb/h:
                        fruit_boxes.append((cx, cy, bw, bh))

        # 2. BOMBS via Color
        bomb_boxes = detect_bombs_color(hsv, game_mask, h, w)
        
        # 3. SAVE
        basename = os.path.splitext(os.path.basename(img_path))[0]
        with open(os.path.join(labels_dir, f"{basename}.txt"), 'w') as f:
            for b in fruit_boxes: f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
            for b in bomb_boxes: f.write(f"1 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

        if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(all_images)}...")
        
        # Save limited previews
        if i < 20 or basename in ['1', '32', '200', 'img53', '262']:
            for b in fruit_boxes:
                px1, py1 = int((b[0]-b[2]/2)*w), int((b[1]-b[3]/2)*h)
                px2, py2 = int((b[0]+b[2]/2)*w), int((b[1]+b[3]/2)*h)
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
            for b in bomb_boxes:
                px1, py1 = int((b[0]-b[2]/2)*w), int((b[1]-b[3]/2)*h)
                px2, py2 = int((b[0]+b[2]/2)*w), int((b[1]+b[3]/2)*h)
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(preview_dir, f"{basename}_hybrid.png"), img)

    print("Auto-labeling complete. Check c:\\tmp\\auto_label_previews_v5")

if __name__ == "__main__":
    main()
