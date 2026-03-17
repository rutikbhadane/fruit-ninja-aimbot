import os
import json
import random
import shutil

annotations_dir = "c:/Fruit-Ninja/dataset/annotations"
images_dir = "c:/Fruit-Ninja/dataset/train"
output_images_train = "c:/Fruit-Ninja/dataset/images/train"
output_images_val = "c:/Fruit-Ninja/dataset/images/val"
output_labels_train = "c:/Fruit-Ninja/dataset/labels/train"
output_labels_val = "c:/Fruit-Ninja/dataset/labels/val"

class_mapping = {
    "Fruit": 0,
    "Bomb": 1,
    "fruits": 0,
    "bombs": 1
}

for path in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
    os.makedirs(path, exist_ok=True)

# 1. Parse all JSON annotations and store them indexed by their integer "id"
# The user's label studio tasks have an "id" field (like 1, 2, 3...)
# AND the user's `mergejson.py` specifically iterated over filenames `1`, `2`, `3`... `591`.
json_data_by_index = {}

json_files = os.listdir(annotations_dir)
for json_file in json_files:
    # the filenames are just "1", "2", "3" etc.
    if not json_file.isdigit():
        continue
        
    idx = int(json_file)
    json_path = os.path.join(annotations_dir, json_file)
    
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            continue
            
    if isinstance(data, list) and len(data) > 0:
        task_data = data[0]
    elif isinstance(data, dict):
        task_data = data
    else:
        continue
        
    if "result" not in task_data:
        continue
        
    yolo_annotations = []
    for item in task_data["result"]:
        if item.get("type") == "rectanglelabels":
            val = item["value"]
            label_list = val.get("rectanglelabels", [])
            if not label_list: continue
            
            class_id = class_mapping.get(label_list[0], class_mapping.get(label_list[0].capitalize(), -1))
            if class_id == -1: continue
            
            x_center = max(0.0, min(1.0, (val["x"] + val["width"] / 2) / 100.0))
            y_center = max(0.0, min(1.0, (val["y"] + val["height"] / 2) / 100.0))
            w = max(0.0, min(1.0, val["width"] / 100.0))
            h = max(0.0, min(1.0, val["height"] / 100.0))
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
    if yolo_annotations:
        json_data_by_index[idx] = yolo_annotations

print(f"Loaded {len(json_data_by_index)} annotations.")

# 2. Iterate image files 1.png to 617.png
# We will match annotation "idx" to image "img(idx).png" or just "idx.png".
# Since user ran `ton.py` and `numbering.py`, the images are sequential. 
# BUT there are 617 images and 591 annotations. 
# The user's `numbering.py` sorts alphabetically by default if not careful, 
# but it used `key=lambda x: int(re.search(r'\d+', x).group())` which is PERFECT sequential sort!
# This means: 
# The FIRST annotated image is 1.png, SECOND is 2.png...
# Until we run out of annotations.

pairs = []
missing_images = 0

for idx in sorted(json_data_by_index.keys()):
    # We assume the user's `ton.py` logic perfectly correlated the index 1-591 to 1.png-591.png
    # because they were renamed AFTER labeling.
    img_name = f"{idx}.png"
    img_path = os.path.join(images_dir, img_name)
    
    # Just in case the user named it img1.png
    if not os.path.exists(img_path):
        img_name = f"img{idx}.png"
        img_path = os.path.join(images_dir, img_name)
        
    if os.path.exists(img_path):
        pairs.append({
            "image_path": img_path,
            "annotations": json_data_by_index[idx],
            "filename": img_name
        })
    else:
        missing_images += 1

print(f"Matched {len(pairs)} annotations to images. (Missing: {missing_images})")

# 3. Output

random.seed(42)
random.shuffle(pairs)

split_idx = int(len(pairs) * 0.8)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

def process_pairs(pair_list, dst_images, dst_labels):
    for pair in pair_list:
        shutil.copy(pair["image_path"], os.path.join(dst_images, pair["filename"]))
        
        base_name = os.path.splitext(pair["filename"])[0]
        dst_label = os.path.join(dst_labels, base_name + ".txt")
        
        with open(dst_label, 'w') as f:
            f.write("\n".join(pair["annotations"]))

print("Writing Train...")
process_pairs(train_pairs, output_images_train, output_labels_train)
print("Writing Val...")
process_pairs(val_pairs, output_images_val, output_labels_val)
print("Done!")
