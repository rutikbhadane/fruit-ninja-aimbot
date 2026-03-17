import os
import random
import shutil

def prepare_subset(source_dir, dest_dir, n=150):
    os.makedirs(dest_dir, exist_ok=True)
    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) < n:
        print(f"Warning: Only {len(images)} images found. Picking all.")
        sample = images
    else:
        sample = random.sample(images, n)
    
    for img in sample:
        shutil.copy(os.path.join(source_dir, img), os.path.join(dest_dir, img))
    
    print(f"Copied {len(sample)} images to {dest_dir}")

if __name__ == "__main__":
    src = r"c:\Fruit-Ninja\dataset\images\train"
    dst = r"c:\Fruit-Ninja\dataset\manual_annotation_subset"
    prepare_subset(src, dst)
