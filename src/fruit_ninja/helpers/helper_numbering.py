import os

def rename_dataset_images(folder="dataset"):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort()  # Sort alphabetically

    # First pass: rename to temporary names
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(folder, filename)
        temp_name = f"temp_{i}.png"
        temp_path = os.path.join(folder, temp_name)
        os.rename(old_path, temp_path)

    # Second pass: rename to final sequential names
    temp_files = [f for f in os.listdir(folder) if f.startswith("temp_")]
    temp_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    for i, filename in enumerate(temp_files, start=1):
        old_path = os.path.join(folder, filename)
        new_name = f"{i}.png"
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} -> {new_name}")

if __name__ == "__main__":
    rename_dataset_images(r"dataset\images")
