import os
import re

def rename_images_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter and sort image files based on the numeric part of their names
    image_files = sorted(
        [f for f in files if re.match(r'\d+\.png', f)],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    # Rename the images to have a continuous sequence
    for i, file_name in enumerate(image_files, start=1):
        old_file_path = os.path.join(folder_path, file_name)
        new_file_name = f"img{i}.png"
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file_name}' to '{new_file_name}'")

# Example usage
folder_path = 'dataset/train'
rename_images_in_folder(folder_path)
