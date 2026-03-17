import os

def rename_images_in_folder(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out only the image files (you can add more extensions if needed)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Sort the image files to ensure consistent numbering
    image_files.sort()

    # Rename each image file with consecutive numbering
    for index, filename in enumerate(image_files, start=1):
        # Construct the new file name
        new_filename = f"{index}{os.path.splitext(filename)[1]}"

        # Get the full paths for the old and new filenames
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f"Renamed '{filename}' to '{new_filename}'")

# Example usage
folder_path = r'dataset\train'
rename_images_in_folder(folder_path)
