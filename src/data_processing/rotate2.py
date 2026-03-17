from PIL import Image
import os
# now the code is modified such that if we run this script it will flip the images vertically 
# if we are short of dataset we will run this script 
# giving us more than 650 images flipped
def rotate_images_in_folder(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Filter and process image files
    for i, file_name in enumerate(files):
        file_path = os.path.join(input_folder, file_name)

        try:
            # Open the image file
            with Image.open(file_path) as img:
                # Rotate the image by 180 degrees
                rotated_img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

                # Define the new file name
                new_file_name = f"img({i + 1})_f.png"
                new_file_path = os.path.join(output_folder, new_file_name)

                # Save the rotated image
                rotated_img.save(new_file_path)
                print(f"Saved {new_file_path}")
        except Exception as e:
            print(f"Could not process {file_name}: {e}")

# Example usage
input_folder = r'dataset\train'
output_folder = r'dataset\train'
rotate_images_in_folder(input_folder, output_folder)
