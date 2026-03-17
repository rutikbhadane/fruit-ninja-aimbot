import glob
#import json
"""
# Specify the folder path
folder_path = r"dataset\annotations"  # Replace with the actual folder path

# Specify the output file path
output_file = "merged.json"

# Create an empty list to store the merged data
merged_data = []

# Get all JSON files in the folder
json_files = glob.glob(folder_path)

# Iterate through each JSON file
for file in json_files:
    try:
        # Load the JSON data from the file
        with open(file, "r") as f:
            data = json.load(f)
        # Append the data to the merged_data list
        merged_data.extend(data)
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON file: {file}")
        continue
    except FileNotFoundError:
        print(f"File not found: {file}")
        continue

# Write the merged data to a new JSON file
with open(output_file, "w") as f:
    json.dump(merged_data, f, indent=4)

print(f"Merged JSON files saved to {output_file}")

"""
import json
contents = []
for i in range(1,600):
    with open(f"dataset/annotations/{i}",'r') as file:
        data = json.load(file)
        contents.append(data)
        
output_file = "merged.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)
"""   
for i in contents:
    print(i)
   """       