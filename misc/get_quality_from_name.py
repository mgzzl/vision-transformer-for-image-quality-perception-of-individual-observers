import csv
import os
import re

def get_quality_level_from_name(image_name):
    # Use regular expression to extract the quality level from the image name
    # For example, from "12ILSVRC2013_train_00000419.JPEG_I1_Q6.jpeg",
    # we want to extract the "1" after the second 'I', which indicates the quality level.
    match = re.search(r'_I(\d+)_', image_name)
    if match:
        quality_level = match.group(1)
        return int(quality_level)
    else:
        print(f"No Quality-Level found for {image_name}")


image_folder_path = '../assets/DatasetSubjective/Persons/Person_1_shinyx/imgs'
csv_file_path = '../assets/DatasetSubjective/Persons/Person_1_shinyx/objective.csv'

# Get a list of image files in the directory
image_files = [file for file in os.listdir(image_folder_path)]

# Prepare data for writing to CSV
csv_data = [['Image Name', 'Quality Level']]

# Process each image and get the quality level
for image_name in image_files:
    quality_level = get_quality_level_from_name(image_name)
    csv_data.append([image_name, quality_level])

# Write the data to the CSV file
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"CSV file '{csv_file_path}' created successfully.")
