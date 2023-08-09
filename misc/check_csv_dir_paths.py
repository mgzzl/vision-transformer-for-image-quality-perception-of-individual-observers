import csv
import os

csv_file_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/Test/AccTest/arktipAccTest19-01-2023.csv'
# csv_file_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/Test/TestRating/arktipTest20-01-2023.csv'
image_folder_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/Test/TestImg'
# image_folder_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/DatasetObjective/allDistorted'

# Read the CSV file
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header row if present
    
    exist_count = 0
    not_exist_count = 0
    not_exist_images = []
    
    for row in csv_reader:
        image_name = row[0]
        found = False

        # Recursively search for the image in all subdirectories
        for root, dirs, files in os.walk(image_folder_path):
            if image_name in files:
                found = True
                break

        if found:
            exist_count += 1
        else:
            not_exist_count += 1
            not_exist_images.append(image_name)

    # Print summary
    print(f"Number of images found: {exist_count}")
    print(f"Number of images not found: {not_exist_count}")

    if not_exist_count > 0:
        print("Images not found:")
        for image_name in not_exist_images:
            print(image_name)

