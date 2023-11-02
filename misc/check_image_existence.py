# Import necessary modules
import csv
import os

def check_image_existence(csv_file_path, image_folder_path):
    """
    Reads a CSV file and checks the existence of images in a directory.

    Returns:
        exist_count (int): The number of images found.
        not_exist_count (int): The number of images not found.
        not_exist_images (list): A list of image names that were not found.
    """

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

        return exist_count, not_exist_count, not_exist_images

# Example of how to use the check_image_existence function:
if __name__ == "__main__":
    # Configuration variables
    csv_file_path = 'assets/Test/AccTestCsv/Obs4AccTest.csv'
    image_folder_path = '/assets/Test/DSX'

    exist_count, not_exist_count, not_exist_images = check_image_existence(csv_file_path, image_folder_path)
