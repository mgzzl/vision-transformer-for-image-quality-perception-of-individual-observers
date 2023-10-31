import csv
import os
import re

def get_quality_level_from_name(image_name):
    """
    Extract the quality level (vote) from the image name using regular expressions.

    This function searches for a quality level (vote) within the image name by looking
    for a pattern like "_I{quality_level}_". It returns the extracted quality level as
    an integer.

    Parameters:
    image_name (str): The name of the image containing the quality level information.

    Returns:
    int: The extracted quality level.
    None: If no quality level is found in the image name.

    Example:
    --------
    >>> image_name = "12ILSVRC2013_train_00000419.JPEG_I1_Q6.jpeg"
    >>> get_quality_level_from_name(image_name)
    1

    Output:
    - The function returns the extracted quality level as an integer.
    - If no quality level is found in the image name, it returns None.
    """

    # Use regular expression to extract the quality level (vote) from the image name
    match = re.search(r'_I(\d+)_', image_name)
    if match:
        quality_level = match.group(1)
        return int(quality_level)
    else:
        print(f"No Quality-Level found for {image_name}")
        return None

def process_images_and_create_csv(image_folder_path, csv_file_path):
    """
    Process images in a folder and create a CSV file with image names and quality levels.

    This function processes image files in a specified folder, extracts quality levels from
    their names, and writes the data to a CSV file. The CSV file will contain two columns:
    "Image Name" and "Vote" (quality level).

    Parameters:
    image_folder_path (str): The path to the folder containing image files.
    csv_file_path (str): The path to the CSV file to be created.

    Output:
    - The function creates a CSV file with image names and quality levels.
    - The CSV file is saved at the specified location.
    """

    # Get a list of image files in the directory
    image_files = [file for file in os.listdir(image_folder_path)]

    # Prepare data for writing to CSV
    csv_data = [['Image Name', 'Vote']]

    # Process each image and get the quality level
    for image_name in image_files:
        quality_level = get_quality_level_from_name(image_name)
        if quality_level is not None:
            csv_data.append([image_name, quality_level])

    # Write the data to the CSV file
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"CSV file '{csv_file_path}' created successfully.")

if __name__ == "__main__":
    image_folder_path = '../assets/DatasetSubjective/Persons/Person_1_shinyx/imgs'
    csv_file_path = '../assets/DatasetSubjective/Persons/Person_1_shinyx/objective.csv'
    process_images_and_create_csv(image_folder_path, csv_file_path)
