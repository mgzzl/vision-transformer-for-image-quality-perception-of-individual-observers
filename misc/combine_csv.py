import os
import csv


def combine_csv_files(subdirectory):
    base_name = os.path.splitext(os.path.basename(subdirectory))[0]
    output_file_path = os.path.join(subdirectory, base_name + '_combined.csv')
    csv_files = [file for file in os.listdir(subdirectory) if file.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in the subdirectory: {subdirectory}")
        return

    combined_rows = set()  # Set to store unique rows

    with open(output_file_path, 'w', newline='') as combined_csv:
        writer = csv.writer(combined_csv)

        # Write the header from the first CSV file
        with open(os.path.join(subdirectory, csv_files[0]), 'r') as first_csv:
            reader = csv.reader(first_csv)
            writer.writerows(reader)

        # Append data from the remaining CSV files
        for csv_file in csv_files[1:]:
            with open(os.path.join(subdirectory, csv_file), 'r') as file:
                # Skip the header row in subsequent files
                next(file)
                reader = csv.reader(file)
                for row in reader:
                    if tuple(row) not in combined_rows:  # Check if row is unique
                        writer.writerow(row)
                        combined_rows.add(tuple(row))

    print(f"Combined CSV file saved at: {output_file_path}")


def process_directory(directory):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            combine_csv_files(subdir_path)


# Provide the path to the root directory
# root_directory = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/Assets/DatasetSubjective/scores/Persons'
root_directory = '/home/maxgan/WORKSPACE/UNI/BA/tutorial_matlab_swdi/swdi/image-quality-all/Scores'
combine_csv_files(root_directory)
# process_directory(root_directory)
