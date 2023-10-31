import os
import csv

def combine_csv_files(directory):
    """
    Combines multiple CSV files into one.

    Parameters:
    directory (str): The path to the directory containing CSV files to be combined.

    Output:
    The combined CSV file is saved in the input directory with '_combined' appended to the directory name.
    """

    base_name = os.path.splitext(os.path.basename(directory))[0]
    output_file_path = os.path.join(directory, base_name + '_combined.csv')
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in the directory: {directory}")
        return
    count = 0

    combined_rows = set()  # Set to store unique rows

    with open(output_file_path, 'w', newline='') as combined_csv:
        writer = csv.writer(combined_csv)

        # Write the header from the first CSV file
        with open(os.path.join(directory, csv_files[0]), 'r') as first_csv:
            reader = csv.reader(first_csv)
            writer.writerows(reader)

        # Append data from the remaining CSV files
        for csv_file in csv_files[1:]:
            with open(os.path.join(directory, csv_file), 'r') as file:
                # Skip the header row in subsequent files
                next(file)
                reader = csv.reader(file)
                for row in reader:
                    count += 1
                    writer.writerow(row)
                    d = count - len(combined_rows)

    print(f"Combined CSV file saved at: {output_file_path}")
    print(f"Number of not unique rows: {d}")

# Example of how to use the combine_csv_files function:
if __name__ == "__main__":
    root_directory = '/home/maxgan/WORKSPACE/UNI/BA_Pavel/matlab/results_csv/Obs4'
    combine_csv_files(root_directory)

