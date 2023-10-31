import csv

def compare_csv_files(file1_path, file2_path):
    """
    Compare the content of the first columns of two CSV files.

    This function reads two CSV files, compares the values in their first columns,
    and provides information on whether they are duplicates or if there are missing
    values in one of the files.

    Parameters:
    file1_path (str): The file path to the first CSV file.
    file2_path (str): The file path to the second CSV file.

    Output:
    - If both files have identical values in their first columns, it prints "This is a duplicate."
    - If there are missing values in either file, it prints the number of missing values
      and their names in each file.

    Note:
    - The comparison considers values in the first columns only.
    """

    file1_values = set()
    file2_values = set()

    # Read and store values from the first column of the first CSV file
    with open(file1_path, 'r', newline='') as file1:
        reader1 = csv.reader(file1)
        for row in reader1:
            if row:
                file1_values.add(row[0])

    # Read and store values from the first column of the second CSV file
    with open(file2_path, 'r', newline='') as file2:
        reader2 = csv.reader(file2)
        for row in reader2:
            if row:
                file2_values.add(row[0])

    # Check if all values in the first column of file1 are in file2, and vice versa
    if file1_values == file2_values:
        print("This is a duplicate")
    else:
        # Find missing values and print their count and names
        missing_in_file1 = file2_values - file1_values
        missing_in_file2 = file1_values - file2_values

        print("Number of missing values in file1:", len(missing_in_file1))
        print("Missing values in file1:", missing_in_file1)
        print("Number of missing values in file2:", len(missing_in_file2))
        print("Missing values in file2:", missing_in_file2)

# Example of how to use the compare_csv_files function:
if __name__ == "__main__":
    file1_path = '/home/maxgan/WORKSPACE/UNI/BA_Pavel/matlab/results_csv/Obs4/nictac01-04-2023.csv'
    file2_path = '/home/maxgan/WORKSPACE/UNI/BA_Pavel/matlab/results_csv/Obs4.0/nictacTest10-01-2023.csv'
    compare_csv_files(file1_path, file2_path)

