import os
import scipy.io
import csv

def save_sorttable_to_csv(mat_file_path, output_directory):
    """
    Convert a MATLAB .mat file containing 'sortTable' into a CSV file.

    This function reads a MATLAB .mat file, extracts the 'sortTable' data, and
    saves it as a CSV file in the specified output directory. The CSV file
    will contain the following columns: "Image Name," "Vote," "Duration," and "Timestamp."

    Parameters:
    mat_file_path (str): The file path to the input MATLAB .mat file.
    output_directory (str): The directory where the CSV file will be saved.

    Output:
    The function will save the CSV file containing 'sortTable' data in the specified output directory.

    Note:
    - If 'sortTable' is not found in the .mat file, an error message will be printed, and no CSV file will be created.
    """

    mat = scipy.io.loadmat(mat_file_path)

    if 'sortTable' not in mat:
        print(f"Error: '{mat_file_path}' does not contain 'sortTable' key. Skipping...")
        return

    sort_table = mat['sortTable'][0]

    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
    csv_file_path = os.path.join(output_directory, base_name + '.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "Vote", "Duration", "Timestamp"])
        for i in range(len(sort_table)):
            image_name = mat['imgList'][0][i]
            vote = mat['voteList'][0][i]
            duration = mat['drtList'][0][i]
            timestamp = mat['voteTime'][0][i]

            writer.writerow([image_name[0], vote, duration[0], timestamp[0]])

        print(f"CSV file saved at: {csv_file_path}")


def process_directory(directory, output_directory):
    """
    Process all MATLAB .mat files in a directory and save them as CSV files.

    This function takes a directory containing MATLAB .mat files, processes each file
    using the save_sorttable_to_csv function, and saves the resulting CSV files in
    the specified output directory.

    Parameters:
    directory (str): The directory path containing the .mat files to be processed.
    output_directory (str): The directory where the CSV files will be saved.

    Output:
    The function will process each .mat file in the input directory and save the corresponding
    CSV files in the output directory using the save_sorttable_to_csv function.
    """

    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            mat_file_path = os.path.join(directory, file_name)
            save_sorttable_to_csv(mat_file_path, output_directory)

# Example of how to use the process_directory function:
if __name__ == "__main__":
    directory = '/home/maxgan/WORKSPACE/UNI/BA_Pavel/matlab/results_csv/Obs4.0'
    output_directory = '/home/maxgan/WORKSPACE/UNI/BA_Pavel/matlab/results_csv/Obs4.0'
    process_directory(directory, output_directory)
