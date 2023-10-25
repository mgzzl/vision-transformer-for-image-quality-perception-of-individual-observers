import os
import scipy.io
import csv


def save_sorttable_to_csv(mat_file_path, output_directory):
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
    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            mat_file_path = os.path.join(directory, file_name)
            save_sorttable_to_csv(mat_file_path, output_directory)


# Provide the directory path containing the .mat files
directory = '/home/maxgan/WORKSPACE/UNI/BA_Pavel/matlab/results_csv/Obs5'
output_directory = '/home/maxgan/WORKSPACE/UNI/BA_Pavel/matlab/results_csv/Obs5'

process_directory(directory, output_directory)
