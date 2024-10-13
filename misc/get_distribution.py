import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/home/maxgan/WORKSPACE/UNI/BA/vision-transformer-for-image-quality-perception-of-individual-observers')

def generate_bar_chart_from_csv(csv_directory, csv_file, rating_labels):
    """
    Process a single CSV file, generate a bar chart, and print a rating distribution table.

    This function reads a CSV file, maps numerical ratings to their corresponding labels,
    generates a bar chart showing the distribution of ratings, and prints a rating
    distribution table. It also saves the bar chart as an image.

    Parameters:
    csv_directory (str): The directory path containing the CSV files.
    csv_file (str): The name of the CSV file to process.
    rating_labels (dict): A dictionary mapping numerical ratings to their labels.

    Output:
    -------
    - The function generates a bar chart showing the rating distribution and saves it as an image.
    - It also prints a rating distribution table.

    Note:
    - The function assumes that the CSV file contains a "Vote" column with numerical ratings.
    - The generated bar chart is saved in the same directory as the CSV file.
    """

    # Read the CSV file
    data = pd.read_csv(os.path.join(csv_directory, csv_file), header=0)
    
    # Map rating values to their corresponding labels
    data["Rating_Label"] = data["Vote"].map(rating_labels)

    # Group data by Rating_Label and count occurrences
    class_counts = data["Rating_Label"].value_counts()

    # Reindex the class_counts Series to match the desired order
    class_counts = class_counts.reindex(rating_labels.values())
    digit = ''.join(filter(str.isdigit, csv_file.split('.')[0]))
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind="bar", color='skyblue')
    # plt.title(f"Image Rating Distribution of DS{digit} according to {csv_file.split('.')[0]}",fontsize=15)
    plt.title(f"Image Rating Distribution of DSX according to {csv_file.split('.')[0]}",fontsize=15)
    plt.xlabel("Rating", fontsize=15)
    plt.ylabel("Number of Images", fontsize=15, labelpad=10)
    plt.ylim(0,165)
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    # Save the bar chart as an image
    # output_image_path = os.path.join(csv_directory, f"rating_distribution_DS{digit}_{csv_file.replace('.csv', '.png')}")
    output_image_path = os.path.join(csv_directory, f"rating_distribution_DSX_{csv_file.replace('.csv', '.png')}")
    plt.savefig(output_image_path)

    # Display the table
    print(f"Rating Distribution Table of DS{digit} for {csv_file.split('.')[0]}")
    print(class_counts)

    # Close the plot
    plt.close()

def main():
    # Directory containing the CSV files
    csv_directory = "assets/Test"

    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]

    # Define rating labels
    rating_labels = {1: "Bad", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}

    # Process each CSV file in the directory
    for csv_file in csv_files:
        generate_bar_chart_from_csv(csv_directory, csv_file, rating_labels)

if __name__ == "__main__":
    main()
