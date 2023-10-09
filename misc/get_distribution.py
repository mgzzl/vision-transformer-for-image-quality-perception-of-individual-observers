import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the CSV files
csv_directory = "assets/Test/AccTestCsv"

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]

# Define rating labels
rating_labels = {
    1: "Bad",
    2: "Insufficient",
    3: "Fair",
    4: "Good",
    5: "Excellent"
}

# Function to process a single CSV file
def process_csv(csv_file):
    # Read the CSV file
    data = pd.read_csv(os.path.join(csv_directory, csv_file), header=0)
    
    # Map rating values to their corresponding labels
    data["Rating_Label"] = data["Vote"].map(rating_labels)

    # Group data by Rating_Label and count occurrences
    class_counts = data["Rating_Label"].value_counts()

    # Reindex the class_counts Series to match the desired order
    class_counts = class_counts.reindex(rating_labels.values())

    # Calculate total number of images
    total_images = class_counts.sum()

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind="bar", color='skyblue')
    plt.title(f"Image Rating Distribution for {csv_file.split('.')[0]}")
    plt.xlabel("Rating")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the bar chart as an image
    output_image_path = os.path.join(csv_directory, f"rating_distribution_{csv_file.replace('.csv', '.png')}")
    plt.savefig(output_image_path)

    # Display the table
    print(f"Rating Distribution Table for {csv_file.split('.')[0]}")
    print(class_counts)

    # Close the plot
    plt.close()

# Process each CSV file in the directory
for csv_file in csv_files:
    process_csv(csv_file)

