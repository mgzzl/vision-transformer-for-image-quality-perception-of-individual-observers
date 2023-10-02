import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_path = "../assets/Test/AccTestCsv/shinyxAccTest20-01-2023.csv"
data = pd.read_csv(csv_path, header=0)

# Map rating values to their corresponding labels
rating_labels = {
    1: "Bad",
    2: "Insufficient",
    3: "Fair",
    4: "Good",
    5: "Excellent"
}

# Define the desired order for x-axis labels
desired_order = ["Bad", "Insufficient", "Fair", "Good", "Excellent"]

data["Rating_Label"] = data["Vote"].map(rating_labels)
print(data)
# Group data by Rating_Label and count occurrences
class_counts = data["Rating_Label"].value_counts()

# Reindex the class_counts Series to match the desired order
class_counts = class_counts.reindex(desired_order)
# print(class_counts)

# Calculate total number of images
total_images = class_counts.sum()

# Create a bar chart
plt.figure(figsize=(10, 6))
class_counts.plot(kind="bar", color='skyblue')
plt.title("Image Rating Distribution ObsX (obs1)")
plt.xlabel("Rating")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the bar chart as an image
output_image_path = "../assets/Test/AccTestCsv/rating_distribution_ObsX_obs1.png"
plt.savefig(output_image_path)

# Save the distribution table as a text file
output_table_path = "../assets/Test/AccTestCsv/rating_distribution_objective.txt"
with open(output_table_path, "w") as f:
    f.write("Rating Distribution Table:\n")
    f.write("==========================\n\n")
    for rating_label, count in class_counts.items():
        f.write(f"{rating_label}: {count} images\n")
    f.write(f"\nTotal Images: {total_images} images\n")

# Display the table
print("Rating Distribution Table:")
print(class_counts)

# Close the plot
plt.close()