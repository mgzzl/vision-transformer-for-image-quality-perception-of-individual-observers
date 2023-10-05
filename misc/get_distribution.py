import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_path = "./assets/Test/AccTestCsv/Obs0AccTest.csv"
data = pd.read_csv(csv_path, header=0)

# Map rating values to their corresponding labels
rating_labels = {
    1: "Bad",
    2: "Insufficient",
    3: "Fair",
    4: "Good",
    5: "Excellent"
}

data["Rating_Label"] = data["Vote"].map(rating_labels)
print(data)
# Group data by Rating_Label and count occurrences
class_counts = data["Rating_Label"].value_counts()

# Reindex the class_counts Series to match the desired order
class_counts = class_counts.reindex(rating_labels.values())
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
output_image_path = "./assets/Test/AccTestCsv/rating_distribution_ObsX_obs0.png"
plt.savefig(output_image_path)

# Display the table
print("Rating Distribution Table:")
print(class_counts)

# Close the plot
plt.close()