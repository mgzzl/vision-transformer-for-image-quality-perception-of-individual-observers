import os
import random
import re
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ImageQualityDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None):
        self.csv_file = csv_file
        self.transform = transform

        df = pd.read_csv(csv_file)

        # Check if the CSV file has a header row
        if "Image Name" in df.columns and "Quality Level" in df.columns:
            image_column = "Image Name"
            quality_column = "Quality Level"
        else:
            image_column = df.columns[0]
            quality_column = df.columns[1]

        class_counts = [0] * 5  # Initialize count for each quality level

        self.image_files = []
        for _, row in df.iterrows():
            image_path = os.path.join(image_dir,row[image_column])
            quality_level = int(row[quality_column]) - 1
            self.image_files.append((image_path, quality_level))
            class_counts[quality_level] += 1

        selected_files = []
        for i in range(5):
            class_files = [(img, label) for img, label in self.image_files if label == i]
            random.shuffle(class_files)
            selected_files.extend(class_files)

        random.shuffle(selected_files)
        self.image_files = selected_files

        print(f"Number of images in the dataset: {len(selected_files)}")

    
    def get_true_quality_level(self, image_name):
        # Use regular expression to extract the quality level from the image name
        # For example, from "12ILSVRC2013_train_00000419.JPEG_I1_Q6.jpeg",
        # we want to extract the "1" after the second 'I', which indicates the quality level.
        match = re.search(r'_I(\d+)_', image_name)
        if match:
            quality_level = match.group(1)
            return int(quality_level) - 1
        else:
            print(f"No Quality-Level found for {image_name}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path, label = self.image_files[idx]
        # Load image and apply transformations
        # print(image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label