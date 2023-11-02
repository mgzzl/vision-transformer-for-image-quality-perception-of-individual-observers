import os
import random
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ImageQualityDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None):
        """
        Custom dataset class for image quality prediction.

        This class reads a CSV file with image names and votes (quality levels), and loads images
        from the specified directory. It shuffles the images in the dataset.

        Parameters:
        csv_file (str): Path to the CSV file containing image information.
        image_dir (str): Directory path containing image files.
        transform (callable, optional): A function/transform to apply to the loaded image.

        Output:
        - An instance of the ImageQualityDataset class.

        Note:
        - The CSV file should have a 'Image Name' or 'Vote' column.
        """
        self.csv_file = csv_file
        self.transform = transform
        self.class_files = []
        df = pd.read_csv(csv_file)

        # Determine column names based on the header presence
        if "Image Name" in df.columns and "Vote" in df.columns:
            image_column = "Image Name"
            quality_column = "Vote"
        else:
            image_column = df.columns[0]
            quality_column = df.columns[1]

        class_counts = [0] * 5  # Initialize count for each quality level

        self.image_files = []
        for _, row in df.iterrows():
            image_path = os.path.join(image_dir, row[image_column])
            quality_level = int(row[quality_column]) - 1
            self.image_files.append((image_path, quality_level))
            class_counts[quality_level] += 1
        
        for i in range(len(class_counts)):
            print(f"Number of images in class {i}: {class_counts[i]}")

        random.shuffle(self.image_files)
        print(f"\nNumber of images in the dataset: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path, label = self.image_files[idx]
        # Load image and apply transformations
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
