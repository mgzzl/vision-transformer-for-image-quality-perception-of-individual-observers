import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from itertools import combinations
from misc.visualization import get_attention_maps, normalize_attention_maps
from misc.helpers import trans_norm2tensor, create_vit_model
from misc import transformations
import csv

class ImageAttentionGlobalAvgDataset(Dataset):
    def __init__(self, images_dir, weight_files, image_size, patch_size, layer_idx, device):
        """
        Custom dataset class for computing global attention averages.

        Parameters:
        - images_dir (str): Directory containing image files.
        - weight_files (list): List of paths to pre-trained model weights.
        - image_size (int): Desired size of the input images.
        - patch_size (int): Size of the image patches for attention computation.
        - layer_idx (int): Index of the layer for attention computation.
        - device (torch.device): Device on which the model is loaded.

        Output:
        - An instance of the ImageAttentionGlobalAvgDataset class.
        """
        self.images_dir = images_dir
        self.weight_files = weight_files
        self.image_size = image_size
        self.patch_size = patch_size
        self.device = device
        # Filter filenames to include only JPEG files
        self.image_filenames = [filename for filename in os.listdir(images_dir) if filename.lower().endswith('.jpeg')]
        self.layer_idx = layer_idx

        # Compute and store global averages during initialization
        self.all_global_avgs = self._compute_all_global_avgs()

    def _compute_all_global_avgs(self):
        """
        Compute global averages for all images in the dataset.

        Returns:
        list: List of dictionaries containing filename, global_avg, and average_attention_map.
        """
        all_global_avgs = []
        for image_filename in self.image_filenames:
            global_avg, average_attention_map, org_img, att_maps = self._compute_global_avg(image_filename)
            all_global_avgs.append({'filename': image_filename, 'global_avg': global_avg, 'average_attention_map': average_attention_map, 'org_img': org_img, 'att_maps': att_maps})
        return all_global_avgs

    def _compute_global_avg(self, image_filename):
        """
        Compute global average and average attention map for a single image.

        Parameters:
        - image_filename (str): Filename of the image.

        Returns:
        tuple: Tuple containing global average and average attention map.
        """
        image_path = os.path.join(self.images_dir, image_filename)
        img = Image.open(image_path)
        org_img = img.copy()
        img_tensor = trans_norm2tensor(img, self.image_size)

        img_attentions = []
        for weight_file in self.weight_files:
            model = create_vit_model(weights_path=weight_file)
            _, attention = get_attention_maps(model, img_tensor, self.patch_size, self.device)
            att_ll = attention[self.layer_idx]
            att_mean = np.mean(att_ll, 0)       
            att_mean = normalize_attention_maps(att_mean)
            img_attentions.append(att_mean)

        # Calculate absolute difference attentions for all pairs of attention maps
        absolute_difference_attentions = [
            np.abs(att_a - att_b)
            for img_attentions_a, img_attentions_b in combinations(img_attentions, 2)
            for att_a, att_b in zip(img_attentions_a, img_attentions_b)
        ]

        # Calculate average attention map and global average
        average_attention_map = np.mean(absolute_difference_attentions, axis=0)
        global_avg = np.mean(average_attention_map)
        return global_avg, average_attention_map, org_img, img_attentions

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        return self.all_global_avgs[idx]

    def get_top_global_avg(self, num_images, descending):
        """
        Get the top N images with the highest global averages.

        Parameters:
        - num_images (int): Number of top images to retrieve.
        - descending (bool): If True, retrieve images in descending order of global average.

        Returns:
        list: List of dictionaries containing filename and global_avg for the top images.
        """
        sorted_indices = sorted(
            range(len(self.all_global_avgs)),
            key=lambda i: self.all_global_avgs[i]['global_avg'],
            reverse=descending
        )
        selected_indices = sorted_indices if num_images is None else sorted_indices[:num_images]
        selected_images = [self.all_global_avgs[i] for i in selected_indices]
        return selected_images

    def write_to_csv(self, csv_file_path, num_images=None, descending=True):
        """
        Write the top N images with the highest global averages to a CSV file.

        Parameters:
        - csv_file_path (str): Path to the CSV file.
        - num_images (int): Number of top images to write to the CSV file.
        - descending (bool): If True, write images in descending order of global average.
        """
        selected_images = self.get_top_global_avg(num_images=num_images, descending=descending)
        fieldnames = ['filename', 'global_avg']
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for img_info in selected_images:
                selected_fields = {key: img_info[key] for key in fieldnames}
                writer.writerow(selected_fields)