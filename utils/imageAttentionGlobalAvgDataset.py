import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from itertools import combinations
from misc.visualization import visualize_attention
from misc.helpers import trans_norm2tensor, create_vit_model
import csv

class ImageAttentionGlobalAvgDataset(Dataset):
    def __init__(self, images_dir, weight_files, image_size, patch_size, layer_idx, device):
        self.images_dir = images_dir
        self.weight_files = weight_files
        self.image_size = image_size
        self.patch_size = patch_size
        self.device = device
        self.image_filenames = [filename for filename in os.listdir(images_dir) if filename.lower().endswith('.jpeg')]
        self.layer_idx = layer_idx

        # Compute and store global averages during initialization
        self.all_global_avgs = self._compute_all_global_avgs()

    def _compute_all_global_avgs(self):
        all_global_avgs = []
        for image_filename in self.image_filenames:
            global_avg, average_attention_map = self._compute_global_avg(image_filename)
            all_global_avgs.append({'filename': image_filename, 'global_avg': global_avg, 'average_attention_map': average_attention_map})
        return all_global_avgs

    def _compute_global_avg(self, image_filename):
        image_path = os.path.join(self.images_dir, image_filename)
        img = Image.open(image_path)
        img_tensor = trans_norm2tensor(img, self.image_size)

        img_attentions = []
        for weight_file in self.weight_files:
            model = create_vit_model(weights_path=weight_file)
            _, attention = visualize_attention(model, img_tensor, self.patch_size, self.device)
            att_ll = attention[self.layer_idx]
            att_mean = np.mean(att_ll, 0)
            img_attentions.append(att_mean)

        absolute_difference_attentions = [
            np.abs(att_a - att_b)
            for img_attentions_a, img_attentions_b in combinations(img_attentions, 2)
            for att_a, att_b in zip(img_attentions_a, img_attentions_b)
        ] # n choose 2 => input:5 attentionmaps; calc: n(n-1)/2; Output: 10 attentionmaps

        average_attention_map = np.mean(absolute_difference_attentions, axis=0)
        global_avg = np.mean(average_attention_map)
        return global_avg, average_attention_map

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        return self.all_global_avgs[idx]

    def get_top_global_avg(self, num_images, descending):
        sorted_indices = sorted(
            range(len(self.all_global_avgs)),
            key=lambda i: self.all_global_avgs[i]['global_avg'],
            reverse=descending
        )
        selected_indices = sorted_indices if num_images is None else sorted_indices[:num_images]
        selected_images = [self.all_global_avgs[i] for i in selected_indices]
        return selected_images

    def write_to_csv(self, csv_file_path, num_images=None, descending=True):
        selected_images = self.get_top_global_avg(num_images=num_images, descending=descending)
        fieldnames = ['filename', 'global_avg']
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for img_info in selected_images:
                selected_fields = {key: img_info[key] for key in fieldnames}
                writer.writerow(selected_fields)