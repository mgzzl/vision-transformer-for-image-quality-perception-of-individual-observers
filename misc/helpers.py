import os
import numpy as np
import torch


def find_pth_files(directory_path):
    """
    Find and return a list of full paths to .pth files in the specified directory.

    Args:
        directory_path (str): The directory path to search for .pth files.

    Returns:
        List[str]: A list of full paths to .pth files.
    """
    pth_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pth"):
                pth_files.append(os.path.join(root, file))
    return pth_files

def resize_and_normalize_attention_maps(attention_maps, image_patch_size, image_size):
    """
    Resize and normalize attention maps to match the original image patch dimensions.

    This function takes a list of attention maps and resizes them to match the specified
    image patch dimensions. It also normalizes the attention maps to have values in the
    range [0, 1].

    Parameters:
    attention_maps (list of numpy.ndarray): A list of attention maps to resize and normalize.
    image_patch_size (int): The size of image patches.
    image_size (int): The size of the original image.

    Returns:
    list of numpy.ndarray: A list of resized and normalized attention maps.

    Output:
    - The function returns a list of resized and normalized attention maps.

    Note:
    - The input attention maps should be a list of numpy arrays.
    """

    resized_and_normalized_attention_maps = []
    for attention_map in attention_maps:
        # Resize the attention map to match image patch dimensions
        resized_attention_map = np.zeros((image_size, image_size))
        for i in range(attention_map.shape[0]):
            for j in range(attention_map.shape[1]):
                x_start = i * image_patch_size
                x_end = (i + 1) * image_patch_size
                y_start = j * image_patch_size
                y_end = (j + 1) * image_patch_size

                # Resize the attention map and add it to the corresponding region
                resized_attention_map[x_start:x_end, y_start:y_end] = attention_map[i, j]
                
        # Normalize the attention map to range [0, 1]
        min_value = np.min(resized_attention_map)
        max_value = np.max(resized_attention_map)
        normalized_attention_map = (resized_attention_map - min_value) / (max_value - min_value)

        resized_and_normalized_attention_maps.append(normalized_attention_map)

    return resized_and_normalized_attention_maps

def calculate_true_distributions(labels, sigma, num_classes, device):
    """
    Calculate true distributions based on labels, sigma, and the number of classes.

    This function calculates true distributions for a set of labels using a Gaussian
    probability distribution with the specified sigma value. The resulting distributions
    are normalized to have a sum of 1.

    Parameters:
    labels (list or torch.Tensor): A list or tensor of labels.
    sigma (float): The standard deviation for the Gaussian distribution.
    num_classes (int): The number of classes.
    device (str): The device (CPU or GPU) for tensor computations.

    Returns:
    torch.Tensor: A tensor containing the true distributions.

    Output:
    - The function returns a tensor containing the true distributions.

    Note:
    - The input labels should be a list or a 1D torch tensor.
    """

    true_distributions = torch.empty((len(labels), num_classes), dtype=torch.float32, device=device)

    for j, label in enumerate(labels):
        # Calculate the true distribution for the current label
        distance = torch.abs(torch.arange(num_classes, dtype=torch.float32, device=device) - label)
        probability = torch.exp(-distance**2 / (2 * sigma**2))

        # Normalize the distribution (sum=1)
        true_distribution = probability / probability.sum()

        # Assign the true distribution to the corresponding row in the tensor
        true_distributions[j] = true_distribution

    return true_distributions
