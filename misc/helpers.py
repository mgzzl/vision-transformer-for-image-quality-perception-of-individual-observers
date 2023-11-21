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

def calculate_sigma_from_distributions(probabilities, num_classes, device):
    """
    Calculate sigma from true distributions, the number of classes, and the device.

    This function calculates the sigma value used to generate the true distributions
    based on a set of given true distributions.

    Parameters:
    probabilities (torch.Tensor): A tensor of probabilities.
    num_classes (int): The number of classes.
    device (str): The device (CPU or GPU) for tensor computations.

    Returns:
    float: The calculated sigma value.

    Note:
    - The input probabilities should be a 2D torch tensor.
    """

    # Calculate the mean of the probabilities
    mean_probabilities = torch.mean(probabilities, dim=1)

    # Calculate the squared difference between each probability and the mean
    squared_diff = (probabilities - mean_probabilities.unsqueeze(1))**2

    # Calculate the mean of the squared differences
    mean_squared_diff = torch.mean(squared_diff, dim=1)

    # Take the square root to get the standard deviation
    std_dev = torch.sqrt(mean_squared_diff)

    # print("Standard Deviation:", std_dev)


    # num_samples = probabilities.size(0)
    # sigma_values = torch.empty(num_samples, dtype=torch.float32, device=device)

    # for i in range(num_samples):
    #     # Calculate the mean of the distribution
    #     mean = torch.sum(torch.arange(num_classes, dtype=torch.float32, device=device) * probabilities[i])

    #     # Calculate the squared distance from the mean
    #     distance_squared = torch.sum(((torch.arange(num_classes, dtype=torch.float32, device=device) - mean)**2) * probabilities[i])

    #     # Calculate sigma using the standard deviation formula
    #     sigma = torch.sqrt(distance_squared)

    #     sigma_values[i] = sigma

    return std_dev.cpu().numpy()
