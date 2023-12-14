import csv
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.vit_for_small_dataset import ViT

def create_vit_model(image_size=256, patch_size=16, num_classes=5, dim=1024,  depth=6, heads=16, mlp_dim=2048, emb_dropout=0.1, weights_path=None):
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        emb_dropout=emb_dropout,
    )
    
    if weights_path is not None:
        # Load pretrained weights if a weights file path is provided
        model.load_state_dict(torch.load(weights_path))

    return model

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

def calculate_true_distributions(labels, device, sigma=0.7**2, num_classes=5):
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

    for i, label in enumerate(labels):
        # Calculate the true distribution for the current label
        distance = torch.abs(torch.arange(num_classes, dtype=torch.float32, device=device) - label)
        probability = torch.exp(-distance**2 / (2 * sigma**2))

        # Normalize the distribution (sum=1)
        true_distribution = probability / probability.sum()

        # Assign the true distribution to the corresponding row in the tensor
        true_distributions[i] = true_distribution

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
    - The input probabilities should be a 2D torch tensor. batch_size, probability
    """

    # Calculate the mean of the probabilities
    mean_probabilities = torch.mean(probabilities, dim=1)

    # Calculate the squared difference between each probability and the mean
    squared_diff = (probabilities - mean_probabilities.unsqueeze(1))**2

    # Calculate the mean of the squared differences
    mean_squared_diff = torch.mean(squared_diff, dim=1)

    # Take the square root to get the standard deviation
    std_dev = torch.sqrt(mean_squared_diff)

    return std_dev.cpu().numpy()


def trans_norm2tensor(img, img_size):
    # Convert grayscale to RGB if the image has only one channel
    if img.mode == 'L':
        img = img.convert('RGB')

    # Define the normalization parameters (mean and std)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Define the transformation including normalization
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    img = transform(img)
    return img

def prev_img(img, img_size):
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ])
    img = transform(img)
    return img

def prev_img_gray(img, img_size):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
        ])
        img = transform(img)
        img_bw = img.convert('L')

        # BW with YCbCr
        img_ycbcr = img_bw.convert('YCbCr')
        y, cb, cr = img_ycbcr.split()
        # Convert Y to NumPy arrays
        y_array = np.array(y)

        # Perform the desired contrast adjustment on the Y component
        y_array = np.clip(y_array * 2.0, 0, 255).astype(np.uint8)

        # Merge the Y, Cb, Cr components back into an image
        img_bw_contrast = Image.merge('YCbCr', (Image.fromarray(y_array), cb, cr))

        # Convert back to RGB for display (if needed)
        img_bw_contrast_rgb = img_bw_contrast.convert('RGB')
        
        return img_bw_contrast_rgb

def get_csv_files_from_directory(directory_path):

    csv_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def get_image_paths_from_csv(csv_file_path, num_files):
    # Extract the directory path from the CSV file path
    directory_path = os.path.dirname(csv_file_path)

    # Initialize a list to store the image paths
    image_paths = []

    # Read the CSV file and extract the image filenames
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            image_filename = row[0]
            global_avg = row[1]
            print(f"Visualizing {image_filename} with a global avg of {global_avg}")
            image_path = os.path.join(directory_path, image_filename)
            image_paths.append(image_path)

            # Break the loop when the desired number of files is reached
            if len(image_paths) == num_files:
                break

    return image_paths

def get_image_paths_from_dir(image_dir, num_files=None):
    # Get a list of all files in the specified directory
    all_files = os.listdir(image_dir)

    # Filter files ending with ".jpeg"
    jpeg_files = [file for file in all_files if file.lower().endswith(".jpeg")]

    # Limit the number of files if num_files is specified
    if num_files is not None:
        jpeg_files = jpeg_files[:num_files]

    # Create full paths for the images
    image_paths = [os.path.join(image_dir, file) for file in jpeg_files]

    return image_paths