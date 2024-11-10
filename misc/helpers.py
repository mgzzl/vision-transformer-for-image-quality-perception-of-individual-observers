import csv
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.vit_for_small_dataset import ViT

def create_vit_model(image_size=256, patch_size=16, num_classes=5, dim=1024,  depth=6, heads=16, mlp_dim=2048, emb_dropout=0.1, weights_path=None):
    """
    Create a Vision Transformer (ViT) model.

    Parameters:
    image_size (int): Input image size. Defaults to 256.
    patch_size (int): Patch size. Defaults to 16.
    num_classes (int): Number of output classes. Defaults to 5.
    dim (int): Dimension of the token embeddings. Defaults to 1024.
    depth (int): Depth of the Transformer model. Defaults to 6.
    heads (int): Number of attention heads. Defaults to 16.
    mlp_dim (int): Dimension of the MLP layers. Defaults to 2048.
    emb_dropout (float): Dropout rate applied to the token embeddings. Defaults to 0.1.
    weights_path (str): Path to pretrained weights file. Defaults to None.

    Returns:
    model (torch.nn.Module): ViT model instance.
    """
    # Initialize ViT model with specified parameters
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

def predict_image_class_probabilities(model, img, device):
    """
    Get class probabilities for an input image using the provided model.

    This function takes an input image, preprocesses it, and passes it through the provided model
    to obtain class probabilities using softmax activation. It returns the probabilities as a NumPy array.

    Parameters:
    model (torch.nn.Module): The pre-trained model.
    img (torch.Tensor): The input image tensor.
    device (torch.device): The device to run the model on (e.g., CPU or GPU).

    Returns:
    probabilities (numpy.ndarray): An array containing class probabilities for the input image.
    """
    # Add a batch dimension to the input image
    img = img.unsqueeze(0).to(device)
    # Move the model to the specified device
    model.to(device)

    # Set the model to evaluation mode and disable gradient computation
    with torch.no_grad():
        model.eval()
        # Pass the input image through the model to get the output logits
        output = model(img)

    # Apply softmax activation to obtain class probabilities and convert to NumPy array
    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    return probabilities

def find_model_weights(directory_path):
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

def calculate_label_distributions(labels, device, sigma=0.7**2, num_classes=5):
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


def trans_norm2tensor(img, img_size, transformation_function=None):
    """
    Transform, normalize, and apply a chosen transformation to the input image.

    Parameters:
    - img (PIL.Image): Input image.
    - img_size (int): Desired size for the image.
    - transformation_function (function, optional): Function that applies a transformation to a part of the image. If None, no additional transformation is applied.

    Returns:
    torch.Tensor: Transformed and normalized image tensor.
    """
    # Convert grayscale to RGB if the image has only one channel
    if img.mode == 'L':
        img = img.convert('RGB')

    # Define the normalization parameters (mean and std) - ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Define the transformation including resizing, center cropping, and normalization
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ])

    # Apply the basic transformations (resize, center crop)
    img = transform(img)

    # Apply the optional additional transformation on a part of the image
    if transformation_function is not None:
        # Convert back to PIL Image for applying the part-transformation
        # img_pil = transforms.ToPILImage()(img)
        
        # Apply the selected transformation (e.g., Grayscale Patch, Blur Patch)
        img = transformation_function(img)
        
        # Convert the PIL image back to Tensor and normalize
        # img = transforms.ToTensor()(img_pil)

    img = transforms.ToTensor()(img)
    # Normalize the image
    img = transforms.Normalize(mean=mean, std=std)(img)

    return img

def prev_img(img, img_size, transformation_function=None):
    """
    Process the image for previewing purposes.

    This function applies transformations to resize, center crop the input image, 
    and optionally apply additional transformations to a part of the image.

    Parameters:
    - img (PIL.Image): The input image.
    - img_size (int): The desired size of the output image.
    - transformation_function (function, optional): A function that applies a transformation 
      to a part of the image. If None, no additional transformation is applied.

    Returns:
    img (PIL.Image): The processed and optionally transformed image.
    """
    # Define the transformation for resizing and center cropping
    transform = transforms.Compose([
        transforms.Resize(img_size),     # Resize the image
        transforms.CenterCrop(img_size), # Center crop the image
    ])
    
    # Apply the resize and center crop transformation
    img = transform(img)

    # Apply the optional additional transformation on a part of the image
    if transformation_function is not None:
        img = transformation_function(img)  # Apply the transformation to a specific area of the image

    return img


def prev_img_gray(img, img_size, transformation_function=None):
    """
    Process the image for previewing purposes with grayscale and contrast adjustment.

    This function applies transformations to resize and center crop the input image to a specified size.
    It then converts the image to grayscale and enhances the contrast by scaling the luminance values.
    
    Parameters:
    img (PIL.Image): The input image.
    img_size (int): The desired size of the output image.

    Returns:
    img_bw_contrast_rgb (PIL.Image): The processed image with enhanced contrast in RGB format.
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(img_size),     # Resize the image
        transforms.CenterCrop(img_size), # Center crop the image
    ])
    # Apply the transformation to the input image
    img = transform(img)

        # Apply the optional additional transformation on a part of the image
    if transformation_function is not None:
        # Apply the selected transformation (e.g., Grayscale Patch, Blur Patch)
        img = transformation_function(img)
        
        # Convert the PIL image back to Tensor and normalize
        # img = transforms.ToTensor()(img_pil)

    # Convert the image to grayscale
    img_bw = img.convert('L')

    # Convert grayscale image to YCbCr color space
    img_ycbcr = img_bw.convert('YCbCr')
    y, cb, cr = img_ycbcr.split()
    # Convert Y channel to NumPy arrays
    y_array = np.array(y)

    # Perform contrast adjustment on the Y component
    y_array = np.clip(y_array * 2.0, 0, 255).astype(np.uint8)

    # Merge the adjusted Y, Cb, Cr components back into an image
    img_bw_contrast = Image.merge('YCbCr', (Image.fromarray(y_array), cb, cr))

    # Convert the image back to RGB for display
    img_bw_contrast_rgb = img_bw_contrast.convert('RGB')
    
    return img_bw_contrast_rgb

def find_csv_files(directory_path):
    """
    Get a list of CSV files from the specified directory and its subdirectories.

    Parameters:
    directory_path (str): The path to the directory containing CSV files.

    Returns:
    csv_files (list): A list of paths to CSV files found in the directory and its subdirectories.
    """
    csv_files = []  # Initialize an empty list to store CSV file paths
    # Traverse through the directory and its subdirectories
    for root, _, files in os.walk(directory_path):
        # Iterate over each file in the current directory
        for file in files:
            # Check if the file has a ".csv" extension
            if file.endswith(".csv"):
                # If it does, append the full path of the CSV file to the list
                csv_files.append(os.path.join(root, file))
    return csv_files

def get_image_paths_from_csv(csv_file_path, num_files):
    """
    Get a list of image paths from a CSV file containing image filenames.

    Parameters:
    csv_file_path (str): The path to the CSV file.
    num_files (int): The number of image paths to extract. 0 means extract all.

    Returns:
    image_paths (list): A list of paths to the image files extracted from the CSV.
    """
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
            # Print visualization information
            image_path = os.path.join(directory_path, image_filename)
            image_paths.append(image_path)
            
            # Break the loop when the desired number of files is reached
            if len(image_paths) == num_files:
                break

    return image_paths

def get_image_filenames_by_label(csv_file_path, label:int, img_directory_path, num_files=0):
    """
    Get a list of image paths from a CSV file containing image filenames.

    Parameters:
    csv_file_path (str): The path to the CSV file.
    label (str): The label to filter the images by.
    img_directory_path (str): The path to the directory containing the image files.
    num_files (int): The number of image paths to extract. 0 means extract all.

    Returns:
    image_filenames (list): A list of paths to the image files extracted from the CSV.
    """
    # Initialize a list to store the image paths
    image_paths = []

    # Read the CSV file and extract the image filenames
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            image_filename = row[0]
            vote = row[1]
            # fitler only images with vote of 5
            if int(vote) == label:
                image_path = os.path.join(img_directory_path, image_filename)
                image_paths.append(image_path)

            # Break the loop when the desired number of files is reached
            if len(image_paths) == num_files:
                break

    return image_paths

def get_image_paths_from_dir(image_dir, num_files=None):
    """
    Get a list of image paths from the specified directory.

    This function lists all files in the specified directory, filters out files with ".jpeg" extension,
    and creates full paths for the images. It optionally limits the number of files returned if num_files
    is specified.

    Parameters:
    image_dir (str): The path to the directory containing images.
    num_files (int): The number of image paths to return. If None, return all image paths. Defaults to None.

    Returns:
    image_paths (list): A list of paths to the images in the directory.
    """
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
