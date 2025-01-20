import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from misc.helpers import create_vit_model, prev_img, prev_img_gray, trans_norm2tensor
from model.recorder import Recorder
import seaborn as sns
from PIL import Image

from misc import transformations


def normalize_attention_maps(attentions):
    """
    Normalize 2D attention maps to the range [0, 1].

    Parameters:
    - attentions (numpy.ndarray): Array of shape (num_heads, height, width).

    Returns:
    numpy.ndarray: Normalized attention maps.
    """
    attentions_min = np.min(attentions)
    attentions_max = np.max(attentions)
    normalized_attentions = (attentions - attentions_min) / (attentions_max - attentions_min)
    return normalized_attentions

def get_attention_maps(model, img, patch_size, device):
    """
    Get prediction and attention maps for a given image using a pre-trained AIO.

    Parameters:
    - model: Pre-trained AIO.
    - img (torch.Tensor): Input image tensor.
    - patch_size (int): Size of image patches for attention computation.
    - device (torch.device): Device on which the model is loaded.

    Returns:
    tuple: Tuple containing predicted labels and attention maps.

    Note:
    - Ensure that the input image (img) is transformed and normalized before passing it to this function.
    - from helpers import trans_norm2tensor
    """
    model = Recorder(model).to(device)
    img = img.to(device)
    img = img.unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    
    # Make the prediction
    with torch.no_grad():
        model.eval()
        outputs, attentions = model(img)
        _, preds = torch.max(outputs, dim=1)
        nh = attentions.shape[2]  # number of heads
        nl = attentions.shape[1]  # number of layers
        print("Attention Shape", attentions)
        # Initialize the result tensor for attention maps
        # (nl, nh, w_featmap, h_featmap) 
        # attens.shape[-1] - 1 is the number of tokens (cls + patches) - 1 (exclude cls)
        atts = torch.zeros(nl, nh, attentions.shape[-1] - 1, attentions.shape[-1] - 1)  # Initialize the result tensor

        # Extract attention maps for each layer and head
        for i in range(nl):
            # Extract attention maps for each head in the current layer
            # attentions[0, i, :, 0, 1:] is the attention map for the first token ([CLS] token) in the current layer - cls-to-patch attention
            att = attentions[0, i, :, 0, 1:].reshape(nh, -1)
            print("attentions", att.shape) 
            att = att.reshape(nh, w_featmap, h_featmap)
            print("attentions", att.shape) 
            att = nn.functional.interpolate(att.unsqueeze(0), scale_factor=patch_size, mode="bilinear")[0]
            print("attentions", att.shape) 
            atts[i, :, :, :] = att


        print("Attention Shape after reshaping", atts.shape)
        model.clear()

    return preds.cpu().numpy()[0], atts.cpu().numpy()

def visualize_all_layer_head_attention_maps(model, img, img_size, patch_size, device):
    """
    Visualize attention maps for all layers and heads in a pre-trained AIO.

    Parameters:
    - model: Pre-trained AIO.
    - img: Input image.
    - img_size (int): Desired size of the input image.
    - patch_size (int): Size of image patches for attention computation.
    - device (torch.device): Device on which the model is loaded.
    """
    # img_pre = transformations.apply_blur_patch(img, (100,100))
    img_pre = trans_norm2tensor(img, img_size)
    _, attention = get_attention_maps(model, img_pre, patch_size, device)
    plot_attention_maps_per_layer_and_heads(img, attention)

def plot_attention_maps_per_layer_and_heads(img, attention):
    """
    Plot attention maps for each layer and head.

    Parameters:
    - img: Input image.
    - attention (numpy.ndarray): Array of attention maps.
    """
    n_heads = attention.shape[1]
    n_layers = attention.shape[0]
    image_size = 256

    img = transformations.apply_blur_patch(img, (100,100))
    img_pre = prev_img(img, image_size)
    img_gray = prev_img_gray(img, image_size)

    fig, axes = plt.subplots(n_heads + 1, n_layers + 1, figsize=(20, 50))

    for ax in axes.flat:
        ax.axis('off')

    # Original image
    axes[0, 0].imshow(img_pre)
    axes[0, 0].set_title("Original Image")

    for head_idx in range(n_heads + 1):
        for layer_idx in range(n_layers):
            ax = axes[head_idx, layer_idx + 1]
            if head_idx == 0:
                layer_mean = np.mean(attention[layer_idx], axis=0)
                layer_mean_norm = normalize_attention_maps(layer_mean)
                ax.imshow(img_gray, cmap='gray')
                sns.heatmap(layer_mean_norm, cmap="inferno", alpha=0.7, ax=ax)
                ax.set_title(f"Layer {layer_idx + 1}: Head Mean")
            else:
                head = attention[layer_idx][head_idx - 1]
                head_norm = normalize_attention_maps(head)
                ax.imshow(img_gray, cmap='gray')
                sns.heatmap(head_norm, cmap="inferno", alpha=0.7, ax=ax)
                ax.set_title(f"Layer {layer_idx + 1}: Head {head_idx}")

    plt.show()


def get_attention_maps_across_weights(model, img, img_size, patch_size, depth, weight_files, device):
    """
    Visualize attention maps for multiple weight files of a Vision Transformer model.

    This function generates visualizations for attention maps across different layers and heads 
    of multiple AIOs. It displays attention maps for each weight file in separate rows and
    for each layer in separate columns.

    Parameters:
    model (torch.nn.Module): The Vision Transformer model.
    img (PIL.Image.Image): The input image for visualization.
    img_size (int): The size of the input image.
    patch_size (int): The size of the image patches used in the Vision Transformer.
    depth (int): The depth of the Vision Transformer model.
    weight_files (list): A list of paths to weight files used for visualization.
    device (torch.device): The device to run the model on (e.g., CPU or GPU).

    Returns:
    None: This function does not return any value. It directly visualizes the attention maps.
    """
    # Convert the input image to a tensor and move it to the specified device
    img_tensor = trans_norm2tensor(img, img_size).to(device)
    # Convert the input image to grayscale for visualization
    img_gray = prev_img_gray(img, img_size)

    # Create subplots for visualizing attention maps
    fig, axes = plt.subplots(len(weight_files), depth + 1, figsize=(3 * 7, 2 * len(weight_files)))
    for ax in axes.flat:
        ax.axis('off')

    # Iterate over each weight file
    for i, weight_file in enumerate(weight_files):
        # Load the model with the specified weight file and set it to evaluation mode
        model = create_vit_model(weights_path=weight_file)
        model.eval()

        # Iterate over each layer of the model
        for j in range(depth + 1):
            ax = axes[i, j]
            if j != 0:
                # Visualize attention for the current layer and head
                _, attention = get_attention_maps(model, img_tensor, patch_size, device)
                attention = attention[j - 1]
                att_mean = np.mean(attention, 0)
                att_mean_norm = normalize_attention_maps(att_mean)

                # Plot attention map as a heatmap
                ax.imshow(img_gray)
                sns.heatmap(att_mean_norm, cmap="inferno", alpha=0.7, ax=ax)
                ax.set_title(f'{os.path.splitext(os.path.basename(weight_files[i]))[0]} Layer {j} Head Mean')
            else:
                # Plot the input image in the first column
                ax.imshow(prev_img(img, img_size))

    # Show the visualization
    plt.show()


def get_attention_maps_with_deviation(img, weight_files, image_size, depth, patch_size, device):
    """
    Visualize attention maps for multiple weight files of a Vision Transformer model.

    This function generates visualizations for attention maps across different layers and heads of a AIO. It displays the deviation of attention maps from the average attention
    maps across all weight files. The function takes an input image, a list of weight files, image size, depth of
    the model, patch size, and the device to run the model on.

    Parameters:
    img (PIL.Image.Image): The input image for visualization.
    weight_files (list): A list of paths to weight files used for visualization.
    image_size (int): The size of the input image.
    depth (int): The depth of the Vision Transformer model.
    patch_size (int): The size of the image patches used in the Vision Transformer.
    device (torch.device): The device to run the model on (e.g., CPU or GPU).

    Returns:
    None: This function does not return any value. It directly visualizes the attention maps.
    """
    # Get the number of weight files
    num_weights = len(weight_files)

    # Convert the input image to a tensor and grayscale version
    img_prev = prev_img(img, image_size)
    img_tensor = trans_norm2tensor(img, image_size).to(device)
    img_gray = prev_img_gray(img, image_size)

    # Create subplots for visualizing attention maps
    fig, axes = plt.subplots(num_weights, depth + 1, figsize=(3 * 7, 2 * depth))
    for ax in axes.flat:
        ax.axis('off')

    # Initialize a list to store attention maps per weight file
    img_attentions_per_aio = []

    # Iterate over each weight file
    for i, weight_file in enumerate(weight_files):
        # Load the model with the specified weight file and set it to evaluation mode
        model = create_vit_model(weights_path=weight_file)
        model.eval()
        img_attentions = []

        # Iterate over each layer of the model
        for j in range(depth + 1):
            if j != 0:
                # get attention for the current layer and mean head
                _, attention = get_attention_maps(model, img_tensor, patch_size, device)
                attention = attention[j - 1]
                att_mean = np.mean(attention, 0)
                att_mean_norm = normalize_attention_maps(att_mean)

                # Append attention map to the list
                img_attentions.append(att_mean_norm)

        # Append attention maps for the current weight file to the list
        img_attentions_per_aio.append(img_attentions)

    # Convert the list of attention maps to a numpy array
    img_attentions_per_aio = np.array(img_attentions_per_aio)

    # Calculate the average attention maps across all weight files
    average_layer_attention_maps = np.mean(img_attentions_per_aio, axis=0)

    # Calculate the deviation of attention maps from the average attention maps
    sub_atts = np.subtract(img_attentions_per_aio, average_layer_attention_maps)

    # Visualize attention maps with deviation from the average
    for i in range(sub_atts.shape[0]):  # for each weight file
        for j in range(sub_atts.shape[1] + 1):  # for each layer
            ax = axes[i, j]

            if j != 0:
                ax.imshow(img_gray)
                sns.heatmap(sub_atts[i][j - 1], cmap="seismic", alpha=0.7, ax=ax, vmin=-1, vmax=1)
                ax.set_title(f'{os.path.splitext(os.path.basename(weight_files[i]))[0]} Layer {j} Head Mean')
            else:
                ax.imshow(img_prev)

    # Show the visualization
    plt.show()


def plot_attention_maps_comparison(weight_files, image_paths, image_size, patch_size, output_dir, layer_idx, device):
    """
    Plot attention maps for multiple weight files and input images.

    This function generates visualizations for attention maps across multiple weight files and input images.
    It takes a list of weight files, a list of image paths, image size, patch size, output directory, layer index,
    and device to run the model on. For each input image, it calculates the attention maps for each weight file and
    visualizes the deviation of attention maps from the mean attention map across all weight files.

    Parameters:
    weight_files (list): A list of paths to weight files used for visualization.
    image_paths (list): A list of paths to input images for visualization.
    image_size (int): The size of the input images.
    patch_size (int): The size of the image patches used in the Vision Transformer.
    output_dir (str): The directory path to save the generated visualizations.
    layer_idx (int): The index of the layer for which attention maps are visualized (-1 for the last layer).
    device (torch.device): The device to run the model on (e.g., CPU or GPU).

    Returns:
    None: This function does not return any value. It directly saves the generated visualizations.
    """
    # Initialize Matplotlib figures
    num_weights = len(weight_files)
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(num_images, num_weights + 1, figsize=(3 * (num_weights + 1), 2.5 * num_images))
    
    for ax in axes.flat:
        ax.axis('off')
    
    # Iterate over each input image
    for j, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        img_prev = prev_img(img, image_size)
        img_gray = prev_img_gray(img, image_size)

        aio_attentions = []
        # Get attention maps for each weight file
        for n in range(num_weights):
            model = create_vit_model(weights_path=weight_files[n])
            img_tensor = trans_norm2tensor(img, image_size)
            _, attention = get_attention_maps(model, img_tensor, patch_size, device)
            
            # Determine the layer index
            if layer_idx == -1:
                layer_idx = attention.shape[0] - 1
                
            att_ll = attention[layer_idx]  # num_heads, seq_len, seq_len
            att_mean = np.mean(att_ll, 0)
            att_ll_norm = normalize_attention_maps(att_mean)

            # Remove x% lowest attention values
            # att_ll_norm = np.where(att_ll_norm < np.max(att_ll_norm) * 0.2, 0, att_ll_norm)

            aio_attentions.append(att_ll_norm)

        # Calculate mean attention maps and deviation
        mean_aio_atts = np.mean(aio_attentions, axis=0)
        atts_sub = np.subtract(aio_attentions, mean_aio_atts)
        
        # Plotting
        for i in range(len(atts_sub) + 1):
            ax = axes[j, i]
            if i != 0:
                ax.imshow(img_gray)
                sns.heatmap(atts_sub[i - 1], cmap="seismic", alpha=0.8, ax=ax, vmin=-1, vmax=1)
                ax.set_title(f'{os.path.splitext(os.path.basename(weight_files[i - 1]))[0]} Layer {layer_idx + 1} Attention')
            else:
                ax.imshow(img_prev)
                ax.set_title('Original Image')
    
    # Save the visualizations
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{os.path.basename(os.path.dirname(image_paths[0]))}_layer_{layer_idx + 1}_attention_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()