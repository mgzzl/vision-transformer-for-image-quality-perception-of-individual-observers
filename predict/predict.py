import torch
from PIL import Image
import os
import torch.nn as nn
import sys
sys.path.append('/home/maxgan/WORKSPACE/UNI/BA/vision-transformer-for-image-quality-perception-of-individual-observers')
from misc.visualization import get_attention_maps
from misc.helpers import create_vit_model, trans_norm2tensor

image_size=256
patch_size=16

def predict_images_in_dir(dir_path, model):
    """
    Process all images in a directory and return predictions and attention maps.

    Parameters:
    dir_path (str): The path to the dir containing the images.
    model (torch.nn.Module): The Vision Transformer model (AIO).

    Returns:
    -------
    predictions (list): List of predictions for each image.
    attentions (list): List of attention maps for each image.
    """
    predictions = []
    attentions = []
    
    # Get a list of all image files in the dir
    image_files = [file for file in os.listdir(dir_path) if file.endswith(('.jpg', '.jpeg'))]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Iterate over each image file
    for image_file in image_files:
        # Load and preprocess the image
        image_path = os.path.join(dir_path, image_file)
        image_name = os.path.basename(image_path)
        image_pil = Image.open(image_path).convert('RGB')
        img_pre = trans_norm2tensor(image_pil, image_size)
        pred, attention = get_attention_maps(model, img_pre, patch_size, device)

        print(f"Prediction for image {image_name}: {pred+1}")
        print(f"Attention map shape: {attention.shape} | depth, heads, width, height\n")
        predictions.append(pred)
        attentions.append(attention)

    return predictions, attentions

if __name__ == "__main__":

    model_path = 'results/weights/Cross-Entropy_3_Iter_var_0.4/FINAL/AIO0.pth'
    image_dir = "assets/work_imgs/fg_bg"

    # Load the trained model weights
    model = create_vit_model(weights_path=model_path)
    # Process images in the dir
    predictions, attentions = predict_images_in_dir(image_dir, model)
    