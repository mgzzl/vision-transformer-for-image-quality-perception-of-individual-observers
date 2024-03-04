import torch
from PIL import Image
import os
import torch.nn as nn
import sys
sys.path.append('/home/maxgan/WORKSPACE/UNI/BA/vision-transformer-for-image-quality-perception-of-individual-observers')
from model.recorder import Recorder
from misc.helpers import create_vit_model, trans_norm2tensor

patch_size = 16


def predict_images_in_folder(folder_path, model):
    """
    Process all images in a folder and return predictions and attention maps.

    Parameters:
    folder_path (str): The path to the folder containing the images.
    model (torch.nn.Module): The Vision Transformer model.

    Returns:
    -------
    predictions (list): List of predictions for each image.
    attentions (list): List of attention maps for each image.
    """
    predictions = []
    attentions = []
    
    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg'))]

    # Iterate over each image file
    for image_file in image_files:
        # Load and preprocess the image
        image_path = os.path.join(folder_path, image_file)
        image_pil = Image.open(image_path).convert('RGB')
        image = trans_norm2tensor(image_pil, 256)
        img = image.to(device)

        # Make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size
        # Make the prediction
        with torch.no_grad():
            model.eval()
            prediction, attention = model(img)
            _, preds = torch.max(prediction, dim=1)
            predictions.append(preds.cpu().numpy()[0]+1)
            attentions.append(attention.cpu().numpy())

    return predictions, attentions

if __name__ == "__main__":
    # Load the trained model weights
    model_path = 'results/weights/Cross-Entropy_3_Iter_var_0.4/FINAL/AIO5.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_vit_model(weights_path=model_path)
    model = Recorder(model).to(device)
    model = model.to(device)
    # Directory containing the images
    image_folder = "assets/work_imgs/fg_bg"

    # Process images in the folder
    predictions, attentions = predict_images_in_folder(image_folder, model)
    
    # Print the results
    for i, (prediction, attention) in enumerate(zip(predictions, attentions)):
        print(f"Prediction for image {i + 1}: {prediction}")
        print(f"Attention map shape for image {i + 1}: {attention.shape}\n")