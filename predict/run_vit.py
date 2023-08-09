import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from vit_pytorch.vit_for_small_dataset import ViT
import numpy as np
import math

# Define the ViT model architecture
v = ViT(
    image_size=256,
    patch_size=16,
    num_classes=5,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    emb_dropout=0.1
)

# Load the trained model weights
model_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/results/vit_model_20230628_183411_nEpochs_10_batchsize_16_objective_subjective.pth'
v.load_state_dict(torch.load(model_path))

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),  # Add center cropping
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as during training
])

# Load and preprocess the image
image_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/Assets/Dataset/Persons/Person_1/Bad/12ILSVRC2013_train_00000419.JPEG_I1_Q6.jpeg'  # Replace with the path to your image
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)  # Add batch dimension

# Make the prediction
with torch.no_grad():
    v.eval()
    prediction = v(image)

# Apply softmax to obtain probabilities
probabilities = nn.functional.softmax(prediction, dim=1)
# Compute predicted MOS
quality_levels = np.array([1, 2, 3, 4, 5])
MOS_res = torch.matmul(probabilities, torch.from_numpy(quality_levels).float().to(prediction.device))
# Convert the prediction to a quality level
quality_levels_str = ['Bad', 'Insufficient', 'Fair', 'Good', 'Excellent']
predicted_quality_level = quality_levels_str[int(MOS_res.item())]
print(f"Predicted: {image_path}")
# Print the results
print("Predicted MOS:", MOS_res.item())
print(f"Probabilities: {[probability.item() for probability in probabilities[0]]}")

print(f"Predicted image quality (mos): {predicted_quality_level}")
