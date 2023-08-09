import cv2
from matplotlib import pyplot as plt
import numpy as np
from vit_pytorch.vit_for_small_dataset import ViT
from imageset_handler import ImageQualityDataset

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def prediction_quality(image_path, model):
    # Define the image augmentation transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])

    # image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image_path).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted_label = torch.max(outputs, 1)
        predicted_rating = predicted_label.item() + 1  # Adding 1 to convert 0-based index to 1-based rating
    return predicted_rating

def compare(dataset, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    predictions = []
    ground_truth_ratings = []

    with torch.no_grad():
        for image_path, rating in dataset:
            # image_path = os.path.join(dataset_root, image_path)
            predicted_rating = prediction_quality(image_path, model)

            predictions.append(predicted_rating)
            ground_truth_ratings.append(rating + 1)

    return predictions, ground_truth_ratings

# Load the trained ViT model (the best model obtained from transfer learning)
# model_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/results/vit_model_20230713_231903_Epochs_35_batchsize_64_allDistorted.pth'
model_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/results/vit_model_20230801_131421_epoch_12of20_valLoss_0.680_batchsize_64_lr_0.00_imgs.pth'
# dataset_root = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/DatasetObjective/test'
dataset_root = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/Test/TestImg'
# csv_file = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/DatasetObjective/objective_imagesquality_scores_test.csv'
csv_file = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/Test/AccTest/shinyxAccTest20-01-2023.csv'
results_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/results'


model = ViT(
    image_size=256,
    patch_size=16,
    num_classes=5,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    emb_dropout=0.1
)
model.load_state_dict(torch.load(model_path))

dataset = ImageQualityDataset(csv_file,dataset_root)

# Compare model predictions with ground truth ratings for the subjective datasets
predictions, ground_truth_ratings = compare(dataset, model)

# Create the confusion matrix
conf_matrix = confusion_matrix(ground_truth_ratings, predictions)

figure_name = f"Confusion_Matrix_{os.path.splitext(os.path.basename(model_path))[0] + '.png'}"
figure_path = os.path.join(results_path, figure_name)

# Plot the confusion matrix as a heatmap with annotations
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted Ratings')
plt.ylabel('Ground Truth Ratings')
plt.title('Confusion Matrix')
plt.xticks(np.arange(5), np.arange(1, 6))
plt.yticks(np.arange(5), np.arange(1, 6))

# Add text annotations for true positives and false positives
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        # Customize text color for light blue boxes (when the value is high)
        if conf_matrix[i, j] > conf_matrix.max() / 2:
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white')
        else:
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')
plt.savefig(figure_path)
plt.show()
