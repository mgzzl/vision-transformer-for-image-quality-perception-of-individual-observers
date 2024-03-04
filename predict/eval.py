import csv
import torch
from PIL import Image
import os
import torch.nn as nn
import sys
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import numpy as np
sys.path.append('/home/maxgan/WORKSPACE/UNI/BA/vision-transformer-for-image-quality-perception-of-individual-observers')
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.imageQualityDataset import ImageQualityDataset
from scipy.stats import entropy
from misc.helpers import create_vit_model, calculate_label_distributions, trans_norm2tensor

def evaluate_model(weight_file, test_loader):
    """
    Evaluate AIO with the specified weight file.

    Parameters:
    weight_file (str): The path to the weight file.
    test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
    results (dict): Evaluation results including accuracy, MSE, MSE weighted, mean entropy, mean KL divergence, and classification report.
    """

    print(f'Weights-file: {os.path.basename(weight_file)} will be evaluated')
    # Load the model with different weights
    model = create_vit_model(weights_path=weight_file)
    model.eval()

    # Initialize result lists
    true_labels = []
    test_preds = []
    entropies = []
    true_entropies = []
    weighted_sums = []
    kl_divs = []

    with torch.no_grad():
        for i, (images, image_paths, labels) in enumerate(test_loader, 0):
            # images = images.to(device)
            # labels = labels.to(device)
            outputs = model(images)
            true_labels.extend(labels)

            # Convert logits to probabilities
            probabilities = nn.functional.softmax(outputs, dim=1)
            
            # Calculate the true distribution
            true_distributions = calculate_label_distributions(labels,device='cpu')
            # Get prediction by the maximum probability
            _, preds = torch.max(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            
            formatted_probabilities = ["{:.4f}".format(prob) for prob in probabilities[0]]

            # Calculate Entropy
            entropy_values = entropy(probabilities.numpy(),base=np.exp(1), axis=1)
            true_entropy_values = entropy(true_distributions.numpy(),base=np.exp(1), axis=1)
            # Format entropies in a readable way
            entropies.extend(entropy_values)
            true_entropies.extend(entropy_values)

            # Calculate KL Divergence
            kl_div = torch.nn.functional.kl_div(torch.log(probabilities), true_distributions, reduction='batchmean')
            kl_divs.append(kl_div.item())
            
            # Define weighting factors
            weighting_factors = [0,1,2,3,4]
            # Calculate the weighted sum of probabilities
            weighted_sum = torch.sum(probabilities * torch.tensor(weighting_factors), dim=1).cpu().numpy()
            # Format weighted sum in a readable way
            weighted_sums.extend(weighted_sum)

            # Example printout for the first batch
            if i <= 2:
                print(f'Example Prediction of Batch: {i}')
                print(f'True-Label: {labels.cpu()[0]}')
                print(f'Predicted-Label: {preds.cpu().numpy()[0]}')
                print(f'Weighted Sum of Probability: {round(weighted_sum[0],4)}')  # Weighted Sum of Prob
                print(f'Predicted Probability Distribution: {[round(prob,4) for prob in probabilities[0].numpy()]}')
                print(f'True Probability Distribution: {true_distributions.cpu().numpy()[0]}')
                print(f'Entropy Value: {round(entropy_values[0],4)}') # High Value: spreading; Low Value: concentrated
                print(f'True Entropy Value: {round(true_entropy_values[0],4)}') # High Value: spreading; Low Value: concentrated
                print(f'KL Divergence (batch-mean): {round(kl_div.item(),4)}\n')

    # Calculate the MSE of weighted sum and ground truth
    mse_weighted = mean_squared_error(true_labels, weighted_sums)

    # Calculate the MSE of most likely class and ground truth
    mse = mean_squared_error(true_labels, test_preds)

    # Calculate the Mean Entropy
    mean_entropy = np.mean(entropies)

    # Calculate the Mean KL Divergence
    mean_kl_div = np.mean(kl_divs)

    # Calculate Accuracy
    accuracy = accuracy_score(true_labels, test_preds)

    # Get the unique predicted classes
    unique_preds = set(test_preds)
    target_names  = ["bad", "poor", "fair", "good", "excellent"]

    # Extract only the target names that match the unique predicted classes
    target_names_used = [target_names[i] for i in unique_preds]
    # Generate classification report
    class_report = classification_report(true_labels, test_preds, target_names=target_names_used)

    # Generate confusion matrix
    # confusion = confusion_matrix(true_labels, test_preds, labels=target_names_used)

    print('#'*100)
    print('model summary:')
    # Print summary
    results = {
        "Weights File": os.path.basename(weight_file),
        "Accuracy": accuracy,
        "MSE": mse,
        "MSE weighted": mse_weighted,
        "Mean Entropy": mean_entropy,
        "Mean KL Divergence": mean_kl_div, 
        "Classification Report": "\n"+class_report
    }
    for key, value in results.items():
        print(f"{key}: {value}")
    print('#'*100 + "\n")

    return results

def save_results_to_csv(results, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write rows
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    """
    Evaulate all AIOs based on their real Observer (Obs) and save it in a csv file.
    
    """


    # Define parameters
    image_size = 256
    patch_size = 16
    num_classes = 5
    depth = 6

    batch_size = 128

    # Define model and device
    model = create_vit_model(image_size=image_size, patch_size=patch_size, num_classes=num_classes, depth=depth)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dataset parameters
    csv_dir = 'assets/Test'
    csv_paths = [os.path.join(csv_dir, f'Obs{i}.csv') for i in range(0, 6)]

    # List of different weight files
    weights_dir = 'results/weights/Cross-Entropy_3_Iter_var_0.4/FINAL'
    weight_paths = [os.path.join(weights_dir, f'AIO{i}.pth') for i in range(0, 6)]

    # Testdataset
    dataset_root = 'assets/Test/DSX'

    # Define the normalization parameters (mean and std)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define the transformation including normalization
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    results = []
    for weight_file,csv_file in zip(weight_paths,csv_paths):
        print(f"Loading Testdataset due to {os.path.basename(csv_file)}...")
        # Initialize the test dataset and data loader
        test_dataset = ImageQualityDataset(csv_file, dataset_root, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        result = evaluate_model(weight_file, test_loader)
        results.append(result)

    # Save results to a CSV file
    output_csv_file = 'evaluation_results.csv'
    output_csv_path = os.path.join(weights_dir, output_csv_file)
    save_results_to_csv(results, output_csv_path)
    print(f"Results saved to '{output_csv_path}'")