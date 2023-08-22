import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vit_pytorch.vit_for_small_dataset import ViT
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from utils.imageset_handler import ImageQualityDataset


def train_vit_model(csv_file, dataset_root, results_path, num_epochs=10, batch_size=16, learning_rate=1e-4/2, pretrained_model_path=None, vis=False):    # Define the ViT model architecture
 
    ############################# BUILD MODEL #############################
    v = ViT(
        image_size=256,
        patch_size=16,
        num_classes=5,  # Number of classes for image quality levels
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        emb_dropout=0.1
    )

    # Define the image augmentation transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])

    if pretrained_model_path:
        v.load_state_dict(torch.load(pretrained_model_path))
        print("Pretrained Model loaded..")

    # Create the dataset object
    dataset = ImageQualityDataset(csv_file,dataset_root,transform=transform)

    ############################# VISUALISATION #############################
    if vis:
        # Get an example image and its label
        example_idx = 0  # Change this index to see different examples
        example_image, example_label = dataset[example_idx]
        print(f"Image {example_image} has a rating of {example_label+1}")
        # Reverse the normalization to visualize the image
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        example_image = example_image * std[:, None, None] + mean[:, None, None]
        example_image = example_image.clamp(0, 1)

        # Convert the torch tensor to a numpy array for plotting
        example_image_np = example_image.numpy().transpose(1, 2, 0)

        # Plot the example image
        plt.imshow(example_image_np)
        plt.title(f"Example Image (Quality Level {example_label+1})")
        plt.axis('off')
        plt.show()

    test_size = 0.2
    num_train = int(len(dataset)* (1-test_size))
    num_val = len(dataset) - num_train
    
    assert len(dataset) == num_train+num_val
    print('Splitting Dataset..')
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
    print("Dataset splitted into train and validation...")
    print(f"Number of Data to train: {num_train}")
    print(f"Number of Data to validate: {num_val}")

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(v.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v.to(device)

    best_val_loss = float('inf')
    best_model_weights = None
    figure_path = None

    # Lists to store the loss values
    train_losses = []
    val_losses = []
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training
        v.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for _, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = v(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)  # Get predicted labels
            train_preds.extend(preds.cpu().numpy())  # Extend the list of predictions
            train_labels.extend(labels.cpu().numpy())  # Extend the list of true labels
            train_accuracy = accuracy_score(train_labels, train_preds)

        # Validation
        v.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for _, (images, labels) in enumerate(val_dataloader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = v(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)  # Get predicted labels
                val_preds.extend(preds.cpu().numpy())  # Extend the list of predictions
                val_labels.extend(labels.cpu().numpy())  # Extend the list of true labels


                val_loss += loss.item() * images.size(0)

        val_accuracy = accuracy_score(val_labels, val_preds)
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Acc: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}')

        # Calculate and store the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            # Delete the previously saved best model
            if best_model_weights is not None:
                os.remove(best_model_path)

            # Update the best validation loss and save the new best model
            best_val_loss = val_loss
            best_model_weights = v.state_dict().copy()

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            last_folder = os.path.basename(dataset_root)
            # Use the timestamp and transfer learning information as a name extension
            model_name = f"vit_model_{timestamp}_epoch_{epoch+1}of{num_epochs}_valLoss_{best_val_loss:.3f}_valAcc_{val_accuracy:.3f}_batchsize_{batch_size}_lr_{learning_rate:.1f}_{last_folder}.pth"
            best_model_path = os.path.join(results_path, model_name)
            torch.save(best_model_weights, best_model_path)


    # Save the Matplotlib figure with the same basename as the saved model
    figure_name = os.path.splitext(model_name)[0] + '.png'
    figure_path = os.path.join(results_path, figure_name)

    # Plot the losses
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(figure_path)

    if vis:
        plt.show()

    print("Training completed!")




#Define the paths to dataset directory and results directory

#OBJETIVE
# dataset_root = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/DatasetObjective/allDistorted'
# csv_file = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/DatasetObjective/objective_imagesquality_scores.csv'

#SUBJECTIVE
dataset_root = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/DatasetSubjective/Persons/Person_1_shinyx/imgs'
csv_file = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/assets/DatasetSubjective/Persons/Person_1_shinyx/img_scores.csv'

#RESULTS
results_path = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/results'

#PRETRAINED PATH
pretrained_model = '/home/maxgan/WORKSPACE/UNI/BA/TIQ/results/vit_model_20230713_231903_Epochs_35_batchsize_64_allDistorted.pth'
# pretrained_model = None

# Run the training process
train_vit_model(csv_file, dataset_root, results_path, batch_size=64, num_epochs=20 ,vis=False, pretrained_model_path=pretrained_model)

