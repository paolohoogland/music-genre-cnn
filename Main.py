import os
import pandas as pd

import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

from DatasetManager import DatasetManager
from SpectrogramDataset import SpectrogramDataset

from MLP import MLP

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()  
    total_loss = 0.0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        # Zero the gradients because PyTorch accumulates gradients by default
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs 
        outputs = model(features)

        # Compute the loss
        loss = loss_fn(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the loss
        total_loss += loss.item() * features.size(0)

    # Average the loss over the entire dataset
    epoch_loss = total_loss / len(dataloader.dataset)
    print(f"Training loss: {epoch_loss:.4f}")
    return epoch_loss

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()  
    total_loss = 0.0
    correct_predictions = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass: compute predicted outputs 
        outputs = model(features)

        # Compute the loss
        loss = loss_fn(outputs, labels)

        # Accumulate the loss
        total_loss += loss.item() * features.size(0)

        _, predicted = torch.max(outputs, 1) # _ : ignore the first return value
        correct_predictions += (predicted == labels).sum().item()

    # Average the loss over the entire dataset
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_accuracy = correct_predictions / len(dataloader.dataset)
    print(f"Validation loss: {epoch_loss:.4f}")
    print(f"Validation accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy

def test_model(model_path, model, dataloader, loss_fn, device):
    print("\n--- Starting Final Test ---")
    # We load the best saved model state
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_loss, test_accuracy = validate_one_epoch(model, dataloader, loss_fn, device)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

def main(args):
    torch.manual_seed(42)  # For reproducibility

    dataset_mgr_instance = DatasetManager()

    # Flag allowing creation of the dataset
    if args.feature_dataset_creation:
        dataset_mgr_instance.create_feature_dataset()
        return
    else:
        dataset_mgr_instance.get_feature_dataset()

    try:
        all_genres = dataset_mgr_instance.all_genres_list
        num_genres = len(all_genres)
        genre_to_index = {genre: i for i, genre in enumerate(all_genres)}
    except Exception as e:
        print(f"Error creating genre list: {e}")
        return

    # Create training, validation, and test sets
    try:
        train_set, val_set, test_set = dataset_mgr_instance.create_sets()
        num_features = len(train_set[0][1])
        print("Data sets created successfully.")
    except Exception as e:
        print(f"Error creating data sets: {e}")
        return

    # Create spectrogram datasets
    try:
        train_dataset = SpectrogramDataset(train_set, genre_to_index, transform=None, model_type='mlp')
        val_dataset = SpectrogramDataset(val_set, genre_to_index, transform=None, model_type='mlp')
        test_dataset = SpectrogramDataset(test_set, genre_to_index, transform=None, model_type='mlp')

        print("Spectrogram datasets created successfully.")
    except Exception as e:
        print(f"Error creating spectrogram datasets: {e}")
        return

    try:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print("Data loaders created successfully.")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return

    # Create an MLP (Multi-Layer Perceptron) 
    model = MLP(num_genres=num_genres, num_features=num_features)

    # Prints the model architecture
    # We will see every layer, its input shape, and output shape
    print("\n" + "="*40)
    print("INITIALIZING MODEL ARCHITECTURE")
    print(model)
    print("="*40 + "\n")

    try:
        # Device setup using CUDA (GPU) if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
    except Exception as e:
        print(f"Error setting up device: {e}")
        return

    try:
        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        print("Loss function and optimizer set up successfully.")
    except Exception as e:
        print(f"Error setting up loss function and optimizer: {e}")
        return

    try:
        highest_val_acc = 0.0
        model_save_path = "best_complex_model.pth"
        
        # Patience is used to prevent useless training
        patience = 20 
        epochs_without_improvement = 0

        print("Starting training...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    except Exception as e:
        print(f"Error setting up optimizer/scheduler: {e}")
        return

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
            val_loss, val_accuracy = validate_one_epoch(model, val_dataloader, loss_fn, device)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

            scheduler.step(val_accuracy)

            if val_accuracy > highest_val_acc:
                highest_val_acc = val_accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved with accuracy: {highest_val_acc:.4f}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"No improvement in validation accuracy for {epochs_without_improvement} epochs.")

            if epochs_without_improvement == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    try:
        test_model(model_save_path, model, test_dataloader, loss_fn, device)
    except Exception as e:
        print(f"Error during testing: {e}")
        return

    print("Training completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN for Music Genre Classification")

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')

    parser.add_argument('--feature_dataset_creation', action='store_true', help='Flag to create the feature dataset')

    args = parser.parse_args()
    main(args)