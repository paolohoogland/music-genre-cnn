import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

from DatasetManager import DatasetManager
from SpectrogramDataset import SpectrogramDataset
from CNN import CNN

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()  
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients because PyTorch accumulates gradients by default
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs 
        outputs = model(images)

        # Compute the loss
        loss = loss_fn(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the loss
        total_loss += loss.item() * images.size(0)

    # Average the loss over the entire dataset
    epoch_loss = total_loss / len(dataloader.dataset)
    print(f"Training loss: {epoch_loss:.4f}")
    return epoch_loss

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()  
    total_loss = 0.0
    correct_predictions = 0

    # Disable gradient calculation for memory efficiency
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass: compute predicted outputs 
            outputs = model(images)

            # Compute the loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # Accumulate the loss
            _, predicted = torch.max(outputs, 1) # _ : ignore the first return value
            correct_predictions += (predicted == labels).sum().item()

    # Average the loss over the entire dataset
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_accuracy = correct_predictions / len(dataloader.dataset)
    print(f"Validation loss: {epoch_loss:.4f}")
    print(f"Validation accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy

def test_model(model, dataloader, loss_fn, device):
    print("\n--- Starting Final Test ---")
    # We load the best saved model state
    model.load_state_dict(torch.load("most_precise_cnn.pth", weights_only=True))
    test_loss, test_accuracy = validate_one_epoch(model, dataloader, loss_fn, device)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")


def main(args):
    dataset_mgr_instance = DatasetManager()

    try:
        all_genres = dataset_mgr_instance.get_genre_list()
        num_genres = len(all_genres)
        genre_to_index = {genre: i for i, genre in enumerate(all_genres)} # Dictionary (genre_name: index)

        print("Genre list created successfully.")
    except Exception as e:
        print(f"Error creating genre list: {e}")

    try:
        train_set, val_set, test_set = dataset_mgr_instance.create_sets()
        print("Data sets created successfully.")
    except Exception as e:
        print(f"Error creating data sets: {e}")

    try:
        train_transform  = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((CNN.IMG_HEIGHT, CNN.IMG_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally to prevent overfitting
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Randomly translate and scale images
            transforms.ToTensor()
        ])

        val_and_test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((CNN.IMG_HEIGHT, CNN.IMG_WIDTH)),
            transforms.ToTensor()
        ])
        print("Data transform created successfully.")
    except Exception as e:
        print(f"Error creating data transform: {e}")

    try:
        train_dataset = SpectrogramDataset(train_set, genre_to_index, train_transform)
        val_dataset = SpectrogramDataset(val_set, genre_to_index, val_and_test_transform)
        test_dataset = SpectrogramDataset(test_set, genre_to_index, val_and_test_transform)

        print("Spectrogram datasets created successfully.")
    except Exception as e:
        print(f"Error creating spectrogram datasets: {e}")

    try:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print("Data loaders created successfully.")
    except Exception as e:
        print(f"Error creating data loaders: {e}")

    try:
        model = CNN(num_genres=num_genres)
        print("CNN model created successfully.")
        print(model)
    except Exception as e:
        print(f"Error creating CNN model: {e}")

    print("All components initialized successfully.")
    print("Starting training...")

    try:
        # Device setup using CUDA (GPU) if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        #     print(f"Using device: {device}")
        # else:
        #     device = torch.device("cpu")
        #     print(f"Using device: {device}")
        model.to(device)
    except Exception as e:
        print(f"Error setting up device: {e}")

    try:
        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Adam is a good optimizer
        # Weight decay is used to prevent overfitting
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                                     
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

        print("Loss function and optimizer set up successfully.")
    except Exception as e:
        print(f"Error setting up loss function and optimizer: {e}")

    highest_val_acc = 0.0
    model_save_path = "most_precise_cnn.pth"

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
            val_loss, val_accuracy = validate_one_epoch(model, val_dataloader, loss_fn, device)

            print(f"Epoch {epoch + 1} completed: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            scheduler.step(val_accuracy)

            if val_accuracy > highest_val_acc:
                highest_val_acc = val_accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved with accuracy: {highest_val_acc:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")

    try:
        test_model(model, test_dataloader, loss_fn, device)
    except Exception as e:
        print(f"Error during testing: {e}")

    print("Training completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN for Music Genre Classification")

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=CNN.BATCH_SIZE, help='Batch size for training and validation')

    args = parser.parse_args()

    main(args)