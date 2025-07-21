from torchvision import transforms
from torch.utils.data import DataLoader

from DatasetManager import DatasetManager
from SpectrogramDataset import SpectrogramDataset
from CNN import CNN

def main():
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
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((CNN.IMG_HEIGHT, CNN.IMG_WIDTH)),
            transforms.ToTensor()
        ])
        print("Data transform created successfully.")
    except Exception as e:
        print(f"Error creating data transform: {e}")

    try:
        train_dataset = SpectrogramDataset(train_set, genre_to_index, data_transform)
        val_dataset = SpectrogramDataset(val_set, genre_to_index, data_transform)
        # test_dataset = SpectrogramDataset(test_set, genre_to_index, data_transform)

        print("Spectrogram datasets created successfully.")
    except Exception as e:
        print(f"Error creating spectrogram datasets: {e}")

    try:
        train_dataloader = DataLoader(train_dataset, batch_size=CNN.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=CNN.BATCH_SIZE, shuffle=False)
        # test_dataloader = DataLoader(test_dataset, batch_size=CNN.BATCH_SIZE, shuffle=False)
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

if __name__ == "__main__":
    main()