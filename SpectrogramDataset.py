import torch
from PIL import Image
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    # Contains all methods required for a PyTorch Dataset

    # Set up the dataset with data & transforms
    def __init__(self, data, class_to_index, transform=None):
        self.data_list = data # List (image_path, label)
        self.class_to_index = class_to_index # Dictionary mapping class names to indexes
        self.transform = transform # Transform to apply to the images

    # Get total items in the dataset
    def __len__(self):
        return len(self.data_list)
    
    # Define how to get an item from the dataset
    def __getitem__(self, index):
        img_path, numeric_features, class_name = self.data_list[index]

        image = Image.open(img_path)
        label = self.class_to_index[class_name]

        if self.transform:
            image = self.transform(image) # Apply any transformations (like normalization, resizing)

        # Convert numeric features to a tensor
        numeric_features_tensor = torch.tensor(numeric_features, dtype=torch.float32)

        return image, numeric_features_tensor, label
