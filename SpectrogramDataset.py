import torch
from PIL import Image
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    # Contains all methods required for a PyTorch Dataset

    # Set up the dataset with data & transforms
    def __init__(self, data, class_to_index, transform, model_type):
        self.data_list = data # List (image_path, label)
        self.class_to_index = class_to_index # Dictionary mapping class names to indexes
        self.transform = transform # Transform to apply to the images
        self.model_type = model_type # Type of model (e.g., 'cnn', 'mlp')

    # Get total items in the dataset
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        # Unpack all the raw data for the given index
        img_path, numeric_features, class_name = self.data_list[index]

        # Always process the label and numeric features
        label = self.class_to_index[class_name]
        numeric_features_tensor = torch.tensor(numeric_features, dtype=torch.float32)

        # --- Use the model_type flag to decide what to return ---
        if self.model_type == 'mlp':
            # For the MLP experiment, we only need the features and the label
            return numeric_features_tensor, label
        