import torch
import torch.nn as nn

class CNN(nn.Module):
    IMG_HEIGHT = 128
    IMG_WIDTH = 256
    BATCH_SIZE = 32 

    def __init__(self, num_genres=10):
        super(CNN, self).__init__()

        # Three convolutional layers
        # Input channels: 1 (grayscale), Output channels: 32, 64, 128 respectively
        # All three inside the self.features block for adaptability
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), # Activation function, introduces non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsampling the feature maps by 2
        
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dummy input
        # If we add another convolutional layer, it will still work
        dummy_input = torch.randn(1, 1, self.IMG_HEIGHT, self.IMG_WIDTH) # (Batch, Channels, H, W)
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.numel() # Get the number of elements 

        # Flatten layer
        self.flatten = nn.Flatten() # Converts the 2D feature maps into a 1D vector

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=flattened_size, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Dropout for regularization
            nn.Linear(in_features=256, out_features=num_genres)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        res = self.classifier(x)
        return res