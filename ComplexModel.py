import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ComplexModel(nn.Module):
    IMG_HEIGHT = 128
    IMG_WIDTH = 256
    BATCH_SIZE = 16 

    def __init__(self, num_genres, num_features, freeze_features=True):
        super(ComplexModel, self).__init__()

        # IMAGENET1K_V2 is a set of pre-trained weights for ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.image_model = resnet50(weights=weights)

        if freeze_features:
            for param in self.image_model.parameters():
                param.requires_grad = False

        # Get the number of output features from the ResNet model feature extraction layers
        nb_features = self.image_model.fc.in_features

        # Remove the final classifier layer
        self.image_model.fc = nn.Identity()  

        self.numeric_model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.6)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(nb_features + 64, 256),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(256, num_genres)
        )

    def forward(self, image_data, numeric_data):
        image_features = self.image_model(image_data)
        numeric_features = self.numeric_model(numeric_data)

        # Concatenate the features from both modalities
        combined_features = torch.cat((image_features, numeric_features), dim=1)

        # Pass the combined features through the fusion layer
        output = self.fusion_layer(combined_features)
        return output
