import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetModel(nn.Module):
    IMG_HEIGHT = 128
    IMG_WIDTH = 256
    BATCH_SIZE = 16 

    def __init__(self, num_genres=10, freeze_features=True):
        super(ResNetModel, self).__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.resnet = resnet50(weights=weights)

        # Freeze the feature extraction layers
        # This is so that only the classifier layers are trained
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace the final classifier layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_genres)
        )

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        return x