import torch
import torch.nn as nn

class MLP(nn.Module):
    BATCH_SIZE = 16 

    def __init__(self, num_genres, num_features):
        super(MLP, self).__init__()

        # A simple but effective Multi-Layer Perceptron (MLP)
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.6),

            # Second hidden layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.6),

            # Output layer
            nn.Linear(128, num_genres)
        )

    def forward(self, x):
        return self.network(x)