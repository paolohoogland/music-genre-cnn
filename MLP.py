import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, size, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection
    

class MLP(nn.Module):
    def __init__(self, num_genres, num_features):
        super(MLP, self).__init__()

        # Multi-Layer Perceptron (MLP)
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            ResidualBlock(512, dropout=0.5),
            # ResidualBlock(512, dropout=0.4),

            # Second hidden layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            ResidualBlock(256, dropout=0.3),

            # Third hidden layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            # Fourth hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(64, num_genres)
        )

    def forward(self, x):
        return self.network(x)