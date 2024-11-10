import torch
from torch import nn
from torch.nn import functional as F

# Accuracy 84.62%
class CustomCNN(nn.Module):
    def __init__(self):
        """
        Initializes the CustomCNN model with the following architecture:

        - Four convolutional blocks (block1 to block4), each consisting of:
        - Convolutional layers with Batch Normalization and Leaky ReLU activation.
        - Max Pooling and Dropout for regularization.

        - Fully connected layers for classification, with Batch Normalization applied to the first two layers.

        The model is designed to process input images with three channels (e.g., RGB) and output ten class scores.
        """
        super(CustomCNN, self).__init__()

        # First block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        # Second block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        # Third block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        # Fourth block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Pass through the blocks
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input to the network, of shape `(batch_size, channels, height, width)`.

        Returns
        -------
        torch.Tensor
            Output of the network, of shape `(batch_size, 10)`.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Flatten before passing to fully connected layers
        x = x.view(-1, 512 * 2 * 2)

        # Fully connected layers with BatchNorm and Dropout
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)), 0.1)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)), 0.1)
        x = F.dropout(x, 0.5)
        x = self.fc3(x)

        return x