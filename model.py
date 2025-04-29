# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A simple CNN for CIFAR-10 compatible with CPU training."""

    def __init__(self):
        super().__init__()
        # Input shape: (Batch, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Output: (Batch, 16, 32, 32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (Batch, 16, 16, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output: (Batch, 32, 16, 16)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (Batch, 32, 8, 8)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (Batch, 64, 8, 8)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (Batch, 64, 4, 4)

        # Flatten the output for the fully connected layers
        # 64 channels * 4 width * 4 height = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer (logits)
        return x


if __name__ == '__main__':
    # Quick test of the model architecture
    model = SimpleCNN()
    print(model)
    # Test with a dummy input batch (Batch=4, Channels=3, Height=32, Width=32)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (4, 10)
