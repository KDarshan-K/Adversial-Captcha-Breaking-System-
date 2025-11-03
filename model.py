import torch.nn as nn
import torch.nn.functional as F

# Import shared settings from config.py
from config import IMAGE_HEIGHT, IMAGE_WIDTH, CAPTCHA_LENGTH, N_CHARS


class CaptchaCNN(nn.Module):
    """
    A simple Convolutional Neural Network for solving the captcha.

    The model takes an image (1, 64, 128) and outputs a tensor
    (5, 36) representing the scores for each of the 5 characters
    across all 36 possible classes.
    """

    def __init__(self):
        super(CaptchaCNN, self).__init__()

        # --- Convolutional Layers ---
        # Input: (Batch, 1, 64, 128)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # (Batch, 16, 64, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (Batch, 16, 32, 64)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # (Batch, 32, 32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (Batch, 32, 16, 32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # (Batch, 64, 16, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (Batch, 64, 8, 16)

        # --- Fully Connected Layers ---

        # Calculate the flattened size after conv/pool layers
        # (64 channels) * (8 height) * (16 width)
        self.fc_input_size = 64 * 8 * 16

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout = nn.Dropout(0.6)

        # --- Output Layer ---
        # We need to predict CAPTCHA_LENGTH (5) characters,
        # and each character can be one of N_CHARS (36) classes.
        # So, the output size is 5 * 36
        self.fc_output_size = CAPTCHA_LENGTH * N_CHARS
        self.fc2 = nn.Linear(512, self.fc_output_size)

    def forward(self, x):
        # Pass through conv layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output for the FC layers
        # x.size(0) is the batch size
        x = x.view(x.size(0), -1)  # Flattens to (Batch, fc_input_size)

        # Pass through FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Reshape the final output
        # From (Batch, 5 * 36) to (Batch, 5, 36)
        # This gives us a separate output vector for each of the 5 characters
        x = x.view(-1, CAPTCHA_LENGTH, N_CHARS)

        return x

