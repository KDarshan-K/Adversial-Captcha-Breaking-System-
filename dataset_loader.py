import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

# Import shared settings from config.py
from config import CHAR_SET, CAPTCHA_LENGTH


class CaptchaDataset(Dataset):
    """
    Custom Dataset for loading captcha images from a *directory path*.
    Assumes the filename (before .png) is the label.
    e.g., "2b827.png" has label "2b827"
    """

    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset from a *directory path*.

        Args:
            data_dir (str): The path to the folder containing the images.
                            (e.g., "C:/Users/YourUser/Desktop/MyCaptchas")
            transform (callable, optional): A torchvision transform.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Find all .png files in the directory
        search_path = os.path.join(data_dir, "*.png")
        self.image_paths = glob.glob(search_path)

        if not self.image_paths:
            print(f"Warning: No .png files found in directory: {data_dir}")
            print(f"Searched for pattern: {search_path}")
            print("Please check the path. Example path: 'C:/Users/YourUser/Desktop/MyCaptchas'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Extract label from filename
        # e.g., "C:/Users/YourUser/Desktop/MyCaptchas/2b827.png" -> "2b827"
        filename = os.path.basename(image_path)
        label_str = os.path.splitext(filename)[0]

        # Load image
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
        except FileNotFoundError:
            print(f"Error: File not found at {image_path}. Skipping.")
            # Return dummy data to avoid a crash
            return torch.zeros(1, 64, 128), torch.zeros(CAPTCHA_LENGTH, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        # Convert text label to tensor of indices
        try:
            label = torch.tensor([CHAR_SET.find(char) for char in label_str], dtype=torch.long)

            # Check for invalid characters
            if -1 in label:
                print(f"Warning: Filename '{filename}' contains characters not in CHAR_SET. Skipping.")
                return image, torch.zeros(CAPTCHA_LENGTH, dtype=torch.long)

            # Ensure label is the correct length
            if len(label) != CAPTCHA_LENGTH:
                print(f"Warning: Filename '{filename}' has length {len(label)}, expected {CAPTCHA_LENGTH}. Skipping.")
                return image, torch.zeros(CAPTCHA_LENGTH, dtype=torch.long)

        except Exception as e:
            print(f"Error processing label for {filename}: {e}")
            return image, torch.zeros(CAPTCHA_LENGTH, dtype=torch.long)

        return image, label

