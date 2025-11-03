import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import os
import sys

# Import our custom modules
from config import (
    MODEL_PATH, BATCH_SIZE, N_CHARS, IMAGE_HEIGHT, IMAGE_WIDTH, TEST_DIR
)
from dataset_loader import CaptchaDataset
from model import CaptchaCNN


def test_model():
    """
    Loads the best trained model and evaluates it on the
    "sacred" test set.
    """

    # 1. Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train.py to train and save the model first.")
        return

    # 2. Check if the test directory exists
    if not os.path.isdir(TEST_DIR):
        print(f"Error: Test directory not found at {TEST_DIR}")
        print(f"Please check your 'TEST_DIR' path in config.py")
        print(f"Looking for: {os.path.abspath(TEST_DIR)}")
        return

    print(f"--- Loading final model from {MODEL_PATH} ---")

    # 3. Define the *non-augmented* transform for testing
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    # 4. Load the test dataset
    print(f"Loading test data from: {TEST_DIR}")
    test_dataset = CaptchaDataset(data_dir=TEST_DIR, transform=test_transform)

    if len(test_dataset) == 0:
        print("No images found in the test directory.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Load the model architecture and its saved weights
    model = CaptchaCNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set to evaluation mode (turns off dropout)

    # 6. Run the evaluation
    criterion = nn.CrossEntropyLoss()
    total_test_loss = 0
    correct_test_chars = 0
    total_test_chars = 0

    with torch.no_grad():  # We don't need gradients for testing
        for images, labels in test_loader:
            outputs = model(images)

            outputs_reshaped = outputs.view(-1, N_CHARS)
            labels_reshaped = labels.view(-1)

            loss = criterion(outputs_reshaped, labels_reshaped)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 2)
            correct_test_chars += (predicted == labels).sum().item()
            total_test_chars += labels.numel()

    # 7. Report the final, official score
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = (correct_test_chars / total_test_chars) * 100

    print("\n" + "=" * 40)
    print("      FINAL BASELINE MODEL SCORE      ")
    print("=" * 40)
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Accuracy (per-char): {test_accuracy:.2f}%")
    print("=" * 40)
    print("\nThis is the 'baseline accuracy' you will use to")
    print("compare against your 'robust' model later.")


if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # A simple way to ensure the script exits cleanly in all cases
        sys.exit(0)

