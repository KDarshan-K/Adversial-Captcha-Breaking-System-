import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as transforms
import os

# Import our custom modules
from config import (
    MODEL_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    N_CHARS, IMAGE_HEIGHT, IMAGE_WIDTH
)
from dataset_loader import CaptchaDataset
from model import CaptchaCNN


def train_model(model, train_loader, val_loader, epochs, lr):
    """
    Training and validation loop.
    (This function remains the same as it just needs the data loaders)
    """
    print("--- Starting Training ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()  # Set model to training mode
        total_train_loss = 0
        correct_train_chars = 0
        total_train_chars = 0

        for i, (images, labels) in enumerate(train_loader):
            # images: [Batch, 1, H, W]
            # labels: [Batch, CaptchaLength]

            optimizer.zero_grad()
            outputs = model(images)  # Output: [Batch, CaptchaLength, N_Chars]

            # Reshape for loss calculation
            # [Batch * CaptchaLength, N_Chars]
            outputs_reshaped = outputs.view(-1, N_CHARS)
            # [Batch * CaptchaLength]
            labels_reshaped = labels.view(-1)

            loss = criterion(outputs_reshaped, labels_reshaped)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 2)  # [Batch, CaptchaLength]
            correct_train_chars += (predicted == labels).sum().item()
            total_train_chars += labels.numel()  # total chars in batch

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = (correct_train_chars / total_train_chars) * 100

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        correct_val_chars = 0
        total_val_chars = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)

                outputs_reshaped = outputs.view(-1, N_CHARS)
                labels_reshaped = labels.view(-1)

                loss = criterion(outputs_reshaped, labels_reshaped)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 2)
                correct_val_chars += (predicted == labels).sum().item()
                total_val_chars += labels.numel()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = (correct_val_chars / total_val_chars) * 100

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Save the best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved with Val Acc: {best_val_acc:.2f}%")

    print("--- Training Finished ---")


# In your train.py

def main():
    # 1. !!! DEFINE TWO TRANSFORMS !!!

    # Transform for TRAINING data (with augmentations)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomAffine(degrees=10, shear=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    # Transform for VALIDATION & TEST data (no augmentations)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    # 2. Define your dataset folder paths
    DATASET_BASE_DIR = "A:\Datasets\captcha"  # Change this
    train_dir = os.path.join(DATASET_BASE_DIR, "train")
    val_dir = os.path.join(DATASET_BASE_DIR, "val")
    test_dir = os.path.join(DATASET_BASE_DIR, "test")

    # ... (add your path checking 'if not os.path.isdir...') ...

    # 3. !!! APPLY THE CORRECT TRANSFORM !!!
    print(f"Loading training data from: {train_dir}")
    train_dataset = CaptchaDataset(
        data_dir=train_dir,
        transform=train_transform  # <--- Use train_transform
    )

    print(f"Loading validation data from: {val_dir}")
    val_dataset = CaptchaDataset(
        data_dir=val_dir,
        transform=val_transform  # <--- Use val_transform
    )

    # ... (the rest of your code: create DataLoaders, train, etc.) ...

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data loaded successfully.")
    print(f"Training set:   {len(train_dataset)} images (with augmentation)")
    print(f"Validation set: {len(val_dataset)} images (no augmentation)")

    # 5. Create and train the model
    model = CaptchaCNN()
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    print(f"--- Training complete. Best model saved to {MODEL_PATH} ---")


if __name__ == "__main__":
    main()