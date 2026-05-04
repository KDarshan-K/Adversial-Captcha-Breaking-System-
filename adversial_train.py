import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import os
import sys
import time
import traceback

# Import our custom modules
from config import (
    TRAIN_DIR, VAL_DIR, ROBUST_MODEL_PATH, MODEL_PATH, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHARS,
    ADV_TRAIN_EPSILON, CAPTCHA_LENGTH
)
# --- THIS IMPORT IS NOW FIXED ---
from dataset_loader import CaptchaDataset
from model import CaptchaCNN


# --- --- --- --- --- --- --- --- --- --- ---
# --- 1. COPY YOUR FGSM ATTACK FUNCTION ---
# --- --- --- --- --- --- --- --- --- --- ---
def fgsm_attack(model, images, labels, epsilon, device):
    """
    FGSM (Fast Gradient Sign Method) attack.
    (Copied from your evaluate_attacks.py)
    """
    images = images.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)

    # Put model in eval mode for the attack
    model.eval()

    # Forward pass
    outputs = model(images)  # (batch, 5, 36)

    # Calculate loss for all 5 characters
    loss = 0
    for char_idx in range(CAPTCHA_LENGTH):
        loss += nn.functional.cross_entropy(
            outputs[:, char_idx, :],  # (batch, 36)
            labels[:, char_idx]  # (batch,)
        )

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Create adversarial examples
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    # Put model back in train mode
    model.train()

    return perturbed_images.detach()


# --- --- --- --- --- --- --- --- --- --- ---
# --- 2. MODIFIED TRAINING FUNCTION ---
# --- --- --- --- --- --- --- --- --- --- ---
def train_adversarial_model(model, train_loader, val_loader, epochs, lr, device):
    print("--- Starting Adversarial Training ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Add the Learning Rate Scheduler (this helped last time)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
        # verbose=True  <- This line was removed as it's not supported in your PyTorch version
    )

    best_val_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        model.train()  # Set model to training mode
        total_train_loss = 0
        correct_train_chars = 0
        total_train_chars = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # --- --- --- --- --- --- --- --- ---
            # --- 1. GENERATE ADVERSARIAL IMAGES ---
            adv_images = fgsm_attack(model, images, labels, ADV_TRAIN_EPSILON, device)

            # --- 2. COMBINE BATCHES ---
            # We will train on *both* clean and adversarial images
            combined_images = torch.cat([images, adv_images], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            # --- 3. STANDARD TRAINING STEP ---
            optimizer.zero_grad()

            # Forward pass on the *combined* batch
            outputs = model(combined_images)

            # Reshape for loss calculation
            outputs_reshaped = outputs.view(-1, N_CHARS)
            labels_reshaped = combined_labels.view(-1)

            loss = criterion(outputs_reshaped, labels_reshaped)

            # Backward pass
            loss.backward()
            optimizer.step()
            # --- --- --- --- --- --- --- --- ---

            total_train_loss += loss.item()

            # Calculate training accuracy (optional, but good to see)
            _, predicted = torch.max(outputs, 2)
            correct_train_chars += (predicted == combined_labels).sum().item()
            total_train_chars += combined_labels.numel()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = (correct_train_chars / total_train_chars) * 100

        # --- Validation Phase (No change here) ---
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        correct_val_chars = 0
        total_val_chars = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                # --- THIS TYPO IS NOW FIXED ---
                outputs_reshaped = outputs.view(-1, N_CHARS)
                labels_reshaped = labels.view(-1)

                loss = criterion(outputs_reshaped, labels_reshaped)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 2)
                correct_val_chars += (predicted == labels).sum().item()
                total_val_chars += labels.numel()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = (correct_val_chars / total_val_chars) * 100
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Time: {epoch_time:.0f}s, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Save the *best* robust model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), ROBUST_MODEL_PATH)
            print(f"New best *robust* model saved to {ROBUST_MODEL_PATH} with Val Acc: {best_val_acc:.2f}%")

    print("--- Adversarial Training Finished ---")


# --- --- --- --- --- --- --- --- --- --- ---
# --- 3. MAIN FUNCTION (Loads base model) ---
# --- --- --- --- --- --- --- --- --- --- ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    # We use augmented transforms for training
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomAffine(degrees=10, shear=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        print(f"Error: Train/Val directory not found.")
        print(f"Please check your 'DATASET_BASE_DIR' in config.py")
        return

    print(f"Loading training data from: {TRAIN_DIR}")
    train_dataset = CaptchaDataset(data_dir=TRAIN_DIR, transform=train_transform)

    print(f"Loading validation data from: {VAL_DIR}")
    val_dataset = CaptchaDataset(data_dir=VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data loaded successfully.")
    print(f"Training set:   {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")

    # --- Load MODEL ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Base model '{MODEL_PATH}' not found!")
        print("Please run train.py to create the base model first.")
        return

    print(f"Loading *base* model from {MODEL_PATH} to continue training...")
    model = CaptchaCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # --- Train ---
    # We train for *fewer* epochs, as adversarial training is slow and powerful
    adv_train_epochs = 30

    # --- THIS BLOCK IS NOW FIXED (Removed 'S'more,') ---
    train_adversarial_model(
        model,
        train_loader,
        val_loader,
        epochs=adv_train_epochs,
        lr=LEARNING_RATE / 10,  # Use a *smaller* learning rate for fine-tuning
        device=device
    )

    print(f"--- Adversarial Training complete. Best robust model saved to {ROBUST_MODEL_PATH} ---")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print("\n--- A CRITICAL ERROR OCCURRED ---")
        print(f"ERROR: {e}")
        print("\n--- FULL TRACEBACK ---")
        traceback.print_exc()
    finally:
        try:
            sys.exit(0)
        except SystemExit:
            pass

