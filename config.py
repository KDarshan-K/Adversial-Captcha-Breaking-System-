# config.py - Centralized Configuration

import os

# --- Dataset Settings ---
DATASET_BASE_DIR = r"A:\Datasets\captcha"  # Change this to your dataset location
TRAIN_DIR = os.path.join(DATASET_BASE_DIR, "train")
VAL_DIR = os.path.join(DATASET_BASE_DIR, "val")
TEST_DIR = os.path.join(DATASET_BASE_DIR, "test")

# --- Image Settings ---
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 1  # Grayscale

# --- CAPTCHA Settings ---
CAPTCHA_LENGTH = 5  # Number of characters in each CAPTCHA
CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyz"  # All possible characters
N_CHARS = len(CHAR_SET)  # 36 classes (0-9, A-Z)

# --- Training Settings ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50

# --- Model Paths ---
MODEL_PATH = "captcha_model.pth"
ROBUST_MODEL_PATH = "captcha_model_robust.pth"  # NEW: Path for adversarially trained model

# --- Adversarial Training Settings ---
ADV_TRAIN_EPSILON = 0.1  # Epsilon value for FGSM during adversarial training
ADV_TRAIN_ALPHA = 0.5    # Mix ratio: 0.5 means 50% clean + 50% adversarial images
