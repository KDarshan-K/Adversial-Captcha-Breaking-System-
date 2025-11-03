import string
import os

# --- Dataset Config ---
CAPTCHA_LENGTH = 5
CHAR_SET = string.digits + string.ascii_lowercase # "0123456789abcdefghijklmnopqrstuvwxyz"
N_CHARS = len(CHAR_SET)

# --- Image Config ---
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64

# --- Directory Config ---
BASE_DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
VAL_DIR = os.path.join(BASE_DATA_DIR, "val")
TEST_DIR = os.path.join(BASE_DATA_DIR, "test")

# --- Model & Training Config ---
MODEL_PATH = "captcha_model.pth"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# --- Attack Config ---
ATTACK_EPSILON = 0.1 # How "strong" the attack should be
