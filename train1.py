import os
import string
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import CRNN
from dataset import CaptchaImagesDataset
from utils import LabelConverter

# ---------------------------
# Settings
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char_set = string.ascii_lowercase + string.digits
label_converter = LabelConverter(char_set)
vocab_size = label_converter.get_vocab_size()

# Paths
train_folder = "A:/Datasets/samples/"
val_folder = ""
output_path = "output/weight.pth"

# Hyperparameters
batch_size = 32
num_epochs = 30
learning_rate = 0.001

# ---------------------------
# Datasets & Loaders
# ---------------------------
train_dataset = CaptchaImagesDataset(train_folder, label_converter)
val_dataset = CaptchaImagesDataset(val_folder, label_converter)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# Model, Loss, Optimizer
# ---------------------------
model = CRNN(vocab_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)  # reduces LR on plateau

# ---------------------------
# Training Loop
# ---------------------------
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        images, labels = batch  # unpack tuple
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # shape: [batch, seq_len, vocab_size]

        # Flatten outputs and labels for CrossEntropyLoss
        outputs_flat = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # ---------------------------
    # Validation
    # ---------------------------
