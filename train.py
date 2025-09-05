# train.py
import os
import string
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from model import CRNN  # your CRNN model
from utils import LabelConverter, write_log, write_figure  # your utils


# -----------------------------
# Dataset Class
# -----------------------------
class CaptchaImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Load image paths and labels
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.splitext(file)[0]  # filename without extension
                self.data.append((os.path.join(root_dir, file), label))

        if len(self.data) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]

        try:
            image = Image.open(image_path).convert('RGB')  # ensure 3 channels
        except Exception as e:
            print(f"Skipping corrupted image: {image_path}, error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for {image_path}, error: {e}")
                return self.__getitem__((idx + 1) % len(self))

        return image, label


# -----------------------------
# Data Loader Helper
# -----------------------------
def get_loader(root_dir, batch_size=16, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((50, 200)),  # match CRNN input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CaptchaImagesDataset(root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return loader


# -----------------------------
# Loss Function
# -----------------------------
def calculate_loss(inputs, texts, label_converter, device):
    criterion = nn.CTCLoss(blank=0)

    inputs = inputs.log_softmax(2)
    input_size, batch_size, _ = inputs.size()
    input_size = torch.full(size=(batch_size,), fill_value=input_size, dtype=torch.int32)

    encoded_texts, text_lens = label_converter.encode(texts)
    loss = criterion(inputs, encoded_texts.to(device),
                     input_size.to(device),
                     text_lens.to(device))
    return loss


# -----------------------------
# Training Step
# -----------------------------
def fit(epoch, model, optimizer, label_converter, device, data_loader, phase='training'):
    model.train() if phase == 'training' else model.eval()
    running_loss = 0

    for images, labels in tqdm(data_loader, desc=f"{phase} epoch {epoch}"):
        images = images.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            outputs = model(images)
        else:
            with torch.no_grad():
                outputs = model(images)

        loss = calculate_loss(outputs, labels, label_converter, device)
        running_loss += loss.item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader)
    print(f"[{epoch}][{phase}] loss: {epoch_loss:.4f}")
    return epoch_loss


# -----------------------------
# Main Training Loop
# -----------------------------
def train():
    print("Starting training...")

    batch_size = 16
    num_epochs = 30
    learning_rate = 0.1

    label_converter = LabelConverter(char_set=string.ascii_lowercase + string.digits)
    vocab_size = label_converter.get_vocab_size()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CRNN(vocab_size=vocab_size).to(device)

    train_loader = get_loader('A:/Datasets/samples/train', batch_size=batch_size)
    val_loader = get_loader('A:/Datasets/samples/val', batch_size=batch_size, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = fit(epoch, model, optimizer, label_converter, device, train_loader, phase='training')
        val_loss = fit(epoch, model, optimizer, label_converter, device, val_loader, phase='validation')

        if epoch == 0 or val_loss <= np.min(val_losses):
            os.makedirs('output', exist_ok=True)
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        write_figure('output', train_losses, val_losses)
        write_log('output', epoch, train_loss, val_loss)

        scheduler.step(val_loss)


if __name__ == "__main__":
    train()
