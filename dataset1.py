# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CaptchaImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Gather all image paths
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.data.append((os.path.join(root_dir, fname), os.path.splitext(fname)[0]))

        if len(self.data) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.data)

    from PIL import Image

    def __getitem__(self, idx):
        image_path, label = self.data[idx]

        try:
            image = Image.open(image_path).convert('RGB')  # force 3 channels
        except Exception as e:
            print(f"Skipping corrupted image: {image_path}, error: {e}")
            return self.__getitem__((idx + 1) % len(self))  # try next image

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for {image_path}, error: {e}")
                return self.__getitem__((idx + 1) % len(self))

        return image, label


def get_loader(root_dir, batch_size=16, shuffle=True, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CaptchaImagesDataset(root_dir=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, loader  # same loader for train/val; replace with separate folders if needed
