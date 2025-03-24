import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BirdImageDataset(Dataset):
    def __init__(self, root_dir, image_size=256, augment=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.image_paths = []
        self.labels = []

        self._load_images()
        self.transform = self._build_transform()

    def _load_images(self):
        """Load images while filtering based on size & aspect ratio."""
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(("jpg", "jpeg", "png")):
                    img_path = os.path.join(root, file)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            aspect_ratio = width / height
                            if 0.65 <= aspect_ratio <= 1.6 and min(width, height) >= 256:
                                self.image_paths.append(img_path)
                                self.labels.append(self._extract_label(root))
                    except Exception as e:
                        print(f"Skipping {img_path}: {e}")

    def _extract_label(self, path):
        folder_name = os.path.basename(path)
        try:
            label = int(folder_name.split('.')[0])
        except ValueError:
            raise ValueError(f"Invalid folder format: {folder_name}. Expected 'XXX.ClassName'.")
        return label

    def _build_transform(self):
        base_transforms = [
            transforms.CenterCrop(self.image_size),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        if self.augment:
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),  # Flip 50% of images
                transforms.ColorJitter(brightness=0.08, contrast=0.05, saturation=0.1, hue=0.05)
            ]
            return transforms.Compose(aug_transforms + base_transforms)
        else:
            return transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return img, label
