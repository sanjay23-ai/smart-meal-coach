"""
Food classification model using transfer learning.

You can start with a lightweight pretrained model (e.g., ResNet18) and fine-tune it
on a food image dataset like Food-101 or your own curated subset.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class FoodImageDataset(Dataset):
    """
    A minimal PyTorch dataset for food images arranged as:
        root/class_name/*.jpg
    """

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_to_idx: Dict[str, int] = {}

        self._index_dataset()

    def _index_dataset(self) -> None:
        classes = sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()],
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_dir = self.root / cls_name
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Create PyTorch dataloaders for training and validation.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_ds = FoodImageDataset(train_dir, transform=train_transform)
    val_ds = FoodImageDataset(val_dir, transform=val_transform)

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, idx_to_class


def build_model(num_classes: int, use_resnet50: bool = True) -> nn.Module:
    """
    Build a transfer-learning model using ResNet50 (or ResNet18) as a starting point.
    ResNet50 provides better accuracy for food classification.
    """
    if use_resnet50:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def predict_food(
    model: nn.Module,
    image: Image.Image,
    idx_to_class: Dict[int, str],
) -> Tuple[str, float]:
    """
    Run inference on a single image and return (predicted_class, confidence).
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    model.eval()
    with torch.no_grad():
        x = transform(image).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        return idx_to_class[int(pred_idx)], float(confidence.item())


if __name__ == "__main__":
    # This main block is for quick sanity checks, not full training.
    print("vision_model module loaded. Implement training loop in a notebook or script.")


