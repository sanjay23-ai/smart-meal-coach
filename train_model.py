"""
Simple training script for the food classification model.

Usage:
    python train_model.py --data_dir data/images --epochs 10 --batch_size 32

This script will:
1. Load images from data/images/<class_name>/*.jpg
2. Split into train/val sets
3. Fine-tune ResNet18
4. Save the trained model as models/food_classifier.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.vision_model import (
    build_model,
    create_dataloaders,
    FoodImageDataset,
)
from src.data_utils import DATA_DIR


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train food classification model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images",
        help="Directory containing food images organized by class",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (increased for 90% accuracy)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate (lower for fine-tuning)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the trained model",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        print("Please organize your images as: data/images/<class_name>/*.jpg")
        return

    # Create dataloaders with proper train/val split
    print("Loading dataset...")
    
    # Split dataset into train/val
    from src.data_utils import example_train_val_split
    from src.vision_model import FoodImageDataset
    from torchvision import transforms
    
    # Load all image paths
    all_samples = []
    class_to_idx = {}
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir = data_dir / cls_name
        for img_path in cls_dir.glob("*.jpg"):
            all_samples.append((img_path, class_to_idx[cls_name]))
        for img_path in cls_dir.glob("*.png"):
            all_samples.append((img_path, class_to_idx[cls_name]))
    
    if len(all_samples) == 0:
        print(f"Error: No images found in {data_dir}")
        return
    
    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"Found {len(train_samples)} training images, {len(val_samples)} validation images")
    
    # Create transforms with more augmentation for better accuracy
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create custom datasets
    class SplitDataset(FoodImageDataset):
        def __init__(self, samples, class_to_idx, transform=None):
            self.root = data_dir
            self.transform = transform
            self.samples = samples
            self.class_to_idx = class_to_idx
    
    train_ds = SplitDataset(train_samples, class_to_idx, transform=train_transform)
    val_ds = SplitDataset(val_samples, class_to_idx, transform=val_transform)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    num_classes = len(idx_to_class)
    print(f"Found {num_classes} classes: {list(idx_to_class.values())[:10]}...")

    # Build model (using ResNet50 for better accuracy)
    print("Building model (ResNet50 for 90%+ accuracy)...")
    model = build_model(num_classes=num_classes, use_resnet50=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    
    # Use learning rate scheduler for better convergence
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop with early stopping for 90% accuracy
    print(f"\nStarting training for {args.epochs} epochs...")
    print("Target: 90%+ validation accuracy")
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10  # Stop if no improvement for 10 epochs

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = output_dir / "food_classifier.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Saved best model (val acc: {val_acc:.2f}%) to {model_path}")
            
            # Early stopping if we reach 90% accuracy
            if val_acc >= 90.0:
                print(f"\n🎉 Target accuracy reached! Stopping early at {val_acc:.2f}%")
                break
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n⏸️  Early stopping: No improvement for {max_patience} epochs")
                break

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir / 'food_classifier.pt'}")
    print(f"\nClass mapping saved to: {output_dir / 'class_mapping.txt'}")
    
    # Save class mapping
    with open(output_dir / "class_mapping.txt", "w") as f:
        for idx, class_name in sorted(idx_to_class.items()):
            f.write(f"{idx}: {class_name}\n")


if __name__ == "__main__":
    main()

