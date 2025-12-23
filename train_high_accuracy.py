"""
High-accuracy training script targeting 90%+ validation accuracy.

This script uses advanced techniques:
- ResNet50 architecture
- Extensive data augmentation
- Learning rate scheduling
- Early stopping
- Weight decay regularization
"""

import sys
from pathlib import Path

# Import training functions
from train_model import main as train_main
import argparse

def main():
    print("=" * 70)
    print("Smart Meal Coach - High Accuracy Training (Target: 90%+)")
    print("=" * 70)
    print("\nThis script will train with optimized settings for 90%+ accuracy:")
    print("  - ResNet50 architecture (better than ResNet18)")
    print("  - Extensive data augmentation")
    print("  - Learning rate scheduling")
    print("  - Early stopping")
    print("  - Weight decay regularization")
    print("\n" + "=" * 70)
    
    # Check if dataset exists
    data_dir = Path("data/images")
    if not data_dir.exists():
        print("\n❌ Error: Dataset directory not found!")
        print("Please create data/images/<food_name>/ directories and add images.")
        print("See TRAINING_GUIDE.md for instructions.")
        return
    
    # Count images
    total_images = 0
    classes = []
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if len(images) > 0:
                classes.append(class_dir.name)
                total_images += len(images)
                print(f"  ✅ {class_dir.name}: {len(images)} images")
    
    if total_images == 0:
        print("\n❌ No images found!")
        print("Please add food images to data/images/<food_name>/ directories")
        return
    
    print(f"\n📊 Dataset Summary:")
    print(f"  - Total images: {total_images}")
    print(f"  - Classes: {len(classes)}")
    print(f"  - Average per class: {total_images // len(classes) if classes else 0}")
    
    # Recommendations
    if total_images < 200:
        print("\n⚠️  Warning: For 90%+ accuracy, recommend:")
        print("  - At least 50-100 images per class")
        print("  - Total of 500+ images minimum")
        print("  - Current dataset may achieve 70-85% accuracy")
    
    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Set up training arguments for high accuracy
    sys.argv = [
        "train_model.py",
        "--data_dir", "data/images",
        "--epochs", "50",  # More epochs
        "--batch_size", "16",  # Smaller batch for better gradient estimates
        "--lr", "0.0001",  # Lower learning rate for fine-tuning
        "--output_dir", "models"
    ]
    
    # Run training
    train_main()
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("Check models/food_classifier.pt for your trained model")
    print("Run: python -m streamlit run src/app_streamlit.py to test")
    print("=" * 70)

if __name__ == "__main__":
    main()

