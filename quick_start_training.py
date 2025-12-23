"""
Quick Start Training Helper

This script helps you get started with training by:
1. Checking if you have images
2. Providing options to get images quickly
3. Starting training when ready
"""

import os
import sys
from pathlib import Path

def check_images():
    """Check how many images we have."""
    data_dir = Path("data/images")
    if not data_dir.exists():
        return 0, []
    
    total = 0
    classes_with_images = []
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            count = len(images)
            total += count
            if count > 0:
                classes_with_images.append((class_dir.name, count))
    
    return total, classes_with_images

def main():
    print("=" * 70)
    print("Smart Meal Coach - Training Setup")
    print("=" * 70)
    
    total_images, classes = check_images()
    
    if total_images == 0:
        print("\n[STATUS] No food images found!")
        print("\nTo train the model, you need food images.")
        print("\nQUICK OPTIONS:")
        print("\n1. Download Food-101 from Kaggle (BEST):")
        print("   https://www.kaggle.com/datasets/dansbecker/food-101")
        print("   - Sign up (free)")
        print("   - Download dataset")
        print("   - Copy images to data/images/<food_name>/")
        print("\n2. Use Your Own Photos (QUICKEST):")
        print("   - Take 5-10 photos of each food with your phone")
        print("   - Transfer to: data/images/pizza/, data/images/burger/, etc.")
        print("   - Minimum: 20-30 images per class")
        print("\n3. Web Search (Educational Use):")
        print("   - Search for food images online")
        print("   - Save to appropriate folders")
        print("\n" + "=" * 70)
        print("\nAfter adding images, run:")
        print("  venv\\Scripts\\python.exe train_model.py")
        print("=" * 70)
        return
    
    print(f"\n[OK] Found {total_images} images across {len(classes)} classes:")
    for cls_name, count in classes:
        print(f"  {cls_name}: {count} images")
    
    if total_images < 50:
        print(f"\n[WARNING] Very few images ({total_images}). Results may be poor.")
        print("Recommendation: Add more images (at least 30-50 per class)")
        response = input("\nProceed with training anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled. Add more images and try again.")
            return
    
    print("\n[INFO] Starting training with optimized settings for 90% accuracy...")
    print("This may take 10-60 minutes depending on your dataset size.")
    print("\nTraining parameters:")
    print("  - Architecture: ResNet50")
    print("  - Epochs: 50 (with early stopping)")
    print("  - Batch size: 16")
    print("  - Learning rate: 0.0001")
    print("  - Target: 90% validation accuracy")
    print("\n" + "=" * 70)
    
    # Import and run training
    import subprocess
    cmd = [
        sys.executable,
        "train_model.py",
        "--data_dir", "data/images",
        "--epochs", "50",
        "--batch_size", "16",
        "--lr", "0.0001",
        "--output_dir", "models"
    ]
    
    print("\nStarting training...\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("[SUCCESS] Training completed!")
        print("\nYour trained model is saved at:")
        print("  models/food_classifier.pt")
        print("\nClass mapping saved at:")
        print("  models/class_mapping.txt")
        print("\nNow run the app:")
        print("  venv\\Scripts\\python.exe -m streamlit run src/app_streamlit.py")
        print("=" * 70)
    else:
        print("\n[ERROR] Training failed. Check errors above.")

if __name__ == "__main__":
    main()

