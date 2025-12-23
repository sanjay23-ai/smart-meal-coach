"""
Complete setup and training script.

This script:
1. Sets up the dataset structure
2. Downloads sample images (if available)
3. Trains the model
4. Updates the app configuration
"""

import subprocess
import sys
from pathlib import Path
import shutil

def check_dataset():
    """Check if dataset exists and has images."""
    data_dir = Path("data/images")
    if not data_dir.exists():
        print("❌ Dataset directory not found. Creating structure...")
        from download_food_dataset import download_sample_images
        download_sample_images()
        return False
    
    # Check if any class directories have images
    has_images = False
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if len(images) > 0:
                has_images = True
                print(f"✅ Found {len(images)} images in {class_dir.name}")
    
    if not has_images:
        print("⚠️  No images found in data/images/")
        print("Please add food images organized as: data/images/<food_name>/*.jpg")
        return False
    
    return True

def train_model():
    """Train the food classification model."""
    print("\n🚀 Starting model training...")
    
    # Check if we have a dataset
    if not check_dataset():
        print("\n❌ Cannot train without images.")
        print("Please add images to data/images/<food_name>/ directories")
        print("Or download Food-101 from: https://www.kaggle.com/datasets/dansbecker/food-101")
        return False
    
    # Run training
    cmd = [
        sys.executable,
        "train_model.py",
        "--data_dir", "data/images",
        "--epochs", "15",
        "--batch_size", "16",  # Smaller batch for limited data
        "--lr", "0.0001",
        "--output_dir", "models"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        return True
    else:
        print("\n❌ Training failed. Check errors above.")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Smart Meal Coach - Model Training Setup")
    print("=" * 60)
    
    success = train_model()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Setup complete! Your model is trained.")
        print("Now run: python -m streamlit run src/app_streamlit.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  Setup incomplete. Please add food images first.")
        print("=" * 60)

