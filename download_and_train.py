"""
Automated script to download sample food images and train model to 90% accuracy.

This script will:
1. Download sample food images from public sources
2. Organize them properly
3. Train the model with optimized settings for 90% accuracy
"""

import os
import sys
from pathlib import Path
import subprocess

def download_sample_images():
    """
    Download sample food images using a simple approach.
    For educational purposes, we'll create a script that helps download images.
    """
    print("=" * 70)
    print("Downloading Sample Food Images for Training")
    print("=" * 70)
    
    data_dir = Path("data/images")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Food classes we need
    food_classes = [
        "pizza",
        "burger", 
        "chicken_breast",
        "fried_chicken",
        "pasta",
        "rice",
        "salad",
        "sandwich",
        "soup",
        "fish"
    ]
    
    print("\n[INFO] To achieve 90% accuracy, we need images.")
    print("Options:")
    print("1. Download Food-101 from Kaggle (recommended)")
    print("2. Use your own photos")
    print("3. Use web scraping (educational use only)")
    
    print("\n[INFO] Creating download helper script...")
    
    # Create a download script
    download_script = """# Quick image download helper
# This script helps you download food images

import os
from pathlib import Path

# Option 1: Use Food-101 dataset from Kaggle
print("To download Food-101:")
print("1. Go to: https://www.kaggle.com/datasets/dansbecker/food-101")
print("2. Download the dataset")
print("3. Extract and copy images to data/images/<food_name>/")

# Option 2: Use Python to download sample images (requires internet)
try:
    import requests
    from PIL import Image
    from io import BytesIO
    
    # Sample image URLs (replace with actual food image URLs)
    # For educational purposes only - make sure you have permission
    print("\\n[INFO] You can use Python requests to download images")
    print("Example:")
    print("  response = requests.get(image_url)")
    print("  img = Image.open(BytesIO(response.content))")
    print("  img.save('data/images/pizza/image1.jpg')")
except ImportError:
    print("Install requests: pip install requests")
"""
    
    with open("download_images_helper.py", "w") as f:
        f.write(download_script)
    
    print("[OK] Created download_images_helper.py")
    print("\n[INFO] For now, let's check if you have images...")
    
    return data_dir

def check_and_train():
    """Check for images and train if available."""
    data_dir = Path("data/images")
    
    # Count images
    total_images = 0
    classes_with_images = []
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            count = len(images)
            total_images += count
            if count > 0:
                classes_with_images.append((class_dir.name, count))
    
    if total_images == 0:
        print("\n" + "=" * 70)
        print("[ERROR] No images found!")
        print("=" * 70)
        print("\nTo train a model to 90% accuracy, you need food images.")
        print("\nQUICK SOLUTION - Download Food-101:")
        print("1. Go to: https://www.kaggle.com/datasets/dansbecker/food-101")
        print("2. Sign up/login (free)")
        print("3. Download the dataset")
        print("4. Extract it")
        print("5. Copy images to: data/images/<food_name>/")
        print("   Example: Copy pizza images to data/images/pizza/")
        print("\nMinimum for 90% accuracy: 50-100 images per class")
        print("Recommended: 100+ images per class")
        print("\nAfter adding images, run:")
        print("  venv\\Scripts\\python.exe train_model.py --epochs 50 --batch_size 16 --lr 0.0001")
        print("=" * 70)
        return False
    
    print(f"\n[OK] Found {total_images} images across {len(classes_with_images)} classes:")
    for cls_name, count in classes_with_images:
        print(f"  {cls_name}: {count} images")
    
    if total_images < 200:
        print(f"\n[WARNING] Low image count ({total_images}).")
        print("For 90% accuracy, recommend:")
        print("  - At least 50-100 images per class")
        print("  - Total of 500+ images minimum")
        print("  - Current may achieve 70-85% accuracy")
        
        response = input("\nProceed with training anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled. Add more images for better accuracy.")
            return False
    
    print("\n" + "=" * 70)
    print("Starting Training for 90% Accuracy")
    print("=" * 70)
    print("\nTraining parameters:")
    print("  - Model: ResNet50")
    print("  - Epochs: 50 (with early stopping)")
    print("  - Batch size: 16")
    print("  - Learning rate: 0.0001")
    print("  - Target: 90% validation accuracy")
    print("  - Early stopping: Enabled")
    print("=" * 70)
    
    # Run training
    cmd = [
        sys.executable,
        "train_model.py",
        "--data_dir", str(data_dir),
        "--epochs", "50",
        "--batch_size", "16",
        "--lr", "0.0001",
        "--output_dir", "models"
    ]
    
    print("\n[INFO] Starting training...\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("[SUCCESS] Training completed!")
        print("=" * 70)
        print("\nYour trained model is saved at:")
        print("  models/food_classifier.pt")
        print("\nClass mapping saved at:")
        print("  models/class_mapping.txt")
        print("\nNow run the app:")
        print("  venv\\Scripts\\python.exe -m streamlit run src/app_streamlit.py")
        print("=" * 70)
        return True
    else:
        print("\n[ERROR] Training failed. Check errors above.")
        return False

if __name__ == "__main__":
    # Create directories
    download_sample_images()
    
    # Check and train
    check_and_train()

