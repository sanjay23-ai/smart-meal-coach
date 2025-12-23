"""
Quick training script with automatic dataset setup.

This script helps you get started quickly by:
1. Creating dataset structure
2. Providing instructions to add images
3. Training the model when ready
"""

import os
import sys
from pathlib import Path

def create_dataset_structure():
    """Create the dataset directory structure."""
    data_dir = Path("data/images")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Common foods that match nutrition CSV
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
    
    print("Creating dataset structure...")
    for food_class in food_classes:
        class_dir = data_dir / food_class
        class_dir.mkdir(exist_ok=True)
    
    print(f"✅ Created directories in {data_dir}")
    return data_dir

def check_images():
    """Check if images exist."""
    data_dir = Path("data/images")
    total_images = 0
    classes_with_images = []
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if len(images) > 0:
                classes_with_images.append(class_dir.name)
                total_images += len(images)
                print(f"  ✅ {class_dir.name}: {len(images)} images")
    
    return total_images, classes_with_images

def main():
    print("=" * 60)
    print("Smart Meal Coach - Quick Training Setup")
    print("=" * 60)
    
    # Create structure
    data_dir = create_dataset_structure()
    
    # Check for images
    print("\nChecking for images...")
    total_images, classes = check_images()
    
    if total_images == 0:
        print("\n⚠️  No images found!")
        print("\nTo train the model, you need to add food images.")
        print("\nOption 1: Download Food-101 from Kaggle")
        print("  https://www.kaggle.com/datasets/dansbecker/food-101")
        print("  Extract and copy images to: data/images/<food_name>/")
        print("\nOption 2: Use your own photos")
        print("  Take photos of foods and organize them as:")
        print("  data/images/pizza/your_image1.jpg")
        print("  data/images/burger/your_image2.jpg")
        print("  etc.")
        print("\nOption 3: Use web-scraped images (for educational purposes)")
        print("  You can download sample images from food websites")
        print("\nMinimum recommended: 20-30 images per class for decent results")
        print("  (More is better! 100+ per class for production)")
        
        response = input("\nDo you want to proceed with training anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Add images and run again.")
            return
    
    if total_images > 0:
        print(f"\n✅ Found {total_images} images across {len(classes)} classes")
        print(f"Classes: {', '.join(classes)}")
        
        if total_images < 50:
            print("\n⚠️  Warning: Very few images. Results may be poor.")
            print("Recommendation: Add more images (at least 20-30 per class)")
        
        response = input("\nProceed with training? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
        
        # Run training
        print("\n🚀 Starting training...")
        import subprocess
        cmd = [
            sys.executable,
            "train_model.py",
            "--data_dir", str(data_dir),
            "--epochs", "15",
            "--batch_size", "16",
            "--lr", "0.0001"
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ Training completed!")
            print("Your model is saved at: models/food_classifier.pt")
            print("Class mapping saved at: models/class_mapping.txt")
            print("\nNow run: python -m streamlit run src/app_streamlit.py")
            print("=" * 60)
        else:
            print("\n❌ Training failed. Check errors above.")
    else:
        print("\n⚠️  Cannot train without images. Please add images first.")

if __name__ == "__main__":
    main()

