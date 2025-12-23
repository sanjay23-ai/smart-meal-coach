"""
Download a small food dataset for training.

This script downloads sample food images from Food-101 or creates a minimal dataset
for quick training and testing.
"""

import os
import urllib.request
from pathlib import Path
import zipfile
import shutil

# Common food classes that match our nutrition CSV
FOOD_CLASSES = [
    "pizza",
    "burger",
    "chicken_breast",
    "fried_chicken",
    "pasta",
    "rice",
    "salad",
    "sandwich",
    "soup",
    "fish",
]

def download_sample_images():
    """
    Download sample food images from a public source.
    For now, we'll create a script that downloads from Food-101 or uses placeholder.
    """
    data_dir = Path("data/images")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample dataset structure...")
    print("Note: For a real project, download Food-101 from Kaggle:")
    print("  https://www.kaggle.com/datasets/dansbecker/food-101")
    print("\nFor now, creating empty directories. You can add your own images there.")
    
    for food_class in FOOD_CLASSES:
        class_dir = data_dir / food_class
        class_dir.mkdir(exist_ok=True)
        print(f"Created directory: {class_dir}")
    
    print(f"\n✅ Dataset structure created at: {data_dir}")
    print("\nNext steps:")
    print("1. Download Food-101 dataset from Kaggle")
    print("2. Extract and copy images to data/images/<class_name>/")
    print("3. Or add your own food photos to these directories")
    print("4. Then run: python train_model.py")


if __name__ == "__main__":
    download_sample_images()

