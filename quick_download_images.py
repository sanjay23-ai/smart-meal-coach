"""
Quick script to download sample food images for training.

This script helps you get started quickly by downloading sample images.
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("Quick Image Download Helper")
    print("=" * 70)
    
    print("\nTo train a model to 90% accuracy, you need food images.")
    print("\nBEST OPTION: Download Food-101 from Kaggle")
    print("- Go to: https://www.kaggle.com/datasets/dansbecker/food-101")
    print("- Download and extract")
    print("- Copy images to: data/images/<food_name>/")
    
    print("\nALTERNATIVE: Use Python to download images")
    print("\nHere's a sample script you can use:")
    
    sample_code = '''
# Example: Download food images using requests (educational use only)
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

# Create directories
foods = ["pizza", "burger", "chicken_breast", "pasta", "rice", "salad"]
for food in foods:
    Path(f"data/images/{food}").mkdir(parents=True, exist_ok=True)

# Download sample images (replace URLs with actual food image URLs)
# Make sure you have permission to use these images!
image_urls = {
    "pizza": ["url1", "url2", ...],  # Add actual URLs
    "burger": ["url1", "url2", ...],
    # etc.
}

for food, urls in image_urls.items():
    for i, url in enumerate(urls[:50]):  # Download 50 per food
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            img.save(f"data/images/{food}/image_{i+1}.jpg")
            print(f"Downloaded {food} image {i+1}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
'''
    
    print(sample_code)
    
    print("\n" + "=" * 70)
    print("RECOMMENDED: Use Food-101 Dataset")
    print("=" * 70)
    print("\nSteps:")
    print("1. Go to: https://www.kaggle.com/datasets/dansbecker/food-101")
    print("2. Click 'Download' (requires Kaggle account - free)")
    print("3. Extract the zip file")
    print("4. Copy images:")
    print("   - From: food-101/images/pizza/*.jpg")
    print("   - To: data/images/pizza/*.jpg")
    print("5. Repeat for burger, chicken_breast, pasta, rice, salad, etc.")
    print("\nMinimum: 50 images per food class")
    print("For 90% accuracy: 100+ images per class")
    print("\nAfter adding images, run:")
    print("  venv\\Scripts\\python.exe train_model.py --epochs 50 --batch_size 16 --lr 0.0001")
    print("=" * 70)

if __name__ == "__main__":
    main()

