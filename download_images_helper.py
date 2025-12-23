# Quick image download helper
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
    print("\n[INFO] You can use Python requests to download images")
    print("Example:")
    print("  response = requests.get(image_url)")
    print("  img = Image.open(BytesIO(response.content))")
    print("  img.save('data/images/pizza/image1.jpg')")
except ImportError:
    print("Install requests: pip install requests")
