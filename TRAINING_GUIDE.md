# Training Guide - Smart Meal Coach

## 🎯 Goal: Train the Model to Recognize Real Foods

Currently, the model predicts generic `class_0`, `class_1`, etc. To make it recognize actual foods like "pizza", "burger", "chicken", you need to train it on real food images.

## 📥 Step 1: Get Food Images

You have 3 options:

### Option A: Download Food-101 Dataset (Recommended for Best Results)

1. **Download from Kaggle:**
   - Go to: https://www.kaggle.com/datasets/dansbecker/food-101
   - Download the dataset (requires Kaggle account, free)
   - Extract the zip file

2. **Organize images:**
   ```
   data/images/
     ├── pizza/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     ├── burger/
     │   ├── image1.jpg
     │   └── ...
     └── ...
   ```

3. **Select 10-20 common food classes** that match your nutrition CSV:
   - pizza, burger, chicken_breast, fried_chicken, pasta, rice, salad, sandwich, soup, fish, etc.
   - Copy 50-100 images per class to `data/images/<food_name>/`

### Option B: Use Your Own Photos (Quick Start)

1. Take photos of foods you eat regularly
2. Organize them:
   ```
   data/images/pizza/your_photo1.jpg
   data/images/burger/your_photo2.jpg
   ```
3. **Minimum:** 20-30 images per food class
4. **Recommended:** 50-100+ images per class

### Option C: Web Scraping (Educational Use Only)

For educational purposes, you can download images from food websites. Make sure you have permission and follow terms of service.

## 🚀 Step 2: Train the Model

### Quick Method (Interactive):

```bash
python quick_train.py
```

This script will:
- Check if you have images
- Guide you through the process
- Train the model automatically

### Manual Method:

```bash
python train_model.py --data_dir data/images --epochs 15 --batch_size 16
```

**Parameters:**
- `--epochs`: Number of training iterations (15-20 is good for start)
- `--batch_size`: Images per batch (16 for limited RAM, 32+ if you have more)
- `--lr`: Learning rate (default 0.0001 is good)

### What Happens During Training:

1. **Loading:** Images are loaded and split into train (80%) and validation (20%)
2. **Training:** Model learns to recognize foods (this takes 10-30 minutes depending on your data)
3. **Saving:** Best model is saved to `models/food_classifier.pt`
4. **Mapping:** Class names are saved to `models/class_mapping.txt`

## ✅ Step 3: Test Your Trained Model

After training completes:

```bash
python -m streamlit run src/app_streamlit.py
```

Upload a food image - it should now show real food names like "Pizza", "Burger", etc. instead of "class_0"!

## 📊 Expected Results

- **With 20-30 images per class:** ~60-70% accuracy (good for demo)
- **With 50-100 images per class:** ~75-85% accuracy (good for portfolio)
- **With 100+ images per class:** ~85-95% accuracy (production-ready)

## 🐛 Troubleshooting

**"No images found" error:**
- Make sure images are in `data/images/<food_name>/` folders
- Check file extensions (.jpg, .png are supported)
- Verify folder names match food names in nutrition CSV

**Low accuracy:**
- Add more images (at least 50 per class)
- Train for more epochs (try 20-30)
- Ensure images are clear and show the food clearly

**Out of memory:**
- Reduce batch_size (try 8 or 4)
- Use fewer images per class
- Close other applications

## 💡 Tips for Best Results

1. **Image Quality:** Use clear, well-lit photos
2. **Variety:** Include different angles, lighting, backgrounds
3. **Consistency:** Keep food names consistent (use underscores: `chicken_breast` not `chicken breast`)
4. **Match Nutrition CSV:** Use food names that exist in `data/nutrition/nutrition_table.csv`

## 🎓 For Your Resume/Portfolio

After training, you can say:
- "Trained a ResNet18-based food classifier achieving 80%+ accuracy on 10 food classes"
- "Fine-tuned transfer learning model on Food-101 dataset"
- "Implemented data augmentation and train/val splitting for robust model training"

Good luck! 🚀

