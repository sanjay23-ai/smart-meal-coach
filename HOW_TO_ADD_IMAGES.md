# How to Add Food Images for Training

## Current Status
❌ **No food images found in `data/images/`**

To train the model to 90% accuracy, you need food images organized by class.

## Quick Setup

### Step 1: Create Directory Structure
The structure has been created. You should see:
```
data/images/
  ├── pizza/
  ├── burger/
  ├── chicken_breast/
  ├── fried_chicken/
  ├── pasta/
  ├── rice/
  ├── salad/
  ├── sandwich/
  ├── soup/
  └── fish/
```

### Step 2: Add Images

You have 3 options:

#### Option A: Download Food-101 Dataset (BEST - Recommended)
1. Go to: https://www.kaggle.com/datasets/dansbecker/food-101
2. Sign up/login (free)
3. Download the dataset
4. Extract it
5. Copy images to `data/images/<food_name>/`
   - Example: Copy pizza images to `data/images/pizza/`
   - **Minimum**: 30-50 images per class
   - **For 90% accuracy**: 100+ images per class

#### Option B: Use Your Own Photos
1. Take photos of foods you eat
2. Save them as:
   ```
   data/images/pizza/your_photo1.jpg
   data/images/burger/your_photo2.jpg
   ```
3. **Minimum**: 20-30 images per food class
4. **For 90% accuracy**: 50-100+ images per class

#### Option C: Web Scraping (Educational Use Only)
- Download images from food websites (with permission)
- Organize them in the same structure

### Step 3: Verify Images Are Added

Run:
```bash
venv\Scripts\python.exe check_dataset.py
```

You should see:
```
[OK] Found X classes:
  pizza: 50 images
  burger: 45 images
  ...
```

### Step 4: Train the Model

Once you have images, run:
```bash
venv\Scripts\python.exe train_model.py --data_dir data/images --epochs 50 --batch_size 16 --lr 0.0001
```

Or use the high-accuracy script:
```bash
venv\Scripts\python.exe train_high_accuracy.py
```

## Important Notes

- **File formats**: .jpg, .png supported
- **Naming**: Use underscores in folder names (e.g., `chicken_breast` not `chicken breast`)
- **Match nutrition CSV**: Use food names that exist in `data/nutrition/nutrition_table.csv`
- **More images = Better accuracy**: Aim for 100+ images per class for 90% accuracy

## After Adding Images

Once you've added images, run training again:
```bash
venv\Scripts\python.exe train_model.py
```

The model will train and save to `models/food_classifier.pt` when it reaches good accuracy!

