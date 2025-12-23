# 🎯 Auto-Train Model to 90% Accuracy

## Current Status
❌ **No food images found in `data/images/`**

To train a model to **90% accuracy**, you need food images first.

## 🚀 Quick Solution

### Option 1: Download Food-101 Dataset (BEST - Recommended)

1. **Go to Kaggle:**
   - https://www.kaggle.com/datasets/dansbecker/food-101
   - Sign up/login (free)

2. **Download the dataset:**
   - Click "Download" button
   - Extract the zip file

3. **Organize images:**
   ```
   data/images/
     ├── pizza/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ... (50-100 images)
     ├── burger/
     │   └── ... (50-100 images)
     └── ...
   ```

4. **Run training:**
   ```bash
   venv\Scripts\python.exe train_model.py --epochs 50 --batch_size 16 --lr 0.0001
   ```

### Option 2: Use Your Own Photos

1. Take 50-100 photos of each food
2. Save them in `data/images/<food_name>/` folders
3. Run training

### Option 3: Use Automated Script

Run:
```bash
venv\Scripts\python.exe download_and_train.py
```

This will guide you through the process.

## 📊 Requirements for 90% Accuracy

- **Minimum:** 50 images per class
- **Recommended:** 100+ images per class
- **Total:** 500+ images minimum
- **Classes:** 10 food types (pizza, burger, chicken, etc.)

## ✅ After Adding Images

Run the training script:
```bash
venv\Scripts\python.exe train_model.py --epochs 50 --batch_size 16 --lr 0.0001
```

The script will:
- ✅ Use ResNet50 (better than ResNet18)
- ✅ Apply data augmentation
- ✅ Use learning rate scheduling
- ✅ Stop automatically when 90% accuracy is reached
- ✅ Save the best model

## 🎓 Expected Results

- **With 50 images/class:** 80-90% accuracy ✅
- **With 100+ images/class:** 90-95% accuracy 🎯

## ⚠️ Important Notes

1. **Image Quality:** Use clear, well-lit photos
2. **Variety:** Include different angles, lighting, backgrounds
3. **Consistency:** Match food names in nutrition CSV
4. **Training Time:** 20-60 minutes depending on dataset size

---

**Ready to train? Add images first, then run:**
```bash
venv\Scripts\python.exe train_model.py --epochs 50 --batch_size 16 --lr 0.0001
```

