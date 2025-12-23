# 🚀 Quick Start: Train Your Model to 90% Accuracy

## ✅ What I've Fixed

1. **Protein Values**: Increased portion size estimation (200g base instead of 120g) for more realistic protein values
2. **Model Architecture**: Upgraded to ResNet50 (better accuracy than ResNet18)
3. **Training Improvements**:
   - More data augmentation (color jitter, affine transforms)
   - Learning rate scheduling
   - Early stopping (stops when 90% accuracy reached)
   - Weight decay regularization
   - Increased default epochs to 50

## 📥 Step 1: Get Food Images

You **MUST** have food images to train. Choose one:

### Option A: Download Food-101 (Recommended)
```bash
# Go to: https://www.kaggle.com/datasets/dansbecker/food-101
# Download and extract
# Copy images to: data/images/<food_name>/
```

### Option B: Use Your Own Photos
- Take photos of foods
- Organize: `data/images/pizza/photo1.jpg`, `data/images/burger/photo2.jpg`, etc.
- **Minimum**: 30-50 images per class
- **For 90% accuracy**: 100+ images per class recommended

## 🎯 Step 2: Train the Model

### Easy Method (Recommended):
```bash
python train_high_accuracy.py
```

This script will:
- Check your dataset
- Train with optimized settings
- Stop automatically when 90% accuracy is reached
- Save the best model

### Manual Method:
```bash
python train_model.py --data_dir data/images --epochs 50 --batch_size 16 --lr 0.0001
```

## ⏱️ Training Time

- **Small dataset (200-500 images)**: 10-20 minutes
- **Medium dataset (500-1000 images)**: 20-40 minutes  
- **Large dataset (1000+ images)**: 40-60+ minutes

## 📊 Expected Results

- **With 30-50 images/class**: 70-80% accuracy
- **With 50-100 images/class**: 80-90% accuracy ✅
- **With 100+ images/class**: 90-95% accuracy 🎯

## ✅ Step 3: Test Your Model

After training completes:

```bash
python -m streamlit run src/app_streamlit.py
```

Upload a food image - it should now show:
- ✅ Real food names (Pizza, Burger, etc.)
- ✅ Accurate nutrition values
- ✅ Higher confidence scores (>0.7)
- ✅ Realistic protein values

## 🐛 Troubleshooting

**"No images found" error:**
- Make sure images are in `data/images/<food_name>/` folders
- Check file extensions (.jpg, .png)

**Low accuracy (<80%):**
- Add more images (at least 50 per class)
- Ensure images are clear and show food clearly
- Train for more epochs

**Out of memory:**
- Reduce batch_size: `--batch_size 8` or `--batch_size 4`
- Close other applications

## 💡 Tips for 90% Accuracy

1. **More Data = Better**: Aim for 100+ images per class
2. **Quality Matters**: Use clear, well-lit photos
3. **Variety**: Include different angles, lighting, backgrounds
4. **Consistent Naming**: Match food names in nutrition CSV

## 🎓 After Training

Your model will be saved at:
- `models/food_classifier.pt` - The trained model
- `models/class_mapping.txt` - Food class names

The app will automatically load it and show real food names!

---

**Ready to train? Run: `python train_high_accuracy.py`**

