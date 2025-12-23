# Quick Start Guide - Smart Meal Coach

## ✅ Fixed Issues

1. **Nutrition CSV Created**: `data/nutrition/nutrition_table.csv` now exists with 70+ common foods
2. **App Updated**: Better error messages and handling for untrained models
3. **Training Script**: `train_model.py` ready to train your model

## 🚀 How to Run the App (Right Now)

1. **Activate your virtual environment:**
   ```bash
   cd "C:\Users\K sanjay\Downloads\computervision"
   .\.venv\Scripts\activate
   ```

2. **Run the Streamlit app:**
   ```bash
   python -m streamlit run src/app_streamlit.py
   ```

3. **Upload an image** - The app will now:
   - ✅ Show predictions (even if untrained, it will use placeholder classes)
   - ✅ Display calories and nutrition info (because nutrition CSV exists)
   - ✅ Allow you to log meals
   - ✅ Show recommendations

## 📊 Current Status

- **Model**: Untrained (predicts `class_0`, `class_1`, etc.)
- **Nutrition Data**: ✅ Ready (70+ foods in CSV)
- **App**: ✅ Working

## 🎯 Next Steps to Improve Food Detection

### Option 1: Use Pre-trained Food Models (Quick Demo)

You can download a pre-trained model from:
- Food-101 models on Hugging Face
- Or use a simple API-based approach for demo

### Option 2: Train Your Own Model (Best for Resume)

1. **Download a food dataset:**
   - Food-101: https://www.kaggle.com/datasets/dansbecker/food-101
   - Or create your own: Take photos of 10-20 common foods

2. **Organize images:**
   ```
   data/images/
     ├── pizza/
     │   ├── img1.jpg
     │   └── img2.jpg
     ├── burger/
     │   ├── img1.jpg
     │   └── img2.jpg
     └── ...
   ```

3. **Train the model:**
   ```bash
   python train_model.py --data_dir data/images --epochs 20 --batch_size 32
   ```

4. **Update the app** to load your trained model:
   - Edit `src/app_streamlit.py` line 68:
     ```python
     model = load_trained_model(
         model_path=Path("models/food_classifier.pt"), 
         num_classes=20  # Your number of classes
     )
     ```

## 📝 For Your Resume/Portfolio

**Project Title:** Smart Meal Coach - AI-Powered Nutrition Tracking System

**Description:**
- Built an end-to-end deep learning application using PyTorch and Streamlit
- Implemented food recognition using transfer learning (ResNet18)
- Created nutrition estimation system with 70+ food database
- Developed personalized meal recommendation engine
- Built interactive web interface for meal logging and tracking

**Technologies:** Python, PyTorch, Streamlit, Computer Vision, Transfer Learning, Data Science

## 🐛 Troubleshooting

**If you get import errors:**
- Make sure you're in the project root directory
- Activate virtual environment: `.\.venv\Scripts\activate`
- Install dependencies: `python -m pip install -r requirements.txt`

**If nutrition lookup fails:**
- Check that `data/nutrition/nutrition_table.csv` exists
- Make sure food names in your model match names in the CSV

**If model predictions are poor:**
- This is expected if the model is untrained
- Train the model using `train_model.py` or use a pre-trained model

