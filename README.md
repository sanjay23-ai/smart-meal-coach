# 🍽️ Smart Meal Coach – AI-Powered Nutrition Tracking System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)

### Overview

Smart Meal Coach is an end-to-end data science and deep learning project that:
- Uses computer vision to recognize foods from images.
- Estimates portion sizes and calories.
- Tracks a user's nutrition over time.
- Recommends healthier, personalized meals based on their habits and goals.

This project is designed to showcase practical skills for data science / AI internships:
- Deep learning (CNNs / transformers for images).
- Data engineering and analytics.
- Recommendation systems.
- Building a simple web app for demonstration.

### Project Structure

- `data/`
  - Raw and processed data (food images, nutrition tables, user logs).
- `notebooks/`
  - Jupyter notebooks for EDA, model experiments, and reports.
- `src/`
  - `data_utils.py` – data loading, preprocessing, and utilities.
  - `vision_model.py` – food classification model (transfer learning).
  - `nutrition.py` – nutrition table loading and calorie estimation.
  - `tracking.py` – user meal logging and analytics.
  - `recommender.py` – personalized meal recommendation logic.
  - `app_streamlit.py` – Streamlit demo app (upload image → predictions + suggestions).

### Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Start Jupyter Lab/Notebook for exploration:

```bash
jupyter lab
```

4. Run the Streamlit demo:

```bash
streamlit run src/app_streamlit.py
```

The app will open in your browser at `http://localhost:8501`

### Features

- 🎯 **Food Recognition**: Upload food images and get instant recognition
- 📊 **Nutrition Tracking**: Automatic calorie and macro estimation
- 📈 **Daily Analytics**: Track your nutrition over time
- 💡 **Smart Recommendations**: Personalized meal suggestions based on your goals
- 🤖 **Train Your Own Model**: Built-in training interface in Streamlit (target: 90%+ accuracy)
- 🔄 **Pre-trained Model**: Works immediately without training using ImageNet weights

### Datasets (You Need to Download)

- Food images:
  - Food-101 (Kaggle or official dataset).
  - Any additional food image datasets you like.
- Nutrition tables:
  - USDA FoodData Central exports or Kaggle nutrition datasets.

Place downloaded data under `data/` and update the paths in `data_utils.py` and `nutrition.py`.

### High-Level Pipeline

1. User uploads a food image.
2. Vision model predicts one or more food classes (e.g., “pizza”, “salad”).
3. Detected foods are mapped to nutrition entries.
4. Portion size is approximated from image regions (simple heuristic to start).
5. Calories and macros are logged for the user and day.
6. Recommender suggests healthier or complementary meals based on history and goals.

### How to Talk About This Project in Interviews

- **Problem**: People find it hard to consistently log and understand their meals.
- **Solution**: Built a vision-based food logger with personalized recommendations.
- **Tech Highlights**:
  - Transfer learning with pretrained CNN/ViT for food classification.
  - Simple portion estimation + nutrition lookup.
  - User-level analytics and visualizations.
  - Neural-network-based scoring for meal recommendations.
  - Deployed an interactive demo using Streamlit.

You can extend this project with more advanced models (e.g., segmentation, transformers, or LLM-based coaching) as you grow.


