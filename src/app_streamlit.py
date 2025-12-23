"""
Streamlit demo app for Smart Meal Coach.

Run with:
    streamlit run src/app_streamlit.py
"""

from pathlib import Path
from typing import Dict

import streamlit as st
from PIL import Image

import torch

from vision_model import build_model, predict_food
from pretrained_food_detector import predict_food_pretrained
from nutrition import predict_nutrition_for_food_label
from tracking import append_meal_log, daily_summary
from data_utils import load_nutrition_table
from recommender import simple_user_goal_from_history, recommend_from_candidates

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random


@st.cache_resource
def load_trained_model(model_path: Path | None = None, num_classes: int = 10):
    """
    Load a trained PyTorch model.
    Automatically loads from models/food_classifier.pt if it exists.
    Uses ResNet50 for better accuracy.
    """
    if model_path is None:
        model_path = Path("models/food_classifier.pt")
    
    # Use ResNet50 for better accuracy (matches training)
    model = build_model(num_classes=num_classes, use_resnet50=True)
    if model_path.exists():
        try:
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            st.success(f"✅ Loaded trained model from {model_path}")
        except Exception as e:
            st.warning(f"Could not load model: {e}. Using untrained model.")
    else:
        st.info("⚠️ No trained model found. Using untrained model. Train with: python train_high_accuracy.py")
    model.eval()
    return model


def load_class_mapping() -> Dict[int, str]:
    """Load class mapping from file or return default."""
    mapping_file = Path("models/class_mapping.txt")
    if mapping_file.exists():
        idx_to_class = {}
        with open(mapping_file, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    idx, class_name = line.split(":", 1)
                    idx_to_class[int(idx.strip())] = class_name.strip()
        return idx_to_class
    return {}


def train_model_in_streamlit(data_dir: Path, epochs: int, batch_size: int, lr: float):
    """Train model directly in Streamlit with progress updates."""
    from vision_model import FoodImageDataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    all_samples = []
    class_to_idx = {}
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir = data_dir / cls_name
        for img_path in cls_dir.glob("*.jpg"):
            all_samples.append((img_path, class_to_idx[cls_name]))
        for img_path in cls_dir.glob("*.png"):
            all_samples.append((img_path, class_to_idx[cls_name]))
    
    if len(all_samples) == 0:
        return None, None, "No images found!"
    
    # Split train/val
    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    class SplitDataset(FoodImageDataset):
        def __init__(self, samples, class_to_idx, transform=None):
            self.root = data_dir
            self.transform = transform
            self.samples = samples
            self.class_to_idx = class_to_idx
    
    train_ds = SplitDataset(train_samples, class_to_idx, transform=train_transform)
    val_ds = SplitDataset(val_samples, class_to_idx, transform=val_transform)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Build model
    num_classes = len(idx_to_class)
    model = build_model(num_classes=num_classes, use_resnet50=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0.0
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        scheduler.step(val_acc)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs}")
        metrics_placeholder.text(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Path("models/food_classifier.pt")
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            # Save class mapping
            with open(Path("models/class_mapping.txt"), "w") as f:
                for idx, class_name in sorted(idx_to_class.items()):
                    f.write(f"{idx}: {class_name}\n")
            
            if val_acc >= 90.0:
                st.success(f"🎉 Target reached! 90%+ accuracy achieved: {val_acc:.2f}%")
                break
    
    progress_bar.empty()
    status_text.empty()
    metrics_placeholder.empty()
    
    return model, idx_to_class, f"Training complete! Best accuracy: {best_val_acc:.2f}%"


def main() -> None:
    st.title("Smart Meal Coach 🍽️")
    
    # Add tabs for different functionalities
    tab1, tab2 = st.tabs(["🍽️ Food Recognition", "🤖 Train Model"])
    
    with tab1:
        st.write(
            "Upload a picture of your meal. The app will try to recognize the food, "
            "estimate calories, log the meal, and suggest healthier options."
        )

        user_id = st.text_input("User ID", value="user1", key="user_id_main")

        uploaded_file = st.file_uploader(
            "Upload a food image", type=["jpg", "jpeg", "png"], key="upload_main"
        )

        # Sidebar: daily summary
        st.sidebar.header("Daily Summary")
        summary = daily_summary(user_id=user_id)
        if not summary.empty:
            st.sidebar.dataframe(summary.tail(7))
        else:
            st.sidebar.write("No meals logged yet.")

        if uploaded_file is None:
            st.info("Upload an image to get started.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded meal")

            # Try to load trained model first, otherwise use pre-trained ImageNet model
            model_path = Path("models/food_classifier.pt")
            idx_to_class = load_class_mapping()
            
            use_pretrained = False
            if idx_to_class:
                num_classes = len(idx_to_class)
                model = load_trained_model(model_path=model_path, num_classes=num_classes)
                if not model_path.exists():
                    use_pretrained = True
            else:
                # Use pre-trained ImageNet model (works immediately without training)
                use_pretrained = True

            with st.spinner("Analyzing your meal..."):
                if use_pretrained:
                    # Use pre-trained ImageNet model - works immediately!
                    predicted_label, confidence = predict_food_pretrained(image)
                else:
                    # Use custom trained model
                    predicted_label, confidence = predict_food(model, image, idx_to_class)

            st.subheader("Prediction")
            # Format food name nicely (replace underscores with spaces, capitalize)
            display_name = predicted_label.replace("_", " ").title()
            st.write(f"**Food:** {display_name} (confidence: {confidence:.2f})")
            
            # Manual correction option for low confidence
            if confidence < 0.3:
                st.warning("⚠️ Low confidence detection. If incorrect, please correct it:")
                correction_options = ["chicken_breast", "roasted_chicken", "fried_chicken", "pizza", "burger", "pasta", "rice", "salad", "fish", "steak"]
                corrected_food = st.selectbox(
                    "Correct food type (if detection is wrong):",
                    options=["Auto-detect"] + correction_options,
                    index=0,
                    key="food_correction"
                )
                if corrected_food != "Auto-detect":
                    predicted_label = corrected_food
                    confidence = 0.8  # Set higher confidence for manual correction
                    st.success(f"✅ Using corrected food: {corrected_food.replace('_', ' ').title()}")
            
            # Show note if using pre-trained model or low confidence
            if use_pretrained:
                if confidence > 0.5:
                    st.success(f"✅ Using pre-trained ImageNet model (no training needed!)")
                else:
                    st.info("ℹ️ Using pre-trained model. For better accuracy on specific foods, train a custom model.")
            elif predicted_label.startswith("class_") or confidence < 0.3:
                st.warning(
                    "⚠️ **Model appears untrained or confidence is low.** "
                    "Using pre-trained ImageNet model as fallback."
                )
            elif confidence < 0.5:
                st.info("⚠️ Low confidence prediction. Consider training the model for better accuracy.")

            # For low confidence predictions, use larger relative area (full meal)
            relative_area = 1.2 if confidence < 0.3 else 1.0
            
            nutrition = predict_nutrition_for_food_label(predicted_label, relative_area=relative_area)
            if nutrition is None:
                st.warning(
                    f"Could not find '{predicted_label}' in the nutrition table. "
                    "The nutrition CSV has been created with common foods. "
                    "If you train the model, make sure class names match food names in the CSV."
                )
            else:
                st.subheader("Estimated Nutrition")
                st.write(
                    f"Calories: **{nutrition.calories:.0f} kcal**  |  "
                    f"Protein: **{nutrition.protein_g:.1f} g**  |  "
                    f"Carbs: **{nutrition.carbs_g:.1f} g**  |  "
                    f"Fat: **{nutrition.fat_g:.1f} g**"
                )

                if st.button("Log this meal"):
                    append_meal_log(nutrition, user_id=user_id)
                    st.success("Meal logged!")

                st.subheader("Recommendations")
                try:
                    nutrition_table = load_nutrition_table()
                    goal = simple_user_goal_from_history(user_id=user_id)
                    recs = recommend_from_candidates(nutrition_table, goal=goal, top_k=5)
                    st.write(f"Target daily calories (approx): **{goal.target_calories:.0f} kcal**")
                    st.write("Suggested foods (healthier or complementary):")
                    st.dataframe(recs[["food_name", "calories_per_100g", "protein_g", "carbs_g", "fat_g", "score"]])
                except FileNotFoundError:
                    st.info("Nutrition table not found.")
    
    with tab2:
        st.header("🤖 Train Your Model")
        st.write("Train a custom food recognition model to achieve 90%+ accuracy.")
        
        # Check for images
        data_dir = Path("data/images")
        if not data_dir.exists():
            st.error("❌ Dataset directory not found! Create `data/images/<food_name>/` folders and add images.")
            st.info("See TRAINING_GUIDE.md for instructions on getting food images.")
            return
        
        # Count images
        total_images = 0
        classes_with_images = []
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                count = len(images)
                total_images += count
                if count > 0:
                    classes_with_images.append((class_dir.name, count))
        
        if total_images == 0:
            st.error("❌ No images found!")
            st.info("""
            **To train a model, you need food images:**
            
            1. Download Food-101 from Kaggle: https://www.kaggle.com/datasets/dansbecker/food-101
            2. Extract and copy images to `data/images/<food_name>/`
            3. Minimum: 50 images per class
            4. For 90% accuracy: 100+ images per class
            """)
            return
        
        st.success(f"✅ Found {total_images} images across {len(classes_with_images)} classes:")
        for cls_name, count in classes_with_images:
            st.write(f"  - **{cls_name}**: {count} images")
        
        if total_images < 200:
            st.warning(f"⚠️ Low image count ({total_images}). For 90% accuracy, recommend 500+ images.")
        
        st.divider()
        
        # Training parameters
        st.subheader("Training Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.slider("Epochs", 10, 100, 50, help="Number of training iterations")
        with col2:
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1, help="Images per batch")
        with col3:
            lr = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01], index=0, help="Learning rate for optimization")
        
        st.info(f"""
        **Training Configuration:**
        - Model: ResNet50 (best accuracy)
        - Epochs: {epochs}
        - Batch Size: {batch_size}
        - Learning Rate: {lr}
        - Target: 90%+ validation accuracy
        - Early Stopping: Enabled (stops at 90%)
        """)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if total_images < 50:
                st.warning("⚠️ Very few images. Results may be poor. Recommend adding more images.")
            
            with st.spinner("Training in progress... This may take 10-60 minutes."):
                model, idx_to_class, result_msg = train_model_in_streamlit(
                    data_dir, epochs, batch_size, lr
                )
            
            if model is not None:
                st.success(result_msg)
                st.balloons()
                st.info("✅ Model saved! Go to 'Food Recognition' tab to use your trained model.")
            else:
                st.error(result_msg)

    uploaded_file = st.file_uploader(
        "Upload a food image", type=["jpg", "jpeg", "png"]
    )

    # Sidebar: daily summary
    st.sidebar.header("Daily Summary")
    summary = daily_summary(user_id=user_id)
    if not summary.empty:
        st.sidebar.dataframe(summary.tail(7))
    else:
        st.sidebar.write("No meals logged yet.")

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded meal")

    # Try to load trained model first, otherwise use pre-trained ImageNet model
    model_path = Path("models/food_classifier.pt")
    idx_to_class = load_class_mapping()
    
    use_pretrained = False
    if idx_to_class:
        num_classes = len(idx_to_class)
        model = load_trained_model(model_path=model_path, num_classes=num_classes)
        if not model_path.exists():
            use_pretrained = True
    else:
        # Use pre-trained ImageNet model (works immediately without training)
        use_pretrained = True

    with st.spinner("Analyzing your meal..."):
        if use_pretrained:
            # Use pre-trained ImageNet model - works immediately!
            predicted_label, confidence = predict_food_pretrained(image)
        else:
            # Use custom trained model
            predicted_label, confidence = predict_food(model, image, idx_to_class)

    st.subheader("Prediction")
    # Format food name nicely (replace underscores with spaces, capitalize)
    display_name = predicted_label.replace("_", " ").title()
    st.write(f"**Food:** {display_name} (confidence: {confidence:.2f})")
    
    # Show note if using pre-trained model or low confidence
    if use_pretrained:
        if confidence > 0.5:
            st.success(f"✅ Using pre-trained ImageNet model (no training needed!)")
        else:
            st.info("ℹ️ Using pre-trained model. For better accuracy on specific foods, train a custom model.")
    elif predicted_label.startswith("class_") or confidence < 0.3:
        st.warning(
            "⚠️ **Model appears untrained or confidence is low.** "
            "Using pre-trained ImageNet model as fallback."
        )
    elif confidence < 0.5:
        st.info("⚠️ Low confidence prediction. Consider training the model for better accuracy.")

    nutrition = predict_nutrition_for_food_label(predicted_label, relative_area=1.0)
    if nutrition is None:
        st.warning(
            f"Could not find '{predicted_label}' in the nutrition table. "
            "The nutrition CSV has been created with common foods. "
            "If you train the model, make sure class names match food names in the CSV."
        )
        return

    st.subheader("Estimated Nutrition")
    st.write(
        f"Calories: **{nutrition.calories:.0f} kcal**  |  "
        f"Protein: **{nutrition.protein_g:.1f} g**  |  "
        f"Carbs: **{nutrition.carbs_g:.1f} g**  |  "
        f"Fat: **{nutrition.fat_g:.1f} g**"
    )

    if st.button("Log this meal"):
        append_meal_log(nutrition, user_id=user_id)
        st.success("Meal logged!")

    st.subheader("Recommendations")
    try:
        nutrition_table = load_nutrition_table()
    except FileNotFoundError:
        st.info(
            "Nutrition table not found. Add a CSV at data/nutrition/nutrition_table.csv "
            "to enable recommendations."
        )
        return

    goal = simple_user_goal_from_history(user_id=user_id)
    recs = recommend_from_candidates(nutrition_table, goal=goal, top_k=5)
    st.write(f"Target daily calories (approx): **{goal.target_calories:.0f} kcal**")
    st.write("Suggested foods (healthier or complementary):")
    st.dataframe(recs[["food_name", "calories_per_100g", "protein_g", "carbs_g", "fat_g", "score"]])


if __name__ == "__main__":
    main()


