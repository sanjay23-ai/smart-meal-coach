"""
Nutrition utilities: mapping foods to calories and macronutrients.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Use absolute-style import so it works when running app_streamlit.py directly
from data_utils import load_nutrition_table


@dataclass
class NutritionInfo:
    food_name: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


def find_closest_food(
    predicted_label: str,
    nutrition_df: pd.DataFrame,
    label_column: str = "food_name",
) -> Optional[pd.Series]:
    """
    Map model-predicted class label to a row in the nutrition table.
    This is a simple string-based lookup that you can later improve with
    fuzzy matching (e.g., rapidfuzz) or manually curated mappings.
    """
    matches = nutrition_df[
        nutrition_df[label_column].str.lower() == predicted_label.lower()
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def estimate_portion_calories(
    nutrition_row: pd.Series,
    portion_grams: float = 150.0,
) -> NutritionInfo:
    """
    Estimate calories and macros for a given portion in grams.
    Assumes the nutrition table is per 100g.
    """
    factor = portion_grams / 100.0
    return NutritionInfo(
        food_name=str(nutrition_row.get("food_name", "unknown")),
        calories=float(nutrition_row.get("calories_per_100g", 0.0) * factor),
        protein_g=float(nutrition_row.get("protein_g", 0.0) * factor),
        carbs_g=float(nutrition_row.get("carbs_g", 0.0) * factor),
        fat_g=float(nutrition_row.get("fat_g", 0.0) * factor),
    )


def simple_portion_from_area(relative_area: float, food_type: str = "general") -> float:
    """
    Convert a relative image area (0-1) into a crude portion estimate in grams.
    
    Adjusts portion size based on food type for more realistic protein values.
    """
    # Base portions by food type (in grams)
    base_portions = {
        "chicken": 300.0,  # Chicken dishes are typically larger portions
        "meat": 250.0,     # Meat dishes
        "fish": 200.0,     # Fish portions
        "general": 250.0,  # Default for mixed meals
    }
    
    # Determine food type
    base_portion = base_portions.get(food_type, base_portions["general"])
    
    # For full meals (relative_area ~1.0), increase portion
    if relative_area >= 0.8:
        base_portion *= 1.5  # Full meal = larger portion
    
    portion = base_portion * max(0.6, min(relative_area, 2.5))
    return portion


def predict_nutrition_for_food_label(
    predicted_label: str,
    relative_area: float = 1.0,
) -> Optional[NutritionInfo]:
    """
    High-level helper that:
    - Loads the nutrition table.
    - Finds the best matching food.
    - Converts relative area to portion size.
    - Returns estimated NutritionInfo.
    """
    df = load_nutrition_table()
    row = find_closest_food(predicted_label, df)
    if row is None:
        return None

    # Determine food type for better portion estimation
    food_type = "general"
    label_lower = predicted_label.lower()
    if "chicken" in label_lower or "hen" in label_lower or "cock" in label_lower:
        food_type = "chicken"
    elif "meat" in label_lower or "steak" in label_lower or "beef" in label_lower or "pork" in label_lower:
        food_type = "meat"
    elif "fish" in label_lower or "salmon" in label_lower:
        food_type = "fish"
    
    portion_grams = simple_portion_from_area(relative_area, food_type=food_type)
    return estimate_portion_calories(row, portion_grams=portion_grams)


if __name__ == "__main__":
    try:
        info = predict_nutrition_for_food_label("pizza", relative_area=0.8)
        print("Example nutrition estimate:\n", info)
    except FileNotFoundError as e:
        print(e)


