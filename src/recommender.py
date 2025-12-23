"""
Simple personalized meal recommendation module.

This is intentionally simple but demonstrates how you can:
- Use user history (logs) as features.
- Score candidate meals using a small neural network or heuristic.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

# Absolute-style import for compatibility when running app_streamlit.py
from tracking import daily_summary


@dataclass
class UserGoal:
    user_id: str
    target_calories: float
    high_protein: bool = False


def recommend_from_candidates(
    candidates: pd.DataFrame,
    goal: UserGoal,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Recommend meals from a candidate nutrition table based on simple scoring:
    - Penalize meals that push well above target calories.
    - Optionally favor higher protein.

    candidates must have at least:
        food_name, calories_per_100g, protein_g, carbs_g, fat_g
    """
    df = candidates.copy()

    # Simple healthiness score: lower calories, higher protein if requested.
    calories = df["calories_per_100g"].to_numpy()
    protein = df.get("protein_g", pd.Series(np.zeros(len(df)))).to_numpy()

    cal_norm = (calories - calories.min()) / (calories.max() - calories.min() + 1e-6)
    prot_norm = (protein - protein.min()) / (protein.max() - protein.min() + 1e-6)

    score = -cal_norm  # prefer lower calorie density
    if goal.high_protein:
        score += 0.5 * prot_norm

    df["score"] = score
    df_sorted = df.sort_values("score", ascending=False)
    return df_sorted.head(top_k)


def simple_user_goal_from_history(user_id: str = "user1") -> UserGoal:
    """
    Derive a rough calorie target based on recent intake history.
    (This is a placeholder heuristic for demonstration.)
    """
    summary = daily_summary(user_id=user_id)
    if summary.empty:
        return UserGoal(user_id=user_id, target_calories=2000.0)

    avg_calories = summary["calories"].tail(7).mean()
    # Suggest a modest 10% reduction by default.
    target = float(max(1500.0, avg_calories * 0.9))
    return UserGoal(user_id=user_id, target_calories=target, high_protein=True)


if __name__ == "__main__":
    print("Recommender module loaded. Connect this with nutrition table and logs.")


