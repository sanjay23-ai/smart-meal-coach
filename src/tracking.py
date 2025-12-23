"""
User meal logging and basic analytics.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd

# Absolute-style import for running from src as a script/module
from nutrition import NutritionInfo


LOGS_CSV = Path("data") / "logs" / "meal_logs.csv"


@dataclass
class MealLog:
    user_id: str
    timestamp: datetime
    food_name: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


def append_meal_log(nutrition: NutritionInfo, user_id: str = "user1") -> None:
    """
    Append a meal log row to LOGS_CSV.
    """
    LOGS_CSV.parent.mkdir(parents=True, exist_ok=True)

    log = MealLog(
        user_id=user_id,
        timestamp=datetime.utcnow(),
        food_name=nutrition.food_name,
        calories=nutrition.calories,
        protein_g=nutrition.protein_g,
        carbs_g=nutrition.carbs_g,
        fat_g=nutrition.fat_g,
    )

    df_new = pd.DataFrame([log.to_dict()])
    if LOGS_CSV.exists():
        df_old = pd.read_csv(LOGS_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(LOGS_CSV, index=False)


def load_logs() -> pd.DataFrame:
    """
    Load all meal logs as a DataFrame.
    """
    if LOGS_CSV.exists():
        return pd.read_csv(LOGS_CSV, parse_dates=["timestamp"])
    return pd.DataFrame(
        columns=[
            "user_id",
            "timestamp",
            "food_name",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
        ]
    )


def daily_summary(user_id: str = "user1") -> pd.DataFrame:
    """
    Summarize daily total calories and macros for a user.
    """
    df = load_logs()
    if df.empty:
        return df

    df_user = df[df["user_id"] == user_id].copy()
    df_user["date"] = df_user["timestamp"].dt.date
    grouped = (
        df_user.groupby("date")[["calories", "protein_g", "carbs_g", "fat_g"]].sum()
    )
    return grouped.reset_index()


if __name__ == "__main__":
    print("Current logs path:", LOGS_CSV)
    print("Loaded logs:")
    print(load_logs().head())


