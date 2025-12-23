"""
Utilities for loading and preprocessing food image datasets and nutrition tables.

This file is intentionally lightweight and educational. As an undergraduate project,
focus on:
- Understanding where the data comes from.
- Writing clear, well-documented preprocessing steps.
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image


DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
NUTRITION_CSV = DATA_DIR / "nutrition" / "nutrition_table.csv"


def load_image_paths(limit: int | None = None) -> List[Path]:
    """
    Load paths to food images from IMAGES_DIR.

    Assumes a structure like:
        data/images/<class_name>/*.jpg

    You can adapt this to match Food-101 or any Kaggle dataset you download.
    """
    image_paths: List[Path] = []
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images directory not found at {IMAGES_DIR}. "
            "Please place your dataset under data/images."
        )

    for class_dir in IMAGES_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.jpg"):
            image_paths.append(img_path)

    if limit is not None:
        image_paths = image_paths[:limit]

    return image_paths


def load_image(path: Path, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image and resize to a fixed size for model input.
    Returns a NumPy array with shape (H, W, C).
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size)
    return np.array(img, dtype=np.float32) / 255.0


def load_nutrition_table() -> pd.DataFrame:
    """
    Load nutrition table CSV.

    The CSV should contain at least:
        - food_name
        - calories_per_100g
        - protein_g
        - carbs_g
        - fat_g

    You can extend this with more nutrients as needed.
    """
    if not NUTRITION_CSV.exists():
        raise FileNotFoundError(
            f"Nutrition table not found at {NUTRITION_CSV}. "
            "Download a nutrition dataset (e.g., USDA/Kaggle) and "
            "save it as nutrition_table.csv."
        )

    df = pd.read_csv(NUTRITION_CSV)
    return df


def example_train_val_split(
    image_paths: List[Path], val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """
    Simple train/validation split for image paths.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(image_paths))
    rng.shuffle(indices)

    split_idx = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_paths = [image_paths[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    return train_paths, val_paths


if __name__ == "__main__":
    # Quick manual test to ensure paths and CSV are set correctly.
    print("Data directory:", DATA_DIR.resolve())
    if IMAGES_DIR.exists():
        print("Number of image files:", len(load_image_paths(limit=1000)))
    else:
        print("Images directory not found. Please add your dataset to data/images.")

    try:
        nutrition = load_nutrition_table()
        print("Loaded nutrition rows:", len(nutrition))
    except FileNotFoundError as e:
        print(e)


