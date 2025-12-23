"""Quick script to check if dataset exists."""
from pathlib import Path

data_dir = Path("data/images")
if not data_dir.exists():
    print("[ERROR] No data/images directory found!")
    print("Creating structure...")
    data_dir.mkdir(parents=True, exist_ok=True)
    print("[OK] Created data/images/")
    print("\n[WARNING] You need to add food images!")
    print("Organize as: data/images/<food_name>/*.jpg")
    print("See START_TRAINING.md for instructions")
else:
    classes = [c.name for c in data_dir.iterdir() if c.is_dir()]
    total_images = 0
    print(f"[OK] Found {len(classes)} classes:")
    for cls in classes:
        images = list((data_dir/cls).glob("*.jpg")) + list((data_dir/cls).glob("*.png"))
        total_images += len(images)
        print(f"  {cls}: {len(images)} images")
    print(f"\nTotal images: {total_images}")
    if total_images == 0:
        print("\n[WARNING] No images found! Add images to train the model.")

