import os
import random
import shutil
from pathlib import Path

random.seed(42)

SOURCE_DIR = Path("../data")
OUTPUT_DIR = Path("../data_resplit")

CLASSES = ["NORMAL", "PNEUMONIA"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Remove old resplit folder if it exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# Create folders
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

for cls in CLASSES:
    all_images = []

    # Collect images from old train/val/test folders
    for old_split in ["train", "val", "test"]:
        folder = SOURCE_DIR / old_split / cls
        images = list(folder.glob("*"))
        all_images.extend(images)

    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = all_images[:train_end]
    val_files = all_images[train_end:val_end]
    test_files = all_images[val_end:]

    print(f"{cls}: total={total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    for file in train_files:
        shutil.copy(file, OUTPUT_DIR / "train" / cls / file.name)

    for file in val_files:
        shutil.copy(file, OUTPUT_DIR / "val" / cls / file.name)

    for file in test_files:
        shutil.copy(file, OUTPUT_DIR / "test" / cls / file.name)

print("\nDataset resplit completed successfully!")