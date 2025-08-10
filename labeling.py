#!/usr/bin/env python3

import os
import csv
import random
from pathlib import Path

# ========== CONFIG ==========
DATASET_DIR = Path("dataset/images")  # Folder with class subfolders
OUTPUT_DIR = Path("dataset/labels")
TRAIN_SPLIT = 0.8

# ========== CREATE OUTPUT DIR ==========
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== GATHER ALL IMAGES ==========
data = []

for class_dir in DATASET_DIR.iterdir():
    if class_dir.is_dir():
        label = class_dir.name
        for image_path in class_dir.glob("*.jpg"):
            data.append((str(image_path.resolve()), label))

# ========== SHUFFLE AND SPLIT ==========
random.shuffle(data)
split_index = int(len(data) * TRAIN_SPLIT)
train_data = data[:split_index]
val_data = data[split_index:]

# ========== SAVE CSV ==========
def save_csv(data_list, filename):
    with open(OUTPUT_DIR / filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(data_list)

save_csv(train_data, "train.csv")
save_csv(val_data, "val.csv")

print(f"[INFO] Auto-labeling complete ✅")
print(f"       Total images : {len(data)}")
print(f"       Train split  : {len(train_data)}")
print(f"       Val split    : {len(val_data)}")
print(f"       Labels saved to: {OUTPUT_DIR}")

