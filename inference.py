#!/usr/bin/env python3

import os
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from train_cnn import ImageDataset  # reuses the label_map logic

# ========== CONFIG ==========
MODEL_PATH = Path("outputs/model.pt")
INFER_DIR = Path("inference/test")
OUTPUT_DIR = Path("outputs/predictions")
LABEL_CSV = Path("dataset/labels/train.csv")  # we reuse the train.csv to get class-to-id map
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== INIT ==========
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load label map from training data
label_map = ImageDataset(LABEL_CSV).label_map
idx_to_class = {v: k for k, v in label_map.items()}

# Transforms
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load font for labeling
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# ========== INFERENCE ==========
image_paths = list(INFER_DIR.glob("*.jpg"))

print(f"[INFO] Found {len(image_paths)} images to infer.")

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = torch.argmax(outputs, 1).item()
        label = idx_to_class[pred_idx]

    # Draw label on image
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (200, 30)], fill="black")
    draw.text((5, 5), f"Predicted: {label}", font=font, fill="white")

    save_path = OUTPUT_DIR / f"{image_path.stem}_{label}.jpg"
    image.save(save_path)

print(f"[INFO] Inference complete ✅")
print(f"[INFO] Predictions saved to {OUTPUT_DIR}")
