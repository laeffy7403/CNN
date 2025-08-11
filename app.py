import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image

# ====== CONFIG ======
MODEL_PATH = "outputs/model.pt"
DATASET_DIR = "dataset/images"  # where your class folders are
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Auto-load class names from dataset folder ======
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
NUM_CLASSES = len(class_names)

# ====== Rebuild model architecture ======
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ====== Image Transform ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====== Flask App ======
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Save uploaded file temporarily
        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        # Process image
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            pred_class = class_names[predicted.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item() * 100

        return render_template("result.html", image_path=img_path, label=pred_class, confidence=confidence)

    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)  # folder for uploads
    app.run(debug=True)
