#!/usr/bin/env python3

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, f1_score, precision_score, recall_score
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pathlib import Path

# ========== CONFIG ==========
LABEL_DIR = Path("dataset/labels")
OUTPUT_DIR = Path("outputs")
BATCH_SIZE = 32
EPOCHS = 10
INPUT_SIZE = 224  # ResNet18 default
NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== DATASET ==========
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        self.image_paths = df["filepath"].tolist()
        self.labels = df["label"].tolist()
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.label_map[self.labels[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

# ========== MAIN FUNCTION ==========
def main():
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = ImageDataset(LABEL_DIR / "train.csv", transform=transform)
    val_data = ImageDataset(LABEL_DIR / "val.csv", transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ========== MODEL ==========
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_data.label_map))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []

    print(f"[INFO] Starting training on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        train_acc_list = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ========== SAVE MODEL ==========
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    # ========== EVALUATION ==========
    classes = list(train_data.label_map.keys())
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.clf()

    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")

    with open(OUTPUT_DIR / "results.txt", "w") as f:
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")

                # ========== LOSS CURVES ==========
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "loss_ curve.png")
        plt.clf()

        # ========== METRICS OVER EPOCHS (F1, Precision, Recall) ==========
        # You can store metrics during training if you want more detailed graphs over time
        # For now, we just plot the final static values to visualize

        metrics = {"F1 Score": f1, "Precision": precision, "Recall": recall}
        plt.bar(metrics.keys(), metrics.values(), color=["blue", "orange", "green"])
        plt.title("Final Classification Metrics")#which the itle to classification
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.savefig(OUTPUT_DIR / "final_scores_bar.png")
        plt.clf()

        # Calculate accuracy for this epoch
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        train_acc_list.append(accuracy)

        # Plot Accuracy Curve
        plt.figure()
        plt.plot(train_acc_list, label='Training Accuracy', marker='o')
        plt.title('Training Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig("outputs/accuracy_curve.png")
        plt.close()



    # ========== PRECISION-RECALL CURVE ==========
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_preds, pos_label=1)
    plt.plot(recall_vals, precision_vals, marker='.')
    plt.title('Precision-Recall Curve (class 1)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(OUTPUT_DIR / "precision_recall_curve.png")
    plt.clf()

    # ========== LABEL DISTRIBUTION ==========
    label_counts = pd.Series(all_labels).value_counts().sort_index()
    label_names = [classes[i] for i in label_counts.index]
    plt.bar(label_names, label_counts)
    plt.title("Label Distribution (Val Set)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig(OUTPUT_DIR / "labels_distribution.png")
    plt.clf()

    print(f"[INFO] Training complete ✅")
    print(f"[INFO] Outputs saved to {OUTPUT_DIR}")

# ========== SAFE ENTRY POINT ==========
if __name__ == "__main__":
    main()
                          