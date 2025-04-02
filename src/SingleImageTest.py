# test_single_image.py

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from torchvision import models
from tkinter import Tk, filedialog

# Constants
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
LABELS = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'UNKNOWN']

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Preprocess image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor

# Select file using file dialog
def pick_image_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an X-ray image",
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

def main():
    img_path = pick_image_file()
    if not img_path:
        print("❌ No file selected.")
        return

    print(f"📂 Selected image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to read the image.")
        return

    model = load_model()
    input_tensor = preprocess_image(img).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        prediction = LABELS[predicted_idx]
        print(f"✅ Prediction: {prediction}")

if __name__ == "__main__":
    main()
