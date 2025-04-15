# backend/server.py
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
LABELS = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'UNKNOWN']
IMAGE_SIZE = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image provided."}), 400

    image_file = request.files['image']
    try:
        img = Image.open(image_file).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            prediction_idx = torch.argmax(probabilities).item()
            prediction = LABELS[prediction_idx]

        confidences = {label: float(probabilities[i]) for i, label in enumerate(LABELS)}

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidences": confidences,
            "source": "Our AI Engine",
            "description": f"The uploaded chest X-ray is classified as {prediction} with confidence levels provided. (ma zedta lal confidence level hon, eza bedak men shila idk)"
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
