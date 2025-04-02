import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
from data_loading import XrayDataset, LABEL_MAP  # Uses your defined dataset and label map

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best_model.pth"
TEST_DIR = BASE_DIR / "cleaned_data"
BATCH_SIZE = 32
NUM_CLASSES = 4  # Only: NORMAL, PNEUMONIA, TUBERCULOSIS

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Evaluate the model
def evaluate(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    if total > 0:
        accuracy = correct / total
        print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    else:
        print("❌ No samples found. Please check your dataset path.")

# Main function
def main():
    print(f"📂 Evaluating model on test set: {TEST_DIR}")
    test_dataset = XrayDataset(TEST_DIR, split="test")
    test_dataset.samples = [
        (path, label) for path, label in test_dataset.samples if label in [0, 1, 2]
    ]  # Exclude UNKNOWN
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model()
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
