import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 10
LABEL_MAP = {'NORMAL': 0, 'PNEUMONIA': 1, 'TUBERCULOSIS': 2, 'UNKNOWN': 3}

# Set BASE_DIR to the project root (parent of src/)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "cleaned_data"

print("BASE_DIR:", BASE_DIR)
print("DATA_PROCESSED_DIR:", DATA_PROCESSED_DIR)

# Custom Dataset for loading .npy images and labels
class XrayDataset(Dataset):
    def __init__(self, base_dir, split='train'):
        self.samples = []
        self.label_map = LABEL_MAP
        base_path = Path(base_dir) / split
        for class_name, label in self.label_map.items():
            class_path = base_path / class_name
            if not class_path.exists():
                print(f"Directory not found: {class_path}")
                continue
            for file in class_path.glob("*.npy"):
                self.samples.append((file, label))
        print(f"Found {len(self.samples)} samples in {base_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        img = np.load(file_path)
        # Convert from (H, W, C) to (C, H, W)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        return img_tensor, label

# Training function for one epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    # Create datasets
    train_dataset = XrayDataset(DATA_PROCESSED_DIR, split='train')
    val_dataset   = XrayDataset(DATA_PROCESSED_DIR, split='val')
    test_dataset  = XrayDataset(DATA_PROCESSED_DIR, split='test')

    # Create DataLoaders (adjust num_workers if needed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(LABEL_MAP))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print("Training epoch:")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BASE_DIR / "best_model.pth")
            print("  Saved best model.")
    print("Training complete.")

if __name__ == '__main__':
    mp.freeze_support()  # For Windows support when using multiprocessing
    main()
