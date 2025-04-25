# Located in: src/Model/data_loading.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
# Import the top-level amp module
from torch.amp import autocast, GradScaler  # <--- Updated import
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
import time
import warnings
from collections import Counter

# Try importing albumentations
try:
    import albumentations as A
    import cv2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Optional dependency 'albumentations' not found. Augmentation disabled.")
    print("Install with: pip install albumentations opencv-python")

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
ETA_MIN_LR = 1e-7
DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 5
LABEL_MAP = {'NORMAL': 0, 'PNEUMONIA': 1, 'TUBERCULOSIS': 2, 'UNKNOWN': 3}
IMAGE_SIZE_EXPECTED = (512, 512)

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # src/
PROJECT_ROOT = BASE_DIR.parent  # xray_diagnosis_ML/
DATA_PROCESSED_DIR = BASE_DIR / "data_processed"
MODEL_SAVE_PATH = BASE_DIR / "best_model.pth"

# --- Updated Mean/Std ---
CUSTOM_MEAN_LOAD = 0.233119  # <<< Placeholder - Replace with actual value
CUSTOM_STD_LOAD = 0.072080   # <<< Placeholder - Replace with actual value
# --- End Updated Values ---

if not (0 < CUSTOM_MEAN_LOAD < 1 and 0 < CUSTOM_STD_LOAD < 0.5):
    warnings.warn(f"!!! Custom Mean/Std values ({CUSTOM_MEAN_LOAD:.4f}, {CUSTOM_STD_LOAD:.4f}) seem unusual. !!!")

print(f"Script Base Directory (src): {BASE_DIR}")
print(f"Project Root Directory: {PROJECT_ROOT}")
print(f"Processed Data Input Directory: {DATA_PROCESSED_DIR}")
print(f"Model Save Path: {MODEL_SAVE_PATH}")
print(f"Using Custom Mean: {CUSTOM_MEAN_LOAD:.6f}, Custom Std: {CUSTOM_STD_LOAD:.6f}")


# --- XrayDataset Class (Keep robust version) ---
class XrayDataset(Dataset):
    # ... (Keep implementation as before) ...
    def __init__(self, base_dir, split='train', transform=None):
        self.samples = [];
        self.label_map = LABEL_MAP;
        self.split = split
        self.transform = transform;
        self.targets = []
        base_path = Path(base_dir) / self.split
        print(f"\nLoading data for split: '{self.split}' from {base_path}")
        if not base_path.exists(): print(f"❌ WARNING: Dir not found: {base_path}"); self.samples = []; return
        found_samples = 0
        for class_name, label in self.label_map.items():
            class_path = base_path / class_name
            if not class_path.exists(): continue
            count = 0
            try:
                files_in_cat = list(class_path.glob("*.npy"))
                for file in files_in_cat:
                    self.samples.append((file, label));
                    self.targets.append(label);
                    count += 1
                if count > 0: print(f"  Found {count} samples in {class_path.name}"); found_samples += count
            except Exception as e:
                print(f"  ❌ Error scanning {class_path}: {e}")
        print(f"✅ Found {found_samples} total samples for split '{self.split}'.")
        if found_samples == 0 and base_path.exists(): print(f"❌ WARNING: Found 0 samples in dir: {base_path}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples): return None, None
        file_path, label = self.samples[idx]
        try:
            img = np.load(file_path)
            if img is None or not isinstance(img, np.ndarray): return None, None
            if img.shape[0] != IMAGE_SIZE_EXPECTED[0] or img.shape[1] != IMAGE_SIZE_EXPECTED[1]: return None, None
            if np.isnan(img).any() or np.isinf(img).any(): return None, None
            if self.transform:
                try:
                    augmented = self.transform(image=img);
                    img = augmented['image']
                    if np.isnan(img).any() or np.isinf(img).any(): return None, None
                except Exception:
                    pass
            img_tensor = torch.tensor(img).permute(2, 0, 1).float()
            return img_tensor, label
        except FileNotFoundError:
            return None, None
        except Exception:
            return None, None


# --- Collate function (Keep robust version) ---
def collate_fn_filter_none(batch):
    # ... (Keep implementation as before) ...
    original_size = len(batch)
    batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
    filtered_size = len(batch)
    if not batch: return None, None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"❌ Collate Error: {e}. Skip batch."); return None, None


# --- Training function (Update autocast) ---
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, scheduler):
    model.train();
    running_loss = 0.0;
    correct = 0;
    total = 0;
    batch_num = 0;
    start_time = time.time();
    nan_batches = 0
    use_amp = (device.type == 'cuda') and (scaler is not None);
    loader_len = len(loader)
    print(f"  Train on: {device} {'with AMP' if use_amp else '(AMP off)'} ({loader_len} batches)")
    for images, labels in loader:
        batch_num += 1;
        if images is None or labels is None: continue
        images = images.to(device, non_blocking=True);
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # <<< UPDATED AUTOCAST SYNTAX >>>
        with autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                      enabled=use_amp):
            # <<< END UPDATE >>>
            try:
                outputs = model(images); loss = criterion(outputs, labels)
            except RuntimeError as e:
                print(f"❌ FwdRuntime B{batch_num} {images.shape}: {e}"); continue
            except Exception as e:
                print(f"❌ FwdError B{batch_num}: {e}"); continue

        if torch.isnan(loss) or torch.isinf(loss): print(
            f"❌ NaN/Inf Loss B{batch_num}! Skip."); nan_batches += 1; continue

        if use_amp:  # Scaler handles backward/step
            scaler.scale(loss).backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);
            scaler.step(optimizer);
            scaler.update()
        else:  # Standard backward/step
            loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);
            optimizer.step()

        if not (torch.isnan(loss) or torch.isinf(loss)): running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1);
        correct += (preds == labels).sum().item();
        total += images.size(0)

        if batch_num % (loader_len // 4 + 1) == 0 or batch_num == loader_len:
            current_loss_val = loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else float('nan')
            print(f"    Batch {batch_num}/{loader_len}, Loss: {current_loss_val:.4f}")

    if nan_batches > 0: print(f"  ⚠️ Train: {nan_batches} NaN/Inf loss batches.")
    epoch_loss = running_loss / total if total > 0 else float('nan');
    epoch_acc = correct / total if total > 0 else 0.0
    epoch_time = time.time() - start_time;
    print(f"  Train finished in {epoch_time:.2f}s");
    return epoch_loss, epoch_acc


# --- Evaluation function (Update autocast) ---
def evaluate(model, loader, criterion, device):
    model.eval();
    running_loss = 0.0;
    correct = 0;
    total = 0;
    start_time = time.time()
    use_amp = (device.type == 'cuda');
    loader_len = len(loader)
    print(f"  Eval on: {device} {'with AMP ctx' if use_amp else '(AMP ctx off)'} ({loader_len} batches)")
    with torch.no_grad():
        for images, labels in loader:
            if images is None or labels is None: continue
            images = images.to(device, non_blocking=True);
            labels = labels.to(device, non_blocking=True)

            # <<< UPDATED AUTOCAST SYNTAX >>>
            with autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                          enabled=use_amp):
                # <<< END UPDATE >>>
                try:
                    outputs = model(images)
                    if criterion:
                        loss = criterion(outputs, labels)
                        if not (torch.isnan(loss) or torch.isinf(loss)): running_loss += loss.item() * images.size(0)
                except RuntimeError as e:
                    print(f"❌ EvalRuntime {images.shape}: {e}"); continue
                except Exception as e:
                    print(f"❌ EvalError: {e}"); continue

            _, preds = torch.max(outputs, 1);
            correct += (preds == labels).sum().item();
            total += images.size(0)

    epoch_loss = running_loss / total if total > 0 and criterion else float('nan');
    epoch_acc = correct / total if total > 0 else 0.0
    epoch_time = time.time() - start_time;
    print(f"  Eval finished in {epoch_time:.2f}s");
    return epoch_loss, epoch_acc


# --- Main Function (Keep robust version) ---
def main():
    # Device & Scaler
    if torch.cuda.is_available():
        device = torch.device("cuda"); print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu"); print("⚠️ Using CPU.")
    scaler = GradScaler(enabled=(device.type == 'cuda'));
    print(f"   AMP GradScaler {'enabled' if scaler.is_enabled() else 'disabled'}.")

    # Augmentations
    train_transform = None
    if ALBUMENTATIONS_AVAILABLE:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.06, 0.06),
                rotate=(-8, 8),
                cval=0,
                mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
            A.CoarseDropout(max_holes=8,
                            max_height=32,
                            max_width=32,
                            min_height=16,
                            min_width=16,
                            fill_value=0,
                            p=0.1)
        ])

        print("Albumentations defined for training.")
    val_test_transform = None

    # Datasets
    print("\n--- Loading Datasets ---");
    try:
        train_dataset = XrayDataset(DATA_PROCESSED_DIR, split='train', transform=train_transform)
        val_dataset = XrayDataset(DATA_PROCESSED_DIR, split='val', transform=val_test_transform)
        test_dataset = XrayDataset(DATA_PROCESSED_DIR, split='test', transform=val_test_transform)
    except Exception as dataset_err:
        print(f"\n❌ Dataset Init Error: {dataset_err}"); return
    if len(train_dataset) == 0 or len(val_dataset) == 0: print("\n❌ Train/Val dataset empty."); return

    # Class Weights
    print("\n--- Class Weights ---")
    class_counts = Counter(train_dataset.targets);
    total_samples = len(train_dataset.targets);
    class_weights = []
    for i in range(len(LABEL_MAP)):
        count = class_counts.get(i, 0);
        if count == 0:
            print(f"  WARN Class {i} 0 samples."); class_weights.append(0.0)
        else:
            weight = total_samples / (len(LABEL_MAP) * count); class_weights.append(weight); print(
                f"  Class {i}: {count} samples, weight={weight:.3f}")
    class_weights_tensor = torch.tensor(class_weights).float().to(device);
    print(f"Weights tensor: {class_weights_tensor}")

    # Sampler
    print("\n--- WeightedRandomSampler ---");
    sample_weights = [class_weights[label] for label in train_dataset.targets];
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor),
                                    replacement=True)
    print(f"Sampler created for {len(sample_weights_tensor)} samples.")

    # DataLoaders
    num_workers = 0;
    print(f"\n--- DataLoaders (num_workers={num_workers}) ---")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers,
                              collate_fn=collate_fn_filter_none, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn_filter_none, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn_filter_none, pin_memory=True if device.type == 'cuda' else False)
    print("DataLoaders created (Train uses Sampler).")

    # Model
    print("\n--- Model Setup ---");
    print("Loading ResNet18...")
    try:
        model = models.resnet18(weights='IMAGENET1K_V1');
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_features, len(LABEL_MAP)));
        print(f"Added Dropout({DROPOUT_RATE}), replaced FC for {len(LABEL_MAP)} classes.");
        model = model.to(device);
        print(f"Model on {device}.")
    except Exception as model_err:
        print(f"\n❌ Model Setup Error: {model_err}"); return

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss();
    print("Using CrossEntropyLoss (Sampler handles imbalance).")
    # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # Optional: Add weights back
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY);
    print(f"Optimizer: AdamW (LR={LEARNING_RATE}, WD={WEIGHT_DECAY}).")
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN_LR);
    print(f"Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS}, eta_min={ETA_MIN_LR})")
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True) # Alternative

    # Training Loop
    print("\n--- Starting Training ---");
    best_val_loss = float('inf');
    epochs_no_improve = 0;
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---");
        current_lr = optimizer.param_groups[0]['lr'];
        print(f"  LR: {current_lr:.7f}")
        epoch_start_time = time.time()
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                                    scheduler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history['train_loss'].append(train_loss);
            history['train_acc'].append(train_acc);
            history['val_loss'].append(val_loss);
            history['val_acc'].append(val_acc)
        except Exception as epoch_err:
            print(f"\n❌❌ ERROR EPOCH {epoch + 1}: {epoch_err}"); break
        epoch_duration = time.time() - epoch_start_time
        train_loss_str = f"{train_loss:.4f}" if not np.isnan(train_loss) else "nan";
        train_acc_str = f"{train_acc:.4f}"
        val_loss_str = f"{val_loss:.4f}" if not np.isnan(val_loss) else "nan";
        val_acc_str = f"{val_acc:.4f}"
        print("-" * 50);
        print(f"Epoch {epoch + 1} Summary (Duration: {epoch_duration:.2f}s):");
        print(f"  Train Loss: {train_loss_str}, Train Acc: {train_acc_str}");
        print(f"  Val   Loss: {val_loss_str}, Val   Acc: {val_acc_str}");
        print("-" * 50)

        # Scheduler Step
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss if not np.isnan(val_loss) else float('inf'))

        # Checkpoint & Early Stopping
        if not np.isnan(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss;
                epochs_no_improve = 0
                try:
                    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(),
                                                                                          MODEL_SAVE_PATH); print(
                        f"  ✅ Saved best model (Loss: {best_val_loss:.4f}) to {MODEL_SAVE_PATH}")
                except Exception as e:
                    print(f"  ❌ Error saving: {e}")
            else:
                epochs_no_improve += 1; print(
                    f"  Val loss no improve {epochs_no_improve} epochs. Best: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1; print("  ⚠️ NaN val loss.")
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE: print(
            f"\n🛑 Early stopping after {epochs_no_improve} epochs."); break

    print("\n--- Training complete ---");
    # Old line 379:
    # print(f"Lowest Val Loss: {best_val_loss:.4f if best_val_loss != float('inf') else 'N/A'}")

    # --- CORRECTED PRINTING ---
    # First, determine the string representation
    if best_val_loss != float('inf'):
        # If we have a valid best loss, format it
        best_val_loss_str = f"{best_val_loss:.4f}"
    else:
        # Otherwise, use 'N/A'
        best_val_loss_str = "N/A"

    # Then, print the resulting string
    print(f"Lowest Validation Loss achieved: {best_val_loss_str}")
    # --- END CORRECTION ---

    # Final Evaluation
    if Path(MODEL_SAVE_PATH).exists():
        print("\n--- Evaluating best saved model on test set ---");
        try:
            print(f"Reloading best from {MODEL_SAVE_PATH}...");
            final_model = models.resnet18(weights=None);
            num_features = final_model.fc.in_features
            final_model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_features, len(LABEL_MAP)));
            final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device));
            final_model.to(device);
            final_model.eval();
            print("Reloaded.")
            test_loss, test_acc = evaluate(final_model, test_loader, criterion, device)  # Use same criterion instance
            test_loss_str = f"{test_loss:.4f}" if not np.isnan(test_loss) else "nan";
            test_acc_str = f"{test_acc:.4f}"
            print("-" * 50);
            print(f"Final Test Results:");
            print(f"  Test Loss: {test_loss_str}");
            print(f"  Test Acc:  {test_acc_str}");
            print("-" * 50)
            # If you want detailed metrics here, import evaluate_detailed from evaluate_model.py
            # from evaluate_model import evaluate_detailed
            # evaluate_detailed(final_model, test_loader, device, LABEL_MAP)
        except Exception as e:
            print(f"❌ Error reloading/evaluating: {e}")
    else:
        print("\n--- Skip test eval (best model not found) ---")


if __name__ == '__main__':
    mp.freeze_support();
    main()