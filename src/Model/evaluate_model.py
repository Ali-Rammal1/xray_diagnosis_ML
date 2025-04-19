# Located in: src/Model/evaluate_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path

# Import XrayDataset and LABEL_MAP from data_loading module
try:
    from data_loading import XrayDataset, LABEL_MAP, collate_fn_filter_none  # Import collate_fn too
except ImportError:
    print("❌ Failed import from data_loading. Ensure evaluate_model.py is in src/Model/.")
    exit(1)

# Import metrics calculation functions
try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
except ImportError:
    print("❌ Install scikit-learn: pip install scikit-learn pandas"); exit(1)
import numpy as np

try:
    import pandas as pd;

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# --- Paths (Corrected based on normalize_all.py output and data_loading save path) ---
# Script is in src/Model/
BASE_DIR = Path(__file__).resolve().parent.parent  # src/
PROJECT_ROOT = BASE_DIR.parent  # xray_diagnosis_ML/

# Processed data is at project root level
DATA_PROCESSED_DIR = BASE_DIR / "data_processed"
# Model is loaded from src/ (where data_loading.py saved it)
MODEL_PATH = BASE_DIR / "best_model.pth"

# --- Constants ---
BATCH_SIZE = 32  # Can increase if GPU memory allows
NUM_CLASSES = len(LABEL_MAP)
DROPOUT_RATE = 0.5  # MUST match the dropout used during training

# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation using device: {DEVICE}")


# --- Load the Model (Keep robust version) ---
def load_model(model_path: Path, num_classes: int, device: torch.device):
    print(f"\n--- Loading Model for Evaluation from {model_path} ---")
    model = models.resnet18(weights=None);
    num_features = model.fc.in_features
    # Replicate the exact final layer structure used in training
    model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_features, num_classes))
    print(f"Architecture created: Dropout(p={DROPOUT_RATE}), {num_classes} classes.")
    try:
        if not model_path.exists(): raise FileNotFoundError(f"❌ Model not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device));
        print("State dictionary loaded.")
        model.to(device);
        model.eval();
        print(f"Model ready on {device}.");
        return model
    except FileNotFoundError as e:
        print(f"{e}"); exit(1)
    except RuntimeError as e:
        print(f"❌ Error loading state dict: {e}\n   (Check architecture/dropout/corruption)"); exit(1)
    except Exception as e:
        print(f"❌ Unexpected error loading model: {e}"); exit(1)


# --- Evaluate Model - Detailed Metrics (Keep robust version) ---
def evaluate_detailed(model, dataloader, device, label_map):
    all_preds = [];
    all_labels = []
    model.eval();
    print("\n--- Running Evaluation Loop ---")
    with torch.no_grad():
        batch_num = 0
        for inputs, labels in dataloader:
            batch_num += 1
            if inputs is None or labels is None: continue  # Skip bad batches
            inputs = inputs.to(device, non_blocking=True);
            labels = labels.to(device, non_blocking=True)
            try:
                outputs = model(inputs);
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy());
                all_labels.extend(labels.cpu().numpy())
            except RuntimeError as e:
                print(f"❌ EvalRuntime B{batch_num}: {e}"); continue
            except Exception as e:
                print(f"❌ EvalError B{batch_num}: {e}"); continue
            if batch_num % 20 == 0 or batch_num == len(dataloader): print(
                f"  Processed batch {batch_num}/{len(dataloader)}")  # Less frequent printing
    print("Evaluation loop finished.")
    if not all_labels or not all_preds: print("❌ No samples processed."); return

    print("\n--- Calculating Performance Metrics ---");
    try:
        target_names = list(label_map.keys());
        report_labels = list(label_map.values())
        print("\nClassification Report:")
        report = classification_report(all_labels, all_preds, labels=report_labels, target_names=target_names, digits=4,
                                       zero_division=0);
        print(report)
        accuracy = accuracy_score(all_labels, all_preds);
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                                                     average='macro', zero_division=0,
                                                                                     labels=report_labels)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                                                              average='weighted',
                                                                                              zero_division=0,
                                                                                              labels=report_labels)
        print(f"\nMacro Avg Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1-Score: {f1_macro:.4f}");
        print(
            f"Weighted Avg Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1-Score: {f1_weighted:.4f}");
        print("\nConfusion Matrix (Rows: True, Cols: Pred):")
        cm = confusion_matrix(all_labels, all_preds, labels=report_labels)
        if PANDAS_AVAILABLE:
            try:
                cm_df = pd.DataFrame(cm, index=target_names, columns=target_names); print(cm_df)
            except Exception as pd_err:
                print(f"(Pandas err: {pd_err})"); print(cm)
        else:
            print("(Install pandas for table view)"); print(cm)
    except Exception as e:
        print(f"\n❌ Metrics Calc Error: {e}")


# --- Main Function (Corrected Paths) ---
def main():
    print("--- Model Evaluation Script ---")
    print(f"Evaluating model: {MODEL_PATH}")
    print(f"Using test data from: {DATA_PROCESSED_DIR / 'test'}")  # Correct path
    print(f"Label Map: {LABEL_MAP}")

    # Load Test Dataset (Correct path)
    try:
        test_dataset = XrayDataset(DATA_PROCESSED_DIR, split="test", transform=None)
    except Exception as dataset_err:
        print(f"\n❌ Test Dataset Init Error: {dataset_err}"); return
    if len(test_dataset) == 0: print("\n❌ Test dataset empty."); return
    print(f"\n🧪 Found {len(test_dataset)} test samples.")

    # DataLoader (using robust collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                             collate_fn=collate_fn_filter_none, pin_memory=True if DEVICE.type == 'cuda' else False)

    # Load Model (Correct path)
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

    # Evaluate
    evaluate_detailed(model, test_loader, DEVICE, LABEL_MAP)
    print("\n--- Evaluation Script Finished ---")


if __name__ == "__main__": main()
