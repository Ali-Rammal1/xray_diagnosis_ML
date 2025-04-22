# Located in: backend/server.py
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2  # Use OpenCV for preprocessing consistent with normalize_all.py
from flask_cors import CORS
from pathlib import Path
import warnings

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Constants - MUST MATCH normalize_all.py & training ---
LABELS = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'UNKNOWN']
IMAGE_SIZE = (512, 512)  # Match the size used in normalize_all.py
NUM_CLASSES = len(LABELS)
DROPOUT_RATE = 0.5  # Match dropout used in data_loading.py

# !!! IMPORTANT: REPLACE THESE WITH VALUES FROM normalize_all.py (512x512) RUN !!!
# These are the stats calculated by normalize_all.py for the 512x512 custom processing
CUSTOM_MEAN = 0.232080  # <<< Placeholder - Replace with actual value
CUSTOM_STD = 0.070931  # <<< Placeholder - Replace with actual value
# !!! END IMPORTANT !!!

# --- Replicate preprocessing settings from normalize_all.py ---
USE_CLAHE = True  # Must match normalize_all.py
WINDOW_LEVEL_PARAMS = {'wl': 600, 'ww': 1500}  # Must match (or be None)
INTERPOLATION = cv2.INTER_AREA  # Must match
EPSILON = 1e-6  # Use consistent epsilon
# --- End Preprocessing Constants ---

# --- Check if mean/std placeholders are still present ---
if CUSTOM_MEAN == 0.491234 or CUSTOM_STD == 0.256789:
    warnings.warn("!!! SERVER: PLACEHOLDER custom mean/std detected. Replace with actual values! !!!")
# --- End Check ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server using device: {device}")

# --- Path logic: server.py in backend/, model in src/ ---
project_root = Path(__file__).parent.parent.resolve()  # Should be xray_diagnosis_ML/
# Model path points to where data_loading.py saved it (src/best_model.pth)
model_path = project_root / "src" / "best_model.pth"
print(f"Server attempting to load model from: {model_path}")


# --- Utility functions matching normalize_all.py (Keep robust versions) ---
def window_level(img: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Apply window-leveling, output is [0, 1]"""
    mn = wl - ww / 2.0
    mx = wl + ww / 2.0
    denominator = mx - mn
    if np.isclose(denominator, 0): return np.where(img >= wl, 1.0, 0.0).astype(np.float32)
    img_clipped = np.clip(img.astype(np.float32), mn, mx)
    return (img_clipped - mn) / (denominator + EPSILON)


def preprocess_image_for_inference(img_bytes: bytes) -> torch.Tensor:
    """Preprocesses raw image bytes using the same pipeline as training data."""
    try:#Debug #1
        # 1. Decode image bytes using OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Read as color
        if img is None: raise ValueError("Could not decode image using OpenCV.")

        # 2. To Grayscale Float32
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 3. Window/Level (outputs [0,1])
        if WINDOW_LEVEL_PARAMS:
            gray_wl = window_level(gray, **WINDOW_LEVEL_PARAMS)
            if np.isnan(gray_wl).any() or np.isinf(gray_wl).any(): raise ValueError("NaN/Inf after WL")
            # Prep for CLAHE: scale [0,1] to [0,255] uint8
            gray_uint8_for_clahe = np.clip(gray_wl * 255.0, 0, 255).astype(np.uint8)
        else:
            # Prep for CLAHE: scale original [0,255] to uint8
            gray_uint8_for_clahe = np.clip(gray, 0, 255).astype(np.uint8)

        # 4. CLAHE (operates on uint8, output is uint8)
        if USE_CLAHE:
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray_uint8_for_clahe)
                # Scale CLAHE output back to [0,1] float32
                gray = gray_clahe.astype(np.float32) / 255.0
            except Exception as e:
                raise ValueError(f"CLAHE Error: {e}")
        elif WINDOW_LEVEL_PARAMS:
            gray = gray_wl  # Use the [0,1] output from WL if CLAHE is off
        else:
            gray = gray / 255.0  # Scale original to [0,1] if WL and CLAHE are off

        # 5. Resize
        gray_resized = cv2.resize(gray, IMAGE_SIZE, interpolation=INTERPOLATION)

        # Check before normalization
        if np.isnan(gray_resized).any() or np.isinf(gray_resized).any(): raise ValueError("NaN/Inf before Norm")

        # 6. Normalize using CUSTOM stats (CRITICAL STEP)
        if CUSTOM_MEAN is None or CUSTOM_STD is None:
            raise ValueError("CUSTOM_MEAN and CUSTOM_STD must be set in server.py")
        std_dev = CUSTOM_STD if not np.isclose(CUSTOM_STD, 0) else EPSILON
        norm_gray = (gray_resized - CUSTOM_MEAN) / (std_dev + EPSILON)  # Add epsilon

        # Check after normalization
        if np.isnan(norm_gray).any() or np.isinf(norm_gray).any(): raise ValueError("NaN/Inf after Norm")

        # 7. Stack grayscale to 3 channels (H, W, C)
        norm_3c = np.stack([norm_gray] * 3, axis=-1)

        # 8. Convert to PyTorch Tensor (C, H, W) and add batch dimension (B, C, H, W)
        # Use torch.from_numpy for efficiency, ensure float32
        tensor = torch.from_numpy(norm_3c.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return tensor

    except Exception as e:
        # Catch any preprocessing error and raise a specific type if needed
        app.logger.error(f"Preprocessing failed: {e}")
        raise ValueError(f"Image preprocessing failed: {e}")


# --- End Utility functions ---

# --- Load Model (ResNet18 with Dropout) ---
model = None  # Initialize model variable
try:
    print("Creating server model architecture (ResNet18)...")
    model_instance = models.resnet18(weights=None)  # Load architecture
    num_features = model_instance.fc.in_features

    # IMPORTANT: Replicate the final layer structure used during training
    model_instance.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(num_features, NUM_CLASSES)
    )
    print(f"Architecture created for {NUM_CLASSES} classes with Dropout(p={DROPOUT_RATE}).")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading state dictionary from {model_path}...")
    # Load state dict onto the correct device directly
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    print("State dictionary loaded.")

    model_instance.to(device)  # Move model to device
    model_instance.eval()  # Set to evaluation mode
    model = model_instance  # Assign to the global 'model' variable
    print(f"✅ Model loaded successfully and set to eval mode on {device}.")

except FileNotFoundError as e:
    print(f"❌ FATAL: {e}")
    print("   Server cannot start without the model file.")
except Exception as e:
    # Catch all other exceptions during model loading
    print(f"❌ FATAL: An unexpected error occurred loading the model: {e}")
    model = None  # Ensure model remains None if loading fails


# --- End Load Model ---

# --- Predict Route ---
@app.route('/predict', methods=['POST'])
def predict():
    # Check if model loaded correctly

    if model is None:
        return jsonify(
            {"success": False, "message": "Model is not loaded or failed to load."}), 503  # Service Unavailable

    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image file found in the request."}), 400

    image_file = request.files['image']

    # Check if filename is present (basic check)
    if image_file.filename == '':
        return jsonify({"success": False, "message": "No selected file or empty filename."}), 400

    try:
        # Read image bytes directly from the file stream
        img_bytes = image_file.read()
        if not img_bytes:
            return jsonify({"success": False, "message": "Image file is empty."}), 400

        # Preprocess the image using the consistent function
        input_tensor = preprocess_image_for_inference(img_bytes).to(device)

        # Perform inference
        with torch.no_grad():  # Ensure no gradients are calculated
            output = model(input_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            # Get the predicted class index
            prediction_idx = torch.argmax(probabilities).item()
            # Get the predicted class label
            prediction = LABELS[prediction_idx]

            # *** Get the raw probability (0-1) of the PREDICTED class ***
            pred_probability = probabilities[prediction_idx].item()

        # Create dictionary of confidences for all classes
        confidences = {label: float(probabilities[i]) for i, label in enumerate(LABELS)}
        # Calculate percentage just for the description string if needed
        pred_confidence_percent = pred_probability * 100

        # *** Return successful prediction WITH the correct 'confidence' key ***
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidences": confidences,  # Map of all confidences
            "confidence": pred_probability, # <<< CORRECT KEY (probability 0-1) for frontend logic
            "source": "Our AI Engine",
            # Description can still include percentage for internal/logging use if needed
            "description": f"Classified as {prediction} ({pred_confidence_percent:.1f}% confidence)."
        })

    except ValueError as e:
        # Handle specific errors raised by preprocessing
        app.logger.error(f"Image processing error: {str(e)}")
        return jsonify({"success": False, "message": f"Error processing image: {str(e)}"}), 400
    except RuntimeError as e:
        # Handle potential CUDA errors during inference
        app.logger.error(f"Model inference runtime error: {str(e)}")
        return jsonify({"success": False, "message": "Error during model prediction."}), 500
    except Exception as e:
        # Catch any other unexpected errors
        app.logger.error(f"Unexpected prediction error: {str(e)}")
        # Consider logging traceback for debugging: app.logger.exception("Prediction failed")
        return jsonify({"success": False, "message": "An unexpected error occurred during prediction."}), 500


# --- End Predict Route ---

if __name__ == "__main__":
    # Check if model loaded before starting server
    if model is None:
        print("❌ Model failed to load. Server cannot currently start.")
    else:
        print(f"🚀 Starting Flask development server on http://0.0.0.0:5000...")
        # Use host='0.0.0.0' to make server accessible on network (use with caution)
        # use_reloader=True enables auto-restart on code changes (good for dev)
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)