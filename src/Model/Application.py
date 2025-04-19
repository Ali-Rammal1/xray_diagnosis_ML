# Located in: src/Model/Application.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox  # Import messagebox
from PIL import Image, ImageTk
import threading
from torchvision import models
import os
import warnings
from typing import Union, Optional  # Need these for type hints

# --- Constants - MUST MATCH normalize_all.py & training ---
IMAGE_SIZE = (512, 512)  # Match the size used in normalize_all.py
DISPLAY_SIZE = (400, 400)  # Keep display size manageable for UI
LABELS = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'UNKNOWN']
LABEL_COLORS = {'NORMAL': '#4CAF50', 'PNEUMONIA': '#F44336', 'TUBERCULOSIS': '#FF9800', 'UNKNOWN': '#9E9E9E'}
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
    warnings.warn("!!! APPLICATION: PLACEHOLDER custom mean/std detected. Replace with actual values! !!!")


# --- End Check ---


class XRayClassifierApp:

    def __init__(self, root):
        self.root = root
        self.root.title("X-Ray Diagnostic Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # --- Paths based on script location src/Model/ ---
        self.base_dir = Path(__file__).resolve().parent.parent  # src/
        # Model loaded from src directory (where data_loading.py saved it)
        self.model_path = self.base_dir / "best_model.pth"  # src/best_model.pth
        print(f"App - Script Base Dir (src/): {self.base_dir}")
        print(f"App - Loading model from: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"App - Using device: {self.device}")

        self.model_loaded = False
        self.model = None
        self.current_image_display = None  # For Tkinter display
        self.current_image_cv2 = None  # For OpenCV processing

        # Setup UI elements
        self.setup_ui()

        # Start loading the model in a background thread
        self.status_label.config(text="Status: Initializing model...")
        # Use daemon=True so thread exits if main app exits
        threading.Thread(target=self.load_model_thread, daemon=True).start()

    @staticmethod
    def window_level(img: np.ndarray, wl: float, ww: float) -> np.ndarray:
        """Static method for window leveling, output [0,1]"""
        mn = wl - ww / 2.0
        mx = wl + ww / 2.0
        denominator = mx - mn
        if np.isclose(denominator, 0): return np.where(img >= wl, 1.0, 0.0).astype(np.float32)
        img_clipped = np.clip(img.astype(np.float32), mn, mx)
        return (img_clipped - mn) / (denominator + XRayClassifierApp.EPSILON)

    def setup_ui(self):
        """Creates the user interface elements."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(self.main_frame, text="X-Ray Diagnostic Assistant", font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 10))

        # Description
        desc = ttk.Label(self.main_frame, text="Upload a chest X-ray image for analysis", font=("Arial", 10))
        desc.pack(pady=(0, 20))

        # Image Display Area
        self.image_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="groove", width=self.DISPLAY_SIZE[0],
                                     height=self.DISPLAY_SIZE[1])
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)  # Prevent frame resizing to content

        self.placeholder_text = ttk.Label(self.image_frame, text="No image selected\nClick 'Upload X-ray' to begin",
                                          font=("Arial", 12), justify=tk.CENTER)
        self.placeholder_text.pack(fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(self.image_frame)  # Label to hold the image

        # Control Panel (Upload Button)
        ctrl_frame = ttk.Frame(self.main_frame)
        ctrl_frame.pack(fill=tk.X, pady=10)
        style = ttk.Style()
        style.configure("Upload.TButton", font=("Arial", 11, "bold"), padding=6)
        self.upload_button = ttk.Button(ctrl_frame, text="Upload X-ray", command=self.upload_image,
                                        style="Upload.TButton")
        self.upload_button.pack(side=tk.LEFT, padx=5)

        # Results Panel
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Analysis Results")
        self.results_frame.pack(fill=tk.X, pady=10, padx=5)
        res_inner = ttk.Frame(self.results_frame)
        res_inner.pack(fill=tk.X, padx=10, pady=10)

        # Status Label
        self.status_label = ttk.Label(res_inner, text="Status: Ready", font=("Arial", 10))
        self.status_label.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        # Prediction Label
        self.prediction_label = ttk.Label(res_inner, text="Prediction: None", font=("Arial", 12, "bold"))
        self.prediction_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Confidence Bars Area
        self.confidence_frame = ttk.Frame(res_inner)
        self.confidence_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=10)
        self.confidence_bars = {}
        self.confidence_labels = {}
        for i, label in enumerate(self.LABELS):
            # Class Name Label
            lbl = ttk.Label(self.confidence_frame, text=f"{label}:", width=12, anchor=tk.W)
            lbl.grid(row=i, column=0, sticky=tk.W, pady=2, padx=(0, 5))
            # Progress Bar
            prog = ttk.Progressbar(self.confidence_frame, length=300, mode='determinate', maximum=100)
            prog.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            self.confidence_frame.grid_columnconfigure(1, weight=1)  # Make progress bar expand
            # Percentage Value Label
            val_lbl = ttk.Label(self.confidence_frame, text="0.0%", width=7, anchor=tk.E)
            val_lbl.grid(row=i, column=2, sticky=tk.E, pady=2, padx=(5, 0))
            self.confidence_bars[label] = prog
            self.confidence_labels[label] = val_lbl

        # Footer
        footer = ttk.Frame(self.main_frame)
        footer.pack(fill=tk.X, pady=(10, 0), side=tk.BOTTOM)
        disclaimer = ttk.Label(footer, text="Disclaimer: Educational prototype only. Not for medical diagnosis.",
                               font=("Arial", 8), foreground="gray")
        disclaimer.pack(side=tk.LEFT)
        dev_info = ttk.Label(footer, text=f"Using: {self.device}", font=("Arial", 8), foreground="gray")
        dev_info.pack(side=tk.RIGHT)

    def load_model_thread(self):
        """Loads the model in a background thread."""
        try:
            print(f"App: Background thread starting model load...")
            # Create the same architecture as used in training
            model_instance = models.resnet18(weights=None)
            num_features = model_instance.fc.in_features
            # IMPORTANT: Replicate exact final layer structure from training
            model_instance.fc = nn.Sequential(
                nn.Dropout(p=DROPOUT_RATE),
                nn.Linear(num_features, NUM_CLASSES)
            )
            print(f"App: Model architecture created.")

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            print(f"App: Loading state dict from {self.model_path}")
            model_instance.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"App: State dict loaded.")

            model_instance.to(self.device)
            model_instance.eval()  # Set to evaluation mode
            self.model = model_instance
            self.model_loaded = True
            # Update UI from the main thread using root.after
            self.root.after(0, lambda: self.status_label.config(text=f"Status: Model ready on {self.device}"))
            print(f"App: Model loaded successfully.")

        except FileNotFoundError as e:
            error_msg = f"Status: Model file not found!"
            print(f"App Error: {e}")
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
            self.root.after(0, lambda: messagebox.showerror("Model Load Error",
                                                            f"{e}\nPlease ensure 'best_model.pth' exists in the 'src' directory."))
        except Exception as e:
            error_msg = f"Status: Error loading model!"
            # Print detailed error to console for debugging
            import traceback
            print(f"App Error loading model:\n{traceback.format_exc()}")
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
            # Show error popup to user
            self.root.after(0, lambda: messagebox.showerror("Model Load Error",
                                                            f"An error occurred loading the model:\n{e}"))

    def upload_image(self):
        """Handles the image upload process."""
        # Define allowed file types
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        file_path = filedialog.askopenfilename(title="Select X-ray Image", filetypes=filetypes)

        # If user cancels dialog
        if not file_path:
            return

        # Reset previous results
        self.prediction_label.config(text="Prediction: None", foreground="black")
        for label in self.LABELS:
            self.confidence_bars[label]['value'] = 0
            self.confidence_labels[label].config(text="0.0%")

        # Load and display the selected image
        self.load_and_display_image(file_path)

        # If image loaded successfully, try to analyze
        if self.current_image_cv2 is not None:
            if self.model_loaded:
                self.analyze_image()
            else:
                self.status_label.config(text="Status: Model still loading, please wait...")
        else:
            # Error message handled in load_and_display_image
            pass

    def load_and_display_image(self, image_path_str: str):
        """Loads image using OpenCV for processing and PIL for display."""
        try:
            image_path = Path(image_path_str)
            # Load with OpenCV (BGR format) for processing consistency
            self.current_image_cv2 = cv2.imread(str(image_path))
            if self.current_image_cv2 is None:
                raise IOError(f"OpenCV could not read image: {image_path.name}")

            # Load with PIL for Tkinter display (handles various formats well)
            display_image_pil = Image.open(image_path)
            # Resize for display (thumbnail preserves aspect ratio)
            display_image_pil.thumbnail(self.DISPLAY_SIZE)
            self.current_image_display = ImageTk.PhotoImage(display_image_pil)

            # Update the image label in the UI
            if self.placeholder_text.winfo_ismapped():
                self.placeholder_text.pack_forget()  # Hide placeholder
                self.image_label.pack(fill=tk.BOTH, expand=True)  # Show image label

            self.image_label.config(image=self.current_image_display)
            # Keep a reference to prevent garbage collection
            self.image_label.image = self.current_image_display

            self.status_label.config(text=f"Status: Loaded: {image_path.name}")

        except Exception as e:
            error_msg = f"Status: Error loading image!"
            print(f"Error loading/displaying {image_path_str}: {e}")
            self.status_label.config(text=error_msg)
            messagebox.showerror("Image Load Error", f"Could not load or display image:\n{e}")
            # Reset image variables
            self.current_image_cv2 = None
            self.current_image_display = None
            # Clear image display area
            self.image_label.pack_forget()
            self.placeholder_text.pack(fill=tk.BOTH, expand=True)  # Show placeholder again

    def preprocess_image(self, img_cv2: np.ndarray) -> torch.Tensor:
        """
        Preprocesses the loaded OpenCV image (BGR) to match training.
        Returns a Tensor ready for the model.
        """
        # --- Uses the exact same logic as server.py's preprocess ---
        try:
            # 1. To Grayscale Float32
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # 2. Window/Level (outputs [0,1])
            if self.WINDOW_LEVEL_PARAMS:
                gray_wl = self.window_level(gray, **self.WINDOW_LEVEL_PARAMS)
                if np.isnan(gray_wl).any() or np.isinf(gray_wl).any(): raise ValueError("NaN/Inf after WL")
                # Prep for CLAHE: scale [0,1] to [0,255] uint8
                gray_uint8_for_clahe = np.clip(gray_wl * 255.0, 0, 255).astype(np.uint8)
            else:
                # Prep for CLAHE: scale original [0,255] to uint8
                gray_uint8_for_clahe = np.clip(gray, 0, 255).astype(np.uint8)

            # 3. CLAHE (operates on uint8, output is uint8)
            if self.USE_CLAHE:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray_uint8_for_clahe)
                # Scale CLAHE output back to [0,1] float32
                gray = gray_clahe.astype(np.float32) / 255.0
            elif self.WINDOW_LEVEL_PARAMS:
                gray = gray_wl  # Use the [0,1] output from WL if CLAHE is off
            else:
                gray = gray / 255.0  # Scale original to [0,1] if WL and CLAHE are off

            # 4. Resize
            gray_resized = cv2.resize(gray, self.IMAGE_SIZE, interpolation=self.INTERPOLATION)

            # Check before normalization
            if np.isnan(gray_resized).any() or np.isinf(gray_resized).any(): raise ValueError("NaN/Inf before Norm")

            # 5. Normalize using CUSTOM stats
            if self.CUSTOM_MEAN is None or self.CUSTOM_STD is None:
                raise ValueError("CUSTOM_MEAN/STD values are not set in the application!")
            std_dev = self.CUSTOM_STD if not np.isclose(self.CUSTOM_STD, 0) else self.EPSILON
            norm_gray = (gray_resized - self.CUSTOM_MEAN) / (std_dev + self.EPSILON)

            # Check after normalization
            if np.isnan(norm_gray).any() or np.isinf(norm_gray).any(): raise ValueError("NaN/Inf after Norm")

            # 6. Stack grayscale to 3 channels (H, W, C)
            norm_3c = np.stack([norm_gray] * 3, axis=-1)

            # 7. Convert to PyTorch Tensor (C, H, W) and add batch dimension (B, C, H, W)
            tensor = torch.from_numpy(norm_3c.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            return tensor

        except Exception as e:
            # Raise the error to be caught in analyze_image
            raise ValueError(f"Preprocessing failed: {e}")

    def analyze_image(self):
        """Performs model inference on the current image."""
        if not self.model_loaded:
            self.status_label.config(text="Status: Model not ready.")
            messagebox.showwarning("Model Not Ready", "The analysis model is still loading. Please wait.")
            return
        if self.current_image_cv2 is None:
            self.status_label.config(text="Status: No valid image loaded.")
            messagebox.showerror("No Image", "Please upload a valid image first.")
            return

        self.status_label.config(text="Status: Analyzing image...")
        self.root.update_idletasks()  # Update UI to show "Analyzing..."

        try:
            # Preprocess the image using the instance method
            input_tensor = self.preprocess_image(self.current_image_cv2).to(self.device)

            # Perform inference
            with torch.no_grad():  # Disable gradients for inference
                output = self.model(input_tensor)
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                # Get predicted class index
                predicted_idx = torch.argmax(probabilities).item()
                prediction = self.LABELS[predicted_idx]

            # Update UI with results
            self.prediction_label.config(text=f"Prediction: {prediction}",
                                         foreground=self.LABEL_COLORS.get(prediction, "black"))

            # Update confidence bars and labels
            for i, label in enumerate(self.LABELS):
                confidence = probabilities[i].item() * 100
                self.confidence_bars[label]['value'] = confidence
                self.confidence_labels[label].config(text=f"{confidence:.1f}%")

            self.status_label.config(text="Status: Analysis complete")

        except ValueError as e:  # Catch specific preprocessing errors
            error_msg = f"Status: Preprocessing Error!"
            print(f"Preprocessing Error during analysis: {e}")
            self.status_label.config(text=error_msg)
            messagebox.showerror("Processing Error", f"Failed to preprocess image:\n{e}")
        except Exception as e:
            # Catch other errors during analysis (e.g., model runtime error)
            error_msg = f"Status: Analysis Error!"
            import traceback
            print(f"Analysis Error:\n{traceback.format_exc()}")
            self.status_label.config(text=error_msg)
            messagebox.showerror("Analysis Error", f"An unexpected error occurred during analysis:\n{e}")


# --- Main execution block ---
if __name__ == "__main__":
    # Basic check for placeholder constants
    if CUSTOM_MEAN == 0.491234 or CUSTOM_STD == 0.256789:
        print("=" * 60)
        print(" WARNING: Using placeholder CUSTOM_MEAN/CUSTOM_STD values! ")
        print("          Replace these in Application.py with the actual values")
        print("          calculated by normalize_all.py for accurate results.")
        print("=" * 60)

    # Create the main Tkinter window and run the app
    root = tk.Tk()
    app = XRayClassifierApp(root)
    root.mainloop()
