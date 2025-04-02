import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
from torchvision import models
import os

class XRayClassifierApp:
    # Constants
    IMAGE_SIZE = (224, 224)
    DISPLAY_SIZE = (400, 400)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    LABELS = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'UNKNOWN']
    LABEL_COLORS = {
        'NORMAL': '#4CAF50',  # Green
        'PNEUMONIA': '#F44336',  # Red
        'TUBERCULOSIS': '#FF9800',  # Orange
        'UNKNOWN': '#9E9E9E'  # Gray
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("X-Ray Diagnostic Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Set icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
            
        # Setup paths
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.model_path = self.base_dir / "best_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model loading flag
        self.model_loaded = False
        self.model = None
        self.current_image = None
        self.current_image_path = None
        
        self.setup_ui()
        
        # Start loading the model in a separate thread
        self.status_label.config(text="Status: Loading model...")
        threading.Thread(target=self.load_model_thread).start()
    
    def setup_ui(self):
        # Create frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title and description
        title_label = ttk.Label(
            self.main_frame, 
            text="X-Ray Diagnostic Assistant", 
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        description = ttk.Label(
            self.main_frame,
            text="Upload a chest X-ray image to detect potential respiratory conditions",
            font=("Arial", 10)
        )
        description.pack(pady=(0, 20))
        
        # Image area
        self.image_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="groove")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.placeholder_text = ttk.Label(
            self.image_frame,
            text="No image selected\nClick 'Upload X-ray' to begin",
            font=("Arial", 12),
            justify=tk.CENTER
        )
        self.placeholder_text.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(self.image_frame)
        
        # Control panel
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Upload button with styling
        style = ttk.Style()
        style.configure("Upload.TButton", font=("Arial", 11, "bold"))
        
        self.upload_button = ttk.Button(
            control_frame,
            text="Upload X-ray",
            command=self.upload_image,
            style="Upload.TButton"
        )
        self.upload_button.pack(side=tk.LEFT, padx=5)
        
        # Result panel
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Analysis Results")
        self.results_frame.pack(fill=tk.X, pady=10)
        
        # Results row
        results_inner_frame = ttk.Frame(self.results_frame)
        results_inner_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Status label
        self.status_label = ttk.Label(
            results_inner_frame,
            text="Status: Ready",
            font=("Arial", 10)
        )
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Prediction label
        self.prediction_label = ttk.Label(
            results_inner_frame,
            text="Prediction: None",
            font=("Arial", 12, "bold")
        )
        self.prediction_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Confidence bars frame
        self.confidence_frame = ttk.Frame(results_inner_frame)
        self.confidence_frame.grid(row=2, column=0, sticky=tk.EW, pady=10)
        
        # Create confidence bars for each label
        self.confidence_bars = {}
        self.confidence_labels = {}
        
        for i, label in enumerate(self.LABELS):
            # Label name
            lbl = ttk.Label(self.confidence_frame, text=f"{label}:", width=12, anchor=tk.W)
            lbl.grid(row=i, column=0, sticky=tk.W, pady=2)
            
            # Progress bar for confidence
            prog = ttk.Progressbar(self.confidence_frame, length=300, mode='determinate')
            prog.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            
            # Percentage label
            val_lbl = ttk.Label(self.confidence_frame, text="0%", width=6)
            val_lbl.grid(row=i, column=2, sticky=tk.W, pady=2)
            
            self.confidence_bars[label] = prog
            self.confidence_labels[label] = val_lbl
        
        # Footer
        footer_frame = ttk.Frame(self.main_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        disclaimer = ttk.Label(
            footer_frame,
            text="Disclaimer: This is a prototype tool and should not replace professional medical diagnosis.",
            font=("Arial", 8),
            foreground="gray"
        )
        disclaimer.pack(side=tk.LEFT)
        
        # Info about device
        device_info = ttk.Label(
            footer_frame,
            text=f"Using: {self.device}",
            font=("Arial", 8),
            foreground="gray"
        )
        device_info.pack(side=tk.RIGHT)
    
    def load_model_thread(self):
        try:
            # Create model architecture
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(self.LABELS))
            
            # Load weights
            try:
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.model = model
                self.model_loaded = True
                self.root.after(0, lambda: self.status_label.config(text=f"Status: Model loaded successfully (ResNet18)"))
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                self.root.after(0, lambda: self.status_label.config(text=f"Status: {error_msg}"))
                print(error_msg)
        except Exception as e:
            error_msg = f"Error initializing model: {str(e)}"
            self.root.after(0, lambda: self.status_label.config(text=f"Status: {error_msg}"))
            print(error_msg)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an X-ray image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
            
        self.current_image_path = file_path
        self.load_and_display_image(file_path)
        
        if self.model_loaded:
            self.analyze_image()
        else:
            self.status_label.config(text="Status: Model not loaded yet. Please wait.")
    
    def load_and_display_image(self, image_path):
        try:
            # Load image with OpenCV for analysis
            self.current_image = cv2.imread(image_path)
            
            # Load image with PIL for display
            display_image = Image.open(image_path)
            
            # Resize while maintaining aspect ratio
            display_image.thumbnail((self.DISPLAY_SIZE[0], self.DISPLAY_SIZE[1]))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Update display
            if self.placeholder_text.winfo_ismapped():
                self.placeholder_text.pack_forget()
                self.image_label.pack(fill=tk.BOTH, expand=True)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Update status
            self.status_label.config(text=f"Status: Image loaded from {Path(image_path).name}")
            
        except Exception as e:
            self.status_label.config(text=f"Status: Error loading image - {str(e)}")
    
    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
        return img_tensor
    
    def analyze_image(self):
        if not self.current_image is not None:
            self.status_label.config(text="Status: No image to analyze")
            return
            
        self.status_label.config(text="Status: Analyzing image...")
        
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(self.current_image).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # Get probabilities using softmax
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Get predicted class
                predicted_idx = torch.argmax(output, dim=1).item()
                prediction = self.LABELS[predicted_idx]
                
                # Update prediction label with color
                self.prediction_label.config(
                    text=f"Prediction: {prediction}",
                    foreground=self.LABEL_COLORS.get(prediction, "black")
                )
                
                # Update confidence bars
                for i, label in enumerate(self.LABELS):
                    confidence = probabilities[i].item() * 100
                    self.confidence_bars[label]['value'] = confidence
                    self.confidence_labels[label].config(text=f"{confidence:.1f}%")
                
                self.status_label.config(text=f"Status: Analysis complete")
                
        except Exception as e:
            self.status_label.config(text=f"Status: Error during analysis - {str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = XRayClassifierApp(root)
    root.mainloop()