import cv2
import numpy as np
from pathlib import Path

# Go up one level from src/ to reach data/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_processed"

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def preprocess_image(img, method='imagenet'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    if method == 'imagenet':
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img

def normalize_all_images(data_dir, output_dir, method='imagenet'):
    for split in ['train', 'val', 'test']:
        split_path = data_dir / split
        for category in ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'UNKNOWN']:
            category_path = split_path / category
            if not category_path.exists():
                print(f"Skipping missing folder: {category_path}")
                continue

            output_category_path = output_dir / split / category
            output_category_path.mkdir(parents=True, exist_ok=True)

            image_files = list(category_path.glob("*"))
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: couldn't read {img_path}")
                    continue

                norm_img = preprocess_image(img, method=method)
                output_file = output_category_path / (img_path.stem + ".npy")
                np.save(output_file, norm_img)
    
    print("All images normalized and saved to:", output_dir)

# Entry point
if __name__ == "__main__":
    normalize_all_images(DATA_DIR, OUTPUT_DIR)
