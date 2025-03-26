from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Set path to test image folder
TEST_IMAGE_PATH = Path("../data")  # or wherever your images are

# List image files (filter common image formats)
image_files = [f for f in TEST_IMAGE_PATH.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

print(f"Found {len(image_files)} images in {TEST_IMAGE_PATH}")

# Display first few images
def show_test_images(images, n=5):
    plt.figure(figsize=(15, 3))
    for i, img_file in enumerate(images[:n]):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(img_file.name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_test_images(image_files)
