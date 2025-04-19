# Located in: src/Model/normalize_all.py
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union
import sys
import traceback  # Import traceback for detailed error printing

import cv2
import numpy as np

# ======================================
# Configurable parameters
# ======================================
IMAGE_SIZE: Tuple[int, int] = (512, 512)
PREPROCESS_METHOD: str = "custom"
CUSTOM_MEAN: Optional[float] = None  # Will be calculated
CUSTOM_STD: Optional[float] = None  # Will be calculated
USE_CLAHE: bool = True
INTERPOLATION: int = cv2.INTER_AREA
WINDOW_LEVEL_PARAMS: Optional[dict] = {"wl": 600, "ww": 1500}
EPSILON: float = 1e-6

# ======================================
# Path Calculations (Defensive)
# ======================================
try:
    # Ensure __file__ is defined and resolve paths
    SCRIPT_DIR = Path(__file__).resolve().parent  # src/Model/
    SRC_DIR = SCRIPT_DIR.parent  # src/
    PROJECT_ROOT = SRC_DIR.parent  # xray_diagnosis_ML/
    DATA_DIR_INPUT = SRC_DIR / "data"  # src/data/
    DATA_DIR_OUTPUT = SRC_DIR / "data_processed"  # xray_diagnosis_ML/data_processed/
    PATHS_DEFINED = True
except NameError:
    # Fallback if __file__ is not available (e.g., interactive session)
    print("Warning: __file__ not defined. Using relative paths from current directory.")
    PROJECT_ROOT = Path(".").resolve()  # Use resolved current working directory
    DATA_DIR_INPUT = PROJECT_ROOT / "src" / "data"
    DATA_DIR_OUTPUT = PROJECT_ROOT / "data_processed"
    PATHS_DEFINED = False  # Indicate that paths might be relative


# ======================================
# Utility functions
# ======================================
def _apply_window_level(img: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Helper for window leveling, returns image scaled [0, 1]."""
    mn = wl - ww / 2.0
    mx = wl + ww / 2.0
    denominator = mx - mn
    if np.isclose(denominator, 0):
        # warnings.warn(f"WL width near zero ({denominator}). Clipping.", RuntimeWarning) # Less verbose
        return np.where(img >= wl, 1.0, 0.0).astype(np.float32)
    clipped = np.clip(img.astype(np.float32), mn, mx)
    return (clipped - mn) / (denominator + EPSILON)


def _apply_clahe(img_uint8: np.ndarray) -> np.ndarray:
    """Helper to apply CLAHE."""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_uint8)
    except Exception as e:
        # warnings.warn(f"CLAHE application failed: {e}", RuntimeWarning) # Less verbose
        raise e  # Re-raise to be caught by caller


def _resize_image(img: np.ndarray, size: Tuple[int, int], interpolation: int) -> np.ndarray:
    """Helper to resize image."""
    try:
        return cv2.resize(img, size, interpolation=interpolation)
    except Exception as e:
        # warnings.warn(f"Resize failed: {e}", RuntimeWarning) # Less verbose
        raise e  # Re-raise


def _normalize_custom(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Helper for custom normalization."""
    std_dev = std if not np.isclose(std, 0) else EPSILON
    return (img - mean) / (std_dev + EPSILON)


def compute_dataset_stats(
        input_data_dir: Path,
        split: str = "train",
        categories: Tuple[str, ...] = ("NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"),
) -> Optional[Tuple[float, float]]:
    """
    Computes mean and std deviation on the preprocessed pixels of the training set.
    Returns (mean, std) or None if calculation fails.
    """
    print(f"\n--- Computing stats on '{split}' split in '{input_data_dir}' for size {IMAGE_SIZE} ---")
    pixel_values_list: List[np.ndarray] = []
    processed_count = skipped_count = 0
    found_any_data = False

    for category in categories:
        category_path = input_data_dir / split / category
        if not category_path.exists():
            print(f"  Info: Skipping missing category '{category_path.name}' in '{split}'")
            continue

        print(f"  Processing Category: {category_path.name}")
        image_files = list(category_path.glob("*"))
        if not image_files:
            print(f"  Warning: No files found in {category_path}")
            continue
        found_any_data = True

        cat_processed = cat_skipped = 0
        for idx, img_path in enumerate(image_files, 1):
            # Basic file type check
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                cat_skipped += 1
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                cat_skipped += 1
                continue

            # Apply preprocessing steps sequentially to get the data for stats
            try:
                gray_float = img.astype(np.float32)

                if WINDOW_LEVEL_PARAMS:
                    processed_gray = _apply_window_level(gray_float, **WINDOW_LEVEL_PARAMS)  # Output is [0,1]
                    # Prep for CLAHE if used
                    gray_uint8_for_clahe = np.clip(processed_gray * 255.0, 0, 255).astype(np.uint8)
                else:
                    processed_gray = gray_float / 255.0  # Scale to [0,1] if no WL
                    gray_uint8_for_clahe = np.clip(gray_float, 0, 255).astype(np.uint8)  # Prep original for CLAHE

                if USE_CLAHE:
                    processed_gray_clahe = _apply_clahe(gray_uint8_for_clahe)
                    processed_gray = processed_gray_clahe.astype(np.float32) / 255.0  # Back to [0,1]

                processed_gray_resized = _resize_image(processed_gray, IMAGE_SIZE, INTERPOLATION)

                # Final check before adding pixels for stats
                if np.isnan(processed_gray_resized).any() or np.isinf(processed_gray_resized).any():
                    raise ValueError("NaN/Inf values after processing steps")

                # Add valid pixels (flattened array) to the list
                pixel_values_list.append(processed_gray_resized.flatten())
                cat_processed += 1

            except Exception as e:
                # Log error minimally during stats computation
                # print(f"    Skipping stats for {img_path.name} due to error: {e}")
                cat_skipped += 1

            # Optional: Progress indicator within category
            # if idx % 100 == 0: print(f"    {category}: Stats {idx}/{len(image_files)}...", end='\r')

        print(f"    {category}: Completed. Valid for stats = {cat_processed}, Skipped = {cat_skipped}.")
        processed_count += cat_processed
        skipped_count += cat_skipped

    # --- Final Stats Calculation ---
    if not found_any_data:
        print(f"❌ Error: No category directories found in {input_data_dir / split}.")
        return None
    if not pixel_values_list:
        print(f"❌ Error: No valid images processed in {input_data_dir / split} to calculate stats.")
        print(f"   (Processed={processed_count}, Skipped={skipped_count})")
        return None

    print(f"\n Calculating stats from {processed_count} images (skipped {skipped_count})...")
    try:
        all_pixels = np.concatenate(pixel_values_list)
        mean = float(np.mean(all_pixels))
        std = float(np.std(all_pixels))
        # Prevent zero standard deviation
        if np.isclose(std, 0, atol=EPSILON):
            warnings.warn(f"Calculated std dev is near zero ({std}). Using epsilon.", RuntimeWarning)
            std = EPSILON
        return mean, std
    except Exception as e:
        print(f"❌ Error during final stats calculation: {e}")
        return None


def process_and_save_image(
        img_path: Path,
        output_path: Path,
        method: str,
        mean_val: Optional[float],  # Renamed for clarity
        std_val: Optional[float]  # Renamed for clarity
) -> bool:
    """Processes a single image and saves it to the output path. Returns True on success."""
    img = cv2.imread(str(img_path))
    if img is None: return False
    try:
        gray_float = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if WINDOW_LEVEL_PARAMS:
            processed_gray = _apply_window_level(gray_float, **WINDOW_LEVEL_PARAMS)
            gray_uint8_for_clahe = np.clip(processed_gray * 255.0, 0, 255).astype(np.uint8)
        else:
            processed_gray = gray_float / 255.0
            gray_uint8_for_clahe = np.clip(gray_float, 0, 255).astype(np.uint8)

        if USE_CLAHE:
            processed_gray_clahe = _apply_clahe(gray_uint8_for_clahe)
            processed_gray = processed_gray_clahe.astype(np.float32) / 255.0

        processed_gray_resized = _resize_image(processed_gray, IMAGE_SIZE, INTERPOLATION)

        if np.isnan(processed_gray_resized).any() or np.isinf(processed_gray_resized).any():
            raise ValueError("NaN/Inf before normalization")

        if method == "custom":
            if mean_val is None or std_val is None: raise ValueError("Mean/std required for custom norm")
            normalized_gray = _normalize_custom(processed_gray_resized, mean_val, std_val)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        if np.isnan(normalized_gray).any() or np.isinf(normalized_gray).any():
            raise ValueError("NaN/Inf after normalization")

        # Stack to 3 channels (H, W, C format)
        final_image = np.stack([normalized_gray] * 3, axis=-1).astype(np.float32)

        # Save the processed image
        np.save(output_path, final_image)
        return True

    except Exception as e:
        # print(f"  [!] Error processing {img_path.name}: {e}") # Less verbose during mass processing
        return False


def run_normalization_pipeline(
        input_dir: Path,  # e.g., src/data
        output_dir: Path,  # e.g., data_processed
        method: str,
        mean_val: Optional[float],
        std_val: Optional[float]
) -> None:
    """Runs the normalization process for all splits and categories."""
    print(f"\n--- Running Normalization Pipeline ---")
    print(f"  Input Dir:  {input_dir}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Method: {method}, Size: {IMAGE_SIZE}")

    total_saved_count = total_skipped_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure base output exists

    for split in ("train", "val", "test"):
        split_input_path = input_dir / split
        split_output_path = output_dir / split
        print(f"\n Processing Split: '{split}'")

        if not split_input_path.exists():
            print(f"  Warning: Input directory not found, skipping split: {split_input_path}")
            continue

        split_output_path.mkdir(parents=True, exist_ok=True)
        split_saved = split_skipped = 0

        for category in ("NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"):
            category_input_path = split_input_path / category
            category_output_path = split_output_path / category

            if not category_input_path.exists():
                # print(f"  Info: Skipping missing category: {category_input_path.name}")
                continue

            category_output_path.mkdir(parents=True, exist_ok=True)
            image_files = list(category_input_path.glob("*"))
            print(f"  Category '{category}': Found {len(image_files)} potential files.")
            cat_saved = cat_skipped = 0

            for idx, img_path in enumerate(image_files, 1):
                # Basic file check
                if not img_path.is_file() or img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tif',
                                                                             '.tiff']:
                    cat_skipped += 1
                    continue

                output_npy_path = category_output_path / f"{img_path.stem}.npy"
                success = process_and_save_image(
                    img_path, output_npy_path, method, mean_val, std_val
                )
                if success:
                    cat_saved += 1
                else:
                    cat_skipped += 1

                # Progress indicator
                if idx % 100 == 0:
                    print(f"    {category}: Processed {idx}/{len(image_files)}...", end='\r')

            print(
                f"    {category}: Completed {len(image_files)}. Saved={cat_saved}, Skipped={cat_skipped}." + " " * 15)  # Clear line
            split_saved += cat_saved
            split_skipped += cat_skipped

        print(f" Split '{split}' summary: Saved={split_saved}, Skipped={split_skipped}")
        total_saved_count += split_saved
        total_skipped_count += split_skipped

    print("-" * 50)
    print(f"Normalization Pipeline Finished.")
    print(f"  Total Images Saved: {total_saved_count}")
    print(f"  Total Images Skipped: {total_skipped_count}")
    print(f"  Output Location: {output_dir}")
    print("-" * 50)


# ======================================
# Script entry point
# ======================================
if __name__ == "__main__":
    print("--- Image Normalization Script ---")
    if not PATHS_DEFINED:
        print("Warning: Using paths relative to current directory. Ensure you run from project root.")
    print(f"Project Root Dir: {PROJECT_ROOT}")
    print(f"Input Data Dir:   {DATA_DIR_INPUT}")
    print(f"Output Data Dir:  {DATA_DIR_OUTPUT}")

    final_mean = CUSTOM_MEAN
    final_std = CUSTOM_STD

    # --- Compute Statistics ---
    if PREPROCESS_METHOD == "custom":
        stats_result = compute_dataset_stats(DATA_DIR_INPUT, split="train")
        if stats_result:
            final_mean, final_std = stats_result
            print(f"\n📊 Calculated Stats (Size: {IMAGE_SIZE}):")
            print(f"  Mean: {final_mean:.6f}")
            print(f"  Std:  {final_std:.6f}\n")
            print("!! REMEMBER TO UPDATE THESE VALUES in other scripts !!")
        else:
            print("\n❌ Failed to compute dataset statistics. Cannot proceed with custom normalization.")
            exit(1)

    # --- Run Normalization ---
    try:
        run_normalization_pipeline(
            DATA_DIR_INPUT,
            DATA_DIR_OUTPUT,
            method=PREPROCESS_METHOD,
            mean_val=final_mean,
            std_val=final_std
        )
        print("\n✅ Normalization Script Finished Successfully.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the normalization pipeline:")
        traceback.print_exc()
        exit(1)
