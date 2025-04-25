import hashlib
import shutil
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "data_processed"
DEST_DIR = BASE_DIR / "cleaned_data"
LABELS = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'UNKNOWN']


def hash_file(path):
    """Returns a hash of the file content."""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def collect_train_hashes(train_dir):
    """Collect hashes of all train files."""
    print("🔍 Collecting hashes from training set...")
    hashes = set()
    for label in LABELS:
        for file in (train_dir / label).glob("*.npy"):
            hashes.add(hash_file(file))
    print(f"✅ Collected {len(hashes)} unique hashes from training set.")
    return hashes


def copy_clean_split(split, train_hashes):
    """Copy files from split (val/test/train) if they're not in train_hashes."""
    print(f"\n📦 Processing split: {split}")
    src_split_dir = SOURCE_DIR / split
    dst_split_dir = DEST_DIR / split
    copied = 0
    skipped = 0

    for label in LABELS:
        src_label_dir = src_split_dir / label
        dst_label_dir = dst_split_dir / label
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for file in src_label_dir.glob("*.npy"):
            file_hash = hash_file(file)
            if split == "train" or file_hash not in train_hashes:
                shutil.copy(file, dst_label_dir / file.name)
                copied += 1
            else:
                skipped += 1
    print(f"✅ Copied {copied} files, 🚫 Skipped {skipped} duplicates in '{split}'")


def main():
    train_dir = SOURCE_DIR / "train"
    DEST_DIR.mkdir(exist_ok=True)

    train_hashes = collect_train_hashes(train_dir)

    # Always copy training data completely
    copy_clean_split("train", train_hashes)
    copy_clean_split("val", train_hashes)
    copy_clean_split("test", train_hashes)

    print(f"\n🎉 Cleaning complete. Cleaned data saved to: {DEST_DIR}")


if __name__ == "__main__":
    main()
