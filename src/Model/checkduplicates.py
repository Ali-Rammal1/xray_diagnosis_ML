import hashlib
from pathlib import Path

# Set base path relative to this script (assumed inside /src/)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "cleaned_data"


def hash_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def find_duplicates(data_dir):
    hashes = {}
    data_dir = Path(data_dir)
    for split in ['train', 'val', 'test']:
        split_path = data_dir / split
        if not split_path.exists():
            continue
        for file in split_path.rglob("*.npy"):
            file_hash = hash_file(file)
            if file_hash in hashes:
                print(f"🔁 Duplicate found:\n  {file}\n  == {hashes[file_hash]}\n")
            else:
                hashes[file_hash] = file
    print("✅ Duplicate check complete.")


if __name__ == "__main__":
    print("Checking for duplicates in:", DATA_PROCESSED_DIR)
    find_duplicates(DATA_PROCESSED_DIR)
