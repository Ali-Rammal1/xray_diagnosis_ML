import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp" / "test"
LABELS = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']  # skip UNKNOWN

def hash_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def remove_duplicates_in_temp():
    print(f"🔍 Scanning merged folder: {TEMP_DIR}")
    seen_hashes = {}
    removed = 0

    for label in LABELS:
        folder = TEMP_DIR / label
        if not folder.exists():
            print(f"⚠️ Skipping missing folder: {folder}")
            continue

        for file in folder.glob("*.npy"):
            h = hash_file(file)
            if h in seen_hashes:
                print(f"🗑️ Removing duplicate: {file.name} (same as {seen_hashes[h].name})")
                file.unlink()
                removed += 1
            else:
                seen_hashes[h] = file

    print(f"\n✅ Finished scanning. Removed {removed} duplicate(s) from temp/test.")

if __name__ == "__main__":
    remove_duplicates_in_temp()
