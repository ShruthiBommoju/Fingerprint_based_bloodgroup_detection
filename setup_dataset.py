"""
STEP 1: Run this first to download and set up the dataset.

Usage:
    python setup_dataset.py

Requirements:
    - Kaggle account + API key (kaggle.json)
    - OR manually download from:
      https://www.kaggle.com/datasets/rajumavinmar/finger-prints-based-blood-group-dataset
      and place the zip in this folder.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

DATASET_DIR = Path("dataset")
BLOOD_GROUPS = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]


def download_via_kaggle_api():
    """Download using kaggle CLI."""
    print("Attempting download via Kaggle API...")
    print("Make sure kaggle.json is at ~/.kaggle/kaggle.json")
    ret = os.system(
        "kaggle datasets download -d rajumavinmar/finger-prints-based-blood-group-dataset "
        "--path . --unzip"
    )
    return ret == 0


def find_and_organize():
    """Find downloaded images and organize into dataset/ folder."""
    print("\nLooking for downloaded images...")

    # Common locations after unzip
    search_roots = [Path("."), Path("finger-prints-based-blood-group-dataset")]
    found_any = False

    for root in search_roots:
        if not root.exists():
            continue
        for bg in BLOOD_GROUPS:
            src = root / bg
            if src.exists() and any(src.iterdir()):
                dst = DATASET_DIR / bg
                dst.mkdir(parents=True, exist_ok=True)
                count = 0
                for img in src.iterdir():
                    if img.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                        shutil.copy2(img, dst / img.name)
                        count += 1
                print(f"  {bg}: {count} images → dataset/{bg}/")
                found_any = True

    return found_any


def manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           MANUAL DATASET SETUP INSTRUCTIONS                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Go to this URL:                                          ║
║     https://www.kaggle.com/datasets/                         ║
║     rajumavinmar/finger-prints-based-blood-group-dataset     ║
║                                                              ║
║  2. Click "Download" (you need a free Kaggle account)        ║
║                                                              ║
║  3. Extract the zip file                                     ║
║                                                              ║
║  4. Place the folders like this:                             ║
║     fingerprint_blood_group/                                 ║
║     └── dataset/                                             ║
║         ├── A-/    (images here)                             ║
║         ├── A+/                                              ║
║         ├── AB-/                                             ║
║         ├── AB+/                                             ║
║         ├── B-/                                              ║
║         ├── B+/                                              ║
║         ├── O-/                                              ║
║         └── O+/                                              ║
║                                                              ║
║  5. Then run: python train.py                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def verify_dataset():
    """Check dataset is ready."""
    print("\n📊 Dataset Summary:")
    print("─" * 40)
    total = 0
    all_good = True
    for bg in BLOOD_GROUPS:
        folder = DATASET_DIR / bg
        if folder.exists():
            imgs = [f for f in folder.iterdir()
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']]
            count = len(imgs)
            total += count
            status = "✅" if count > 50 else "⚠️ (low)"
            print(f"  {bg:<6}: {count:>4} images  {status}")
            if count == 0:
                all_good = False
        else:
            print(f"  {bg:<6}:    0 images  ❌ MISSING")
            all_good = False
    print("─" * 40)
    print(f"  Total : {total} images")

    if all_good:
        print("\n✅ Dataset ready! Run: python train.py")
    else:
        print("\n❌ Dataset incomplete. Follow instructions above.")
    return all_good


if __name__ == "__main__":
    print("=" * 60)
    print("  Fingerprint Blood Group — Dataset Setup")
    print("=" * 60)

    DATASET_DIR.mkdir(exist_ok=True)

    # Check if already set up
    existing = sum(
        len(list((DATASET_DIR / bg).glob("*")))
        for bg in BLOOD_GROUPS
        if (DATASET_DIR / bg).exists()
    )
    if existing > 100:
        print(f"Dataset already found ({existing} files).")
        verify_dataset()
        sys.exit(0)

    # Try Kaggle API
    try:
        import kaggle
        success = download_via_kaggle_api()
        if success:
            find_and_organize()
    except Exception as e:
        print(f"Kaggle API not available: {e}")
        print("Falling back to manual setup instructions...")

    # Check zip in current dir
    zips = list(Path(".").glob("*.zip"))
    if zips:
        print(f"\nFound zip: {zips[0]} — extracting...")
        with zipfile.ZipFile(zips[0], 'r') as z:
            z.extractall(".")
        find_and_organize()

    if not verify_dataset():
        manual_instructions()
