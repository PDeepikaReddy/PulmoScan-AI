"""
Download the Tuberculosis (TB) Chest X-ray dataset from Kaggle.

Prerequisites
-------------
1. Create a free Kaggle account: https://www.kaggle.com
2. Account -> Settings -> "Create New API Token". This downloads `kaggle.json`.
3. Place `kaggle.json` at:
       Windows : C:\\Users\\<you>\\.kaggle\\kaggle.json
       Linux/Mac: ~/.kaggle/kaggle.json
   (This script will also pick it up from the project folder if present.)

Then run:
    python download_data.py
"""
import os
import shutil
import sys
import zipfile

import config


def _ensure_credentials():
    """Make the Kaggle API find kaggle.json from a few common locations."""
    home_kaggle = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    local_kaggle = os.path.join(config.BASE_DIR, "kaggle.json")

    if os.path.exists(home_kaggle):
        return True
    if os.path.exists(local_kaggle):
        os.makedirs(os.path.dirname(home_kaggle), exist_ok=True)
        shutil.copy(local_kaggle, home_kaggle)
        print(f"Copied kaggle.json -> {home_kaggle}")
        return True
    return False


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not _ensure_credentials():
        print(
            "ERROR: kaggle.json not found.\n"
            "  1. Go to https://www.kaggle.com/settings -> 'Create New API Token'\n"
            "  2. Save the downloaded kaggle.json into either:\n"
            f"       {os.path.join(os.path.expanduser('~'), '.kaggle')}\n"
            f"       {config.BASE_DIR}  (this project folder)\n"
            "  3. Re-run:  python download_data.py"
        )
        sys.exit(1)

    # Import after credentials are in place (kaggle authenticates on import).
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {config.KAGGLE_DATASET}")
    api.dataset_download_files(
        config.KAGGLE_DATASET, path=config.DATA_DIR, unzip=True, quiet=False
    )

    if os.path.isdir(config.TRAIN_DIR):
        normal = os.path.join(config.TRAIN_DIR, "Normal")
        tb = os.path.join(config.TRAIN_DIR, "Tuberculosis")
        n_normal = len(os.listdir(normal)) if os.path.isdir(normal) else 0
        n_tb = len(os.listdir(tb)) if os.path.isdir(tb) else 0
        print("\nDownload complete.")
        print(f"  Normal images       : {n_normal}")
        print(f"  Tuberculosis images : {n_tb}")
        print(f"  Location            : {config.TRAIN_DIR}")
        print("\nNext step:  python train.py")
    else:
        print(
            "\nDownloaded, but expected folder not found at:\n"
            f"  {config.TRAIN_DIR}\n"
            f"Inspect {config.DATA_DIR} and update config.TRAIN_DIR if the "
            "folder name differs."
        )


if __name__ == "__main__":
    main()
