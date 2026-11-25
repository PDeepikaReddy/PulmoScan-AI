"""
Download the Tuberculosis chest X-ray dataset from a PUBLIC GitHub mirror
(no Kaggle account/credentials required).

Source mirror:
  https://github.com/sorna-fast/tb-chest-xray-classifier
  -> datasets/.../TB_Chest_Radiography_Database/{Normal,Tuberculosis}

The images are the standard `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`
(Normal + Tuberculosis). This script grabs a balanced subset so a model can be
trained quickly on a CPU. Run a wider range for the full dataset.

Usage:
    python fetch_github_dataset.py            # default balanced subset
    python fetch_github_dataset.py 1 3500     # wider index range
"""
import os
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

ssl._create_default_https_context = ssl._create_unverified_context

RAW = (
    "https://raw.githubusercontent.com/sorna-fast/tb-chest-xray-classifier/"
    "master/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/"
    "versions/3/TB_Chest_Radiography_Database"
)


def _download(cls, idx):
    fname = f"{cls}-{idx}.png"
    dest_dir = os.path.join(config.TRAIN_DIR, cls)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    url = f"{RAW}/{cls}/{fname}"
    try:
        urllib.request.urlretrieve(url, dest)
        return os.path.getsize(dest) > 0
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return False


def fetch_range(start, end, workers=24):
    counts = {"Normal": 0, "Tuberculosis": 0}
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for cls in ("Normal", "Tuberculosis"):
            for i in range(start, end + 1):
                tasks.append((cls, ex.submit(_download, cls, i)))
        for cls, fut in tasks:
            if fut.result():
                counts[cls] += 1
    return counts


def main():
    # Default: indices 101..760 -> ~660 per class (skips low indices kept as a
    # clean holdout for evaluation). Override via CLI args.
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 760
    print(f"Downloading Normal & Tuberculosis images, index {start}..{end} ...")
    counts = fetch_range(start, end)
    print(f"  Normal       : {counts['Normal']}")
    print(f"  Tuberculosis : {counts['Tuberculosis']}")
    print(f"  Saved under  : {config.TRAIN_DIR}")
    print("\nNext step:  python train.py")


if __name__ == "__main__":
    main()
