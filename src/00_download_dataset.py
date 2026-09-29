"""
STEP 0: Download PTB-XL v1.0.3 from PhysioNet into data/raw/ptbxl (~3 GB zipped).

Skip this if you already have the dataset; just point `data.raw_dir` in
configs/config.yaml at it (or symlink it to data/raw/ptbxl).

Usage:
  PYTHONPATH=. python3 src/00_download_dataset.py
"""

import os
import shutil
import urllib.request
import zipfile

from src.data import load_config

URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"


def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    if os.path.exists(config["data"]["database_csv"]):
        print(f"PTB-XL already present at {raw_dir}")
        return

    parent = os.path.dirname(raw_dir.rstrip("/"))
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, "ptbxl.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading {URL}\n  -> {zip_path}")
        urllib.request.urlretrieve(URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        top = zf.namelist()[0].split("/")[0]
        zf.extractall(parent)
    shutil.move(os.path.join(parent, top), raw_dir)
    os.remove(zip_path)
    print(f"Done: {raw_dir}")


if __name__ == "__main__":
    main()
