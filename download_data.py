"""
Download the panoramic dental dataset from Kaggle.

Usage:
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    python download_data.py

The dataset is extracted to the path set by DATASET_ROOT in config.py
(default: ../dataset_root/ relative to this file).
"""

import os
import subprocess
import sys
import zipfile

import config


def main() -> None:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        sys.exit(
            "Error: set KAGGLE_USERNAME and KAGGLE_KEY as environment variables before running.\n"
            "Get a fresh API key at: https://www.kaggle.com/settings  (API -> Create New Token)"
        )

    dest = config.DATASET_ROOT
    images_dir = os.path.join(dest, "images_cut")

    if os.path.isdir(images_dir):
        print(f"Dataset already present at {dest}. Nothing to do.")
        return

    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "panoramic-dental-dataset.zip")

    print(f"Downloading dataset to {dest} ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", "thunderpede/panoramic-dental-dataset",
            "-p", dest,
        ],
        env={**os.environ, "KAGGLE_USERNAME": username, "KAGGLE_KEY": key},
        check=False,
    )

    if result.returncode != 0:
        sys.exit("kaggle download failed — check your credentials and try again.")

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)

    os.remove(zip_path)
    print(f"Done. Dataset available at {dest}")


if __name__ == "__main__":
    main()
