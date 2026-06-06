"""
Download and extract the MovieLens 1M dataset.
Usage: python download_data.py
"""

import os
import zipfile
import requests

URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "ml-1m.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-1m")


def download(url: str, dest: str) -> None:
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%", end="", flush=True)
    print()


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        download(URL, ZIP_PATH)
    else:
        print(f"Archive already exists at {ZIP_PATH}, skipping download.")

    if not os.path.isdir(EXTRACT_DIR):
        print(f"Extracting to {DATA_DIR} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        print("Extraction complete.")
    else:
        print(f"Data already extracted at {EXTRACT_DIR}.")

    for fname in ("ratings.dat", "movies.dat", "users.dat"):
        path = os.path.join(EXTRACT_DIR, fname)
        if os.path.exists(path):
            print(f"  OK  {fname}")
        else:
            print(f"  MISSING  {fname}")

    print("\nDone. Run `python pipeline.py` to train the model.")


if __name__ == "__main__":
    main()
