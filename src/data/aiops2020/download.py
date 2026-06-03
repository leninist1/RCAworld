"""Download AIOps Challenge 2020 dataset.

Sources:
- Tsinghua Cloud: https://cloud.tsinghua.edu.cn/f/c1ea3426ce444bc9baae/
- Google Drive: https://drive.google.com/file/d/1nkEsD1g7THm_T58KwUQZ7o-b174fdx-n/view
"""
import os
import sys
import argparse
import subprocess
import hashlib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EXPECTED_MD5 = "fac7fe1b4e048c81ef88874334b73534"


def download_tsinghua_cloud(output_path: str):
    """Download from Tsinghua Cloud using wget."""
    url = "https://cloud.tsinghua.edu.cn/f/c1ea3426ce444bc9baae/?dl=1"
    print(f"Downloading from Tsinghua Cloud...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")

    cmd = [
        "wget", "-O", output_path,
        "--progress=bar:force",
        "-c",  # resume
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        print("Download complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False


def download_google_drive(output_path: str):
    """Download from Google Drive using gdown."""
    file_id = "1nkEsD1g7THm_T58KwUQZ7o-b174fdx-n"
    url = f"https://drive.google.com/uc?id={file_id}&export=download"

    try:
        import gdown
        print(f"Downloading from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        print("Download complete.")
        return True
    except ImportError:
        print("gdown not installed. Trying pip install gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Google Drive download failed: {e}")
        return False


def verify_md5(filepath: str) -> bool:
    """Verify file MD5."""
    print(f"Verifying MD5...")
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    actual = md5.hexdigest()
    print(f"Expected: {EXPECTED_MD5}")
    print(f"Actual:   {actual}")
    return actual == EXPECTED_MD5


def extract_zip(zip_path: str, extract_dir: str):
    """Extract zip file."""
    print(f"Extracting {zip_path} to {extract_dir}...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    print("Extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Download AIOps 2020 dataset")
    parser.add_argument("--source", choices=["tsinghua", "google"], default="tsinghua",
                        help="Download source")
    parser.add_argument("--extract", action="store_true", help="Extract after download")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "aiops2020_stage1.zip")

    if os.path.exists(zip_path):
        print(f"File already exists: {zip_path}")
        if verify_md5(zip_path):
            print("MD5 verified, skipping download.")
        else:
            print("MD5 mismatch, re-downloading...")
            os.remove(zip_path)
        if not os.path.exists(zip_path):
            pass
        else:
            if args.extract:
                extract_zip(zip_path, RAW_DIR)
            return

    success = False
    if args.source == "tsinghua":
        success = download_tsinghua_cloud(zip_path)
    elif args.source == "google":
        success = download_google_drive(zip_path)

    if success and os.path.exists(zip_path):
        verify_md5(zip_path)
        if args.extract:
            extract_zip(zip_path, RAW_DIR)


if __name__ == "__main__":
    main()
