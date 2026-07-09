import urllib.request
import os
import sys
import time
from pathlib import Path

files = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights"
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "assets" / "models"

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading... {percent}%, {speed} KB/s")
    sys.stdout.flush()

print("Downloading Standard YOLOv3 Model (This is ~240MB, please wait)...")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

for filename, url in files.items():
    target_file = MODEL_DIR / filename
    if not target_file.exists():
        print(f"\nSource: {url}")
        try:
            urllib.request.urlretrieve(url, str(target_file), reporthook)
            print(f"\nSaved {target_file}")
        except Exception as e:
            print(f"\nFailed to download {filename}: {e}")
    else:
        print(f"\n{target_file} already exists.")
