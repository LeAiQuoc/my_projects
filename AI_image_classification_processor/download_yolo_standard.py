import urllib.request
import os
import sys
import time

files = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights"
}

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

for filename, url in files.items():
    if not os.path.exists(filename):
        print(f"\nSource: {url}")
        try:
            urllib.request.urlretrieve(url, filename, reporthook)
            print(f"\nSaved {filename}")
        except Exception as e:
            print(f"\nFailed to download {filename}: {e}")
    else:
        print(f"\n{filename} already exists.")
