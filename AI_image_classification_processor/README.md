# AI Image Classification Processor

A computer vision playground that combines three practical workflows:

1. Shape detection and contour-based sorting.
2. YOLO object detection with category-based crop export.
3. Image utilities (color conversion, resizing, framing, center marking, and face detection).

## Project Structure

```text
AI_image_classification_processor/
  assets/
    cascades/
      haarcascade_frontalface_default.xml
    images/
      Assorted_forks.jpg
      dog.jpg
      shapes.jpg
    models/
      coco.names
      yolov3.cfg
      yolov3-tiny.cfg
      yolov3.weights (optional, downloaded)
      yolov3-tiny.weights (optional fallback)
    videos/
      test_video.mp4
  output/
    Figure_1.png ... Figure_4.png
    output_original.jpg
    output_framed.jpg
    output_center.jpg
    crops/
      human/
      vehicles/
      animals/
      sport_lifestyle/
      kitchen/
      food/
      in_house/
      misc/
  src/
    project1.py
    project2.py
    project3.py
  tools/
    download_yolo_standard.py
  requirements.txt
  README.md
```

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Run from project root (`AI_image_classification_processor`).

### 1) Shape Detection

```bash
python src/project1.py
```

Input: `assets/images/shapes.jpg`

Output: Console table of detected shapes and contour areas.

### 2) YOLO Classification + Cropping

If `assets/models/yolov3.weights` is missing, download it first:

```bash
python tools/download_yolo_standard.py
```

Then run:

```bash
python src/project2.py
```

Inputs:
- `assets/images/Assorted_forks.jpg`
- `assets/images/dog.jpg`
- `assets/videos/test_video.mp4`

Outputs:
- Crops saved under `output/crops/<category>/...`

### 3) Image Processing Utilities

```bash
python src/project3.py
```

Input: `assets/images/shapes.jpg`

Outputs:
- `output/output_original.jpg`
- `output/output_framed.jpg`
- `output/output_center.jpg`
- Matplotlib preview windows for transformations and face detection.

## Notes

- YOLOv3 standard weights are large (~240 MB) and inference is slower than Tiny YOLO.
- If OpenCV was built without CUDA, the pipeline runs on CPU automatically.
- Face detection uses local cascade file first (`assets/cascades/...`), then OpenCV default fallback.
