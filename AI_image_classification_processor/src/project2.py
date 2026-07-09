import cv2
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "assets" / "models"
IMAGE_DIR = PROJECT_ROOT / "assets" / "images"
VIDEO_DIR = PROJECT_ROOT / "assets" / "videos"
CROPS_DIR = PROJECT_ROOT / "output" / "crops"

class image_classification:
    def __init__(self):
       yolov3_weights = MODEL_DIR / "yolov3.weights"
       yolov3_cfg = MODEL_DIR / "yolov3.cfg"
       yolov3_tiny_weights = MODEL_DIR / "yolov3-tiny.weights"
       yolov3_tiny_cfg = MODEL_DIR / "yolov3-tiny.cfg"

        # Using Standard YOLOv3 (better than tiny, but slower)
        # Check if standard weights exist, else fall back to tiny? 
        # For now, forcing standard since user asked for it.
       if yolov3_weights.exists() and yolov3_cfg.exists():
             print("Loading YOLOv3 Standard Model...")
           self.net = cv2.dnn.readNet(str(yolov3_weights), str(yolov3_cfg))
        else:
             print("Loading YOLOv3 Tiny Model...")
           self.net = cv2.dnn.readNet(str(yolov3_tiny_weights), str(yolov3_tiny_cfg))
        
        # Try to enable GPU (CUDA) if available
        # This requires OpenCV to be built with CUDA support. 
        # If not, it will fallback to CPU silently or ignore the request.
        try:
             # Just set CPU for now to ensure stability
             # The error confirms the build is not CUDA enabled
             self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
             self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
             print("Using CPU backend (Standard build detected).")
        except:
             pass

        with open(MODEL_DIR / "coco.names", "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.layer_names = self.net.getLayerNames()
        try:
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
             self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        
        self.categories = {
            "Human": ["person"],
            "Vehicles": ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"],
            "Animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "Sport": ["backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
            "Kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
            "Food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
            "In house": ["chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        }
        
        self.cat_folders = {
            "Human": CROPS_DIR / "human",
            "Vehicles": CROPS_DIR / "vehicles",
            "Animal": CROPS_DIR / "animals",
            "Sport": CROPS_DIR / "sport_lifestyle",
            "Kitchen": CROPS_DIR / "kitchen",
            "Food": CROPS_DIR / "food",
            "In house": CROPS_DIR / "in_house",
            "MISC": CROPS_DIR / "misc"
        }
        
        # Create directories
        for folder in self.cat_folders.values():
            folder.mkdir(parents=True, exist_ok=True)

    def get_category_folder(self, label):
        for cat, items in self.categories.items():
            if label in items:
                return self.cat_folders[cat]
        return self.cat_folders["MISC"]

    def process_image(self, image_source, is_video_frame=False, frame_count=0):
        print(f"Processing {'video frame ' + str(frame_count) if is_video_frame else image_source}...")
        # image_source can be a path (str) or an image array (numpy)
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                print(f"Error reading {image_source}")
                return
            img_name = Path(image_source).stem
        else:
            img = image_source
            img_name = f"frame_{frame_count}"

        height, width, channels = img.shape
        print(f"Image W:{width} H:{height}")

        # Detecting objects
        # Standard YOLOv3 uses 416x416 or 608x608
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        print("Inference complete.")

        class_ids = []
        confidences = []
        boxes = []

        # Analyze detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Debug print for any detection > 0.5
                if confidence > 0.5:
                     print(f"DEBUG: Found {self.classes[class_id]} with confidence {confidence:.2f}")

                # Requirement: confidence more than 90%
                if confidence > 0.9:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.9, 0.4)

        count_dict = {}

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                
                # Ensure crop is within bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w > 0 and h > 0:
                    crop_img = img[y:y+h, x:x+w]
                    
                    folder = self.get_category_folder(label)
                    

                    if label not in count_dict:
                        count_dict[label] = 1
                    else:
                        count_dict[label] += 1
                        
                    filename = f"{img_name}_{label}_{count_dict[label]}.jpg"
                    save_path = folder / filename
                    
                    cv2.imwrite(str(save_path), crop_img)
                    print(f"Saved {label} to {save_path} (Conf: {confidences[i]:.2f})")

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            return

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            

            #  every 30 frames (appro 1 sec) 
            if frame_num % 30 == 0:
                print(f"Processing frame {frame_num}...")
                self.process_image(frame, is_video_frame=True, frame_count=frame_num)
            
            frame_num += 1
            
        cap.release()
        print("Video processing complete.")


if __name__ == "__main__":
    classifier = image_classification()
    
    # Example usage:
    print("--- Processing Forks ---")
    forks_image = IMAGE_DIR / "Assorted_forks.jpg"
    if forks_image.exists():
        classifier.process_image(str(forks_image))
    print("--- Processing Dog ---")
    dog_image = IMAGE_DIR / "dog.jpg"
    if dog_image.exists():
        classifier.process_image(str(dog_image))
        
    print("Project 2 intialized.")
    print("--- Processing Video ---")
    demo_video = VIDEO_DIR / "test_video.mp4"
    if demo_video.exists():
        classifier.process_video(str(demo_video))
