import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "assets" / "images"
CASCADE_FILE = PROJECT_ROOT / "assets" / "cascades" / "haarcascade_frontalface_default.xml"
OUTPUT_DIR = PROJECT_ROOT / "output"

class MyImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
       
        self.height, self.width = self.original_image.shape[:2]

    def _show_image(self, img, title, cmap=None, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        if cmap:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def bgr_2_rgb_convertor(self):
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self._show_image(rgb_image, "RGB Image")
        return rgb_image

    def bgr_2_gray_scale_convertor(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self._show_image(gray_image, "Grayscale Image", cmap='gray')
        return gray_image

    def _50_percent_resizer(self):
        new_width = int(self.width * 0.5)
        new_height = int(self.height * 0.5)
        resized_bgr = cv2.resize(self.original_image, (new_width, new_height))
        
        # Show in Real RGB (convert for display)
        # Using half the default figsize to reflect the size change visually
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        self._show_image(resized_rgb, "50% Resized Image", figsize=(5, 4))
        
        return resized_bgr

    def image_writer(self, output_image_path):
        # Save validation
        
        cv2.imwrite(output_image_path, self.original_image)
        print(f"Image saved to {output_image_path}")

    def frame_it(self, output_image_with_frame_path):
        # Draw RED frame (rectangle) 20px thickness
        img_copy = self.original_image.copy()
        
        # Rectangle around the image
        # Top-left (0,0), Bottom-right (width-1, height-1)
        thickness = 20
        # Drawing a rectangle at the border

        cv2.rectangle(img_copy, (0, 0), (self.width, self.height), (0, 0, 255), thickness)
        
        cv2.imwrite(output_image_with_frame_path, img_copy)
        
        # return the BGR image used for saving.
        return img_copy

    def find_center(self, output_image_with_center_path):
        img_copy = self.original_image.copy()
        
        center_x, center_y = self.width // 2, self.height // 2
        
        # Draw BLUE point (Circle)
        cv2.circle(img_copy, (center_x, center_y), 10, (255, 0, 0), -1) # Filled circle
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "image center"
        
        # Text size calculation for centering
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + 30 
        
        cv2.putText(img_copy, text, (text_x, text_y), font, 1, (255, 0, 0), 2)
        
        cv2.imwrite(output_image_with_center_path, img_copy)
        return img_copy

    def detect_faces(self):
        # Use CascadeClassifier
           if CASCADE_FILE.exists():
               casc_path = str(CASCADE_FILE)
        else:
             casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
             
        face_cascade = cv2.CascadeClassifier(casc_path)
        if face_cascade.empty():
            print(f"Error: Could not load classifier from {casc_path}")
            return self.original_image, 0
            
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        img_copy = self.original_image.copy()
        
        # Draw RED rectangles 
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
        faces_counter = len(faces)
        
        # Show the result 
        rgb_faces = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        self._show_image(rgb_faces, f"Faces Detected: {faces_counter}")
        
        return img_copy, faces_counter

if __name__ == "__main__":
    # Test Project 3
    try:
        print("Testing MyImageProcessor on shapes.jpg...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        processor = MyImageProcessor(str(IMAGE_DIR / "shapes.jpg"))
        
        processor.bgr_2_rgb_convertor()
        processor.bgr_2_gray_scale_convertor()
        processor._50_percent_resizer()
        processor.image_writer(str(OUTPUT_DIR / "output_original.jpg"))
        processor.frame_it(str(OUTPUT_DIR / "output_framed.jpg"))
        processor.find_center(str(OUTPUT_DIR / "output_center.jpg"))
        img_faces, count = processor.detect_faces()
        print(f"Faces found: {count}")
        print("Project 3 Test Complete.")
        
    except Exception as e:
        print(f"Error: {e}")
