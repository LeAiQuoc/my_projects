import cv2
from pathlib import Path

def detect_shape(contour):
    # Approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    shape_name = "unidentified"
    num_vertices = len(approx)
    
    if num_vertices == 3:
        shape_name = "triangle"
    elif num_vertices == 4:
        # Check if it's a square or rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # A square will have an aspect ratio close to 1
        shape_name = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif num_vertices == 5:
        shape_name = "pentagon"
    elif num_vertices == 6:
        shape_name = "hexagon"
    else:
        # Assume circle if many vertices
        shape_name = "circle"
        
    return shape_name

def project1():
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / "assets" / "images" / "shapes.jpg"
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to binary image
    # Using Otsu's binarization for automatic thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes_data = []

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter small noise
        if area < 100:
            continue
            
        shape_name = detect_shape(contour)
        shapes_data.append({"name": shape_name, "area": area})

    # Sort by area descending
    sorted_shapes = sorted(shapes_data, key=lambda x: x["area"], reverse=True)

    # Print the list
    print(f"{'Sort':<5} {'Shape name':<15} {'Area':<10}")
    print("-" * 30)
    
    for i, item in enumerate(sorted_shapes, 1):
        print(f"{i:<5} {item['name']:<15} {item['area']:.1f}")

if __name__ == "__main__":
    project1()
