import cv2
import numpy as np

def preprocess_image(image, blur_ksize=5, thresh_block_size=11, thresh_c=2):
    """
    Applies image processing steps: grayscale, blurring, and adaptive thresholding.
    """
    # 1. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # 2. Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    # 3. Thresholding (Otsu's thresholding combined with binary inversion)
    # Since shapes are darker than background, we invert to get white shape on black background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Morphological operations to remove noise and fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def classify_shape_by_geometry(contour):
    """
    Classifies a shape based on its geometric properties: vertex count, circularity, and convexity.
    """
    # Calculate perimeter (arc length) and approximate polygon
    peri = cv2.arcLength(contour, True)
    # 2% of perimeter is standard for approximation
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    num_vertices = len(approx)
    
    # Calculate area and bounding box
    area = cv2.contourArea(contour)
    if area < 50:  # Noise threshold
        return "unknown", approx
        
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h
    
    # Calculate Convex Hull and Convexity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    
    # Calculate Circularity: 4 * pi * Area / Perimeter^2
    circularity = (4 * np.pi * area) / (peri ** 2) if peri > 0 else 0
    
    # Classification logic based on geometry
    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        # Check aspect ratio to distinguish square from rectangle
        # For a square, aspect ratio is close to 1.0
        if 0.88 <= aspect_ratio <= 1.12:
            shape = "square"
        else:
            shape = "rectangle"
    elif num_vertices == 5:
        # A star can sometimes approximate to 5 vertices if epsilon is large, but convexity is low
        if convexity < 0.85:
            shape = "star"
        else:
            shape = "pentagon"
    elif num_vertices == 6:
        shape = "hexagon"
    elif num_vertices >= 7:
        # Could be star or circle
        # A star is non-convex (low convexity, low circularity)
        # A circle is convex and highly circular
        if convexity < 0.85 or circularity < 0.75:
            shape = "star"
        else:
            shape = "circle"
    else:
        # Fallback using circularity & convexity
        if circularity > 0.82:
            shape = "circle"
        elif convexity < 0.80:
            shape = "star"
        else:
            shape = "unknown"
            
    return shape, approx

def detect_shapes_traditional(image):
    """
    Finds and classifies shapes in an image using traditional image processing.
    Returns:
        results: list of dicts containing 'label_name', 'bbox' (xmin, ymin, xmax, ymax), and 'contour'
    """
    # Preprocess
    binary = preprocess_image(image)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for c in contours:
        # Filter small noise
        if cv2.contourArea(c) < 100:
            continue
            
        shape_name, approx = classify_shape_by_geometry(c)
        if shape_name == "unknown":
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(c)
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        
        results.append({
            'label_name': shape_name,
            'bbox': [xmin, ymin, xmax, ymax],
            'contour': c,
            'approx': approx
        })
        
    return results
