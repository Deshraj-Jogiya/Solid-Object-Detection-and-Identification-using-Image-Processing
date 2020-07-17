import os
import cv2
import numpy as np
import torch
from src.model import ShapeDetectorNet
from src.dataset import SHAPE_CLASSES, LABEL_TO_SHAPE

def load_model(model_path, device=None):
    """
    Loads the trained ShapeDetectorNet model from disk.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = ShapeDetectorNet(num_classes=len(SHAPE_CLASSES))
    # Load state dict with map_location to handle CPU/GPU crossing
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_single_image(model, image_path, device=None):
    """
    Runs the deep learning model to locate and classify a single shape in an image.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Read and preprocess the image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    h, w, _ = orig_img.shape
    
    # Preprocess image for network (RGB, resized, normalized, CHW, batch dimension)
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, pred_bbox = model(img_tensor)
        
        # Get class prediction and confidence
        probs = torch.softmax(logits, dim=1)
        conf, class_idx = torch.max(probs, dim=1)
        class_idx = class_idx.item()
        conf = conf.item()
        shape_name = LABEL_TO_SHAPE[class_idx]
        
        # Scale bounding box back to original image size
        # pred_bbox shape is [1, 4] with normalized [xmin, ymin, xmax, ymax]
        bbox = pred_bbox[0].cpu().numpy()
        xmin = int(bbox[0] * w)
        ymin = int(bbox[1] * h)
        xmax = int(bbox[2] * w)
        ymax = int(bbox[3] * h)
        
        # Clip bounding box
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))
        
    return {
        'label_name': shape_name,
        'confidence': conf,
        'bbox': [xmin, ymin, xmax, ymax]
    }

def predict_multi_shape_hybrid(model, image_path, device=None):
    """
    Combines traditional contour extraction for localization and
    the deep learning model for classification. Allows detecting multiple shapes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    h, w, _ = orig_img.shape
    
    # 1. Traditional Contour Localization
    # Preprocess image
    from src.traditional import preprocess_image
    binary = preprocess_image(orig_img)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for c in contours:
        # Filter small noise contours
        if cv2.contourArea(c) < 120:
            continue
            
        # Get bounding box coordinates
        rx, ry, rw, rh = cv2.boundingRect(c)
        
        # Pad bounding box slightly for cleaner classification crop
        pad = int(max(rw, rh) * 0.1)
        xmin = max(0, rx - pad)
        ymin = max(0, ry - pad)
        xmax = min(w - 1, rx + rw + pad)
        ymax = min(h - 1, ry + rh + pad)
        
        crop = orig_img[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue
            
        # 2. Deep Learning Classification on Cropped Region
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (64, 64))
        crop_tensor = crop_resized.astype(np.float32) / 255.0
        crop_tensor = np.transpose(crop_tensor, (2, 0, 1))
        crop_tensor = torch.tensor(crop_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = model(crop_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, class_idx = torch.max(probs, dim=1)
            
            shape_name = LABEL_TO_SHAPE[class_idx.item()]
            confidence = conf.item()
            
        detections.append({
            'label_name': shape_name,
            'confidence': confidence,
            'bbox': [xmin, ymin, xmax, ymax]
        })
        
    return detections

def draw_predictions(image_path, detections, output_path=None):
    """
    Draws bounding boxes and labels with confidence scores on the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    for det in detections:
        xmin, ymin, xmax, ymax = det['bbox']
        label = det['label_name']
        conf = det.get('confidence', None)
        
        # Color: Green/Blue mix for a clean look
        color = (255, 120, 0) # BGR: Bright Blue/Orange
        
        # Draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Text label
        label_text = f"{label}"
        if conf is not None:
            label_text += f" ({conf:.2f})"
            
        # Background box for text to improve readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        cv2.rectangle(img, (xmin, ymin - text_height - 6), (xmin + text_width + 4, ymin), color, -1)
        # White text
        cv2.putText(img, label_text, (xmin + 2, ymin - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to {output_path}")
        
    return img
