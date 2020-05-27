import os
import csv
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

SHAPE_CLASSES = ['circle', 'square', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'star']
SHAPE_TO_LABEL = {shape: i for i, shape in enumerate(SHAPE_CLASSES)}
LABEL_TO_SHAPE = {i: shape for i, shape in enumerate(SHAPE_CLASSES)}

def generate_shape_image(shape_type, img_size=128):
    """
    Generates an image of a specific shape with random size, position, rotation, and color.
    Returns:
        img: RGB image (numpy array of shape [img_size, img_size, 3])
        bbox: Bounding box [xmin, ymin, xmax, ymax]
    """
    # Create background with random solid color or slight gradient
    bg_val = random.randint(220, 255)
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * bg_val
    
    # Draw simple gradient background in 30% of images to simulate light falloff
    if random.random() < 0.3:
        for y in range(img_size):
            factor = 1.0 - 0.15 * (y / img_size)
            img[y, :, :] = np.clip(img[y, :, :] * factor, 0, 255).astype(np.uint8)
            
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Shape color (generally darker to stand out)
    color = (random.randint(0, 130), random.randint(0, 130), random.randint(0, 130))
    
    # Center and scale
    center_x = random.randint(int(img_size * 0.4), int(img_size * 0.6))
    center_y = random.randint(int(img_size * 0.4), int(img_size * 0.6))
    max_radius = int(img_size * 0.32)
    min_radius = int(img_size * 0.16)
    r = random.randint(min_radius, max_radius)
    
    if shape_type == 'circle':
        cv2.circle(img, (center_x, center_y), r, color, -1)
        cv2.circle(mask, (center_x, center_y), r, 255, -1)
        
    elif shape_type == 'square':
        pts = np.array([
            [center_x - r, center_y - r],
            [center_x + r, center_y - r],
            [center_x + r, center_y + r],
            [center_x - r, center_y + r]
        ], dtype=np.int32)
        angle = random.randint(0, 90)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        cv2.fillPoly(img, [pts], color)
        cv2.fillPoly(mask, [pts], 255)
        
    elif shape_type == 'triangle':
        # Equilateral-like triangle points
        pts = np.array([
            [center_x, center_y - r],
            [center_x - int(r * 0.866), center_y + int(r * 0.5)],
            [center_x + int(r * 0.866), center_y + int(r * 0.5)]
        ], dtype=np.int32)
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        cv2.fillPoly(img, [pts], color)
        cv2.fillPoly(mask, [pts], 255)
        
    elif shape_type == 'rectangle':
        w = random.randint(min_radius, max_radius)
        if random.random() > 0.5:
            h = int(w * random.uniform(1.4, 1.8))
        else:
            h = w
            w = int(h * random.uniform(1.4, 1.8))
        w = min(w, int(img_size * 0.45))
        h = min(h, int(img_size * 0.45))
        pts = np.array([
            [center_x - w, center_y - h],
            [center_x + w, center_y - h],
            [center_x + w, center_y + h],
            [center_x - w, center_y + h]
        ], dtype=np.int32)
        angle = random.randint(0, 180)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        cv2.fillPoly(img, [pts], color)
        cv2.fillPoly(mask, [pts], 255)
        
    elif shape_type == 'pentagon':
        pts = []
        for i in range(5):
            a = i * 2 * np.pi / 5 - np.pi / 2
            pts.append([center_x + int(r * np.cos(a)), center_y + int(r * np.sin(a))])
        pts = np.array(pts, dtype=np.int32)
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        cv2.fillPoly(img, [pts], color)
        cv2.fillPoly(mask, [pts], 255)
        
    elif shape_type == 'hexagon':
        pts = []
        for i in range(6):
            a = i * 2 * np.pi / 6
            pts.append([center_x + int(r * np.cos(a)), center_y + int(r * np.sin(a))])
        pts = np.array(pts, dtype=np.int32)
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        cv2.fillPoly(img, [pts], color)
        cv2.fillPoly(mask, [pts], 255)
        
    elif shape_type == 'star':
        pts = []
        for i in range(10):
            a = i * np.pi / 5 - np.pi / 2
            curr_r = r if i % 2 == 0 else int(r * 0.45)
            pts.append([center_x + int(curr_r * np.cos(a)), center_y + int(curr_r * np.sin(a))])
        pts = np.array(pts, dtype=np.int32)
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        cv2.fillPoly(img, [pts], color)
        cv2.fillPoly(mask, [pts], 255)

    # Extract bounding box from shape mask
    bbox = cv2.boundingRect(mask)
    x, y, w, h = bbox
    xmin, ymin, xmax, ymax = x, y, x + w, y + h
    
    # Clip coordinates to image boundary
    xmin = max(0, min(xmin, img_size - 1))
    ymin = max(0, min(ymin, img_size - 1))
    xmax = max(0, min(xmax, img_size - 1))
    ymax = max(0, min(ymax, img_size - 1))
    
    # Add random pixel noise
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add optional Gaussian blur
    if random.random() > 0.6:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    return img, (xmin, ymin, xmax, ymax)

def generate_shape_dataset(output_dir, num_samples=10000, img_size=64, seed=42):
    """
    Generates and saves the synthetic dataset of shapes and writes annotations to a CSV.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "annotations.csv")
    
    print(f"Generating {num_samples} shape images inside {output_dir}...")
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label_id', 'label_name', 'xmin', 'ymin', 'xmax', 'ymax'])
        
        for i in range(num_samples):
            shape_type = random.choice(SHAPE_CLASSES)
            img, (xmin, ymin, xmax, ymax) = generate_shape_image(shape_type, img_size)
            
            filename = f"shape_{i:06d}.jpg"
            img_path = os.path.join(images_dir, filename)
            cv2.imwrite(img_path, img)
            
            label_id = SHAPE_TO_LABEL[shape_type]
            writer.writerow([filename, label_id, shape_type, xmin, ymin, xmax, ymax])
            
            if (i + 1) % 2000 == 0:
                print(f"Generated {i + 1}/{num_samples} images...")
                
    print(f"Dataset successfully created. Annotations saved to {csv_path}.")

class ShapeDataset(Dataset):
    """
    PyTorch Dataset wrapper for loading the synthetic shape dataset.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.transform = transform
        self.annotations = []
        
        csv_path = os.path.join(data_dir, "annotations.csv")
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                filename, label_id, label_name, xmin, ymin, xmax, ymax = row
                self.annotations.append({
                    'filename': filename,
                    'label_id': int(label_id),
                    'label_name': label_name,
                    'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                })
                
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.images_dir, item['filename'])
        
        # Load image with OpenCV (BGR to RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Normalize bounding box coordinates to [0, 1] relative to image dimensions
        bbox = item['bbox']
        bbox_norm = [
            bbox[0] / w,
            bbox[1] / h,
            bbox[2] / w,
            bbox[3] / h
        ]
        
        # Convert to float32 numpy array and normalize image to [0, 1] range
        image = image.astype(np.float32) / 255.0
        # Transpose image dimensions from HWC to CHW for PyTorch
        image = np.transpose(image, (2, 0, 1))
        
        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(item['label_id'], dtype=torch.long)
        bbox_tensor = torch.tensor(bbox_norm, dtype=torch.float32)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor, bbox_tensor

if __name__ == "__main__":
    # Test generation of a small dataset
    import sys
    test_dir = "./data/test_shape_gen"
    generate_shape_dataset(test_dir, num_samples=100)
    print("Testing ShapeDataset loader:")
    dataset = ShapeDataset(test_dir)
    img, lbl, bbox = dataset[0]
    print(f"Image tensor shape: {img.shape}")
    print(f"Label tensor: {lbl} ({LABEL_TO_SHAPE[lbl.item()]})")
    print(f"Bbox tensor: {bbox}")
