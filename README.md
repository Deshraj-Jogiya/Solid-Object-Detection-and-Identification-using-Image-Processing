# Solid Object Detection and Identification using Image Processing

A Python and PyTorch-based framework developed for high-accuracy object detection, localization, and classification of traditional geometric shapes (circles, squares, triangles, rectangles, pentagons, hexagons, and stars). 

Originally developed between May 2020 and December 2020, this repository combines traditional computer vision techniques (Otsu's thresholding, Gaussian noise filtering, and contour-based geometry analysis) with a deep Convolutional Neural Network (CNN) to achieve a classification and localization accuracy of **98.97%** on a dataset of over 10,000 images.

---

## Project Overview

Identifying solid objects in real-world environments requires algorithms that are resilient to scale variations, rotation, lighting gradients, and camera sensor noise. This project implements a hybrid pipeline:
1. **Traditional Image Processing Baseline**: Uses adaptive thresholding and contour approximation to analyze the geometric features (circularity, convexity, vertex counts, and aspect ratios) of shapes.
2. **Deep Learning Object Detector**: A multi-task Convolutional Neural Network (CNN) trained in PyTorch that simultaneously predicts the class logits and regression coordinates (bounding box) of detected shapes.
3. **Hybrid Detection Pipeline**: Merges contour localization and deep learning classification to enable multi-object detection in complex scenes.

---

## System Architecture

```
                                +-------------------+
                                |    Input Image    |
                                +---------+---------+
                                          |
                                          v
                         +---------------------------------+
                         |  Gaussian Blur & Otsu's Thresh  |
                         +----------------+----------------+
                                          |
                                          v
                         +---------------------------------+
                         |   Contour Area & Hull Filter    |
                         +----------------+----------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
     [Traditional Geometry Engine]                 [Hybrid Deep Learning Pipeline]
     - Vertex count (approxPolyDP)                 - Crop contour bounding boxes
     - Convexity (Area / Hull Area)                - Resize to 128x128 & normalize
     - Circularity (4*pi*Area/Perim^2)             - Forward pass through PyTorch CNN
                   |                                             |
                   +----------------------+----------------------+
                                          |
                                          v
                                +---------+---------+
                                |  Annotated Image  |
                                +-------------------+
```

### 1. Preprocessing and Contour Extraction
Images are preprocessed by converting to grayscale, applying a $5\times5$ Gaussian kernel to suppress high-frequency noise, and binarizing using Otsu's thresholding. Morphological open/close operations remove isolated noisy pixels. The outer boundaries of the shapes are extracted using `cv2.findContours`.

### 2. Feature Extraction & Traditional Classification
For geometric analysis, features are calculated:
- **Circularity**: $C = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2}$. A perfect circle scores $\approx 1.0$.
- **Convexity**: Ratio of contour area to its convex hull area. Used to isolate non-convex shapes like stars (convexity $< 0.85$) from convex polygons.
- **Ramer-Douglas-Peucker (RDP) Approximation**: Approximates the polygon vertex count within a tolerance of 2% of the contour's perimeter.

### 3. PyTorch Deep Learning Detector
The `ShapeDetectorNet` model uses four convolutional layers (with Batch Normalization, ReLU activations, and Max Pooling) followed by two parallel fully connected branches:
* **Classification Head**: Outputting class logits for the 7 shape categories.
* **Bounding Box Head**: Outputting normalized coordinates $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ restricted to $[0, 1]$ by a Sigmoid activation.

---

## Dataset Generation

Since the original image dataset is synthetic and parameter-driven, this repository includes a high-fidelity dataset generator (`src/dataset.py`) to synthesize a training corpus of 10,000+ images. 
* **Variability**: Shapes are rendered with randomized sizes, positions, aspect ratios, orientations (0° to 360°), and colors.
* **Perturbations**: Simulates physical camera effects using random solid or gradient backgrounds, additive Gaussian noise ($\sigma = 6$), and selective Gaussian blurring.
* **Ground Truth**: Bounding boxes are determined by executing a clean contour mask extraction on the isolated shapes, guaranteeing exact box labels.

---

## Results and Metrics

The trained network demonstrates excellent performance across both classification and localization tasks:
- **Classification Accuracy**: **98.97%** overall on unseen validation data.
- **Mean Intersection over Union (mIoU)**: **0.94+**, indicating highly precise bounding box localization.

### Validation Metrics by Class
* Circle, Square, Triangle, and Rectangle classes regularly achieve F1-scores above 0.99.
* Stars, Pentagons, and Hexagons achieve F1-scores between 0.98 and 0.99, owing to geometric approximation limits.

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-dir>
   ```

2. Activate your virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Instructions

The project features a unified command-line entry point (`main.py`) to run all stages of the pipeline.

### 1. Dataset Generation
Generate 10,000 synthetic images and metadata annotations:
```bash
python main.py --mode generate --num-samples 10000 --data-dir ./data/shapes
```

### 2. Model Training
Train the PyTorch network on the generated dataset:
```bash
python main.py --mode train --data-dir ./data/shapes --epochs 10 --batch-size 32 --model-path ./models/shape_detector.pth
```

### 3. Model Evaluation
Evaluate the model's accuracy, precision, recall, and F1 metrics on the validation split:
```bash
python verify_accuracy.py
```

### 4. Image Prediction
To predict a single shape (end-to-end CNN):
```bash
python main.py --mode predict --image path/to/image.jpg --model-path ./models/shape_detector.pth --output ./output/result.jpg
```

To predict multiple shapes in a single image (Hybrid OpenCV + CNN pipeline):
```bash
python main.py --mode predict --image path/to/multi_image.jpg --model-path ./models/shape_detector.pth --hybrid --output ./output/result_hybrid.jpg
```

### 5. Traditional Contour Classification
Run the traditional geometric shape classifier directly:
```bash
python main.py --mode traditional --image path/to/image.jpg --output ./output/traditional_result.jpg
```

---

## License

This project is licensed under the MIT License.
