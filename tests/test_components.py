import os
import shutil
import unittest
import numpy as np
import torch
from src.dataset import generate_shape_image, generate_shape_dataset, ShapeDataset, SHAPE_CLASSES
from src.traditional import preprocess_image, detect_shapes_traditional, find_shape_boundaries
from src.model import ShapeDetectorNet

class TestShapeDetectionComponents(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = "./data/temp_test_data"
        os.makedirs(cls.temp_dir, exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_shape_image_generation(self):
        """Test that single shape generation outputs valid image arrays and bounded coords."""
        for shape in SHAPE_CLASSES:
            img, bbox = generate_shape_image(shape, img_size=128)
            self.assertEqual(img.shape, (128, 128, 3))
            self.assertEqual(img.dtype, np.uint8)
            
            xmin, ymin, xmax, ymax = bbox
            self.assertTrue(0 <= xmin < xmax <= 128)
            self.assertTrue(0 <= ymin < ymax <= 128)

    def test_dataset_generation_and_loader(self):
        """Test generation of small dataset and validation of PyTorch ShapeDataset loader."""
        num_samples = 15
        generate_shape_dataset(self.temp_dir, num_samples=num_samples, img_size=128)
        
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "annotations.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "images")))
        
        # Test loader
        dataset = ShapeDataset(self.temp_dir)
        self.assertEqual(len(dataset), num_samples)
        
        img_tensor, label, bbox = dataset[0]
        self.assertEqual(img_tensor.shape, (3, 128, 128))
        self.assertTrue(isinstance(img_tensor, torch.Tensor))
        self.assertTrue(isinstance(label, torch.Tensor))
        self.assertTrue(isinstance(bbox, torch.Tensor))
        
        # Bbox should be normalized to [0, 1]
        for val in bbox:
            self.assertTrue(0.0 <= val.item() <= 1.0)

    def test_traditional_preprocessing(self):
        """Test traditional OpenCV image preprocessing outputs binary mask."""
        img, _ = generate_shape_image("circle", img_size=128)
        binary = preprocess_image(img)
        
        self.assertEqual(binary.shape, (128, 128))
        # Binary image should only contain 0 and 255 values
        unique_vals = np.unique(binary)
        for val in unique_vals:
            self.assertIn(val, [0, 255])

    def test_model_forward_pass(self):
        """Test the ShapeDetectorNet network inputs and outputs shape dimensions."""
        net = ShapeDetectorNet(num_classes=7)
        dummy_input = torch.randn(4, 3, 64, 64)
        logits, bbox = net(dummy_input)
        
        self.assertEqual(logits.shape, (4, 7))
        self.assertEqual(bbox.shape, (4, 4))
        
        # Output bbox should be bounded by Sigmoid to [0, 1]
        self.assertTrue(torch.all(bbox >= 0.0))
        self.assertTrue(torch.all(bbox <= 1.0))

    def test_geometry_cross_check(self):
        """The hybrid check returns the CNN label and flags agreement with contour geometry."""
        from src.hybrid import classify_with_check, geometry_label
        import cv2
        img_rgb, _ = generate_shape_image('square', img_size=64)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.assertIn(geometry_label(img_bgr), SHAPE_CLASSES + [None])
        model = ShapeDetectorNet(num_classes=len(SHAPE_CLASSES))
        model.eval()
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        result = classify_with_check(model, tensor, img_bgr)
        self.assertIn(result["label"], SHAPE_CLASSES)
        self.assertEqual(result["confirmed"], result["geometry_label"] == result["label"])
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)

    def test_canny_boundaries_locate_the_shape(self):
        """Canny boundaries give a box that overlaps the generated shape's true box."""
        import cv2
        img, bbox = generate_shape_image('hexagon', img_size=64)
        edges = find_shape_boundaries(img)
        self.assertEqual(edges.shape, img.shape[:2])
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        ix = max(0, min(x + w - 1, bbox[2]) - max(x, bbox[0]) + 1)
        iy = max(0, min(y + h - 1, bbox[3]) - max(y, bbox[1]) + 1)
        union = w * h + (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) - ix * iy
        self.assertGreater(ix * iy / union, 0.7)

    def test_draw_predictions_adds_the_outline_overlay(self):
        """A detection with a contour gets its outline drawn, green when geometry agrees with the label."""
        import cv2
        from src.inference import OVERLAYS, draw_predictions
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (44, 44), (0, 0, 0), -1)
        path = os.path.join(self.temp_dir, "square.png")
        cv2.imwrite(path, img)
        contour = np.array([[[20, 20]], [[44, 20]], [[44, 44]], [[20, 44]]], dtype=np.int32)
        det = {"label_name": "square", "confidence": 0.9, "bbox": [18, 18, 46, 46], "contour": contour, "geometry_label": "square"}
        out = draw_predictions(path, [det])
        self.assertEqual(len(OVERLAYS), 4)
        self.assertTrue(np.array_equal(out[44, 30], [0, 200, 0]))   # a point on the outline's bottom edge is green

if __name__ == "__main__":
    unittest.main()
