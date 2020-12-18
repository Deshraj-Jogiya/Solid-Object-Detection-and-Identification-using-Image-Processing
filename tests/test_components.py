import os
import shutil
import unittest
import numpy as np
import torch
from src.dataset import generate_shape_image, generate_shape_dataset, ShapeDataset, SHAPE_CLASSES
from src.traditional import preprocess_image, detect_shapes_traditional
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

if __name__ == "__main__":
    unittest.main()
