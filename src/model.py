import torch
import torch.nn as nn

class ShapeDetectorNet(nn.Module):
    """
    A custom Convolutional Neural Network (CNN) in PyTorch that performs joint
    classification and bounding box regression for geometric shape detection.
    """
    def __init__(self, num_classes=7):
        super(ShapeDetectorNet, self).__init__()
        
        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Block 1: Input (3, 64, 64) -> Output (8, 32, 32)
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Input (8, 32, 32) -> Output (16, 16, 16)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Input (16, 16, 16) -> Output (32, 8, 8)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: Input (32, 8, 8) -> Output (64, 4, 4)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classification Head (outputs raw logits for the classes)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Bounding Box Regression Head (outputs normalized xmin, ymin, xmax, ymax)
        self.box_regressor = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()  # Restricts bounding box outputs to the range [0, 1]
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        class_logits = self.classifier(x)
        bbox = self.box_regressor(x)
        
        return class_logits, bbox

if __name__ == "__main__":
    # Test network forward pass with a dummy tensor
    net = ShapeDetectorNet()
    dummy_input = torch.randn(2, 3, 128, 128)
    logits, bbox = net(dummy_input)
    print("Class logits shape:", logits.shape)  # Expected: [2, 7]
    print("Bbox output shape:", bbox.shape)      # Expected: [2, 4]
    print("Bbox output sample (Sigmoid-constrained):\n", bbox)
