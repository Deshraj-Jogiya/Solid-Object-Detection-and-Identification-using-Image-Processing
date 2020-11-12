import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import ShapeDataset, SHAPE_CLASSES
from src.model import ShapeDetectorNet

def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between boxA and boxB.
    Coordinates are expected in [xmin, ymin, xmax, ymax] format.
    """
    xA = torch.max(boxA[:, 0], boxB[:, 0])
    yA = torch.max(boxA[:, 1], boxB[:, 1])
    xB = torch.min(boxA[:, 2], boxB[:, 2])
    yB = torch.min(boxA[:, 3], boxB[:, 3])
    
    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])
    
    unionArea = boxAArea + boxBArea - interArea
    
    return interArea / torch.clamp(unionArea, min=1e-6)

def train_model(data_dir, model_save_path, epochs=10, batch_size=32, lr=0.001, bbox_weight=10.0, device=None):
    """
    Trains the ShapeDetectorNet model on the synthetic shapes dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    full_dataset = ShapeDataset(data_dir)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # Set seed for reproducible splitting
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize network, loss functions, and optimizer
    model = ShapeDetectorNet(num_classes=len(SHAPE_CLASSES)).to(device)
    class_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.SmoothL1Loss()  # Huber loss is more robust to outliers
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_class_loss = 0.0
        train_bbox_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels, bboxes in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_logits, pred_bboxes = model(images)
            
            # Compute loss
            loss_cls = class_criterion(class_logits, labels)
            loss_box = bbox_criterion(pred_bboxes, bboxes)
            loss = loss_cls + bbox_weight * loss_box
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * images.size(0)
            train_class_loss += loss_cls.item() * images.size(0)
            train_bbox_loss += loss_box.item() * images.size(0)
            
            _, predicted = torch.max(class_logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = (train_correct / train_total) * 100
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_class_loss = 0.0
        val_bbox_loss = 0.0
        val_correct = 0
        val_total = 0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, labels, bboxes in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                bboxes = bboxes.to(device)
                
                class_logits, pred_bboxes = model(images)
                
                loss_cls = class_criterion(class_logits, labels)
                loss_box = bbox_criterion(pred_bboxes, bboxes)
                loss = loss_cls + bbox_weight * loss_box
                
                val_loss += loss.item() * images.size(0)
                val_class_loss += loss_cls.item() * images.size(0)
                val_bbox_loss += loss_box.item() * images.size(0)
                
                _, predicted = torch.max(class_logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Compute average IoU
                batch_iou = compute_iou(pred_bboxes, bboxes)
                val_iou += batch_iou.sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = (val_correct / val_total) * 100
        val_epoch_iou = val_iou / val_total
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_epoch_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} (Cls: {train_class_loss/train_total:.4f}, Box: {train_bbox_loss/train_total:.4f}) | "
              f"Train Acc: {train_epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} | "
              f"Val Acc: {val_epoch_acc:.2f}% | "
              f"Val IoU: {val_epoch_iou:.4f}")
        
        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved best model checkpoint to {model_save_path}")
            
    print("Training finished.")
    return best_val_loss

if __name__ == "__main__":
    # Test training function
    import sys
    test_dir = "./data/test_shape_gen"
    model_path = "./models/test_shape_detector.pth"
    train_model(test_dir, model_path, epochs=2, batch_size=8)
