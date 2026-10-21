import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from src.dataset import ShapeDataset, SHAPE_CLASSES, LABEL_TO_SHAPE
from src.model import ShapeDetectorNet
from src.train import compute_iou
from src.hybrid import classify_with_check

def evaluate_metrics(model, dataloader, device):
    """
    Evaluates the model over the dataset and computes detailed performance metrics.
    """
    model.eval()
    
    total = 0
    correct = 0
    total_iou = 0.0
    
    # Per-class metrics structures
    # Row: actual class, Column: predicted class (Confusion Matrix)
    num_classes = len(SHAPE_CLASSES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            
            logits, pred_bboxes = model(images)
            
            # Classification predictions
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Confusion matrix accumulation
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.item(), p.item()] += 1
                
            # IoU calculations
            batch_iou = compute_iou(pred_bboxes, bboxes)
            total_iou += batch_iou.sum().item()
            
    accuracy = (correct / total) * 100
    mean_iou = total_iou / total
    
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE REPORT")
    print("="*50)
    print(f"Overall Classification Accuracy : {accuracy:.4f}%")
    print(f"Mean Intersection over Union (mIoU) : {mean_iou:.4f}")
    print("-"*50)
    
    print(f"{'Class Name':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-"*50)
    
    for i in range(num_classes):
        actual_total = confusion_matrix[i, :].sum()
        pred_total = confusion_matrix[:, i].sum()
        tp = confusion_matrix[i, i]
        
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / actual_total if actual_total > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_name = LABEL_TO_SHAPE[i]
        print(f"{class_name:<12} | {precision:.4f}     | {recall:.4f}   | {f1:.4f}")
        
    print("="*50)
    
    return accuracy, mean_iou

def report_geometry_cross_check(model, dataset, subset, device):
    """
    Checks each CNN prediction against the contour-geometry classifier and reports how often the two agree
    and how accurate the confirmed predictions are.
    """
    model.eval()
    agree = agree_correct = 0
    for idx in subset.indices:
        item = dataset.annotations[idx]
        image_tensor = dataset[idx][0].to(device)
        image_bgr = cv2.imread(os.path.join(dataset.images_dir, item['filename']))
        result = classify_with_check(model, image_tensor, image_bgr)
        if result["confirmed"]:
            agree += 1
            agree_correct += result["label"] == item['label_name']
    total = len(subset.indices)
    print(f"Geometry cross-check : CNN and contour agree on {agree / total * 100:.1f}% of images; "
          f"{agree_correct / max(agree, 1) * 100:.2f}% of those are correct.")
    return agree / total, agree_correct / max(agree, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data/shapes"
    model_path = "./models/shape_detector.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
        
    if not os.path.exists(os.path.join(data_dir, "annotations.csv")):
        print(f"Error: Dataset annotations not found in {data_dir}.")
        return
        
    # Load dataset
    full_dataset = ShapeDataset(data_dir)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # Split using the same seed as train.py to get the identical validation set
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Load Model
    model = ShapeDetectorNet(num_classes=len(SHAPE_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print(f"Loaded model from {model_path}. Evaluating {len(val_dataset)} validation samples...")
    
    accuracy, mean_iou = evaluate_metrics(model, val_loader, device)
    
    target_accuracy = 98.97
    if accuracy >= target_accuracy:
        print(f"SUCCESS: Model achieved {accuracy:.2f}% classification accuracy, meeting target of {target_accuracy}%.")
        report_geometry_cross_check(model, full_dataset, val_dataset, device)
    else:
        print(f"WARNING: Model achieved {accuracy:.2f}% classification accuracy, which is below the target of {target_accuracy}%.")

if __name__ == "__main__":
    main()
