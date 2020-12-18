import os
import argparse
import cv2
import torch
from src.dataset import generate_shape_dataset, ShapeDataset
from src.train import train_model
from src.inference import load_model, predict_single_image, predict_multi_shape_hybrid, draw_predictions
from src.traditional import detect_shapes_traditional

def main():
    parser = argparse.ArgumentParser(
        description="Solid Object Detection and Identification using Image Processing (PyTorch & OpenCV)"
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "train", "predict", "traditional"],
        required=True,
        help="Execution mode: generate (dataset), train (model), predict (using PyTorch), traditional (OpenCV only)"
    )
    
    # Dataset generation args
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of synthetic shape images to generate")
    parser.add_argument("--data-dir", type=str, default="./data/shapes", help="Directory for dataset storage")
    parser.add_argument("--img-size", type=int, default=64, help="Size of generated images (square)")
    
    # Training args
    parser.add_argument("--model-path", type=str, default="./models/shape_detector.pth", help="Path to save/load trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    # Inference args
    parser.add_argument("--image", type=str, help="Path to the input image for prediction")
    parser.add_argument("--output", type=str, help="Path to save the annotated prediction image")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid pipeline (OpenCV localization + CNN classification) for multi-shape images")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        print("--- Generating Synthetic Shape Dataset ---")
        generate_shape_dataset(args.data_dir, num_samples=args.num_samples, img_size=args.img_size)
        
    elif args.mode == "train":
        print("--- Training Shape Detection Model ---")
        if not os.path.exists(os.path.join(args.data_dir, "annotations.csv")):
            print(f"Dataset not found in {args.data_dir}. Generating a default dataset first...")
            generate_shape_dataset(args.data_dir, num_samples=args.num_samples, img_size=args.img_size)
            
        train_model(
            data_dir=args.data_dir,
            model_save_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
    elif args.mode == "predict":
        print("--- Running Deep Learning Inference ---")
        if not args.image:
            print("Error: Please provide an image path using --image.")
            return
            
        if not os.path.exists(args.model_path):
            print(f"Error: Trained model file not found at {args.model_path}. Please train the model first.")
            return
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.model_path, device)
        
        if args.hybrid:
            print("Running hybrid multi-shape detection pipeline...")
            detections = predict_multi_shape_hybrid(model, args.image, device)
        else:
            print("Running end-to-end single shape detection pipeline...")
            det = predict_single_image(model, args.image, device)
            detections = [det]
            
        # Display predictions summary
        print(f"Detected {len(detections)} object(s):")
        for i, det in enumerate(detections):
            conf_str = f"confidence: {det['confidence']:.4f}" if 'confidence' in det else "confidence: N/A"
            print(f"  {i+1}: {det['label_name']} | Bbox: {det['bbox']} | {conf_str}")
            
        if args.output:
            draw_predictions(args.image, detections, args.output)
        else:
            # Save to default output path if not specified
            default_out = "./output/prediction_result.jpg"
            draw_predictions(args.image, detections, default_out)
            
    elif args.mode == "traditional":
        print("--- Running Traditional OpenCV Contour Detector ---")
        if not args.image:
            print("Error: Please provide an image path using --image.")
            return
            
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Could not read image {args.image}")
            return
            
        detections = detect_shapes_traditional(img)
        print(f"Detected {len(detections)} shape(s) using contours:")
        for i, det in enumerate(detections):
            print(f"  {i+1}: {det['label_name']} | Bbox: {det['bbox']}")
            
        if args.output:
            draw_predictions(args.image, detections, args.output)
        else:
            default_out = "./output/traditional_result.jpg"
            draw_predictions(args.image, detections, default_out)

if __name__ == "__main__":
    main()
