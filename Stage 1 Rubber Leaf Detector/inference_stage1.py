import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Inference script for Stage 1 Rubber Leaf Detector")
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Path to image, video, folder, or camera index to run prediction on"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold for detections (default: 0.25)"
    )
    parser.add_argument(
        "--show-ignore", 
        action="store_true", 
        help="If set, also displays the 'Ignore Leaf' class detections. Otherwise, only shows 'Rubber Leaf'."
    )
    args = parser.parse_args()

    # 1. Path to the trained weights
    # Resolving path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "stage1_training_results", "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find weights at '{weights_path}'")
        print("Please ensure the 'stage1_training_results' directory is in the same folder as this script.")
        return

    # 2. Load the trained model
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    # 3. Class filtering (Class 0: Ignore Leaf, Class 1: Rubber Leaf)
    # If --show-ignore is not set, we restrict prediction only to Class 1 (Rubber Leaf)
    filter_classes = [1] if not args.show_ignore else None

    # 4. Run prediction
    print(f"Running inference on: {args.source}...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        classes=filter_classes,
        save=True,             # Saves annotated images/videos to runs/detect/predict/
        save_txt=False,         # Change to True if you want coordinate text files
        device='cpu'            # Runs on CPU locally by default, change to 'cuda' (0) if you have an NVIDIA GPU
    )
    
    print("\nInference complete!")
    print(f"Annotated results have been saved to the folder: {results[0].save_dir}")

if __name__ == "__main__":
    main()
