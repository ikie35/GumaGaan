import os
import cv2
import sys
import subprocess
from pathlib import Path

# Ensure required libraries are installed programmatically
try:
    from ultralytics import YOLO
except ImportError:
    print("Required package (ultralytics) not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

def main():
    # Paths configuration
    script_dir = Path(__file__).resolve().parent
    weights_path = script_dir.parent / "Stage 1 Rubber Leaf Detector" / "stage1_training_results" / "weights" / "best.pt"
    input_base_dir = Path(r"D:\school\Masters\Research\ChatGPT\Dataset")
    output_base_dir = Path(r"D:\school\Masters\Research\ChatGPT\Dataset_Cropped")
    
    # Check if weights exist
    if not weights_path.exists():
        print(f"Error: Weights file not found at: {weights_path}")
        return
        
    # Check if input dataset directory exists
    if not input_base_dir.exists():
        print(f"Error: Input dataset directory not found at: {input_base_dir}")
        return

    # Load YOLOv8 model
    print(f"Loading model from {weights_path}...")
    model = YOLO(str(weights_path))

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Walk through the input directory structure
    total_images_processed = 0
    total_crops_saved = 0

    print("Starting dataset cropping...")
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
            file_path = Path(root) / file
            
            # Check if file is an image
            if file_path.suffix.lower() not in valid_extensions:
                continue

            total_images_processed += 1
            
            # Determine relative path to recreate directory structure
            relative_path = file_path.relative_to(input_base_dir)
            subfolder_name = relative_path.parent
            
            # Create corresponding subfolder in the output directory
            output_subfolder = output_base_dir / subfolder_name
            output_subfolder.mkdir(parents=True, exist_ok=True)

            # Load image using OpenCV
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"Warning: Failed to load image: {file_path}")
                continue

            # Run prediction only for Class 1 (Rubber Leaf)
            results = model.predict(
                source=str(file_path),
                conf=0.25,      # You can lower this if you want to crop more leaves
                classes=[1],    # Only detect class 1 (Rubber Leaf)
                device='cpu',   # Run on CPU locally (or change to 0 if CUDA is configured)
                verbose=False
            )

            # Extract detections
            boxes = results[0].boxes
            
            if len(boxes) == 0:
                # No rubber leaf detected in this image
                continue

            img_name_stem = file_path.stem
            img_ext = file_path.suffix

            # Crop and save each detected leaf
            for i, box in enumerate(boxes):
                # Bounding box coordinates in pixels
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # Ensure coordinates are within image boundaries
                h, w, _ = img.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Crop leaf
                cropped_leaf = img[y1:y2, x1:x2]

                if cropped_leaf.size == 0:
                    continue

                # Save cropped image with suffix _crop_X
                output_filename = f"{img_name_stem}_crop_{i}{img_ext}"
                output_file_path = output_subfolder / output_filename
                
                cv2.imwrite(str(output_file_path), cropped_leaf)
                total_crops_saved += 1

            if total_images_processed % 50 == 0:
                print(f"Processed {total_images_processed} images... Created {total_crops_saved} crops.")

    print("\n--- Cropping Task Completed ---")
    print(f"Total source images processed: {total_images_processed}")
    print(f"Total cropped rubber leaf images saved to '{output_base_dir}': {total_crops_saved}")

if __name__ == "__main__":
    main()
