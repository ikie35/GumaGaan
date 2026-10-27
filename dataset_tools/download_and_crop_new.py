import os
import cv2
import yaml
import sys
import subprocess
from pathlib import Path

# Ensure required libraries are installed
try:
    from roboflow import Roboflow
except ImportError:
    print("Required package (roboflow) not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
    from roboflow import Roboflow

def main():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    
    # 1. Download dataset from Roboflow
    print("Initializing Roboflow...")
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    project = rf.workspace("rubber-research").project("yanglabel7class-v2-budv6")
    version = project.version(4)
    
    # Target directory for download
    download_dir = workspace_dir / "yanglabel7class-v2-4"
    print(f"Downloading dataset to: {download_dir}")
    dataset = version.download("yolov8", location=str(download_dir))
    
    dataset_location = Path(dataset.location)
    data_yaml_path = dataset_location / "data.yaml"
    
    if not data_yaml_path.exists():
        print(f"Error: data.yaml not found at {data_yaml_path}")
        return
        
    # 2. Parse data.yaml to get class mapping
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
        
    # Class names list or dict
    class_names = data_yaml.get('names')
    if isinstance(class_names, list):
        class_map = {idx: name for idx, name in enumerate(class_names)}
    elif isinstance(class_names, dict):
        class_map = {int(idx): name for idx, name in class_names.items()}
    else:
        print("Error: Could not parse class names from data.yaml")
        return
        
    print("\nDetected Class Mapping from Roboflow:")
    for idx, name in class_map.items():
        print(f"  Class {idx}: {name}")
        
    # Output directories
    output_raw_dir = workspace_dir / "Stage2_Classifier" / "cropped_yanglabel_v3" / "raw"
    output_mapped_dir = workspace_dir / "Stage2_Classifier" / "cropped_yanglabel_v3" / "mapped"
    
    # Define mapping function to Stage 2 classes: Healthy, PLFD, Other Disease
    def map_class_name(name):
        name_lower = name.lower()
        if 'healthy' in name_lower:
            return 'Healthy'
        elif 'plfd' in name_lower or 'pestalotiopsis' in name_lower or 'phytophthora' in name_lower:
            return 'PLFD'
        else:
            # Any other disease class gets grouped into 'Other Disease'
            return 'Other Disease'
            
    # Process images and labels
    splits = ['train', 'valid', 'test']
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    total_images = 0
    total_crops = 0
    crops_per_raw_class = {name: 0 for name in class_map.values()}
    crops_per_mapped_class = {'Healthy': 0, 'PLFD': 0, 'Other Disease': 0}
    
    print("\nStarting cropping process...")
    for split in splits:
        split_dir = dataset_location / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping split {split} (directory not found).")
            continue
            
        print(f"Processing split '{split}'...")
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in valid_extensions:
                continue
                
            # Find corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
                
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Failed to load image {img_file}")
                continue
                
            h, w, _ = img.shape
            total_images += 1
            
            # Read bounding boxes
            with open(label_file, 'r') as lf:
                lines = lf.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                if class_id not in class_map:
                    print(f"Warning: Unknown class ID {class_id} in {label_file}")
                    continue
                    
                raw_class_name = class_map[class_id]
                mapped_class_name = map_class_name(raw_class_name)
                
                # YOLO format: x_center y_center width height (all normalized)
                x_center, y_center, bw, bh = map(float, parts[1:])
                
                # Convert to pixel coordinates
                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)
                
                # Boundary clipping
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Extract crop
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                # Save to raw folder
                raw_dest_dir = output_raw_dir / raw_class_name
                raw_dest_dir.mkdir(parents=True, exist_ok=True)
                raw_crop_name = f"{img_file.stem}_crop_{idx}{img_file.suffix}"
                cv2.imwrite(str(raw_dest_dir / raw_crop_name), crop)
                
                # Save to mapped folder
                mapped_dest_dir = output_mapped_dir / mapped_class_name
                mapped_dest_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mapped_dest_dir / raw_crop_name), crop)
                
                # Stats
                total_crops += 1
                crops_per_raw_class[raw_class_name] += 1
                crops_per_mapped_class[mapped_class_name] += 1
                
    print("\n--- Cropping Summary ---")
    print(f"Processed {total_images} annotated images.")
    print(f"Generated {total_crops} cropped leaf images.")
    
    print("\nCrops per Original Class (Saved to Stage2_Classifier/cropped_yanglabel_v3/raw/):")
    for name, count in crops_per_raw_class.items():
        print(f"  {name}: {count} crops")
        
    print("\nCrops per Mapped Class (Saved to Stage2_Classifier/cropped_yanglabel_v3/mapped/):")
    for name, count in crops_per_mapped_class.items():
        print(f"  {name}: {count} crops")
        
    print("\nFinished! You can now use these cropped images to balance your Stage 2 dataset.")

if __name__ == "__main__":
    main()
