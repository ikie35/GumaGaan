import os
import cv2
import yaml
import shutil
from pathlib import Path

def main():
    dataset_dir = Path(r"D:\school\Masters\Research\Training\Two stage Rubber Disease\yanglabel7class-v2-4")
    data_yaml_path = dataset_dir / "data.yaml"
    
    if not data_yaml_path.exists():
        print(f"Error: data.yaml not found at {data_yaml_path}")
        return
        
    # 1. Parse data.yaml to get class mapping
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
        
    class_names = data_yaml.get('names')
    if isinstance(class_names, list):
        class_map = {idx: name for idx, name in enumerate(class_names)}
    elif isinstance(class_names, dict):
        class_map = {int(idx): name for idx, name in class_names.items()}
    else:
        print("Error: Could not parse class names from data.yaml")
        return
        
    print("\nClass Mapping in Project:")
    for idx, name in class_map.items():
        print(f"  Class {idx}: {name}")
        
    # Output directory for combined crops
    combined_crops_dir = dataset_dir / "combined"
    
    # Clean/recreate combined directory to ensure it only has cropped images
    if combined_crops_dir.exists():
        print(f"Cleaning existing directory: {combined_crops_dir}")
        shutil.rmtree(combined_crops_dir)
    combined_crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Create target directories for each class name
    for name in class_map.values():
        (combined_crops_dir / name).mkdir(parents=True, exist_ok=True)
        
    splits = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    total_images_processed = 0
    total_crops_saved = 0
    crops_per_class = {name: 0 for name in class_map.values()}
    
    print("\nStarting crop & combine process...")
    for split_folder, prefix in splits.items():
        split_path = dataset_dir / split_folder
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
            
        print(f"Processing split '{split_folder}'...")
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in valid_extensions:
                continue
                
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
                
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            h, w, _ = img.shape
            total_images_processed += 1
            
            with open(label_file, 'r') as lf:
                lines = lf.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                if class_id not in class_map:
                    continue
                    
                class_name = class_map[class_id]
                
                # YOLO format: x_center y_center width height (normalized)
                x_center, y_center, bw, bh = map(float, parts[1:])
                
                # Convert to pixel coordinates
                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)
                
                # Clip to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                # Save crop with split prefix
                crop_name = f"{prefix}_{img_file.stem}_crop_{idx}{img_file.suffix}"
                dest_path = combined_crops_dir / class_name / crop_name
                cv2.imwrite(str(dest_path), crop)
                
                total_crops_saved += 1
                crops_per_class[class_name] += 1
                
        print(f"  Finished split '{split_folder}'. Cumulative crops created: {total_crops_saved}")
        
    print("\n--- Processing Complete ---")
    print(f"Processed {total_images_processed} source images across all splits.")
    print(f"Generated and combined {total_crops_saved} cropped leaf images.")
    print(f"Saved crops by class name in: {combined_crops_dir}")
    
    print("\nCrops per class folder:")
    for name, count in crops_per_class.items():
        print(f"  {name}: {count} crops")

if __name__ == "__main__":
    main()
