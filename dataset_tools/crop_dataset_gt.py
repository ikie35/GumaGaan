import os
import cv2
import re
from pathlib import Path

def clean_stem(stem):
    # Roboflow appends '_jpg', '_jpeg', '_png' etc. to the stem before '.rf.'
    # We clean these out to match the original filename stem
    stem = re.sub(r'_(jpg|jpeg|png|bmp|webp|JPG|JPEG|PNG)$', '', stem)
    return stem.lower()

def main():
    # Paths configuration
    script_dir = Path(__file__).resolve().parent
    dataset_roboflow_dir = script_dir.parent / "Stage 1 Rubber Leaf Detector" / "Rubber-Leaf-Disease-Detection-10"
    input_base_dir = Path(r"D:\school\Masters\Research\ChatGPT\Dataset")
    output_base_dir = Path(r"D:\school\Masters\Research\ChatGPT\Dataset_Cropped_GT")
    
    if not dataset_roboflow_dir.exists():
        print(f"Error: Roboflow dataset directory not found at: {dataset_roboflow_dir}")
        return
        
    if not input_base_dir.exists():
        print(f"Error: Original dataset directory not found at: {input_base_dir}")
        return

    # 1. Build a lookup map of original files
    # Key: cleaned filename stem (lowercase), Value: category folder name (e.g. 'Healthy')
    print("Building lookup map of original images...")
    filename_map = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in valid_extensions:
                original_stem = file_path.stem
                category = file_path.relative_to(input_base_dir).parent
                # Map cleaned stem to category
                filename_map[original_stem.lower()] = category
                # Also map the exact stem just in case
                filename_map[clean_stem(original_stem)] = category

    print(f"Mapped {len(filename_map)} original file keys to categories.")

    # 2. Iterate through Roboflow dataset and crop leaves using annotations
    total_images_processed = 0
    total_crops_saved = 0
    not_found_in_mapping = 0
    
    # YOLO annotations have class mapping: 0: Ignore Leaf, 1: Rubber Leaf
    TARGET_CLASS = 1 # Rubber Leaf
    
    # YOLO format is: split_folder/images and split_folder/labels
    splits = ["train", "valid", "test"]
    
    print("Starting ground-truth dataset cropping...")
    
    for split in splits:
        split_dir = dataset_roboflow_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Warning: Missing split directory: {split_dir}")
            continue
            
        print(f"Processing split: {split}...")
        
        for img_file in os.listdir(images_dir):
            img_path = images_dir / img_file
            if img_path.suffix.lower() not in valid_extensions:
                continue
                
            # Extract Roboflow base stem (before .rf. hash)
            # Example: 3306_1_large_jpg.rf.709f1f9df16c3a033c0beb20c2ac06f0.jpg
            parts = img_path.stem.split(".rf.")
            if len(parts) < 2:
                # If there's no .rf. hash, just take the stem
                rf_stem = img_path.stem
            else:
                rf_stem = parts[0]
                
            # Clean the stem to match original filename
            clean_rf_stem = clean_stem(rf_stem)
            
            # Lookup category folder
            category = filename_map.get(clean_rf_stem)
            if not category:
                # Try fallback of searching if any key is contained in the stem
                for key, val in filename_map.items():
                    if key in clean_rf_stem or clean_rf_stem in key:
                        category = val
                        break
                        
            if not category:
                not_found_in_mapping += 1
                # Skip backgrounds/negatives (like cloud images)
                continue
                
            total_images_processed += 1
            
            # Create output subfolder
            output_subfolder = output_base_dir / category
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            # Find corresponding YOLO annotation label file
            label_file = labels_dir / f"{img_path.stem}.txt"
            if not label_file.exists():
                continue
                
            # Load original image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                continue
                
            h, w, _ = img.shape
            
            # Read annotations from label file
            with open(label_file, "r") as f:
                lines = f.readlines()
                
            crop_idx = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                
                # We only want to crop Class 1 (Rubber Leaf)
                if class_id == TARGET_CLASS:
                    # YOLO normalized coordinates: class x_center y_center width height
                    x_center, y_center, bbox_w, bbox_h = map(float, parts[1:])
                    
                    # Convert to pixel coordinates
                    x1 = int((x_center - bbox_w / 2) * w)
                    y1 = int((y_center - bbox_h / 2) * h)
                    x2 = int((x_center + bbox_w / 2) * w)
                    y2 = int((y_center + bbox_h / 2) * h)
                    
                    # Ensure coordinates are within image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # Crop leaf
                    cropped_leaf = img[y1:y2, x1:x2]
                    
                    if cropped_leaf.size == 0:
                        continue
                        
                    # Save cropped image
                    output_filename = f"{clean_rf_stem}_crop_{crop_idx}{img_path.suffix}"
                    output_file_path = output_subfolder / output_filename
                    
                    cv2.imwrite(str(output_file_path), cropped_leaf)
                    crop_idx += 1
                    total_crops_saved += 1
                    
            if total_images_processed % 100 == 0:
                print(f"Processed {total_images_processed} images... Created {total_crops_saved} crops.")

    print("\n--- Ground-Truth Cropping Completed ---")
    print(f"Total annotated images processed: {total_images_processed}")
    # Distractors/clouds are ignored
    print(f"Total background/negative images skipped: {not_found_in_mapping}")
    print(f"Total cropped rubber leaf images saved to '{output_base_dir}': {total_crops_saved}")

if __name__ == "__main__":
    main()
