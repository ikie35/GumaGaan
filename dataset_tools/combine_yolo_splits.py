import os
import shutil
from pathlib import Path

def main():
    dataset_dir = Path(r"D:\school\Masters\Research\Training\Two stage Rubber Disease\yanglabel7class-v2-4")
    combined_dir = dataset_dir / "combined"
    
    combined_images_dir = combined_dir / "images"
    combined_labels_dir = combined_dir / "labels"
    
    # Create target directories
    combined_images_dir.mkdir(parents=True, exist_ok=True)
    combined_labels_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    print("Starting combination of splits...")
    
    for split_folder, prefix in splits.items():
        split_path = dataset_dir / split_folder
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping split {split_folder} (directories not found).")
            continue
            
        print(f"Combining split '{split_folder}' (prefix: '{prefix}_')...")
        
        # Copy images
        copied_images_count = 0
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in valid_extensions:
                continue
                
            # Create new filename
            new_img_name = f"{prefix}_{img_file.name}"
            shutil.copy2(img_file, combined_images_dir / new_img_name)
            
            # Find and copy corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                new_label_name = f"{prefix}_{label_file.name}"
                shutil.copy2(label_file, combined_labels_dir / new_label_name)
                
            copied_images_count += 1
            
        print(f"  Copied {copied_images_count} image-label pairs from '{split_folder}'.")
        
    # Check total files in combined
    total_images = len(list(combined_images_dir.iterdir()))
    total_labels = len(list(combined_labels_dir.iterdir()))
    
    print("\n--- Combination Complete ---")
    print(f"Saved combined dataset under: {combined_dir}")
    print(f"Total combined images: {total_images}")
    print(f"Total combined labels: {total_labels}")

if __name__ == "__main__":
    main()
