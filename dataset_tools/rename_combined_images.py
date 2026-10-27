import os
from pathlib import Path

def main():
    combined_dir = Path(r"D:\school\Masters\Research\Training\Two stage Rubber Disease\yanglabel7class-v2-4\combined")
    
    if not combined_dir.exists():
        print(f"Error: Combined directory not found at: {combined_dir}")
        return
        
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    print(f"Scanning subfolders in: {combined_dir}...")
    
    # Iterate through all subdirectories
    for subfolder in sorted(combined_dir.iterdir()):
        if not subfolder.is_dir():
            continue
            
        print(f"\nProcessing subfolder: {subfolder.name}")
        
        # Get all image files in the subfolder
        image_files = []
        for file in subfolder.iterdir():
            if file.is_file() and file.suffix.lower() in valid_extensions:
                image_files.append(file)
                
        # Sort image files to ensure consistent ordering
        image_files.sort(key=lambda x: x.name.lower())
        total_files = len(image_files)
        print(f"Found {total_files} images.")
        
        if total_files == 0:
            print("No images to rename in this subfolder.")
            continue
            
        # Step 1: Rename to temporary names to avoid conflicts (e.g. if files named 1.jpg already exist)
        temp_renamed = []
        for idx, img_path in enumerate(image_files, start=1):
            temp_name = f"temp_rename_{idx}_{img_path.name}"
            temp_path = img_path.with_name(temp_name)
            try:
                img_path.rename(temp_path)
                temp_renamed.append((temp_path, img_path.suffix))
            except Exception as e:
                print(f"Error renaming {img_path.name} to temp: {e}")
                return
                
        # Step 2: Rename from temp names to final sequence (1 to n) with class prefix
        class_prefix = subfolder.name.replace(" ", "_")
        for idx, (temp_path, suffix) in enumerate(temp_renamed, start=1):
            final_name = f"{class_prefix}_{idx}{suffix}"
            final_path = temp_path.with_name(final_name)
            try:
                temp_path.rename(final_path)
            except Exception as e:
                print(f"Error renaming temp {temp_path.name} to final {final_name}: {e}")
                
        print(f"Successfully renamed {total_files} images to {class_prefix}_1..{class_prefix}_{total_files} in {subfolder.name}.")

if __name__ == "__main__":
    main()
