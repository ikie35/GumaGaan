import os
from pathlib import Path

def main():
    combined_dir = Path(r"D:\school\Masters\Research\Training\Two stage Rubber Disease\yanglabel7class-v2-4\combined")
    
    if not combined_dir.exists():
        print(f"Error: Combined directory does not exist: {combined_dir}")
        return
        
    print(f"Scanning all subfolders in: {combined_dir} for copy files...")
    
    total_deleted = 0
    total_scanned = 0
    
    # Iterate through all subdirectories in combined folder
    for subfolder in combined_dir.iterdir():
        if subfolder.is_dir():
            print(f"Scanning subfolder: {subfolder.name}...")
            folder_deleted = 0
            folder_scanned = 0
            
            for file in subfolder.iterdir():
                if file.is_file():
                    folder_scanned += 1
                    if " - copy" in file.name.lower():
                        try:
                            file.unlink()
                            folder_deleted += 1
                        except Exception as e:
                            print(f"  Error deleting {file.name}: {e}")
            
            print(f"  Scanned: {folder_scanned} | Deleted: {folder_deleted} | Remaining: {folder_scanned - folder_deleted}")
            total_deleted += folder_deleted
            total_scanned += folder_scanned
            
    print("\n--- Clean Complete ---")
    print(f"Total files scanned across all subfolders: {total_scanned}")
    print(f"Total copy files deleted: {total_deleted}")
    print(f"Total files remaining: {total_scanned - total_deleted}")

if __name__ == "__main__":
    main()
