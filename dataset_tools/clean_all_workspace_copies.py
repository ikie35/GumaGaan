import os
from pathlib import Path

def main():
    workspace_dir = Path(r"D:\school\Masters\Research\Training\Two stage Rubber Disease")
    
    # Folders to exclude from scanning to be safe
    exclude_folders = {'.git', '.gemini', '__pycache__', '.ipynb_checkpoints'}
    
    print(f"Recursively scanning entire workspace for copy files: {workspace_dir}")
    
    total_deleted = 0
    total_scanned = 0
    deleted_files_log = []
    
    for root, dirs, files in os.walk(workspace_dir):
        # Exclude directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_folders]
        
        for file in files:
            file_path = Path(root) / file
            total_scanned += 1
            
            # Check for " - copy" in filename (case-insensitive)
            if " - copy" in file_path.name.lower():
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    total_deleted += 1
                    deleted_files_log.append((file_path.relative_to(workspace_dir), file_size))
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                    
    print("\n--- Workspace-Wide Clean Complete ---")
    print(f"Total files scanned: {total_scanned}")
    print(f"Total copy files deleted: {total_deleted}")
    
    if total_deleted > 0:
        print("\nDeleted Files Details:")
        for rel_path, size in deleted_files_log:
            print(f"  - {rel_path} ({size} bytes)")
    else:
        print("\nNo copy files found in other directories.")

if __name__ == "__main__":
    main()
