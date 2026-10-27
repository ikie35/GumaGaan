from roboflow import Roboflow
from pathlib import Path
import os

def main():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    
    print("Initializing Roboflow...")
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    
    print("Loading project 'stage-2-rubber-leaf-classifier'...")
    project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
    version = project.version(2)
    
    # Destination folder inside Stage2_Classifier
    dest_dir = workspace_dir / "Stage2_Classifier" / "stage-2-rubber-leaf-classifier-2"
    
    print(f"Downloading dataset to: {dest_dir}")
    dataset = version.download("folder", location=str(dest_dir))
    
    # Print structure of downloaded dataset
    print("\n--- Download Complete ---")
    print(f"Dataset location: {dataset.location}")
    
    # List classes/folders in train, valid, test splits
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_path = Path(dataset.location) / split
        if split_path.exists():
            print(f"\nSplit '{split}':")
            for class_folder in split_path.iterdir():
                if class_folder.is_dir():
                    file_count = len(list(class_folder.iterdir()))
                    print(f"  - Folder '{class_folder.name}': {file_count} images")

if __name__ == "__main__":
    main()
