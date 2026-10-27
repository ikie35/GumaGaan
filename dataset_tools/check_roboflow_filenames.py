import os
from roboflow import Roboflow
from pathlib import Path

def main():
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    project = rf.workspace("rubber-research").project("rubber-leaf-disease-detection")
    version = project.version(10)
    dataset = version.download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")
    images_dir = Path(dataset.location) / "train" / "images"
    
    if images_dir.exists():
        files = os.listdir(images_dir)
        print(f"Total images in train: {len(files)}")
        print("First 10 filenames:")
        for f in files[:10]:
            print(f" - {f}")
    else:
        print("Images directory not found!")

if __name__ == "__main__":
    main()
