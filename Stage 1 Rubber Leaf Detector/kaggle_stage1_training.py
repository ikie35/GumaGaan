# Stage 1: Rubber Leaf Detector Training (Kaggle Notebook Script)
# Copy and paste these cells into your Kaggle Notebook.

# --- CELL 1: Install Dependencies ---
# !pip install roboflow ultralytics

# --- CELL 2: Download the Stage 1 Dataset from Roboflow ---
from roboflow import Roboflow
import os
import shutil
from pathlib import Path

print("Downloading dataset from Roboflow...")
rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
project = rf.workspace("rubber-research").project("stage-1-rubber-leaf-detection-single-class")
version = project.version(3)
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")


# --- CELL 3: Train the YOLOv8 Object Detection Model ---
from ultralytics import YOLO

# Choose and Load YOLOv8 model size
# Options: 'nano' (yolov8n.pt), 'small' (yolov8s.pt), 'medium' (yolov8m.pt)
MODEL_SIZE = 'nano' # Set to 'nano' (yolov8n.pt) as requested

model_map = {
    'nano': 'yolov8n.pt',
    'small': 'yolov8s.pt',
    'medium': 'yolov8m.pt'
}
model_name = model_map.get(MODEL_SIZE.lower(), 'yolov8n.pt')
print(f"Loading YOLOv8 {MODEL_SIZE} model ({model_name})...")
model = YOLO(model_name)

# Train the object detection model
# YOLO will automatically detect and use GPU (CUDA) on Kaggle.
print("Starting training...")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,            # Increased to 100 epochs for full convergence
    patience=10,           # Early stopping: stops if no validation improvement for 10 epochs
    imgsz=640,             # Standard resolution for YOLOv8 object detection
    project="rubber-leaf-disease-detection",
    name="stage1_single_class_training",
    # Custom augmentations to improve multi-distance scale and rotational invariance:
    degrees=15.0,          # Random rotation (+/- 15 deg)
    flipud=0.5,            # Random vertical flip (50% chance)
    scale=0.7,             # Random scale/zoom (+/- 70%) to handle different camera distances
    multi_scale=True,      # Vary input resolution (+/- 50%) during training for scale-robustness
)


# --- CELL 4: Evaluate Model Performance (Validation & Held-Out Test Sets) ---
print("Evaluating on Validation Set (conf=0.40)...")
val_metrics = model.val(split="val", conf=0.40)
print(f"Validation mAP50:    {val_metrics.box.map50:.4f}")
print(f"Validation mAP50-95: {val_metrics.box.map:.4f}\n")

print("Evaluating on Unseen Held-Out TEST Set (Academic Gold Standard)...")
test_metrics = model.val(split="test", conf=0.40, plots=True, name="test_eval")
print("--- FINAL HELD-OUT TEST METRICS FOR THESIS ---")
print(f"Test mAP50:    {test_metrics.box.map50:.4f}")
print(f"Test mAP50-95: {test_metrics.box.map:.4f}")

# Save exact numerical Test Metrics to a text report file for thesis table
try:
    test_dir = Path(test_metrics.save_dir)
    train_dir = Path(model.trainer.save_dir)
    test_out_dir = train_dir / "test_evaluation"
    test_out_dir.mkdir(exist_ok=True)
    
    # Copy all test plots into test_evaluation subfolder & top-level of save_dir with clear names
    for test_file in test_dir.glob("*.png"):
        shutil.copy(test_file, test_out_dir / test_file.name)
        if "confusion_matrix" in test_file.name:
            shutil.copy(test_file, train_dir / f"test_{test_file.name}")
            
    # Write test metrics text summary file
    summary_path = train_dir / "test_metrics_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== STAGE 1 LEAF DETECTOR: HELD-OUT TEST SET METRICS ===\n")
        f.write(f"Confidence Threshold: {0.40}\n")
        f.write(f"Precision (P):       {test_metrics.box.mp:.4f}\n")
        f.write(f"Recall (R):          {test_metrics.box.mr:.4f}\n")
        f.write(f"mAP@50:              {test_metrics.box.map50:.4f}\n")
        f.write(f"mAP@50-95:           {test_metrics.box.map:.4f}\n")
    shutil.copy(summary_path, test_out_dir / "test_metrics_summary.txt")
    print(f"Successfully saved Test Set confusion matrices & report to: {summary_path}")
except Exception as e:
    print(f"Warning: Could not copy test metrics to main save_dir: {e}")

# Auto-display Test Set Confusion Matrix in Kaggle Notebook
try:
    from IPython.display import Image, display
    test_cm_path = os.path.join(str(test_metrics.save_dir), "confusion_matrix.png")
    if os.path.exists(test_cm_path):
        print(f"\nDisplaying Test Set Confusion Matrix ({test_cm_path}):")
        display(Image(filename=test_cm_path))
except Exception as e:
    print(f"Note: Could not inline display confusion matrix image: {e}")


# --- CELL 5: Export Model to ONNX (for edge deployment) ---
# Exporting to ONNX is required to compile the model to HEF format for the Hailo-8L NPU later.
import shutil
print("Exporting model to ONNX format...")
onnx_path = model.export(format="onnx", imgsz=640, dynamic=False)
print(f"Model successfully exported to: {onnx_path}")

# Copy ONNX model directly to working directory for convenient download / Hailo NPU compilation
top_level_onnx = "stage1_leaf_detector.onnx"
shutil.copy(onnx_path, top_level_onnx)
print(f"Top-level ONNX output ready at: {top_level_onnx}")


# --- CELL 6: Zip Results for Downloading ---
save_dir = str(model.trainer.save_dir)
print(f"Zipping training results from: {save_dir}")
shutil.make_archive("stage1_training_results", "zip", save_dir)
print("Zip archive created: 'stage1_training_results.zip' (Ready to download from Kaggle output)")

