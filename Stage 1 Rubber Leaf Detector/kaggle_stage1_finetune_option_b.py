"""
================================================================================
Stage 1: Option B - YOLOv8n Leaf Detector Fine-Tuning Script
Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
================================================================================
Objective:
  Fine-tune the existing Stage 1 YOLOv8 Nano checkpoint ('stage1_best.pt') 
  rather than training from scratch. This retains learned leaf morphology while
  adapting the model to:
    1. Suppress false alarms using background-only negative images.
    2. Boost recall on distant, shaded, and high-glare canopy leaves.
    3. Stabilize bounding box regression for clean Stage 2 cropping.

Target Hardware: Raspberry Pi 5 + Hailo-8L AI NPU HAT (via ONNX opset 12)
================================================================================
"""

import os
import sys
import shutil
from pathlib import Path

# --- CELL 1: Dependency Verification ---
def install_dependencies():
    packages = ["ultralytics", "roboflow"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[Setup] Installing {pkg}...")
            os.system(f"pip install {pkg} --quiet")

install_dependencies()

from ultralytics import YOLO
from roboflow import Roboflow


# --- CELL 2: Dataset Acquisition (Roboflow) ---
print("\n" + "=" * 70)
print("[Step 1/5] Downloading Refined Stage 1 Dataset from Roboflow...")
print("=" * 70)

# Update version number if you created a newer dataset version in Roboflow
ROBOFLOW_API_KEY = "tsstVHnAqPFAhbWOYzrg"
WORKSPACE_NAME   = "rubber-research"
PROJECT_NAME     = "stage-1-rubber-leaf-detection-single-class"
DATASET_VERSION  = 3  # Set to your latest version that contains background negatives

try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    dataset = project.version(DATASET_VERSION).download("yolov8")
    data_yaml_path = f"{dataset.location}/data.yaml"
    print(f"[Dataset] Ready at: {data_yaml_path}")
except Exception as e:
    print(f"[Dataset Warning] Could not download via Roboflow API: {e}")
    print("[Dataset] Fallback: Expecting local 'data.yaml' in current directory.")
    data_yaml_path = "data.yaml"


# --- CELL 3: Load Existing Stage 1 Checkpoint (Option B) ---
print("\n" + "=" * 70)
print("[Step 2/5] Loading Existing Stage 1 Checkpoint...")
print("=" * 70)

# Check for existing checkpoint locations (Kaggle working dir, input dir, or local repo)
possible_weights = [
    "stage1_best.pt",
    "best.pt",
    "Stage1_training_results/weights/stage1_best.pt",
    "Stage1_training_results/weights/best.pt",
    "/kaggle/input/stage1-weights/stage1_best.pt",
    "/kaggle/input/stage1-weights/best.pt"
]

checkpoint_path = None
for p in possible_weights:
    if os.path.exists(p):
        checkpoint_path = p
        break

if checkpoint_path:
    print(f"[Model] Found existing Stage 1 weights: {checkpoint_path}")
    model = YOLO(checkpoint_path)
else:
    print("[Model Warning] Pre-trained 'stage1_best.pt' not found in path.")
    print("[Model] Downloading/Loading fallback 'yolov8n.pt' for fine-tuning...")
    model = YOLO("yolov8n.pt")


# --- CELL 4: Fine-Tuning Execution ---
print("\n" + "=" * 70)
print("[Step 3/5] Starting Stage 1 Fine-Tuning (Option B - Gentle Adaptation)...")
print("=" * 70)

# Hyperparameter Strategy for Fine-Tuning Nano:
# - lr0=0.002: 5x lower than scratch (0.01) to protect pre-learned weights.
# - imgsz=800: Higher resolution to detect small/distant canopy leaves.
# - close_mosaic=10: Disables artificial mosaic tiles in final 10 epochs.
# - box=8.5: Increased bounding box regression loss for tighter crops.
results = model.train(
    data=data_yaml_path,
    epochs=45,                 # 40-50 epochs is ideal for fine-tuning
    patience=10,               # Early stopping to prevent overfitting
    imgsz=800,                 # 800px catches distant leaves without VRAM exhaustion
    batch=16,                  # Safe batch size for standard T4/P100 GPUs
    device=0,
    project="stage1_finetuning",
    name="yolov8n_refined_leaves",
    
    # Optimizer & Learning Rate Schedule
    lr0=0.002,                 # Gentle initial learning rate
    lrf=0.01,                  # Final learning rate factor (cosine decay)
    warmup_epochs=2.0,         # Brief warmup for momentum stabilization
    
    # Scale & Geometric Invariance
    scale=0.60,                # Zoom variation +/- 60%
    multi_scale=True,          # Dynamically shifts resolution during training
    degrees=20.0,              # Angular rotation for tilted branches
    flipud=0.5,                # Vertical flip
    fliplr=0.5,                # Horizontal flip
    
    # Environmental Illumination Invariance (Waxy Cuticle Glare & Shade)
    hsv_h=0.015,               # Subtle hue jitter
    hsv_s=0.70,                # Saturation jitter
    hsv_v=0.60,                # Value/Brightness jitter
    
    # Loss & Regularization
    mosaic=1.0,                # Clustered leaf training
    close_mosaic=10,           # Transition to clean natural leaf contours
    box=8.5,                   # High penalty on loose bounding boxes
    cls=0.5                    # Standard single-class classification weight
)


# --- CELL 5: Rigorous Dual Evaluation (Val & Held-Out Test Splits) ---
print("\n" + "=" * 70)
print("[Step 4/5] Evaluating Refined Model on Held-Out Test Split...")
print("=" * 70)

# Evaluate at calibrated operational threshold (conf=0.25)
test_metrics = model.val(split="test", conf=0.25, plots=True, name="test_eval")

print("\n" + "-" * 50)
print("=== REFINED STAGE 1 TEST SET PERFORMANCE ===")
print(f"Test Recall (Sensitivity): {test_metrics.box.mr * 100:.2f}%")
print(f"Test Precision:            {test_metrics.box.mp * 100:.2f}%")
print(f"Test mAP@50:               {test_metrics.box.map50 * 100:.2f}%")
print(f"Test mAP@50-95:            {test_metrics.box.map * 100:.2f}%")
print("-" * 50)

# Save text summary report for thesis documentation
train_dir = Path(model.trainer.save_dir)
summary_file = train_dir / "refined_stage1_metrics.txt"
with open(summary_file, "w") as f:
    f.write("=== REFINED STAGE 1 DETECTOR (OPTION B: YOLOV8N) ===\n")
    f.write(f"Operational Confidence: 0.25\n")
    f.write(f"Test Recall:            {test_metrics.box.mr * 100:.2f}%\n")
    f.write(f"Test Precision:         {test_metrics.box.mp * 100:.2f}%\n")
    f.write(f"Test mAP@50:            {test_metrics.box.map50 * 100:.2f}%\n")
    f.write(f"Test mAP@50-95:         {test_metrics.box.map * 100:.2f}%\n")


# --- CELL 6: Export to ONNX (Hailo-8L Ready) & Package Results ---
print("\n" + "=" * 70)
print("[Step 5/5] Exporting ONNX Model & Packaging Zip Archive...")
print("=" * 70)

try:
    # ONNX export with opset 12 (required for Hailo Dataflow Compiler)
    onnx_path = model.export(format="onnx", imgsz=800, opset=12, dynamic=False)
    print(f"[Export] Saved ONNX model to: {onnx_path}")
    
    # Copy ONNX directly to top-level folder
    shutil.copy(onnx_path, "stage1_refined_detector.onnx")
except Exception as e:
    print(f"[Export Warning] Could not export ONNX: {e}")

# Create ZIP archive of all weights, plots, and confusion matrices
zip_name = "stage1_option_b_refined_results"
shutil.make_archive(zip_name, "zip", str(train_dir))
print(f"\n[Done] Training complete. Download output archive: '{zip_name}.zip'")
