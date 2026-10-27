"""
================================================================================
Stage 1 Single-Class Rubber Leaf Detector: Multi-Generation Benchmark Script
================================================================================
Compares YOLO architectural generations (YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26) 
at the Nano/Lightweight scale on a single-class 'Rubber Leaf' detection dataset.

Outputs:
  - Trained PyTorch models saved as .pt weights (e.g., YOLOv8_best.pt)
  - Standardized metrics (Val mAP@50, Test mAP@50, Test mAP@50-95, Precision@F1max, Recall@F1max, F1max)
  - GPU Inference latency (ms) and training duration (min)
  - Model parameter count (M) and PyTorch weight size (.pt in MB)
  - Auto-generated Markdown & CSV summary reports
================================================================================
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

# Auto-install missing packages if running in Kaggle / Colab notebook
for pkg in ["ultralytics", "roboflow", "pandas", "tabulate"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import pandas as pd
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# 1. EXPERIMENTAL & ROBOFLOW DATASET CONFIGURATION
# ------------------------------------------------------------------------------
# Set USE_ROBOFLOW = True to automatically download via Roboflow API
USE_ROBOFLOW = True              
ROBOFLOW_API_KEY = "tsstVHnAqPFAhbWOYzrg"
ROBOFLOW_WORKSPACE = "rubber-research"
ROBOFLOW_PROJECT = "stage-1-rubber-leaf-detection-single-class"
ROBOFLOW_VERSION = 3

# Fallback path if USE_ROBOFLOW = False
LOCAL_DATA_YAML = "data.yaml"

EPOCHS = 100                     # Maximum training epochs per generation
PATIENCE = 10                    # Early stopping patience (epochs to wait with no val improvement)
IMGSZ = 640                      # Input image resolution
BATCH_SIZE = 16                  # Batch size
DEVICE = 0                       # GPU device (0, 'cuda', or 'cpu')
PROJECT_DIR = "stage1_generations_benchmark"
MODELS_DIR = Path(PROJECT_DIR) / "pt_models"
PLOTS_DIR = Path(PROJECT_DIR) / "plots"

# Candidate YOLO Generation Models (Nano / Lightweight scale)
MODELS_TO_BENCHMARK = [
    {
        "generation": "YOLOv8",
        "model_id": "yolov8n.pt",
        "save_pt_name": "stage1_yolov8n_single_class_best.pt"
    },
    {
        "generation": "YOLOv9",
        "model_id": "yolov9t.pt",
        "save_pt_name": "stage1_yolov9t_single_class_best.pt"
    },
    {
        "generation": "YOLOv10",
        "model_id": "yolov10n.pt",
        "save_pt_name": "stage1_yolov10n_single_class_best.pt"
    },
    {
        "generation": "YOLO11",
        "model_id": "yolo11n.pt",
        "save_pt_name": "stage1_yolo11n_single_class_best.pt"
    },
]

# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def measure_gpu_latency(model, imgsz=640, runs=100, warmup=20):
    """Measures pure inference latency in milliseconds per image on GPU."""
    import torch
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(next(model.model.parameters()).device)
    
    # Warmup
    for _ in range(warmup):
        _ = model.model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        for _ in range(runs):
            starter.record()
            _ = model.model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
        return float(np.mean(timings))
    else:
        start_time = time.perf_counter()
        for _ in range(runs):
            _ = model.model(dummy_input)
        return float((time.perf_counter() - start_time) / runs * 1000)

def extract_f1max_metrics(val_results):
    """Extracts optimal F1, Precision, and Recall at the F1-max threshold."""
    try:
        curves = val_results.box.curves_results
        f1_curve = curves[2][1]       # F1 curve
        prec_curve = curves[0][1]     # Precision curve
        rec_curve = curves[1][1]      # Recall curve
        
        best_idx = np.argmax(f1_curve)
        best_f1 = float(f1_curve[best_idx]) * 100.0
        best_p = float(prec_curve[best_idx]) * 100.0
        best_r = float(rec_curve[best_idx]) * 100.0
    except Exception:
        # Fallback to macro mean box metrics if curve array shape differs
        best_p = float(val_results.box.p[0] * 100.0) if len(val_results.box.p) > 0 else float(val_results.box.mp * 100.0)
        best_r = float(val_results.box.r[0] * 100.0) if len(val_results.box.r) > 0 else float(val_results.box.mr * 100.0)
        best_f1 = float(2 * best_p * best_r / (best_p + best_r + 1e-6))
    return best_p, best_r, best_f1

# ------------------------------------------------------------------------------
# 3. BENCHMARK EXECUTION PIPELINE
# ------------------------------------------------------------------------------
def run_benchmark():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    
    print("=" * 85)
    print(" STARTING STAGE 1 GENERATION BENCHMARK (SINGLE-CLASS RUBBER LEAF)")
    print("=" * 85)

    # 1. Resolve Dataset YAML (Roboflow or Local)
    if USE_ROBOFLOW:
        print(f"\n--> Connecting to Roboflow ({ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION})...")
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_VERSION)
        dataset = version.download("yolov8")
        data_yaml_path = f"{dataset.location}/data.yaml"
        print(f"[OK] Roboflow Dataset ready at: {data_yaml_path}")
    else:
        data_yaml_path = LOCAL_DATA_YAML
        print(f"[INFO] Using local dataset config: {data_yaml_path}")

    for item in MODELS_TO_BENCHMARK:
        gen = item["generation"]
        model_id = item["model_id"]
        save_pt_name = item["save_pt_name"]
        run_name = f"{gen}_single_cls"

        print(f"\n---> Training & Evaluating: {gen} ({model_id})")
        start_train = time.time()
        
        try:
            # 1. Initialize model
            model = YOLO(model_id)
            
            # 2. Train on single-class Rubber Leaf dataset
            train_results = model.train(
                data=data_yaml_path,
                epochs=EPOCHS,
                patience=PATIENCE,      # Early stopping trigger
                imgsz=IMGSZ,
                batch=BATCH_SIZE,
                device=DEVICE,
                project=PROJECT_DIR,
                name=run_name,
                single_cls=True,        # Explicitly enforce single class
                exist_ok=True,
                plots=True,             # Generate training curves & plots
                verbose=False
            )
            train_time_min = (time.time() - start_train) / 60.0
            
            # 3. Dynamically Locate PyTorch best.pt checkpoint
            best_pt = None
            if hasattr(model, 'trainer') and hasattr(model.trainer, 'best') and Path(model.trainer.best).exists():
                best_pt = Path(model.trainer.best)
            elif hasattr(train_results, 'save_dir') and (Path(train_results.save_dir) / "weights" / "best.pt").exists():
                best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
            else:
                candidates = list(Path(PROJECT_DIR).rglob("**/best.pt"))
                matched = [c for c in candidates if run_name in str(c)]
                if matched:
                    best_pt = matched[0]
            
            if not best_pt or not best_pt.exists():
                raise FileNotFoundError(f"Could not locate trained weights 'best.pt' for {gen}")
            
            dest_pt = MODELS_DIR / save_pt_name
            shutil.copy(best_pt, dest_pt)
            best_model = YOLO(dest_pt)
            
            # 4. Validation Split Metrics
            val_res = best_model.val(data=data_yaml_path, split="val", imgsz=IMGSZ, single_cls=True, plots=True, verbose=False)
            val_map50 = float(val_res.box.map50 * 100.0)
            
            # 5. Held-Out Test Split Metrics & Evaluation Plots
            test_res = best_model.val(
                data=data_yaml_path, 
                split="test", 
                imgsz=IMGSZ, 
                single_cls=True, 
                plots=True,            # Generates test PR_curve, F1_curve, confusion matrix
                name=f"{run_name}_test_eval",
                project=PROJECT_DIR,
                verbose=False
            )
            test_map50 = float(test_res.box.map50 * 100.0)
            test_map50_95 = float(test_res.box.map * 100.0)
            p_f1max, r_f1max, f1max = extract_f1max_metrics(test_res)
            
            # 6. Measure Latency
            latency_ms = measure_gpu_latency(best_model, imgsz=IMGSZ)
            
            # 7. Model Parameters & File Sizes
            params_m = sum(p.numel() for p in best_model.model.parameters()) / 1e6
            pt_size_mb = os.path.getsize(dest_pt) / (1024 * 1024)

            # 8. Copy and Organize All Plots (.png, .jpg)
            gen_plot_dir = PLOTS_DIR / gen
            gen_plot_dir.mkdir(parents=True, exist_ok=True)
            
            search_dirs = []
            if hasattr(train_results, 'save_dir'):
                search_dirs.append(Path(train_results.save_dir))
            search_dirs.extend([Path(PROJECT_DIR) / run_name, Path(PROJECT_DIR) / f"{run_name}_test_eval"])
            
            for s_dir in search_dirs:
                if s_dir.exists():
                    for ext in ["*.png", "*.jpg", "*.jpeg", "*.svg"]:
                        for plot_file in s_dir.glob(ext):
                            shutil.copy(plot_file, gen_plot_dir / f"{s_dir.name}_{plot_file.name}")

            results.append({
                "Generation": gen,
                "Model Variant": model_id.replace(".pt", ""),
                "Parameters (M)": round(params_m, 2),
                "Weights (.pt)": f"{pt_size_mb:.2f} MB",
                "Val mAP50 (%)": round(val_map50, 2),
                "Test mAP50 (%)": round(test_map50, 2),
                "Test mAP50-95 (%)": round(test_map50_95, 2),
                "Precision (%)": round(p_f1max, 2),
                "Recall (%)": round(r_f1max, 2),
                "F1 (%)": round(f1max, 2),
                "GPU Latency (ms)": round(latency_ms, 2),
                "Train Time (min)": round(train_time_min, 1),
                "PyTorch File (.pt)": save_pt_name
            })
            print(f"[OK] {gen} Complete: Test mAP50 = {test_map50:.2f}%, F1 = {f1max:.2f}%, Latency = {latency_ms:.2f} ms, Saved: {save_pt_name}")

        except Exception as e:
            print(f"[ERROR] Failed benchmarking {gen}: {str(e)}")
            results.append({
                "Generation": gen,
                "Model Variant": model_id,
                "Parameters (M)": "Error",
                "Weights (.pt)": "Error",
                "Val mAP50 (%)": "Error",
                "Test mAP50 (%)": "Error",
                "Test mAP50-95 (%)": "Error",
                "Precision (%)": "Error",
                "Recall (%)": "Error",
                "F1 (%)": "Error",
                "GPU Latency (ms)": "Error",
                "Train Time (min)": "Error",
                "PyTorch File (.pt)": str(e)
            })

    # --------------------------------------------------------------------------
    # 4. EXPORT SUMMARY REPORTS
    # --------------------------------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = Path(PROJECT_DIR) / "stage1_generation_comparison.csv"
    md_path = Path(PROJECT_DIR) / "stage1_generation_comparison.md"
    
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 85)
    print(" FINAL STAGE 1 GENERATION BENCHMARK SUMMARY")
    print("=" * 85)
    print(df.to_markdown(index=False))
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stage 1 Single-Class Rubber Leaf Detector: Architecture Generation Benchmark\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"\nSaved Models to Directory: {MODELS_DIR}")
    print(f"Saved CSV Report to: {csv_path}")
    print(f"Saved Markdown Report to: {md_path}")

if __name__ == "__main__":
    run_benchmark()
