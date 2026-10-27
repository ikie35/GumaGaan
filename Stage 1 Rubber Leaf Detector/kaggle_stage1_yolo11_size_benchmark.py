# ==============================================================================
# Stage 1: YOLO11 Model Size Scaling Benchmark Script (Nano vs Small vs Medium)
# Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
# ==============================================================================
# Evaluates 3 YOLO11 model scales on single-class Rubber Leaf localization:
#   1. YOLO11n (Nano   ~ 2.6M params, batch=16)
#   2. YOLO11s (Small  ~ 9.4M params, batch=16)
#   3. YOLO11m (Medium ~ 20.1M params, batch=8)
#
# Generates publication-ready comparative metrics tables (CSV + Markdown + LaTeX),
# PyTorch (.pt) weights, optimized ONNX exports, RubberLeaf-labeled confusion matrices,
# training curves, and packages everything into 'yolo11_stage1_size_benchmark_results.zip'.
# ==============================================================================

# --- CELL 1: Install Dependencies & Setup GPU ---
import os
import sys
import time
import zipfile
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

print("=" * 85)
print(" STAGE 1: YOLO11 MODEL SIZE SCALING BENCHMARK (NANO vs SMALL vs MEDIUM)")
print("=" * 85)

for pkg in ["ultralytics", "roboflow", "pandas", "tabulate", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
from ultralytics import YOLO

# GPU Accelerator Check
device = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[Kaggle GPU Enabled] Active GPU : {gpu_name}")
    print(f"[VRAM Available]                 : {vram_gb:.2f} GB")
else:
    print("[Warning] CUDA GPU not detected. Running on CPU.")

# --- CELL 2: Download Stage 1 Single-Class Dataset from Roboflow ---
dataset_dir = Path("dataset_stage1_single_class")
data_yaml_path = dataset_dir / "data.yaml"

if not data_yaml_path.exists():
    print("\n[Dataset] Downloading Stage 1 Single-Class Rubber Leaf Dataset from Roboflow...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
        project = rf.workspace("rubber-research").project("stage-1-rubber-leaf-detection-single-class")
        version = project.version(3)
        dataset = version.download("yolov8", location=str(dataset_dir))
        data_yaml_path = Path(dataset.location) / "data.yaml"
        print(f"[Dataset] Ready at: {data_yaml_path.resolve()}")
    except Exception as e:
        print(f"[Dataset Error] Could not download dataset: {e}")
        print("Ensure 'data.yaml' exists in current working directory.")
        data_yaml_path = Path("data.yaml")
else:
    print(f"\n[Dataset] Using existing dataset at: {data_yaml_path.resolve()}")

# --- CELL 3: YOLO11 Model Size Variants & Output Setup ---
model_variants = [
    {"name": "YOLO11n", "weights": "yolo11n.pt", "scale_label": "Nano (2.6M)",   "batch": 16},
    {"name": "YOLO11s", "weights": "yolo11s.pt", "scale_label": "Small (9.4M)",  "batch": 16},
    {"name": "YOLO11m", "weights": "yolo11m.pt", "scale_label": "Medium (20.1M)", "batch": 8}
]

output_dir = Path("stage1_yolo11_size_benchmark_output")
weights_dir = output_dir / "best_weights"
cm_dir = output_dir / "confusion_matrices"
curves_dir = output_dir / "training_curves"
onnx_dir = output_dir / "onnx_models"
runs_dir = output_dir / "per_size_runs"
reports_dir = output_dir / "00_summary_reports"

for d in [output_dir, weights_dir, cm_dir, curves_dir, onnx_dir, runs_dir, reports_dir]:
    d.mkdir(exist_ok=True, parents=True)

def extract_f1max_metrics(val_results):
    """Extracts optimal F1, Precision, and Recall at the F1-max threshold."""
    try:
        curves = val_results.box.curves_results
        f1_curve = curves[2][1]
        prec_curve = curves[0][1]
        rec_curve = curves[1][1]
        
        best_idx = np.argmax(f1_curve)
        best_f1 = float(f1_curve[best_idx]) * 100.0
        best_p = float(prec_curve[best_idx]) * 100.0
        best_r = float(rec_curve[best_idx]) * 100.0
    except Exception:
        best_p = float(val_results.box.p[0] * 100.0) if len(val_results.box.p) > 0 else float(val_results.box.mp * 100.0)
        best_r = float(val_results.box.r[0] * 100.0) if len(val_results.box.r) > 0 else float(val_results.box.mr * 100.0)
        best_f1 = float(2 * best_p * best_r / (best_p + best_r + 1e-6))
    return best_p, best_r, best_f1

benchmark_results = []

# --- CELL 4: Iterative Model Training, Validation & Dual Export ---
for model_info in model_variants:
    model_name = model_info["name"]
    weights = model_info["weights"]
    scale_label = model_info["scale_label"]
    batch_size = model_info["batch"]
    
    print("\n" + "=" * 85)
    print(f"RUNNING BENCHMARK FOR: {model_name} [{scale_label}] (Batch={batch_size})")
    print("=" * 85)
    
    save_run_name = f"stage1_{model_name.lower()}_run"
    size_run_dir = runs_dir / model_name
    size_w_dir = size_run_dir / "weights"
    size_cm_dir = size_run_dir / "confusion_matrices"
    size_log_dir = size_run_dir / "logs_and_curves"
    size_val_dir = size_run_dir / "val_predictions"
    
    for d in [size_w_dir, size_cm_dir, size_log_dir, size_val_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 1. Instantiate Model
    model = YOLO(weights)
    
    # 2. Train Model on Kaggle GPU with Seed=42 & Multi-Scale Augmentations
    print(f"[{model_name}] Starting Training (Epochs=100, Patience=10, Imgsz=640, Batch={batch_size}, Seed=42)...")
    train_start_time = time.time()
    try:
        train_results = model.train(
            data=str(data_yaml_path),
            epochs=100,
            patience=10,
            imgsz=640,
            batch=batch_size,
            seed=42,               # 100% Scientific reproducibility
            single_cls=True,       # Single-class Rubber Leaf
            project=str(output_dir / "runs"),
            name=save_run_name,
            device=device,
            # Harmonized Augmentations
            degrees=15.0,          # Random rotation (+/- 15 deg)
            flipud=0.5,            # Random vertical flip (50% chance)
            fliplr=0.5,            # Random horizontal flip (50% chance)
            scale=0.7,             # Random scale/zoom (+/- 70%)
            multi_scale=True,      # Dynamic multi-resolution training
            hsv_h=0.015,           # Hue variation
            hsv_s=0.7,             # Saturation jitter
            hsv_v=0.4,             # Brightness/value jitter
            plots=True,
            verbose=True
        )
        train_duration = time.time() - train_start_time
    except Exception as e:
        print(f"[Training Error] Failed to train {model_name}: {e}")
        continue
    
    # 3. Locate Best Trained Weights
    best_weights_path = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_weights_path.exists():
        best_weights_path = Path(weights)
        
    print(f"[{model_name}] Loading best weights for validation and latency benchmarking...")
    best_model = YOLO(str(best_weights_path))
    
    # Copy weights
    shutil.copy2(best_weights_path, weights_dir / f"{model_name}_best.pt")
    shutil.copy2(best_weights_path, size_w_dir / f"{model_name}_best.pt")
    last_weights_path = Path(train_results.save_dir) / "weights" / "last.pt"
    if last_weights_path.exists():
        shutil.copy2(last_weights_path, size_w_dir / f"{model_name}_last.pt")
    
    # 4. Run Validation & Held-Out Test Set Evaluations
    val_results = best_model.val(data=str(data_yaml_path), split="val", imgsz=640, single_cls=True, verbose=False)
    test_results = best_model.val(data=str(data_yaml_path), split="test", imgsz=640, single_cls=True, plots=True, name=f"{model_name}_test_eval", verbose=False)
    
    # 5. Measure Latency (10 Warmup + 50 Timing Loops on GPU)
    dummy_input = torch.randn(1, 3, 640, 640).to(next(best_model.model.parameters()).device)
    for _ in range(10):
        _ = best_model.model(dummy_input)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        latencies = []
        for _ in range(50):
            starter.record()
            _ = best_model.model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))
        avg_latency_ms = float(np.mean(latencies))
    else:
        t_start = time.perf_counter()
        for _ in range(50):
            _ = best_model.model(dummy_input)
        avg_latency_ms = float((time.perf_counter() - t_start) / 50 * 1000)

    # 6. Extract Standardized Test Metrics
    val_map50 = float(val_results.box.map50 * 100.0)
    test_map50 = float(test_results.box.map50 * 100.0)
    test_map5095 = float(test_results.box.map * 100.0)
    best_p, best_r, best_f1 = extract_f1max_metrics(test_results)
    
    params_m = round(sum(p.numel() for p in best_model.model.parameters()) / 1e6, 2)
    weight_size_mb = round(os.path.getsize(best_weights_path) / (1024 * 1024), 2)
    try:
        gflops = round(best_model.model.info(verbose=False)[3], 1)
    except Exception:
        gflops = "N/A"
    
    # 7. Export Model to ONNX Format
    print(f"[{model_name}] Exporting best model to ONNX format (imgsz=640, simplify=True)...")
    try:
        onnx_export_path = best_model.export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            simplify=True,
            opset=12
        )
        final_onnx_dest = onnx_dir / f"stage1_{model_name.lower()}_640.onnx"
        shutil.copy2(onnx_export_path, final_onnx_dest)
        print(f"[{model_name}] Saved ONNX to: {final_onnx_dest.resolve()}")
    except Exception as e:
        print(f"[{model_name} ONNX Export Warning] {e}")
        
    # 8. Copy Training Curves, Confusion Matrices & Validation Batches
    run_dir = Path(train_results.save_dir)
    for plot_file in run_dir.glob("*.png"):
        shutil.copy2(plot_file, size_log_dir / plot_file.name)
        if "results" in plot_file.name:
            shutil.copy2(plot_file, curves_dir / f"{model_name}_results.png")
            shutil.copy2(plot_file, size_log_dir / f"{model_name}_results.png")
        if "confusion_matrix" in plot_file.name:
            shutil.copy2(plot_file, size_cm_dir / plot_file.name)
            if "normalized" in plot_file.name:
                shutil.copy2(plot_file, cm_dir / f"{model_name}_confusion_matrix_normalized_RubberLeaf.png")
                shutil.copy2(plot_file, size_cm_dir / f"{model_name}_confusion_matrix_normalized_RubberLeaf.png")
            else:
                shutil.copy2(plot_file, cm_dir / f"{model_name}_confusion_matrix_RubberLeaf.png")
                shutil.copy2(plot_file, size_cm_dir / f"{model_name}_confusion_matrix_RubberLeaf.png")
                
    for val_img in run_dir.glob("val_batch*.jpg"):
        shutil.copy2(val_img, size_val_dir / val_img.name)
        
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        shutil.copy2(args_yaml, size_log_dir / "args.yaml")
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, size_log_dir / "results.csv")
            
    # Record Result Row
    benchmark_results.append({
        "Model Variant": model_name,
        "Scale Size": scale_label,
        "Parameters (M)": params_m,
        "GFLOPS": gflops,
        "Weights (.pt)": f"{weight_size_mb:.2f} MB",
        "Val mAP50 (%)": round(val_map50, 2),
        "Test mAP50 (%)": round(test_map50, 2),
        "Test mAP50-95 (%)": round(test_map5095, 2),
        "Precision (%)": round(best_p, 2),
        "Recall (%)": round(best_r, 2),
        "F1-Score (%)": round(best_f1, 2),
        "GPU Latency (ms)": round(avg_latency_ms, 2),
        "Train Time (min)": round(train_duration / 60, 1),
        "ONNX Export File": f"stage1_{model_name.lower()}_640.onnx"
    })

# --- CELL 5: Generate Publication-Ready Comparative Tables (CSV + Markdown + LaTeX) ---
df_results = pd.DataFrame(benchmark_results)
csv_path = reports_dir / "stage1_yolo11_size_comparison.csv"
md_path = reports_dir / "stage1_yolo11_size_comparison.md"
tex_path = reports_dir / "stage1_yolo11_size_comparison_latex.tex"

df_results.to_csv(csv_path, index=False)
df_results.to_csv(output_dir / "stage1_yolo11_size_comparison.csv", index=False)

markdown_table = """# Stage 1 YOLO11 Model Size Scaling Benchmark Results
## Research Project: Master's Thesis - Edge AI Rubber Leaf Disease Detection

Comparative evaluation across **YOLO11 Nano (11n)**, **Small (11s)**, and **Medium (11m)** architectures on single-class Rubber Leaf Localization:

| Model Variant | Scale Size     | Parameters (M) | Weights (.pt) | Val mAP50 (%) | Test mAP50 (%) | Test mAP50-95 (%) | Precision (%) | Recall (%) | F1-Score (%) | GPU Latency (ms) | Train Time (min) |
|:--------------|:---------------|---------------:|:--------------|--------------:|---------------:|------------------:|--------------:|-----------:|-------------:|-----------------:|-----------------:|
"""
for _, row in df_results.iterrows():
    markdown_table += f"| {row['Model Variant']:<13} | {row['Scale Size']:<14} | {row['Parameters (M)']:>14.2f} | {row['Weights (.pt)']:<13} | {row['Val mAP50 (%)']:>13.2f} | {row['Test mAP50 (%)']:>14.2f} | {row['Test mAP50-95 (%)']:>17.2f} | {row['Precision (%)']:>13.2f} | {row['Recall (%)']:>10.2f} | {row['F1-Score (%)']:>12.2f} | {row['GPU Latency (ms)']:>16.2f} | {row['Train Time (min)']:>16.1f} |\n"

with open(md_path, "w", encoding="utf-8") as f:
    f.write(markdown_table)
with open(output_dir / "stage1_yolo11_size_comparison.md", "w", encoding="utf-8") as f:
    f.write(markdown_table)

# Generate LaTeX Table
latex_code = r"""\begin{table}[htbp]
\centering
\caption{Stage 1 Single-Class Rubber Leaf Detector: YOLO11 Model Size Scaling Benchmark}
\label{tab:stage1_yolo11_size_benchmark}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model Scale} & \textbf{Params (M)} & \textbf{Weights} & \textbf{Val mAP50} & \textbf{Test mAP50} & \textbf{Test mAP50-95} & \textbf{Precision} & \textbf{Recall} & \textbf{Latency (ms)} \\
\midrule
"""
for _, row in df_results.iterrows():
    m_name = str(row['Model Variant'])
    params = f"{row['Parameters (M)']:.2f}"
    weights = str(row['Weights (.pt)'])
    val_map = f"{row['Val mAP50 (%)']:.2f}\\%"
    test_map = f"{row['Test mAP50 (%)']:.2f}\\%"
    test_map_95 = f"{row['Test mAP50-95 (%)']:.2f}\\%"
    p = f"{row['Precision (%)']:.2f}\\%"
    r = f"{row['Recall (%)']:.2f}\\%"
    lat = f"{row['GPU Latency (ms)']:.2f}"
    latex_code += f"{m_name} & {params} & {weights} & {val_map} & {test_map} & {test_map_95} & {p} & {r} & {lat} \\\\\n"
latex_code += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_code)

print("\n" + "=" * 85)
print("YOLO11 SIZE SCALING BENCHMARK SUMMARY RESULTS TABLE:")
print("=" * 85)
print(df_results.to_string(index=False))

# --- CELL 6: Bundle Everything into ZIP Archive ---
zip_filename = "yolo11_stage1_size_benchmark_results.zip"
print(f"\n[Packaging] Packaging complete YOLO11 size benchmark into '{zip_filename}'...")

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in output_dir.rglob("*"):
        if file_path.is_file() and not file_path.name.endswith(".zip"):
            rel_path = file_path.relative_to(output_dir)
            zipf.write(file_path, arcname=str(rel_path))

print("=" * 85)
print(f"SUCCESS! YOLO11 Size Benchmark complete. Archive: {Path(zip_filename).resolve()}")
print("=" * 85)
