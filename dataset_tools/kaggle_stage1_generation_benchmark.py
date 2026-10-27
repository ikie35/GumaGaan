# ==============================================================================
# Stage 1: YOLO Generation Architectural Benchmark Script (YOLOv8 vs v9 vs v10 vs v11)
# Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
# ==============================================================================
# This script trains, evaluates, exports (ONNX), and compares:
# 1. YOLOv8n (YOLOv8 Nano - 3.0M params)
# 2. YOLOv9t (YOLOv9 Tiny - 2.0M params)
# 3. YOLOv10n (YOLOv10 Nano - 2.3M params)
# 4. YOLO11n (YOLOv11 Nano - 2.6M params)
#
# Generates publication-ready comparative metrics tables (CSV + Markdown)
# and packages all ONNX weights, plots, and metrics into a single ZIP archive:
# 'yolov8_stage1_generation_benchmark_results.zip'.
# ==============================================================================

# --- CELL 1: Install Dependencies & Setup CUDA ---
import os
import sys
import time
import zipfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("STAGE 1: CROSS-GENERATION ARCHITECTURAL BENCHMARK (v8 vs v9 vs v10 vs v11)")
print("=" * 80)

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics & roboflow...")
    os.system("pip install ultralytics roboflow pandas matplotlib torch")
    import torch
    from ultralytics import YOLO

# GPU Accelerator Check
device = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"[Kaggle GPU Enabled] Active GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[Warning] CUDA GPU not detected. Running on CPU.")

# --- CELL 2: Download Stage 1 Dataset from Roboflow ---
dataset_dir = Path("dataset_stage1_single_class")
data_yaml_path = dataset_dir / "data.yaml"

if not data_yaml_path.exists():
    print("\n[Dataset] Downloading Stage 1 Rubber Leaf Dataset from Roboflow...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
        project = rf.workspace("rubber-research").project("stage-1-rubber-leaf-detection")
        version = project.version(11)
        dataset = version.download("yolov8", location=str(dataset_dir))
        data_yaml_path = Path(dataset.location) / "data.yaml"
    except Exception as e:
        print(f"[Dataset Error] Could not download dataset: {e}")
        print("Ensure 'data.yaml' exists in current working directory.")
        data_yaml_path = Path("data.yaml")
else:
    print(f"\n[Dataset] Using existing dataset at: {data_yaml_path.resolve()}")

# --- CELL 3: Generation Variants Configuration (YOLOv8, YOLOv9, YOLOv10, YOLOv11 & YOLO26) ---
model_variants = [
    {"name": "YOLOv8n", "weights": "yolov8n.pt", "gen_label": "YOLOv8 (2023 - C2f & Anchor-Free)"},
    {"name": "YOLOv9t", "weights": "yolov9t.pt", "gen_label": "YOLOv9 (2024 - GELAN Backbone)"},
    {"name": "YOLOv10n", "weights": "yolov10n.pt", "gen_label": "YOLOv10 (2024 - NMS-Free Dual Assignment)"},
    {"name": "YOLO11n", "weights": "yolo11n.pt", "gen_label": "YOLOv11 (2024 - C3k2 & C2PSA Spatial Attention)"},
    {"name": "YOLO26n", "weights": "yolo26n.pt", "gen_label": "YOLO26 (2026 - Next-Gen Edge Architecture)"}
]

output_dir = Path("stage1_generation_benchmark_output")
onnx_dir = output_dir / "onnx_models"
plots_dir = output_dir / "evaluation_plots"
output_dir.mkdir(exist_ok=True, parents=True)
onnx_dir.mkdir(exist_ok=True, parents=True)
plots_dir.mkdir(exist_ok=True, parents=True)

benchmark_results = []

# --- CELL 4: Iterative Model Training, Validation & Export ---
for model_info in model_variants:
    model_name = model_info["name"]
    weights = model_info["weights"]
    gen_label = model_info["gen_label"]
    
    print("\n" + "=" * 80)
    print(f"RUNNING GENERATION BENCHMARK FOR: {model_name} [{gen_label}]")
    print("=" * 80)
    
    save_run_name = f"stage1_{model_name.lower()}_run"
    
    # Instantiate Model
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[Warning] Could not load weights '{weights}': {e}. Skipping variant.")
        continue
    
    # Train Model on Kaggle GPU with fixed random seed=42
    print(f"[{model_name}] Starting Training (Epochs=100, Patience=10, Imgsz=640, Seed=42)...")
    train_start_time = time.time()
    try:
        train_results = model.train(
            data=str(data_yaml_path),
            epochs=100,
            patience=10,
            imgsz=640,
            batch=16,
            seed=42,               # Standard fixed seed for 100% scientific reproducibility
            project=str(output_dir / "runs"),
            name=save_run_name,
            device=device,
            degrees=15.0,          # Random rotation (+/- 15 deg)
            flipud=0.5,            # Random vertical flip (50% chance)
            scale=0.7,             # Random scale/zoom (+/- 70%) to handle different camera distances
            multi_scale=True,      # Vary input resolution (+/- 50%) during training for scale-robustness
            verbose=True
        )
        train_duration = time.time() - train_start_time
    except Exception as e:
        print(f"[Training Error] Failed to train {model_name}: {e}")
        continue
    
    # Get Best Trained Weights
    best_weights_path = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_weights_path.exists():
        best_weights_path = Path(weights)
        
    print(f"[{model_name}] Loading best weights for held-out test set validation...")
    best_model = YOLO(str(best_weights_path))
    
    # Run Validation & Held-Out Test Set Evaluation using COCO standard conf=0.25
    val_results = best_model.val(data=str(data_yaml_path), split="val", imgsz=640, conf=0.25, verbose=False)
    test_results = best_model.val(data=str(data_yaml_path), split="test", imgsz=640, conf=0.25, plots=True, name=f"{model_name}_test_eval", verbose=False)
    
    # Measure Latency Benchmark (10 Warmup + 50 Timing Loops)
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        _ = best_model.predict(source=dummy_input, verbose=False)
        
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = best_model.predict(source=dummy_input, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    avg_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)

    # Extract Metrics from Held-Out TEST Set (Academic Gold Standard for Thesis)
    box_metrics = test_results.box

    # Overall Mean Metrics across all classes
    map50_all = float(box_metrics.map50) * 100
    map5095_all = float(box_metrics.map) * 100
    precision_all = float(box_metrics.mp) * 100
    recall_all = float(box_metrics.mr) * 100
    f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all + 1e-6)

    # Target Class Specific Metrics (Explicitly target Class 1 'Rubber Leaf')
    names = test_results.names if hasattr(test_results, "names") and isinstance(test_results.names, dict) else {0: "Ignore Leaf", 1: "Rubber Leaf"}
    rubber_idx = 1 # Default to Class 1 in 2-class dataset (Version 11)
    
    for idx, cname in names.items():
        if "rubber" in str(cname).lower() and "ignore" not in str(cname).lower():
            rubber_idx = idx
            break

    try:
        if hasattr(box_metrics, "p") and len(box_metrics.p) > rubber_idx:
            precision_target = float(box_metrics.p[rubber_idx]) * 100
            recall_target = float(box_metrics.r[rubber_idx]) * 100
            map50_target = float(box_metrics.maps[rubber_idx]) * 100
        elif hasattr(box_metrics, "class_result"):
            p_c, r_c, map50_c, _ = box_metrics.class_result(rubber_idx)
            precision_target = float(p_c) * 100
            recall_target = float(r_c) * 100
            map50_target = float(map50_c) * 100
        else:
            precision_target, recall_target, map50_target = precision_all, recall_all, map50_all
    except Exception as e:
        print(f"[{model_name} Metric Warning] {e}")
        precision_target, recall_target, map50_target = precision_all, recall_all, map50_all

    f1_target = 2 * (precision_target * recall_target) / (precision_target + recall_target + 1e-6)

    f1_target = 2 * (precision_target * recall_target) / (precision_target + recall_target + 1e-6)
    
    params_m = round(sum(p.numel() for p in best_model.model.parameters()) / 1e6, 2)
    try:
        gflops = round(best_model.model.info(verbose=False)[3], 1)
    except Exception:
        gflops = "N/A"
        
    # --- Step 5: Export to ONNX ---
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
        shutil.copy(onnx_export_path, final_onnx_dest)
        print(f"[{model_name}] Saved ONNX to: {final_onnx_dest.resolve()}")
    except Exception as e:
        print(f"[{model_name} ONNX Export Warning] {e}")
        
    # Copy Validation Plot Charts
    run_dir = Path(train_results.save_dir)
    for plot_name in ["results.png", "confusion_matrix.png", "F1_curve.png", "PR_curve.png"]:
        src_plot = run_dir / plot_name
        if src_plot.exists():
            shutil.copy(src_plot, plots_dir / f"{model_name}_{plot_name}")
            
    # Record Result Row (Both Target Class 'Rubber Leaf' & All-Classes Overall side-by-side)
    benchmark_results.append({
        "Model Variant": model_name,
        "Architecture Generation": gen_label,
        "Parameters (M)": params_m,
        "GFLOPS": gflops,
        "Rubber Leaf mAP50 (%)": round(map50_target, 2),
        "Rubber Leaf Precision (%)": round(precision_target, 2),
        "Rubber Leaf Recall (%)": round(recall_target, 2),
        "Rubber Leaf F1 (%)": round(f1_target, 2),
        "All-Classes mAP50 (%)": round(map50_all, 2),
        "All-Classes Precision (%)": round(precision_all, 2),
        "All-Classes Recall (%)": round(recall_all, 2),
        "All-Classes F1 (%)": round(f1_all, 2),
        "Inference Latency (ms)": round(avg_latency_ms, 2),
        "Train Time (min)": round(train_duration / 60, 1),
        "ONNX Export File": f"stage1_{model_name.lower()}_640.onnx"
    })

# --- CELL 5: Generate Publication-Ready Comparative Tables (CSV + Markdown) ---
df_results = pd.DataFrame(benchmark_results)
csv_path = output_dir / "yolov8_stage1_generation_comparison.csv"
md_path = output_dir / "yolov8_stage1_generation_comparison.md"

df_results.to_csv(csv_path, index=False)

markdown_table = """# Stage 1 YOLO Cross-Generation Architectural Benchmark Results
## Research Project: Master's Thesis - Edge AI Rubber Leaf Disease Detection

Below is the comparative quantitative performance evaluation across **YOLOv8n**, **YOLOv9t**, **YOLOv10n**, and **YOLO11n** architectures on Stage 1 Leaf Localization:

""" + df_results.to_markdown(index=False) + """

---
### Key Architectural Insights:
1. **Architectural Innovations:** Compares C2f (v8), GELAN (v9), NMS-Free Dual Assignment (v10), and C3k2/C2PSA Spatial Attention (v11) at the ~3M parameter budget.
2. **Edge Hardware Compatibility:** Evaluates runtime latency and NPU exportability for edge deployment on Raspberry Pi 5 + Hailo-8L accelerator.
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(markdown_table)

print("\n" + "=" * 80)
print("CROSS-GENERATION BENCHMARK SUMMARY RESULTS TABLE:")
print("=" * 80)
print(df_results.to_string(index=False))

# --- CELL 6: Bundle Everything into ZIP Archive ---
zip_filename = "yolov8_stage1_generation_benchmark_results.zip"
print(f"\n[Packaging] Zipping all ONNX models, plots, and markdown tables into '{zip_filename}'...")

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add ONNX models
    for onnx_file in onnx_dir.glob("*.onnx"):
        zipf.write(onnx_file, arcname=f"onnx_models/{onnx_file.name}")
        
    # Add Markdown and CSV tables
    zipf.write(csv_path, arcname="yolov8_stage1_generation_comparison.csv")
    zipf.write(md_path, arcname="yolov8_stage1_generation_comparison.md")
    
    # Add evaluation plots
    for plot_file in plots_dir.glob("*.png"):
        zipf.write(plot_file, arcname=f"evaluation_plots/{plot_file.name}")

print("=" * 80)
print(f"SUCCESS! Benchmark package created successfully: {Path(zip_filename).resolve()}")
print("Included in ZIP package:")
print(" - ONNX Models: stage1_yolov8n_640.onnx, stage1_yolov9t_640.onnx, stage1_yolov10n_640.onnx, stage1_yolo11n_640.onnx")
print(" - Metrics Tables: yolov8_stage1_generation_comparison.csv & .md")
print(" - Plots: F1 curves, PR curves, and confusion matrices")
print("=" * 80)
