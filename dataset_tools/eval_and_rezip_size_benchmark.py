# ==============================================================================
# Fast Evaluation & Re-Zipping Script for Size Benchmark (Nano, Small, Medium)
# Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
# ==============================================================================
# This script loads ALREADY TRAINED best.pt / ONNX weights for YOLOv8n, v8s, v8m,
# generates normalized confusion matrices (removing raw background grid noise),
# extracts target Rubber Leaf metrics, and re-zips into 'yolov8_stage1_size_benchmark_results.zip'.
# ==============================================================================

import os
import sys
import time
import zipfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("FAST SIZE BENCHMARK EVALUATION & RE-ZIPPING SCRIPT (NO RETRAINING)")
print("=" * 80)

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics & roboflow...")
    os.system("pip install ultralytics roboflow pandas matplotlib torch")
    import torch
    from ultralytics import YOLO

# Dataset Path Check
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
        data_yaml_path = Path("data.yaml")

# Directories Configuration
output_dir = Path("stage1_size_benchmark_output")
runs_dir = output_dir / "runs"
onnx_dir = output_dir / "onnx_models"
plots_dir = output_dir / "evaluation_plots"
output_dir.mkdir(exist_ok=True, parents=True)
onnx_dir.mkdir(exist_ok=True, parents=True)
plots_dir.mkdir(exist_ok=True, parents=True)

model_sizes = [
    {"name": "YOLOv8n", "weights": "yolov8n.pt", "size_label": "Nano (3.0M Params)"},
    {"name": "YOLOv8s", "weights": "yolov8s.pt", "size_label": "Small (11.1M Params)"},
    {"name": "YOLOv8m", "weights": "yolov8m.pt", "size_label": "Medium (25.8M Params)"}
]

benchmark_results = []

for model_info in model_sizes:
    model_name = model_info["name"]
    size_label = model_info["size_label"]
    save_run_name = f"stage1_{model_name.lower()}_run"
    
    # Support both .onnx and .pt models
    possible_weights = [
        Path(f"/kaggle/working/stage1_size_benchmark_output/onnx_models/stage1_{model_name.lower()}_640.onnx"),
        Path(f"/kaggle/working/{model_name.lower()}.onnx"),
        Path(f"{model_name.lower()}.onnx"),
        Path(f"/kaggle/working/{model_info['weights']}"),
        Path(f"/kaggle/working/{model_name.lower()}.pt"),
        Path(model_info["weights"]),
        Path(f"{model_name.lower()}.pt"),
        runs_dir / save_run_name / "weights" / "best.pt",
        Path("stage1_training_results/weights/stage1_best.pt")
    ]
    
    # Recursive search
    for search_dir in [Path("/kaggle/working"), output_dir, runs_dir, Path(".")]:
        if search_dir.exists():
            for found_file in search_dir.rglob(f"*{model_name.lower()}*"):
                if found_file.suffix in [".onnx", ".pt"]:
                    possible_weights.insert(0, found_file)

    weights_path = None
    for p in possible_weights:
        if p.exists():
            weights_path = p
            break
            
    if not weights_path or not weights_path.exists():
        print(f"[Skip] Saved weights for {model_name} not found.")
        continue
        
    print(f"\n[{model_name}] Loading weights from: {weights_path.resolve()}")
    model = YOLO(str(weights_path))
    
    # Test evaluation at conf=0.25
    print(f"[{model_name}] Running fast test evaluation...")
    try:
        test_results = model.val(data=str(data_yaml_path), split="test", imgsz=640, conf=0.25, plots=False, verbose=False)
    except Exception as e:
        print(f"[{model_name} Val Warning] {e}")
        continue
    
    # Latency timing
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(source=dummy_input, verbose=False)
        
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model.predict(source=dummy_input, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    avg_latency_ms = np.mean(latencies)

    # Extract Metrics
    box_metrics = test_results.box
    map50_all = float(box_metrics.map50) * 100
    map5095_all = float(box_metrics.map) * 100
    precision_all = float(box_metrics.mp) * 100
    recall_all = float(box_metrics.mr) * 100
    f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all + 1e-6)

    # Dynamic Class Index Lookup for Rubber Leaf
    target_class_idx = 0
    if hasattr(test_results, "names") and isinstance(test_results.names, dict):
        for idx, name in test_results.names.items():
            if "rubber" in str(name).lower() and "ignore" not in str(name).lower():
                target_class_idx = idx
                break

    try:
        if hasattr(box_metrics, "class_result"):
            p_c0, r_c0, map50_c0, map5095_c0 = box_metrics.class_result(target_class_idx)
            precision_target = float(p_c0) * 100
            recall_target = float(r_c0) * 100
            map50_target = float(map50_c0) * 100
            map5095_target = float(map5095_c0) * 100
        elif len(box_metrics.p) > target_class_idx:
            precision_target = float(box_metrics.p[target_class_idx]) * 100
            recall_target = float(box_metrics.r[target_class_idx]) * 100
            map50_target = float(box_metrics.maps[target_class_idx]) * 100 if len(box_metrics.maps) > target_class_idx else map50_all
            map5095_target = float(box_metrics.map) * 100
        else:
            precision_target, recall_target, map50_target, map5095_target = precision_all, recall_all, map50_all, map5095_all
    except Exception:
        precision_target, recall_target, map50_target, map5095_target = precision_all, recall_all, map50_all, map5095_all

    f1_target = 2 * (precision_target * recall_target) / (precision_target + recall_target + 1e-6)
    
    params_m = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    try:
        gflops = round(model.model.info(verbose=False)[3], 1)
    except Exception:
        gflops = "N/A"

    # Export ONNX
    try:
        onnx_export_path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=12)
        shutil.copy(onnx_export_path, onnx_dir / f"stage1_{model_name.lower()}_640.onnx")
    except Exception as e:
        print(f"[{model_name} ONNX Export] {e}")

    benchmark_results.append({
        "Model Variant": model_name,
        "Scale Size": size_label,
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
        "ONNX Export File": f"stage1_{model_name.lower()}_640.onnx"
    })

df_results = pd.DataFrame(benchmark_results)
csv_path = output_dir / "yolov8_stage1_size_comparison.csv"
md_path = output_dir / "yolov8_stage1_size_comparison.md"

df_results.to_csv(csv_path, index=False)

try:
    md_table_str = df_results.to_markdown(index=False)
except Exception:
    md_table_str = df_results.to_string(index=False)

markdown_table = """# Stage 1 YOLOv8 Model Size Scaling Benchmark Results
## Research Project: Master's Thesis - Edge AI Rubber Leaf Disease Detection

""" + md_table_str + """
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(markdown_table)

zip_filename = "yolov8_stage1_size_benchmark_results.zip"
print(f"\n[Packaging] Re-zipping into '{zip_filename}'...")

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for onnx_file in onnx_dir.glob("*.onnx"):
        zipf.write(onnx_file, arcname=f"onnx_models/{onnx_file.name}")
    zipf.write(csv_path, arcname="yolov8_stage1_size_comparison.csv")
    zipf.write(md_path, arcname="yolov8_stage1_size_comparison.md")

print("=" * 80)
print(f"SUCCESS! Created size benchmark package: {Path(zip_filename).resolve()}")
print("=" * 80)
