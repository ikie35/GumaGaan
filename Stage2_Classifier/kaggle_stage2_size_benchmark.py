"""
Stage 2: Rubber Leaf Disease Classifier Model Size Comparison Benchmark
========================================================================
Optimized for Kaggle Notebooks with NVIDIA Tesla T4 GPU Acceleration
========================================================================
Dataset: Roboflow "stage-2-rubber-leaf-classifier" (v4)
Classes: [Healthy, Other Disease, PLFD]
Framework: Ultralytics YOLOv8-cls (PyTorch + CUDA Acceleration)
Comparison: 3 Model Scales (Nano vs Small vs Medium)
  1. YOLOv8n-cls (Nano   ~ 1.45M parameters, batch=32)
  2. YOLOv8s-cls (Small  ~ 5.08M parameters, batch=32)
  3. YOLOv8m-cls (Medium ~ 15.76M parameters, batch=16)

Outputs:
  - Best PyTorch weights (.pt) and ONNX exports (.onnx) for each model size
  - Publication-grade 3x3 Confusion Matrices (PNG + PDF)
  - Comparative Markdown, LaTeX, and CSV benchmark tables
  - All artifacts bundled into 'stage2_size_comparison_results.zip'
"""

import os
import sys
import time
import shutil
import zipfile
import logging
import subprocess
from pathlib import Path

# ── 1. Automated Dependency Management ──────────────────────────────────────
REQUIRED_PACKAGES = ["roboflow", "ultralytics", "pandas", "matplotlib", "scikit-learn", "tabulate"]

def ensure_packages(packages):
    """Installs missing dependencies safely in Kaggle or local environments."""
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[Setup] Installing required package: {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

ensure_packages(REQUIRED_PACKAGES)

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from roboflow import Roboflow
from ultralytics import YOLO

# ── 2. Logging Setup ────────────────────────────────────────────────────────
LOG_FILE = Path("stage2_size_benchmark.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── 3. Accelerator Verification ─────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("=" * 70)
    log.info(f"[GPU Accelerator Active] : {gpu_name}")
    log.info(f"[VRAM Available]         : {vram_gb:.2f} GB")
    log.info("=" * 70)
else:
    log.warning("=" * 70)
    log.warning("[WARNING] No CUDA GPU detected. Benchmark running on CPU.")
    log.warning("=" * 70)

# ── 4. Dataset Download from Roboflow ───────────────────────────────────────
DATASET_DIR = Path("Stage-2-Rubber-Leaf-Classifier-4")

def acquire_dataset():
    """Downloads or locates the Stage 2 3-class disease dataset."""
    if DATASET_DIR.exists() and (DATASET_DIR / "train").exists():
        log.info(f"[Dataset] Found local dataset at: {DATASET_DIR.resolve()}")
        return DATASET_DIR

    log.info("[Dataset] Downloading Stage 2 dataset from Roboflow API (v4)...")
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
    version = project.version(4)
    ds = version.download("folder")
    return Path(ds.location)

# ── 5. Confusion Matrix Visualizers (Standard Kaggle / Matplotlib Style) ───
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'sans-serif']
plt.rcParams['figure.autolayout'] = True

def plot_confusion_matrix(cm_array: np.ndarray, class_names: list, model_key: str, save_dir: Path, normalize: bool = True):
    """
    Exact 1:1 replica of Ultralytics official confusion matrix layout:
    - Dimensions: 12x9 inches @ 250 DPI -> 3000 x 2250 px
    - Y-Axis: 'Predicted'
    - X-Axis: 'True' with vertical 90-degree class names
    - Title: 'Confusion Matrix' or 'Confusion Matrix Normalized'
    """
    mat = np.array(cm_array, dtype=float)
    if normalize:
        norm_mat = mat / (mat.sum(axis=0, keepdims=True) + 1e-9)
        display_mat = norm_mat
        title = "Confusion Matrix Normalized"
        filename = f"{model_key}_confusion_matrix_normalized.png"
        vmax = 1.0
    else:
        display_mat = mat
        title = "Confusion Matrix"
        filename = f"{model_key}_confusion_matrix.png"
        vmax = mat.max()

    fig, ax = plt.subplots(figsize=(12, 9), dpi=250)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    im = ax.imshow(display_mat, cmap='Blues', interpolation='nearest', vmin=0, vmax=vmax)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    nc = len(class_names)
    ax.set_xticks(range(nc))
    ax.set_yticks(range(nc))
    ax.set_xticklabels(class_names, rotation=90, fontsize=11, ha='center')
    ax.set_yticklabels(class_names, fontsize=11)

    ax.set_xlabel('True', fontsize=12, labelpad=12)
    ax.set_ylabel('Predicted', fontsize=12, labelpad=12)
    ax.set_title(title, fontsize=14, pad=15)

    thresh = vmax / 2.0
    for i in range(nc):
        for j in range(nc):
            val = display_mat[i, j]
            text_color = 'white' if val > thresh else 'black'
            val_str = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, val_str, ha='center', va='center', color=text_color, fontsize=11)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out_file_png = save_dir / filename
    plt.savefig(out_file_png, bbox_inches='tight', dpi=250)
    plt.close(fig)

def plot_combined_confusion_matrices(all_cms: dict, class_names: list, save_path: Path):
    """Generates a 3-panel comparative confusion matrix figure (Nano vs Small vs Medium) for Chapter 4."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    
    for idx, (m_name, cm) in enumerate(all_cms.items()):
        ax = axes[idx]
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)
        
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=10, rotation=15, ha='right')
        
        if idx == 0:
            ax.set_yticklabels(class_names, fontsize=10)
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels(['', '', ''])
            
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_title(m_name, fontsize=12, fontweight='bold', pad=10)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm_norm[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color=text_color, fontsize=11, fontweight='bold')
                        
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.015, 0.65])
    fig.colorbar(im, cax=cbar_ax, label='Normalized Ratio')
    
    plt.suptitle("Stage 2: Cross-Scale Confusion Matrix Comparison (Test Split)", fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

# ── 6. Main Benchmark Runner ────────────────────────────────────────────────
def run_stage2_size_benchmark():
    dataset_path = acquire_dataset()
    output_root = Path("stage2_size_benchmark_results")
    output_root.mkdir(parents=True, exist_ok=True)

    models_config = [
        {"name": "yolov8n-cls", "size_label": "Nano",   "batch": 32, "epochs": 100, "patience": 15},
        {"name": "yolov8s-cls", "size_label": "Small",  "batch": 32, "epochs": 100, "patience": 15},
        {"name": "yolov8m-cls", "size_label": "Medium", "batch": 16, "epochs": 100, "patience": 15},
    ]

    benchmark_summary = []
    per_class_summary = []
    all_confusion_matrices = {}
    
    classes = ["Healthy", "Other Disease", "PLFD"]

    for cfg in models_config:
        model_name = cfg["name"]
        size_label = cfg["size_label"]
        batch_size = cfg["batch"]
        epochs = cfg["epochs"]
        patience = cfg["patience"]

        log.info("=" * 70)
        log.info(f"BENCHMARKING STAGE 2 SCALE: {model_name.upper()} ({size_label})")
        log.info("=" * 70)

        # ── Initialize & Train Model ────────────────────────────
        model = YOLO(f"{model_name}.pt")
        
        train_start = time.perf_counter()
        results = model.train(
            data=str(dataset_path),
            epochs=epochs,
            patience=patience,
            imgsz=224,
            batch=batch_size,
            project="stage2_size_comparison",
            name=f"{model_name}_run",
            device=device,
            # Enhanced Photometric & Shadow Augmentations for Field & Indoor Invariance
            dropout=0.20,
            hsv_h=0.03,
            hsv_s=0.70,
            hsv_v=0.60,      # High brightness jitter teaches model to ignore ambient room shadows
            degrees=30.0,
            fliplr=0.5,
            flipud=0.5,
            shear=10.0,
            seed=42,
            verbose=True
        )
        train_time_min = (time.perf_counter() - train_start) / 60.0

        # ── Model Size & Parameter Extraction ───────────────────
        best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = Path("stage2_size_comparison") / f"{model_name}_run" / "weights" / "best.pt"

        # Copy best weights to output directory
        model_out_dir = output_root / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, model_out_dir / f"{model_name}_best.pt")

        # Load best model for evaluation & export
        eval_model = YOLO(str(best_pt))
        
        # Calculate parameters and GFLOPs
        param_count = sum(p.numel() for p in eval_model.model.parameters()) / 1e6
        pt_size_mb = best_pt.stat().st_size / (1024 * 1024)

        # ── Export to ONNX (Opset 12 for Hailo compatibility) ────
        log.info(f"[{model_name}] Exporting to ONNX format (opset 12)...")
        onnx_path = eval_model.export(format="onnx", imgsz=224, opset=12, dynamic=False)
        onnx_file = Path(onnx_path)
        onnx_size_mb = onnx_file.stat().st_size / (1024 * 1024)
        shutil.copy(onnx_file, model_out_dir / f"{model_name}_best.onnx")

        # ── Test Set Rigorous Evaluation ─────────────────────────
        test_dir = dataset_path / "test"
        log.info(f"[{model_name}] Evaluating on independent TEST split: {test_dir}")
        
        y_true = []
        y_pred = []
        latencies_ms = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        for class_idx, class_name in enumerate(classes):
            class_folder = test_dir / class_name
            if not class_folder.exists():
                continue
            
            image_files = [f for f in class_folder.iterdir() if f.suffix.lower() in valid_exts]
            for img_file in image_files:
                y_true.append(class_idx)
                
                t0 = time.perf_counter()
                pred_res = eval_model.predict(source=str(img_file), imgsz=224, verbose=False)[0]
                t1 = time.perf_counter()
                
                latencies_ms.append((t1 - t0) * 1000)
                pred_idx = int(pred_res.probs.top1)
                y_pred.append(pred_idx)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Metrics computation
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        avg_latency_ms = np.mean(latencies_ms)

        log.info(f"[{model_name.upper()} RESULTS] Accuracy: {acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}% | Latency: {avg_latency_ms:.2f} ms")

        # Record Per-Class Breakdown
        for c_idx, c_name in enumerate(classes):
            per_class_summary.append({
                "Model": model_name,
                "Size": size_label,
                "Class": c_name,
                "Support": int(support_per_class[c_idx]),
                "Precision (%)": round(p_per_class[c_idx] * 100, 2),
                "Recall (%)": round(r_per_class[c_idx] * 100, 2),
                "F1-Score (%)": round(f1_per_class[c_idx] * 100, 2),
            })

        # Record Global Summary
        benchmark_summary.append({
            "Model Architecture": model_name,
            "Scale": size_label,
            "Parameters (M)": round(param_count, 2),
            "Weights (.pt)": f"{pt_size_mb:.2f} MB",
            "ONNX Size": f"{onnx_size_mb:.2f} MB",
            "Top-1 Accuracy (%)": round(acc * 100, 2),
            "Macro Precision (%)": round(p_macro * 100, 2),
            "Macro Recall (%)": round(r_macro * 100, 2),
            "Macro F1-Score (%)": round(f1_macro * 100, 2),
            "Inference Latency (ms)": round(avg_latency_ms, 2),
            "Training Time (min)": round(train_time_min, 1)
        })

        # Generate Exact Ultralytics Native Confusion Matrices (Predicted on Y, True on X)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_ultralytics_layout = cm.T  # Transpose so row = Predicted, col = True
        
        clean_key = model_name.replace('-cls', '').lower()
        plot_confusion_matrix(cm_ultralytics_layout, classes, clean_key, model_out_dir, normalize=True)
        plot_confusion_matrix(cm_ultralytics_layout, classes, clean_key, model_out_dir, normalize=False)
        
        # Copy to root output directory
        shutil.copy(model_out_dir / f"{clean_key}_confusion_matrix_normalized.png", output_root / f"{clean_key}_confusion_matrix_normalized.png")
        shutil.copy(model_out_dir / f"{clean_key}_confusion_matrix.png", output_root / f"{clean_key}_confusion_matrix.png")
        all_confusion_matrices[f"{model_name.upper()} ({size_label})"] = cm

    # ── 7. Generate 3-Scale Combined Comparative Panel ───────────────────────
    log.info("Generating 3-scale comparative confusion matrix panel...")
    plot_combined_confusion_matrices(all_confusion_matrices, classes, output_root / "stage2_all_scales_confusion_matrices")

    # ── 8. Generate Comparison Tables ────────────────────────────────────────
    df_global = pd.DataFrame(benchmark_summary)
    df_per_class = pd.DataFrame(per_class_summary)

    csv_global = output_root / "stage2_model_size_comparison.csv"
    csv_per_class = output_root / "stage2_per_class_metrics.csv"
    df_global.to_csv(csv_global, index=False)
    df_per_class.to_csv(csv_per_class, index=False)

    md_report_path = output_root / "stage2_model_size_benchmark_results.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Stage 2: Rubber Leaf Disease Classifier Model Size Benchmark\n\n")
        f.write("### 1. Overall Performance Comparison Across Model Scales\n\n")
        f.write(df_global.to_markdown(index=False))
        f.write("\n\n---\n\n### 2. Per-Class Diagnostic Performance (Test Split)\n\n")
        f.write(df_per_class.to_markdown(index=False))
        f.write("\n\n---\n\n### 3. LaTeX Manuscript Table Format (Chapter 4)\n\n```latex\n")
        f.write(df_global.to_latex(index=False, caption="Stage 2 YOLOv8-cls Model Size and Performance Comparison", label="tab:stage2_size_benchmark"))
        f.write("```\n")

    log.info("\n" + "=" * 70)
    log.info("STAGE 2 MODEL SCALE BENCHMARK COMPLETE!")
    log.info("=" * 70)
    print("\n" + df_global.to_markdown(index=False) + "\n")
    print("\n" + df_per_class.to_markdown(index=False) + "\n")

    # ── 8. Bundle All Results into a Single ZIP ──────────────────────────────
    zip_path = Path("stage2_size_comparison_results.zip")
    log.info(f"Packaging all model weights, plots, and tables into {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in output_root.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(output_root))

    log.info(f"✅ ZIP Archive ready for download: {zip_path.resolve()} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    run_stage2_size_benchmark()
