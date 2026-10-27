"""
================================================================================
Stage 2: YOLO11-cls Field-Robust & Lesion-Attention Training Script
Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
================================================================================
Dataset: Roboflow "stage-2-rubber-leaf-classifier" (v4)
Classes: [Healthy, Other Disease, PLFD]
Model: YOLO11n-cls (C3k2 + Spatial Attention Backbone)
Resolution: 224x224
Target Deployment: Raspberry Pi 5 + Hailo-8L AI NPU HAT

Key Enhancements for Field Lighting, Angles & Pixel-Level Lesion Attention:
  1. Random Erasing (erasing=0.40): Randomly cuts out image patches, forcing the
     network to look for micro-lesion pixels (margins, necrotic spots) rather than
     cheating on overall leaf brightness/color.
  2. Extreme Brightness Jitter (hsv_v=0.70): Simulates deep canopy shadows and
     bright midday tropical sun glare.
  3. Extreme Saturation Jitter (hsv_s=0.80): Prevents washed-out green leaves from
     being misdiagnosed as fungal chlorosis.
  4. Oblique Perspective & Rotation (shear=15.0, degrees=45.0): Ensures diagnosis
     remains stable across handheld camera tilt angles.
  5. Dropout Regularization (dropout=0.25): Prevents dense head overfitting.

Outputs:
  - PyTorch (.pt) weights and ONNX (.onnx, opset 12 for Hailo-8L compilation)
  - Publication-grade Normalized Confusion Matrices (PNG + PDF)
  - Per-class metrics & comparative tables (CSV, Markdown, LaTeX)
  - Packaged ZIP archive: 'stage2_yolo11n_robust_results.zip'
================================================================================
"""

import gc
import os
import sys
import time
import shutil
import zipfile
import logging
import subprocess
from pathlib import Path

# ── 1. Dependency Management ────────────────────────────────────────────────
REQUIRED_PACKAGES = ["roboflow", "ultralytics", "pandas", "matplotlib", "scikit-learn", "onnxsim"]

def ensure_packages(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[Setup] Installing required package: {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

ensure_packages(REQUIRED_PACKAGES)

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from roboflow import Roboflow
from ultralytics import YOLO

# ── 2. Logging Setup ────────────────────────────────────────────────────────
LOG_FILE = Path("stage2_yolo11n_training.log")
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
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ── 3. Accelerator Verification ─────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("=" * 75)
    log.info(f"[GPU Accelerator Active] : {gpu_name}")
    log.info(f"[VRAM Available]         : {vram_gb:.2f} GB")
    log.info("=" * 75)
else:
    log.warning("=" * 75)
    log.warning("[WARNING] No CUDA GPU detected. Running on CPU.")
    log.warning("=" * 75)

# ── 4. Dataset Acquisition ──────────────────────────────────────────────────
DATASET_DIR = Path("Stage-2-Rubber-Leaf-Classifier-4")

def acquire_dataset():
    if DATASET_DIR.exists() and (DATASET_DIR / "train").exists():
        log.info(f"[Dataset] Found local dataset at: {DATASET_DIR.resolve()}")
        return DATASET_DIR
    log.info("[Dataset] Downloading Stage 2 dataset from Roboflow API (v4)...")
    try:
        rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
        project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
        version = project.version(4)
        ds = version.download("folder")
        return Path(ds.location)
    except Exception as e:
        log.error(f"[Dataset Error] {e}")
        return DATASET_DIR

# ── 5. Confusion Matrix Visualizer ──────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"]
plt.rcParams["figure.autolayout"] = True

def plot_confusion_matrix(cm_array, class_names, model_key, save_dir, normalize=True):
    mat = np.array(cm_array, dtype=float)
    if normalize:
        row_sums    = mat.sum(axis=1, keepdims=True)
        display_mat = mat / (row_sums + 1e-9)
        title       = f"{model_key} -- Normalized Confusion Matrix (Test Split)"
        filename    = f"{model_key}_confusion_matrix_normalized.png"
        vmax        = 1.0
    else:
        display_mat = mat
        title       = f"{model_key} -- Confusion Matrix (Test Split)"
        filename    = f"{model_key}_confusion_matrix.png"
        vmax        = mat.max()

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    im   = ax.imshow(display_mat, cmap="Blues", interpolation="nearest", vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    nc = len(class_names)
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    ax.set_xticklabels(class_names, rotation=30, fontsize=11, ha="right")
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    thresh = vmax / 2.0
    for i in range(nc):
        for j in range(nc):
            val = display_mat[i, j]
            tc  = "white" if val > thresh else "black"
            vs  = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, vs, ha="center", va="center", color=tc, fontsize=11, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / filename, bbox_inches="tight", dpi=300)
    plt.savefig(save_dir / f"{model_key}_confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)

# ── 6. Main Field-Robust Training Pipeline ──────────────────────────────────
def train_field_robust_yolo11n():
    dataset_path = acquire_dataset()
    output_root  = Path("stage2_yolo11n_robust_results")
    weights_dir  = output_root / "best_weights"
    cm_dir       = output_root / "confusion_matrices"
    onnx_dir     = output_root / "onnx_models"
    reports_dir  = output_root / "00_summary_reports"
    for d in [output_root, weights_dir, cm_dir, onnx_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model_name = "yolo11n-cls"
    classes    = ["Healthy", "Other Disease", "PLFD"]
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    log.info("=" * 75)
    log.info(f"STARTING FIELD-ROBUST TRAINING: {model_name.upper()} (Spatial Attention)")
    log.info("=" * 75)

    # Initialize YOLO11n-cls
    model = YOLO(f"{model_name}.pt")

    # ── Train with Field Lighting & Lesion Attention Augmentations ──
    train_start = time.perf_counter()
    try:
        train_results = model.train(
            data=str(dataset_path),
            epochs=100,
            patience=15,
            imgsz=224,
            batch=32,
            project=str(output_root / "runs"),
            name="yolo11n_field_robust_run",
            device=device,
            
            # ── 1. Force Focus on Specific Lesion Pixels ────────
            erasing=0.40,        # 40% Random Erasing cuts out image patches, forcing
                                 # network to find micro-lesion pixels instead of global color
            
            # ── 2. Field Lighting & Sun Glare Robustness ───────
            hsv_v=0.70,          # Extreme brightness jitter (±70%): teaches model to
                                 # ignore deep shadows and specular sunlight glares
            hsv_s=0.80,          # Extreme saturation jitter (±80%): prevents washed-out
                                 # green from mimicking fungal chlorosis
            hsv_h=0.05,          # Accommodates Pi Camera AWB drift
            
            # ── 3. Angle & Oblique Perspective Invariance ─────
            degrees=45.0,        # Full rotation invariance (±45°)
            shear=15.0,          # Oblique perspective distortion (±15°)
            fliplr=0.5,
            flipud=0.5,
            
            # ── 4. Regularization ──────────────────────────────
            dropout=0.25,        # 25% Dropout on dense head prevents texture memorization
            seed=42,
            verbose=True
        )
        train_time_min = (time.perf_counter() - train_start) / 60.0
    except Exception as e:
        log.error(f"[Training Error] {e}")
        return

    # Locate Best Weights
    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = output_root / "runs" / "yolo11n_field_robust_run" / "weights" / "best.pt"

    # Copy plots & weights
    run_dir = Path(model.trainer.save_dir)
    for plot_file in run_dir.glob("*.png"):
        shutil.copy2(plot_file, output_root / plot_file.name)
    shutil.copy2(best_pt, weights_dir / "yolo11n_robust_best.pt")

    # Free training VRAM
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    # Load Best Model for Evaluation & Export
    eval_model  = YOLO(str(best_pt))
    param_count = sum(p.numel() for p in eval_model.model.parameters()) / 1e6
    pt_size_mb  = best_pt.stat().st_size / (1024 * 1024)

    # Official Val Split Metrics
    log.info("[Validation] Running official validation on val split...")
    val_res = eval_model.val(data=str(dataset_path), imgsz=224, device=device, verbose=False)
    top1_val = float(getattr(val_res, "top1", 0.0)) * 100.0
    top5_val = float(getattr(val_res, "top5", 0.0)) * 100.0
    log.info(f"[Validation] Top-1 Accuracy: {top1_val:.2f}% | Top-5: {top5_val:.2f}%")

    # Export to ONNX (opset 12 for Hailo-8L Dataflow Compiler)
    log.info("[Export] Exporting to ONNX format (opset 12, simplify=True)...")
    onnx_size_mb = 0.0
    try:
        onnx_path    = eval_model.export(format="onnx", imgsz=224, opset=12, dynamic=False, simplify=True)
        onnx_file    = Path(onnx_path)
        onnx_size_mb = onnx_file.stat().st_size / (1024 * 1024)
        shutil.copy2(onnx_file, onnx_dir / "yolo11n_robust_best.onnx")
        log.info(f"[Export] ONNX model saved ({onnx_size_mb:.2f} MB)")
    except Exception as e:
        log.warning(f"[ONNX Warning] {e}")

    # ── Rigorous Test Split Evaluation ──────────────────────────────
    test_dir = dataset_path / "test"
    log.info(f"[Evaluation] Testing on independent held-out split: {test_dir}")
    y_true = []; y_pred = []; latencies_ms = []

    # GPU warmup passes
    warmup_img = None
    for class_name in classes:
        _folder = test_dir / class_name
        if _folder.exists():
            _imgs = [f for f in _folder.iterdir() if f.suffix.lower() in valid_exts]
            if _imgs: warmup_img = str(_imgs[0]); break
    if warmup_img:
        log.info("[Warmup] Priming CUDA kernels before timing...")
        for _ in range(5):
            eval_model.predict(source=warmup_img, imgsz=224, device=device, verbose=False)

    for class_idx, class_name in enumerate(classes):
        class_folder = test_dir / class_name
        if not class_folder.exists():
            log.warning(f"Test folder missing: {class_folder}"); continue
        for img_file in [f for f in class_folder.iterdir() if f.suffix.lower() in valid_exts]:
            y_true.append(class_idx)
            pred = eval_model.predict(source=str(img_file), imgsz=224, device=device, verbose=False)
            latencies_ms.append(pred[0].speed["inference"])
            y_pred.append(int(pred[0].probs.top1))

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred) * 100.0
    p_macro, r_macro, f1_macro, _         = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_per, r_per, f1_per, sup_per         = precision_recall_fscore_support(y_true, y_pred, average=None,       zero_division=0)
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    log.info("=" * 75)
    log.info(f"FINAL TEST ACCURACY : {acc:.2f}%")
    log.info(f"MACRO F1-SCORE      : {f1_macro*100.0:.2f}%")
    log.info(f"WEIGHTED F1-SCORE   : {f1_weighted*100.0:.2f}%")
    log.info(f"GPU INFERENCE TIME  : {avg_latency:.2f} ms")
    log.info("=" * 75)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plot_confusion_matrix(cm, classes, "yolo11n_robust", cm_dir, normalize=True)
    plot_confusion_matrix(cm, classes, "yolo11n_robust", cm_dir, normalize=False)
    plot_confusion_matrix(cm, classes, "yolo11n_robust", output_root, normalize=True)

    # ── Summary Reports (CSV, Markdown, LaTeX) ──────────────────────
    summary_data = [{
        "Model": "YOLO11n-cls (Field-Robust)",
        "Parameters (M)": round(param_count, 2),
        "Weights (.pt) MB": round(pt_size_mb, 2),
        "ONNX (MB)": round(onnx_size_mb, 2),
        "Test Accuracy (%)": round(acc, 2),
        "Macro Precision (%)": round(p_macro * 100.0, 2),
        "Macro Recall (%)": round(r_macro * 100.0, 2),
        "Macro F1-Score (%)": round(f1_macro * 100.0, 2),
        "Weighted F1 (%)": round(f1_weighted * 100.0, 2),
        "GPU Latency (ms)": round(avg_latency, 2),
        "Train Time (min)": round(train_time_min, 1)
    }]
    per_class_data = []
    for c_idx, c_name in enumerate(classes):
        per_class_data.append({
            "Disease Class": c_name,
            "Precision (%)": round(p_per[c_idx] * 100.0, 2),
            "Recall (%)": round(r_per[c_idx] * 100.0, 2),
            "F1-Score (%)": round(f1_per[c_idx] * 100.0, 2),
            "Test Support": int(sup_per[c_idx])
        })

    df_summary = pd.DataFrame(summary_data)
    df_per_class = pd.DataFrame(per_class_data)
    df_summary.to_csv(reports_dir / "yolo11n_robust_summary.csv", index=False)
    df_per_class.to_csv(reports_dir / "yolo11n_robust_per_class.csv", index=False)
    df_summary.to_csv(output_root / "yolo11n_robust_summary.csv", index=False)
    df_per_class.to_csv(output_root / "yolo11n_robust_per_class.csv", index=False)

    md_content = "# Stage 2 YOLO11n-cls Field-Robust Training Results\n"
    md_content += "## Master's Thesis: Two-Stage Edge AI Rubber Leaf Disease Detection\n\n"
    md_content += "| Model | Params (M) | .pt (MB) | .onnx (MB) | Accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | Wtd F1 (%) | Latency (ms) | Train (min) |\n"
    md_content += "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for _, row in df_summary.iterrows():
        md_content += f"| **{row['Model']}** | {row['Parameters (M)']:.2f} | {row['Weights (.pt) MB']:.2f} | {row['ONNX (MB)']:.2f} | **{row['Test Accuracy (%)']:.2f}%** | {row['Macro Precision (%)']:.2f}% | {row['Macro Recall (%)']:.2f}% | **{row['Macro F1-Score (%)']:.2f}%** | {row['Weighted F1 (%)']:.2f}% | {row['GPU Latency (ms)']:.2f} ms | {row['Train Time (min)']:.1f} min |\n"
    md_content += "\n### Per-Class Performance Breakdown\n\n"
    md_content += "| Disease Class | Precision (%) | Recall (%) | F1-Score (%) | Test Support |\n"
    md_content += "|:---|---:|---:|---:|---:|\n"
    for _, row in df_per_class.iterrows():
        md_content += f"| **{row['Disease Class']}** | {row['Precision (%)']:.2f}% | {row['Recall (%)']:.2f}% | {row['F1-Score (%)']:.2f}% | {row['Test Support']} |\n"
    (reports_dir / "yolo11n_robust_summary.md").write_text(md_content, encoding="utf-8")
    (output_root / "yolo11n_robust_summary.md").write_text(md_content, encoding="utf-8")

    # Packaging ZIP
    zip_filename = "stage2_yolo11n_robust_results.zip"
    log.info(f"\n[Packaging] Bundling all artifacts into {zip_filename!r}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_root.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith(".zip"):
                zipf.write(file_path, arcname=str(file_path.relative_to(output_root)))
    log.info("=" * 75)
    log.info(f"TRAINING COMPLETE! Results: {Path(zip_filename).resolve()}")
    log.info("=" * 75)

if __name__ == "__main__":
    train_field_robust_yolo11n()
