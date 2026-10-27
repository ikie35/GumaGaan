"""
================================================================================
Stage 2: YOLO11-cls Model Size Comparison Benchmark
Scales: YOLO11n-cls (Nano) vs YOLO11s-cls (Small) vs YOLO11m-cls (Medium)
Masters Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
================================================================================
Dataset: Roboflow stage-2-rubber-leaf-classifier (v4)
Classes: [Healthy, Other Disease, PLFD]
Task: Multi-Class Disease Classification (Stage 2 in Two-Stage Pipeline)
Image Resolution: 224x224 (Classification Standard)
Hardware Accelerator: NVIDIA Tesla T4 GPU (Kaggle) / Raspberry Pi 5 + Hailo-8L

Scales Compared:
  1. YOLO11n-cls  (Nano)    ~ 1.53M parameters, batch=32
  2. YOLO11s-cls  (Small)   ~ 5.09M parameters, batch=32
  3. YOLO11m-cls  (Medium)  ~ 8.40M parameters, batch=16

Outputs:
  - PyTorch (.pt) weights and ONNX (.onnx) exports (opset 12 for Hailo)
  - Publication-grade Normalized Confusion Matrices (PNG + PDF)
  - 3-Panel Cross-Scale Comparative Confusion Matrix for Chapter 4
  - Standardized Comparative Tables (CSV, Markdown, and LaTeX)
  - ZIP archive: stage2_yolo11_size_benchmark_results.zip
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

LOG_FILE = Path("stage2_yolo11_size_benchmark.log")
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
    log.warning("[WARNING] No CUDA GPU detected. Benchmark running on CPU.")
    log.warning("=" * 75)

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

# Confusion Matrix Plotting (sklearn convention: rows=True, cols=Predicted)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"]
plt.rcParams["figure.autolayout"] = True

def plot_confusion_matrix(cm_array, class_names, model_key, save_dir, normalize=True):
    """Rows=True label (Y), Cols=Predicted label (X) -- sklearn convention."""
    mat = np.array(cm_array, dtype=float)
    if normalize:
        row_sums    = mat.sum(axis=1, keepdims=True)
        display_mat = mat / (row_sums + 1e-9)
        title       = f"{model_key} -- Normalized Confusion Matrix"
        filename    = f"{model_key}_confusion_matrix_normalized.png"
        vmax        = 1.0
    else:
        display_mat = mat
        title       = f"{model_key} -- Confusion Matrix"
        filename    = f"{model_key}_confusion_matrix.png"
        vmax        = mat.max()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=250)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    im   = ax.imshow(display_mat, cmap="Blues", interpolation="nearest", vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    nc = len(class_names)
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    ax.set_xticklabels(class_names, rotation=45, fontsize=11, ha="right")
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=12)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=12)
    ax.set_title(title, fontsize=14, pad=15)
    thresh = vmax / 2.0
    for i in range(nc):
        for j in range(nc):
            val = display_mat[i, j]
            tc  = "white" if val > thresh else "black"
            vs  = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, vs, ha="center", va="center", color=tc, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / filename, bbox_inches="tight", dpi=250)
    plt.close(fig)


def plot_combined_confusion_matrices(all_cms, class_names, save_path):
    """3-panel comparative figure. Rows=True, Cols=Predicted."""
    num_models = len(all_cms)
    fig, axes  = plt.subplots(1, num_models, figsize=(6 * num_models, 5.2), dpi=300)
    if num_models == 1:
        axes = [axes]
    last_im = None
    for idx, (m_name, cm) in enumerate(all_cms.items()):
        ax      = axes[idx]
        rsums   = cm.sum(axis=1, keepdims=True)
        cm_norm = cm.astype("float") / (rsums + 1e-9)
        last_im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1.0)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=10, rotation=20, ha="right")
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_title(m_name, fontsize=12, fontweight="bold", pad=10)
        if idx == 0:
            ax.set_yticklabels(class_names, fontsize=10)
            ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
        else:
            ax.set_yticklabels([""] * len(class_names))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm_norm[i, j]
                tc  = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=tc, fontsize=11, fontweight="bold")
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.015, 0.65])
    fig.colorbar(last_im, cax=cbar_ax, label="Normalized Ratio")
    plt.suptitle("Stage 2: YOLO11-cls Scale Comparison Confusion Matrix (Test Split)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_stage2_yolo11_size_benchmark():
    dataset_path = acquire_dataset()
    output_root  = Path("stage2_yolo11_size_benchmark_results")
    weights_dir  = output_root / "best_weights"
    cm_dir       = output_root / "confusion_matrices"
    onnx_dir     = output_root / "onnx_models"
    runs_dir     = output_root / "per_scale_runs"
    reports_dir  = output_root / "00_summary_reports"
    for d in [output_root, weights_dir, cm_dir, onnx_dir, runs_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    models_config = [
        {"name": "yolo11n-cls", "model_id": "yolo11n-cls.pt", "size_label": "Nano",   "batch": 32, "epochs": 100, "patience": 15},
        {"name": "yolo11s-cls", "model_id": "yolo11s-cls.pt", "size_label": "Small",  "batch": 32, "epochs": 100, "patience": 15},
        {"name": "yolo11m-cls", "model_id": "yolo11m-cls.pt", "size_label": "Medium", "batch": 16, "epochs": 100, "patience": 15},
    ]

    benchmark_summary      = []
    per_class_summary      = []
    all_confusion_matrices = {}
    classes    = ["Healthy", "Other Disease", "PLFD"]
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cfg in models_config:
        model_name = cfg["name"]
        model_id   = cfg["model_id"]
        size_label = cfg["size_label"]
        batch_size = cfg["batch"]
        epochs     = cfg["epochs"]
        patience   = cfg["patience"]

        log.info("=" * 75)
        log.info(f"BENCHMARKING STAGE 2 SCALE: {model_name.upper()} ({size_label})")
        log.info("=" * 75)

        try:
            model = YOLO(model_id)
        except Exception as e:
            log.error(f"[Model Init Error] {model_id}: {e}. Skipping.")
            continue

        train_start = time.perf_counter()
        try:
            train_results = model.train(
                data=str(dataset_path),
                epochs=epochs, patience=patience,
                imgsz=224, batch=batch_size,
                project=str(output_root / "runs"),
                name=f"{model_name}_run",
                device=device,
                # Photometric & Shadow Augmentations for Canopy & Room Light Invariance
                dropout=0.20, hsv_h=0.03, hsv_s=0.70,
                hsv_v=0.60, degrees=30.0,
                fliplr=0.5, flipud=0.5, shear=10.0,
                seed=42, verbose=True
            )
            train_time_min = (time.perf_counter() - train_start) / 60.0
        except Exception as e:
            log.error(f"[Training Error] {model_name}: {e}")
            del model; gc.collect(); continue

        best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = output_root / "runs" / f"{model_name}_run" / "weights" / "best.pt"

        # Copy training curve plots before freeing training model
        run_dir       = Path(model.trainer.save_dir)
        model_out_dir = runs_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        for plot_file in run_dir.glob("*.png"):
            shutil.copy2(plot_file, model_out_dir / plot_file.name)

        shutil.copy2(best_pt, model_out_dir / f"{model_name}_best.pt")
        shutil.copy2(best_pt, weights_dir   / f"{model_name}_best.pt")

        # Free training model VRAM before loading eval model
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        eval_model  = YOLO(str(best_pt))
        param_count = sum(p.numel() for p in eval_model.model.parameters()) / 1e6
        pt_size_mb  = best_pt.stat().st_size / (1024 * 1024)

        # Official validation on val split (Ultralytics standard top-1/top-5)
        log.info(f"[{model_name}] Running official validation on validation split...")
        val_results  = eval_model.val(data=str(dataset_path), imgsz=224, device=device, verbose=False)
        top1_acc_val = float(getattr(val_results, "top1", 0.0)) * 100.0
        top5_acc_val = float(getattr(val_results, "top5", 0.0)) * 100.0
        log.info(f"[{model_name}] Val Top-1: {top1_acc_val:.2f}%  |  Top-5: {top5_acc_val:.2f}%")

        # Export to ONNX (opset 12 for Hailo-8L compatibility)
        log.info(f"[{model_name}] Exporting to ONNX (opset 12, simplify=True)...")
        onnx_size_mb = 0.0
        try:
            onnx_path    = eval_model.export(format="onnx", imgsz=224, opset=12, dynamic=False, simplify=True)
            onnx_file    = Path(onnx_path)
            onnx_size_mb = onnx_file.stat().st_size / (1024 * 1024)
            shutil.copy2(onnx_file, model_out_dir / f"{model_name}_best.onnx")
            shutil.copy2(onnx_file, onnx_dir      / f"{model_name}_best.onnx")
            log.info(f"[{model_name}] ONNX saved ({onnx_size_mb:.2f} MB)")
        except Exception as e:
            log.warning(f"[{model_name} ONNX Warning] {e}")

        # Rigorous test split evaluation
        test_dir = dataset_path / "test"
        log.info(f"[{model_name}] Evaluating on independent TEST split: {test_dir}")
        y_true = []; y_pred = []; latencies_ms = []

        # GPU warmup: prime CUDA kernel before timing
        warmup_img = None
        for class_name in classes:
            _folder = test_dir / class_name
            if _folder.exists():
                _imgs = [f for f in _folder.iterdir() if f.suffix.lower() in valid_exts]
                if _imgs: warmup_img = str(_imgs[0]); break
        if warmup_img:
            log.info(f"[{model_name}] Running 5 GPU warmup passes...")
            for _ in range(5):
                eval_model.predict(source=warmup_img, imgsz=224, device=device, verbose=False)

        for class_idx, class_name in enumerate(classes):
            class_folder = test_dir / class_name
            if not class_folder.exists():
                log.warning(f"[{model_name}] Test folder not found: {class_folder}"); continue
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

        log.info(
            f"[{model_name.upper()} TEST] "
            f"Accuracy: {acc:.2f}% | Macro F1: {f1_macro*100:.2f}% | "
            f"Wtd F1: {f1_weighted*100:.2f}% | Latency: {avg_latency:.2f} ms"
        )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        all_confusion_matrices[model_name] = cm
        for save_d in [cm_dir, model_out_dir]:
            plot_confusion_matrix(cm, classes, model_name, save_d, normalize=True)
            plot_confusion_matrix(cm, classes, model_name, save_d, normalize=False)

        benchmark_summary.append({
            "Scale":               size_label,
            "Model":               model_name,
            "Parameters (M)":      round(param_count, 2),
            "Weights (.pt) MB":    round(pt_size_mb, 2),
            "ONNX (MB)":           round(onnx_size_mb, 2),
            "Test Accuracy (%)":   round(acc, 2),
            "Macro Precision (%)": round(p_macro    * 100.0, 2),
            "Macro Recall (%)":    round(r_macro    * 100.0, 2),
            "Macro F1-Score (%)":  round(f1_macro   * 100.0, 2),
            "Wtd F1-Score (%)":    round(f1_weighted * 100.0, 2),
            "GPU Latency (ms)":    round(avg_latency, 2),
            "Train Time (min)":    round(train_time_min, 1)
        })
        for c_idx, c_name in enumerate(classes):
            per_class_summary.append({
                "Scale":         size_label,
                "Model":         model_name,
                "Disease Class": c_name,
                "Precision (%)": round(p_per[c_idx]  * 100.0, 2),
                "Recall (%)":    round(r_per[c_idx]  * 100.0, 2),
                "F1-Score (%)":  round(f1_per[c_idx] * 100.0, 2),
                "Test Support":  int(sup_per[c_idx])
            })

        # VRAM cleanup (prevent T4 OOM between models)
        del eval_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    # Multi-panel confusion matrix figure
    if all_confusion_matrices:
        plot_combined_confusion_matrices(all_confusion_matrices, classes,
                                         cm_dir      / "stage2_yolo11_scale_confusion_matrices")
        plot_combined_confusion_matrices(all_confusion_matrices, classes,
                                         output_root / "stage2_yolo11_scale_confusion_matrices")

    df_summary   = pd.DataFrame(benchmark_summary)
    df_per_class = pd.DataFrame(per_class_summary)

    for p in [reports_dir, output_root]:
        df_summary.to_csv(p   / "stage2_yolo11_size_comparison.csv",   index=False)
        df_per_class.to_csv(p / "stage2_yolo11_per_class_metrics.csv", index=False)

    md_content  = "# Stage 2 YOLO11-cls Scale Comparison Benchmark\n"
    md_content += "## Masters Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection\n\n"
    md_content += "| Scale | Model | Params (M) | .pt (MB) | .onnx (MB) | Accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | Wtd F1 (%) | Latency (ms) | Train (min) |\n"
    md_content += "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for _, row in df_summary.iterrows():
        md_content += (
            f"| **{row['Scale']}** | {row['Model']} | {row['Parameters (M)']:.2f} "
            f"| {row['Weights (.pt) MB']:.2f} | {row['ONNX (MB)']:.2f} "
            f"| **{row['Test Accuracy (%)']:.2f}%** "
            f"| {row['Macro Precision (%)']:.2f}% | {row['Macro Recall (%)']:.2f}% "
            f"| **{row['Macro F1-Score (%)']:.2f}%** | {row['Wtd F1-Score (%)']:.2f}% "
            f"| {row['GPU Latency (ms)']:.2f} ms | {row['Train Time (min)']:.1f} min |\n"
        )
    md_content += "\n| Scale | Model | Disease Class | Precision (%) | Recall (%) | F1-Score (%) | Test Support |\n"
    md_content += "|:---|:---|:---|---:|---:|---:|---:|\n"
    for _, row in df_per_class.iterrows():
        md_content += (
            f"| {row['Scale']} | {row['Model']} | {row['Disease Class']} "
            f"| {row['Precision (%)']:.2f}% | {row['Recall (%)']:.2f}% "
            f"| {row['F1-Score (%)']:.2f}% | {row['Test Support']} |\n"
        )
    for p in [reports_dir, output_root]:
        (p / "stage2_yolo11_size_comparison.md").write_text(md_content, encoding="utf-8")

    latex_code  = "\\begin{table}[htbp]\n\\centering\n"
    latex_code += "\\caption{Stage 2 YOLO11-cls Scale Comparison: Rubber Leaf Disease Classification (Test Split)}\n"
    latex_code += "\\label{tab:stage2_yolo11_size}\n\\resizebox{\\textwidth}{!}{%\n"
    latex_code += "\\begin{tabular}{llccccccc}\n\\toprule\n"
    latex_code += "\\textbf{Scale} & \\textbf{Model} & \\textbf{Params (M)} & \\textbf{.pt (MB)} & \\textbf{.onnx (MB)} & \\textbf{Accuracy} & \\textbf{Macro F1} & \\textbf{Wtd F1} & \\textbf{Latency (ms)} \\\\\\\n\\midrule\n"
    for _, row in df_summary.iterrows():
        latex_code += (
            f"{row['Scale']} & {row['Model']} & {row['Parameters (M)']:.2f} "
            f"& {row['Weights (.pt) MB']:.2f} & {row['ONNX (MB)']:.2f} "
            f"& {row['Test Accuracy (%)']:.2f}\\% "
            f"& {row['Macro F1-Score (%)']:.2f}\\% "
            f"& {row['Wtd F1-Score (%)']:.2f}\\% "
            f"& {row['GPU Latency (ms)']:.2f} \\\\\\\n"
        )
    latex_code += "\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    (reports_dir / "stage2_yolo11_size_comparison_latex.tex").write_text(latex_code, encoding="utf-8")

    log.info("\n" + "=" * 75)
    log.info("STAGE 2 YOLO11-CLS SIZE BENCHMARK SUMMARY:")
    log.info("=" * 75)
    log.info("\n" + df_summary.to_string(index=False))

    zip_filename = "stage2_yolo11_size_benchmark_results.zip"
    log.info(f"\n[Packaging] Bundling all artifacts into {zip_filename!r}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_root.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith(".zip"):
                zipf.write(file_path, arcname=str(file_path.relative_to(output_root)))

    log.info("=" * 75)
    log.info(f"BENCHMARK COMPLETE! Results: {Path(zip_filename).resolve()}")
    log.info("=" * 75)


if __name__ == "__main__":
    run_stage2_yolo11_size_benchmark()
