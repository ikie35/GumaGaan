"""
================================================================================
Stage 2: YOLO Classifier Cross-Generation Benchmark Script
Architectures: YOLOv8-cls vs YOLO11-cls vs YOLO26-cls
Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
================================================================================
Dataset: Roboflow "stage-2-rubber-leaf-classifier" (v4)
Classes: [Healthy, Other Disease, PLFD]
Task: Multi-Class Disease Classification (Stage 2 in Two-Stage Pipeline)
Image Resolution: 224x224 (Classification Standard)
Hardware Accelerator: NVIDIA Tesla T4 GPU (Kaggle) / Raspberry Pi 5 + Hailo-8L Target

Outputs:
  - PyTorch (.pt) weights and ONNX (.onnx) exports (opset 12 for Hailo compilation)
  - Publication-grade Normalized Confusion Matrices (PNG + PDF)
  - 3-Panel Cross-Generation Comparative Confusion Matrix for Chapter 4
  - Standardized Comparative Tables (CSV, Markdown, and LaTeX)
  - Complete packaged ZIP archive: 'stage2_generation_benchmark_results.zip'
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

# ── 1. Automated Dependency Management ──────────────────────────────────────
REQUIRED_PACKAGES = ["roboflow", "ultralytics", "pandas", "matplotlib", "scikit-learn", "onnxsim"]

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
LOG_FILE = Path("stage2_generation_benchmark.log")
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
    log.info("=" * 75)
    log.info(f"[GPU Accelerator Active] : {gpu_name}")
    log.info(f"[VRAM Available]         : {vram_gb:.2f} GB")
    log.info("=" * 75)
else:
    log.warning("=" * 75)
    log.warning("[WARNING] No CUDA GPU detected. Benchmark running on CPU.")
    log.warning("=" * 75)

# ── 4. Dataset Acquisition from Roboflow ────────────────────────────────────
DATASET_DIR = Path("Stage-2-Rubber-Leaf-Classifier-4")

def acquire_dataset():
    """Downloads or locates the Stage 2 3-class disease dataset."""
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
        log.error(f"[Dataset Error] Failed to download via Roboflow: {e}")
        return DATASET_DIR

# ── 5. Confusion Matrix Visualizers (Publication Quality) ───────────────────
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'sans-serif']
plt.rcParams['figure.autolayout'] = True

# Convention used throughout: sklearn confusion_matrix convention
#   cm[i, j] = samples with TRUE label i predicted as class j
#   Rows = True labels  (Y-axis)
#   Cols = Predicted labels (X-axis)

def plot_confusion_matrix(cm_array: np.ndarray, class_names: list, model_key: str, save_dir: Path, normalize: bool = True):
    """
    Generates publication-quality confusion matrix.
    Rows = True label  (Y-axis)
    Cols = Predicted label (X-axis)  — sklearn convention
    When normalize=True, each row is divided by its sum (recall per class).
    """
    mat = np.array(cm_array, dtype=float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        display_mat = mat / (row_sums + 1e-9)
        title = f"{model_key} — Normalized Confusion Matrix"
        filename = f"{model_key}_confusion_matrix_normalized.png"
        vmax = 1.0
    else:
        display_mat = mat
        title = f"{model_key} — Confusion Matrix"
        filename = f"{model_key}_confusion_matrix.png"
        vmax = mat.max()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=250)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    im = ax.imshow(display_mat, cmap='Blues', interpolation='nearest', vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    nc = len(class_names)
    ax.set_xticks(range(nc))
    ax.set_yticks(range(nc))
    ax.set_xticklabels(class_names, rotation=45, fontsize=11, ha='right')  # Predicted
    ax.set_yticklabels(class_names, fontsize=11)                            # True
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=12)
    ax.set_ylabel('True Label', fontsize=12, labelpad=12)
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
    plt.savefig(save_dir / filename, bbox_inches='tight', dpi=250)
    plt.close(fig)

def plot_combined_confusion_matrices(all_cms: dict, class_names: list, save_path: Path):
    """
    Generates multi-panel comparative confusion matrix for thesis Chapter 4.
    Rows = True label  (Y-axis)
    Cols = Predicted label (X-axis)  — sklearn convention
    Each row is normalized to sum to 1.0 (per-class recall view).
    """
    num_models = len(all_cms)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5.2), dpi=300)
    if num_models == 1:
        axes = [axes]

    last_im = None
    for idx, (m_name, cm) in enumerate(all_cms.items()):
        ax = axes[idx]
        # Normalize row-wise: each true-class row sums to 1.0
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm.astype('float') / (row_sums + 1e-9)
        last_im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=10, rotation=20, ha='right')  # Predicted
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_title(m_name, fontsize=12, fontweight='bold', pad=10)

        if idx == 0:
            ax.set_yticklabels(class_names, fontsize=10)                        # True
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels([''] * len(class_names))

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm_norm[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color=text_color, fontsize=11, fontweight='bold')

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.015, 0.65])
    fig.colorbar(last_im, cax=cbar_ax, label='Normalized Ratio')

    plt.suptitle(
        "Stage 2: Cross-Generation Classifier Confusion Matrix Comparison (Test Split)",
        fontsize=13, fontweight='bold', y=0.98
    )
    plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

# ── 6. Main Cross-Generation Benchmark Pipeline ─────────────────────────────
def run_stage2_generation_benchmark():
    dataset_path = acquire_dataset()
    output_root = Path("stage2_generation_benchmark_results")
    weights_dir = output_root / "best_weights"
    cm_dir = output_root / "confusion_matrices"
    onnx_dir = output_root / "onnx_models"
    runs_dir = output_root / "per_generation_runs"
    reports_dir = output_root / "00_summary_reports"

    for d in [output_root, weights_dir, cm_dir, onnx_dir, runs_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3 Architectural Generations for Stage 2 Classification
    models_config = [
        {
            "name": "yolov8n-cls",
            "model_id": "yolov8n-cls.pt",
            "gen_label": "YOLOv8 (2023 - C2f Classifier)",
            "batch": 32,
            "epochs": 100,
            "patience": 15
        },
        {
            "name": "yolo11n-cls",
            "model_id": "yolo11n-cls.pt",
            "gen_label": "YOLO11 (2024 - C3k2 Spatial Attention)",
            "batch": 32,
            "epochs": 100,
            "patience": 15
        },
        {
            "name": "yolo26n-cls",
            "model_id": "yolo26n-cls.pt",
            "gen_label": "YOLO26 (2026 - Next-Gen Edge Classifier)",
            "batch": 32,
            "epochs": 100,
            "patience": 15
        }
    ]

    benchmark_summary = []
    per_class_summary = []
    all_confusion_matrices = {}
    
    classes = ["Healthy", "Other Disease", "PLFD"]

    for cfg in models_config:
        model_name = cfg["name"]
        model_id = cfg["model_id"]
        gen_label = cfg["gen_label"]
        batch_size = cfg["batch"]
        epochs = cfg["epochs"]
        patience = cfg["patience"]

        log.info("=" * 75)
        log.info(f"BENCHMARKING STAGE 2 GENERATION: {model_name.upper()} ({gen_label})")
        log.info("=" * 75)

        # ── Initialize Model with Fallback Support ───────────────
        try:
            model = YOLO(model_id)
        except Exception as e:
            log.warning(f"[Model Init] Could not load '{model_id}': {e}. Attempting fallback initialization...")
            try:
                model = YOLO(f"{model_name}.yaml")
            except Exception as e2:
                log.error(f"[Model Init Error] Failed initializing {model_name}: {e2}. Skipping variant.")
                continue

        # ── Train Model ──────────────────────────────────────────
        train_start = time.perf_counter()
        try:
            train_results = model.train(
                data=str(dataset_path),
                epochs=epochs,
                patience=patience,
                imgsz=224,
                batch=batch_size,
                project=str(output_root / "runs"),
                name=f"{model_name}_run",
                device=device,
                # Photometric & Shadow Augmentations for Canopy & Room Light Invariance
                dropout=0.20,
                hsv_h=0.03,
                hsv_s=0.70,
                hsv_v=0.60,      # Brightness jitter teaches model to ignore shadows
                degrees=30.0,
                fliplr=0.5,
                flipud=0.5,
                shear=10.0,
                seed=42,         # Scientific reproducibility
                verbose=True
            )
            train_time_min = (time.perf_counter() - train_start) / 60.0
        except Exception as e:
            log.error(f"[Training Error] Training failed for {model_name}: {e}")
            del model
            gc.collect()
            continue

        # ── Locate Best Weights ──────────────────────────────────
        best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = output_root / "runs" / f"{model_name}_run" / "weights" / "best.pt"

        # Copy training curve plots before releasing training model
        run_dir = Path(model.trainer.save_dir)
        model_out_dir = runs_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        for plot_file in run_dir.glob("*.png"):
            shutil.copy2(plot_file, model_out_dir / plot_file.name)

        # Save best weights to organized directories
        shutil.copy2(best_pt, model_out_dir / f"{model_name}_best.pt")
        shutil.copy2(best_pt, weights_dir  / f"{model_name}_best.pt")

        # ── Free Training Model VRAM before loading eval model ───
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ── Load Best Weights for Evaluation & Export ────────────
        eval_model = YOLO(str(best_pt))

        # Model parameter count and file size
        param_count = sum(p.numel() for p in eval_model.model.parameters()) / 1e6
        pt_size_mb  = best_pt.stat().st_size / (1024 * 1024)

        # ── Official Validation on Val Split (Ultralytics standard) ──
        # Gives canonical top-1 / top-5 accuracy consistent with published YOLO benchmarks
        log.info(f"[{model_name}] Running official validation on validation split...")
        val_results = eval_model.val(data=str(dataset_path), imgsz=224, device=device, verbose=False)
        top1_acc_val = float(getattr(val_results, 'top1', 0.0)) * 100.0
        top5_acc_val = float(getattr(val_results, 'top5', 0.0)) * 100.0
        log.info(f"[{model_name}] Val Top-1: {top1_acc_val:.2f}%  |  Top-5: {top5_acc_val:.2f}%")

        # ── Export to ONNX (Opset 12 for Hailo-8L Compatibility) ──
        log.info(f"[{model_name}] Exporting to ONNX format (opset 12, simplify=True)...")
        onnx_size_mb = 0.0
        try:
            onnx_path = eval_model.export(format="onnx", imgsz=224, opset=12, dynamic=False, simplify=True)
            onnx_file = Path(onnx_path)
            onnx_size_mb = onnx_file.stat().st_size / (1024 * 1024)
            shutil.copy2(onnx_file, model_out_dir / f"{model_name}_best.onnx")
            shutil.copy2(onnx_file, onnx_dir      / f"{model_name}_best.onnx")
            log.info(f"[{model_name}] ONNX export saved ({onnx_size_mb:.2f} MB)")
        except Exception as e:
            log.warning(f"[{model_name} ONNX Export Warning] {e}")

        # ── Rigorous Test Split Evaluation ───────────────────────
        test_dir = dataset_path / "test"
        log.info(f"[{model_name}] Evaluating on independent TEST split: {test_dir}")

        y_true = []
        y_pred = []
        latencies_ms = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # ── GPU Warmup (prime CUDA kernel before timing) ──────────
        # Collect one sample image for warmup inference
        warmup_img = None
        for class_name in classes:
            _folder = test_dir / class_name
            if _folder.exists():
                _imgs = [f for f in _folder.iterdir() if f.suffix.lower() in valid_exts]
                if _imgs:
                    warmup_img = str(_imgs[0])
                    break
        if warmup_img:
            log.info(f"[{model_name}] Running 5 GPU warmup passes before evaluation...")
            for _ in range(5):
                eval_model.predict(source=warmup_img, imgsz=224, device=device, verbose=False)

        for class_idx, class_name in enumerate(classes):
            class_folder = test_dir / class_name
            if not class_folder.exists():
                continue

            image_files = [f for f in class_folder.iterdir() if f.suffix.lower() in valid_exts]
            for img_file in image_files:
                y_true.append(class_idx)

                # Use Ultralytics internal speed['inference'] for pure GPU kernel time (ms)
                # This excludes Python overhead, image I/O, and preprocessing
                pred = eval_model.predict(source=str(img_file), imgsz=224, device=device, verbose=False)
                latencies_ms.append(pred[0].speed['inference'])
                pred_class_idx = int(pred[0].probs.top1)
                y_pred.append(pred_class_idx)

        # ── Metrics Computation ──────────────────────────────────
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred) * 100.0

        # Macro-averaged (equal weight per class — used for thesis reporting)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        # Weighted-averaged (weighted by class support — handles class imbalance)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        # Per-class breakdown
        p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Average GPU inference latency (warmup handled above — all samples clean)
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        log.info(
            f"[{model_name.upper()} TEST RESULTS] "
            f"Accuracy: {acc:.2f}% | Macro F1: {f1_macro*100:.2f}% | "
            f"Weighted F1: {f1_weighted*100:.2f}% | Latency: {avg_latency:.2f} ms"
        )

        # ── Confusion Matrix (sklearn: rows=True, cols=Predicted) ─
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        all_confusion_matrices[model_name] = cm

        # Generate per-model confusion matrix plots
        plot_confusion_matrix(cm, classes, model_name, cm_dir,        normalize=True)
        plot_confusion_matrix(cm, classes, model_name, cm_dir,        normalize=False)
        plot_confusion_matrix(cm, classes, model_name, model_out_dir, normalize=True)
        plot_confusion_matrix(cm, classes, model_name, model_out_dir, normalize=False)

        # ── Record Summary Metrics ───────────────────────────────
        benchmark_summary.append({
            "Generation":          model_name,
            "Architecture":        gen_label,
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

        # Per-class breakdown
        for c_idx, c_name in enumerate(classes):
            per_class_summary.append({
                "Generation":    model_name,
                "Disease Class": c_name,
                "Precision (%)" : round(p_per[c_idx]  * 100.0, 2),
                "Recall (%)":    round(r_per[c_idx]  * 100.0, 2),
                "F1-Score (%)":  round(f1_per[c_idx] * 100.0, 2),
                "Test Support":  int(sup_per[c_idx])
            })

        # ── VRAM Cleanup (prevent T4 OOM between models) ─────────
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── 7. Multi-Panel Comparative Figure for Manuscript ────────────────────
    if all_confusion_matrices:
        plot_combined_confusion_matrices(all_confusion_matrices, classes,
                                         cm_dir      / "stage2_cross_generation_confusion_matrices")
        plot_combined_confusion_matrices(all_confusion_matrices, classes,
                                         output_root / "stage2_cross_generation_confusion_matrices")

    # ── 8. Generate Publication Tables (CSV + Markdown + LaTeX) ────────────
    df_summary = pd.DataFrame(benchmark_summary)
    df_per_class = pd.DataFrame(per_class_summary)

    # Save CSVs
    df_summary.to_csv(reports_dir / "stage2_generation_comparison.csv", index=False)
    df_summary.to_csv(output_root / "stage2_generation_comparison.csv", index=False)
    df_per_class.to_csv(reports_dir / "stage2_per_class_metrics.csv", index=False)
    df_per_class.to_csv(output_root / "stage2_per_class_metrics.csv", index=False)

    # Generate Markdown Table
    md_content = f"""# Stage 2 Rubber Leaf Disease Classifier: Cross-Generation Benchmark
## Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection

Comparative evaluation across **YOLOv8-cls**, **YOLO11-cls**, and **YOLO26-cls** on the
independent held-out test split (3 disease categories: *Healthy*, *Other Disease*, *PLFD*):

### 1. Overall Architectural Performance Summary

| Generation | Params (M) | .pt (MB) | .onnx (MB) | Accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | Wtd F1 (%) | Latency (ms) | Train (min) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, row in df_summary.iterrows():
        md_content += (
            f"| **{row['Generation']}** "
            f"| {row['Parameters (M)']:.2f} "
            f"| {row['Weights (.pt) MB']:.2f} "
            f"| {row['ONNX (MB)']:.2f} "
            f"| **{row['Test Accuracy (%)']:.2f}%** "
            f"| {row['Macro Precision (%)']:.2f}% "
            f"| {row['Macro Recall (%)']:.2f}% "
            f"| **{row['Macro F1-Score (%)']:.2f}%** "
            f"| {row['Wtd F1-Score (%)']:.2f}% "
            f"| {row['GPU Latency (ms)']:.2f} ms "
            f"| {row['Train Time (min)']:.1f} min |\n"
        )

    md_content += "\n### 2. Per-Class Diagnosis Performance (Precision, Recall, F1-Score)\n\n"
    md_content += "| Generation | Disease Category | Precision (%) | Recall (%) | F1-Score (%) | Test Support |\n"
    md_content += "|:---|:---|---:|---:|---:|---:|\n"
    for _, row in df_per_class.iterrows():
        md_content += f"| {row['Generation']} | {row['Disease Class']} | {row['Precision (%)']:.2f}% | {row['Recall (%)']:.2f}% | {row['F1-Score (%)']:.2f}% | {row['Test Support']} |\n"

    with open(reports_dir / "stage2_generation_comparison.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(output_root / "stage2_generation_comparison.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    # Generate LaTeX Table
    latex_code = r"""\begin{table}[htbp]
\centering
\caption{Stage 2 Rubber Leaf Disease Classifier: Cross-Generation Benchmark (Test Split)}
\label{tab:stage2_generation_benchmark}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Generation} & \textbf{Params (M)} & \textbf{.pt (MB)} & \textbf{.onnx (MB)}
  & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{Wtd F1} & \textbf{Latency (ms)} \\
\midrule
"""
    for _, row in df_summary.iterrows():
        latex_code += (
            f"{row['Generation']} & {row['Parameters (M)']:.2f} "
            f"& {row['Weights (.pt) MB']:.2f} & {row['ONNX (MB)']:.2f} "
            f"& {row['Test Accuracy (%)']:.2f}\\% "
            f"& {row['Macro F1-Score (%)']:.2f}\\% "
            f"& {row['Wtd F1-Score (%)']:.2f}\\% "
            f"& {row['GPU Latency (ms)']:.2f} \\\\\n"
        )
    latex_code += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    with open(reports_dir / "stage2_generation_comparison_latex.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    log.info("\n" + "=" * 75)
    log.info("STAGE 2 GENERATION BENCHMARK SUMMARY:")
    log.info("=" * 75)
    log.info("\n" + df_summary.to_string(index=False))

    # ── 9. Package Artifacts into ZIP Archive ───────────────────────────────
    zip_filename = "stage2_generation_benchmark_results.zip"
    log.info(f"\n[Packaging] Bundling all artifacts into '{zip_filename}'...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_root.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith(".zip"):
                rel_path = file_path.relative_to(output_root)
                zipf.write(file_path, arcname=str(rel_path))

    log.info("=" * 75)
    log.info(f"BENCHMARK COMPLETE! Results archive: {Path(zip_filename).resolve()}")
    log.info("=" * 75)

if __name__ == "__main__":
    run_stage2_generation_benchmark()
