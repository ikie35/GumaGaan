"""
Stage 1: Rubber Leaf Detector Model Size Comparison Script (Single Class)
-------------------------------------------------------------------------
Optimized for Kaggle Notebooks with NVIDIA Tesla T4 GPU Acceleration
-------------------------------------------------------------------------
Dataset: Roboflow "stage-1-rubber-leaf-detection-single-class" (v3)
Task: Single-Class Object Detection for Rubber Leaves
Framework: Ultralytics YOLOv8 (T4 CUDA Acceleration)
Comparison: 3 Model Sizes up to Medium (Nano vs Small vs Medium)
  1. YOLOv8n (Nano   ~ 3.2M parameters, batch=16)
  2. YOLOv8s (Small  ~ 11.2M parameters, batch=16)
  3. YOLOv8m (Medium ~ 25.9M parameters, batch=8)
Outputs: Saves PyTorch (.pt) weights, ONNX exports, Confusion Matrices,
         and packages all results into 'stage1_single_class_size_comparison_results.zip'
"""

import os
import sys
import time
import shutil
import zipfile
import logging
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Fix #8: Dependency check uses subprocess.check_call (with return-code checking)
#         instead of os.system, and installs each package individually.
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = ["roboflow", "ultralytics", "pandas", "matplotlib", "tabulate"]

def _ensure_packages(packages):
    """Install any missing packages using the current Python interpreter."""
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[Setup] Installing missing package: {pkg}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[Setup] Successfully installed: {pkg}")
            except subprocess.CalledProcessError as exc:
                print(f"[Setup] FATAL: Could not install '{pkg}'. Error: {exc}")
                sys.exit(1)

_ensure_packages(REQUIRED_PACKAGES)

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from roboflow import Roboflow
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Fix #10: Logging setup – all output is persisted to a .log file as well as stdout.
# ---------------------------------------------------------------------------
LOG_FILE = Path("stage1_single_class_training.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Use matplotlib default style (matches Ultralytics auto-generated training plots)

# ---------------------------------------------------------------------------
# GPU Accelerator Setup (Kaggle NVIDIA T4)
# ---------------------------------------------------------------------------
device = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("=" * 70)
    log.info(f"[Kaggle T4 GPU Enabled] Active GPU : {gpu_name}")
    log.info(f"[VRAM Available]                   : {vram_gb:.2f} GB")
    log.info("=" * 70)
else:
    log.warning("=" * 70)
    log.warning("[WARNING] CUDA GPU not detected. Running on CPU.")
    log.warning("=" * 70)

def plot_publication_confusion_matrix(cm_array: np.ndarray, model_name: str, save_path: Path) -> None:
    """
    2x2 row-normalised confusion matrix using matplotlib default style
    (matches Ultralytics auto-generated training plot aesthetics).
    Fix #11: cm_array shape is validated before slicing.
    """
    if cm_array.shape[0] < 2 or cm_array.shape[1] < 2:
        log.warning(
            f"[{model_name}] Confusion matrix shape {cm_array.shape} is smaller than "
            "expected 2x2. Skipping plot."
        )
        return

    # Ensure exact 2x2 shape (single class + background)
    cm_array = cm_array[:2, :2].astype(float)

    # Row-normalise to percentages
    cm_sum  = cm_array.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_array, cm_sum, out=np.zeros_like(cm_array), where=cm_sum != 0) * 100

    labels = ["Rubber Leaf\n(Target)", "Background\n(Ignore)"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im   = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy Percentage (%)")

    for i in range(2):
        for j in range(2):
            val   = cm_norm[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.2f}%\n({int(cm_array[i, j])})",
                    ha="center", va="center", color=color)

    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_title(f"{model_name} Stage 1 Localization Confusion Matrix\n(conf=0.45)")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Ground-Truth Class")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main() -> None:

    # 1. Download Single-Class Dataset from Roboflow
    log.info("=" * 70)
    log.info("[1/4] Downloading Stage 1 Single-Class Dataset from Roboflow...")
    log.info("=" * 70)

    # Fix #1: Hardcoded API key removed. Fail fast with a clear error message.
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        log.error(
            "ROBOFLOW_API_KEY environment variable is not set.\n"
            "  Kaggle : Add-ons > Secrets > ROBOFLOW_API_KEY\n"
            "  Local  : export ROBOFLOW_API_KEY=your_key"
        )
        sys.exit(1)

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace("rubber-research").project("stage-1-rubber-leaf-detection-single-class")
    version = project.version(3)
    dataset = version.download("yolov8")

    dataset_path = Path(dataset.location)
    data_yaml    = dataset_path / "data.yaml"
    log.info(f"--> Dataset downloaded and ready at: {data_yaml}")

    # 2. Define Model Variants with Adaptive Batch Size for VRAM Safety
    model_variants = [
        {"name": "YOLOv8n", "weights": "yolov8n.pt", "label": "Nano (3.2M)",  "batch": 16},
        {"name": "YOLOv8s", "weights": "yolov8s.pt", "label": "Small (11.2M)", "batch": 16},
        {"name": "YOLOv8m", "weights": "yolov8m.pt", "label": "Medium (25.9M)", "batch": 8},
    ]

    output_dir = Path("stage1_single_class_size_comparison")
    pt_dir = output_dir / "pt_models"
    onnx_dir = output_dir / "onnx_models"
    cm_dir = output_dir / "confusion_matrices"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    pt_dir.mkdir(exist_ok=True, parents=True)
    onnx_dir.mkdir(exist_ok=True, parents=True)
    cm_dir.mkdir(exist_ok=True, parents=True)

    comparison_results = []
    n_variants         = len(model_variants)

    # 3. Train, Evaluate & Export each Model Variant
    for idx, model_info in enumerate(model_variants, start=1):
        model_name = model_info["name"]
        weights    = model_info["weights"]
        size_label = model_info["label"]
        batch_size = model_info["batch"]

        log.info("\n" + "=" * 70)
        # Fix #4: correct step counter – step 1 was dataset download, variants are 2..N+1
        log.info(f"[{idx + 1}/{n_variants + 1}] Training & Evaluating {model_name} [{size_label}] (Batch={batch_size})")
        log.info("=" * 70)

        model = YOLO(weights)
        save_run_name = f"stage1_single_class_{model_name.lower()}"

        train_start = time.time()
        train_results = model.train(
            data=str(data_yaml),
            epochs=100,
            patience=15,
            imgsz=640,
            batch=batch_size,
            seed=42,
            device=device,
            single_cls=True,
            project=str(output_dir / "runs"),
            name=save_run_name,
            # Spatial augmentations
            degrees=15.0,
            flipud=0.5,
            fliplr=0.5,
            scale=0.7,
            multi_scale=True,
            # Fix #12: HSV colour augmentations tuned for rubber leaf colour variation
            #   hsv_h – hue shift ±1.5% handles lighting temperature variation
            #   hsv_s – saturation jitter handles shadows and canopy overexposure
            #   hsv_v – value jitter handles brightness variance between images
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            plots=True,
            verbose=True,
        )
        train_duration_min = round((time.time() - train_start) / 60, 2)

        best_weights = Path(train_results.save_dir) / "weights" / "best.pt"
        if not best_weights.exists():
            log.warning(f"[{model_name}] best.pt not found in run dir – falling back to initial weights.")
            best_weights = Path(weights)

        best_model = YOLO(str(best_weights))

        log.info(f"\n[{model_name}] Running validation and held-out test evaluation...")

        # Fix #3: val_res is now logged (val mAP50) instead of silently discarded.
        val_res   = best_model.val(
            data=str(data_yaml), split="val",
            imgsz=640, device=device, verbose=False,
        )
        val_map50 = round(float(val_res.box.map50) * 100, 2)
        log.info(f"[{model_name}] Validation mAP50 : {val_map50}%")

        # Fix #2: Pass conf=0.45 (operational threshold) so the confusion matrix
        #         reflects deployment-level precision/recall, not raw grid proposals.
        test_res  = best_model.val(
            data=str(data_yaml), split="test",
            imgsz=640, device=device,
            conf=0.45,
            plots=True,
            name=f"{model_name.lower()}_test_eval",
            verbose=False,
        )

        # Process Confusion Matrix Plots
        # Fix #6: Removed unused test_eval_dir variable.
        try:
            if hasattr(test_res, "confusion_matrix") and hasattr(test_res.confusion_matrix, "matrix"):
                cm_matrix   = test_res.confusion_matrix.matrix
                pub_cm_path = cm_dir / f"{model_name.lower()}_confusion_matrix_publication.png"
                plot_publication_confusion_matrix(cm_matrix, model_name, pub_cm_path)
                log.info(f"--> Saved publication confusion matrix: {pub_cm_path}")
            else:
                log.warning(f"[{model_name}] confusion_matrix attribute not available on test_res.")
        except Exception as exc:
            log.warning(f"[{model_name}] Could not generate confusion matrix plot: {exc}")

        # Metrics Extraction
        p       = round(float(np.mean(test_res.box.p))   * 100, 2)
        r       = round(float(np.mean(test_res.box.r))   * 100, 2)
        map50   = round(float(test_res.box.map50)         * 100, 2)
        map5095 = round(float(test_res.box.map)           * 100, 2)
        # Fix #9: use Ultralytics built-in F1 (test_res.box.f1) when available
        #         instead of manually recalculating from p and r.
        if hasattr(test_res.box, "f1"):
            f1 = round(float(np.mean(test_res.box.f1)) * 100, 2)
        else:
            f1 = round(2 * (p * r) / (p + r + 1e-6), 2)
        params_m       = round(sum(param.numel() for param in best_model.model.parameters()) / 1e6, 2)

        # Native GPU Inference Speed (ms per image)
        avg_latency_ms = round(float(test_res.speed["inference"]), 2)

        # Save PyTorch (.pt) Model Weights
        pt_file       = f"stage1_single_class_{model_name.lower()}_best.pt"
        final_pt_path  = pt_dir / pt_file
        if best_weights.exists():
            shutil.copy(best_weights, final_pt_path)
            log.info(f"--> PyTorch model saved to: {final_pt_path}")
        else:
            # Fix #7: explicit warning instead of silent skip
            log.warning(f"[{model_name}] WARNING: best.pt not found – model weights NOT saved!")

        # Export ONNX Format with Dynamic Shapes
        onnx_file       = f"stage1_single_class_{model_name.lower()}_dynamic.onnx"
        final_onnx_path  = onnx_dir / onnx_file
        try:
            onnx_path = best_model.export(
                format="onnx",
                imgsz=640,
                dynamic=True,
                opset=12,
                device=device,
            )
            shutil.copy(onnx_path, final_onnx_path)
            log.info(f"--> Dynamic ONNX model exported to: {final_onnx_path}")
        except Exception as exc:
            log.warning(f"[{model_name}] ONNX export failed: {exc}")

        # Accumulate results
        comparison_results.append({
            "Model Variant":         model_name,
            "Size Label":            size_label,
            "Parameters (M)":        params_m,
            "Batch Size":            batch_size,
            "Val mAP50 (%)":         val_map50,
            "Test mAP50 (%)":        map50,
            "Test mAP50-95 (%)":     map5095,
            "Precision (%)":         p,
            "Recall (%)":            r,
            "F1-Score (%)":          f1,
            "T4 Latency (ms)":       avg_latency_ms,
            "Train Time (min)":      train_duration_min,
            "PyTorch Model (.pt)":   pt_file,
            "Dynamic ONNX File":     onnx_file,
            "Confusion Matrix Plot": f"{model_name.lower()}_confusion_matrix_publication.png",
        })

    # 4. Generate Comparative Benchmark Table (CSV & Markdown)
    log.info("\n" + "=" * 70)
    log.info(f"[{n_variants + 1}/{n_variants + 1}] Generating benchmark tables and ZIP archive...")
    log.info("=" * 70)

    df       = pd.DataFrame(comparison_results)
    csv_file  = output_dir / "stage1_single_class_size_comparison.csv"
    md_file   = output_dir / "stage1_single_class_size_comparison.md"
    df.to_csv(csv_file, index=False)

    md_content = (
        "# Stage 1 Rubber Leaf Detector (Single Class) - Model Size Comparison\n"
        "## Kaggle NVIDIA Tesla T4 GPU Benchmark\n\n"
        "Comparative evaluation of **3 YOLOv8 Model Sizes** (Nano, Small, Medium) "
        "trained on Roboflow `stage-1-rubber-leaf-detection-single-class` (v3).\n\n"
        f"{df.to_markdown(index=False)}\n"
    )
    with open(md_file, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    # 5. Create Complete ZIP Archive Package
    # Fix #13: ZIP now includes training runs/ weights directories and the log file.
    zip_filename = "stage1_single_class_size_comparison_results.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in pt_dir.glob("*.pt"):
            zf.write(f, arcname=f"pt_models/{f.name}")
        for f in onnx_dir.glob("*.onnx"):
            zf.write(f, arcname=f"onnx_models/{f.name}")
        for f in cm_dir.glob("*.png"):
            zf.write(f, arcname=f"confusion_matrices/{f.name}")
        zf.write(csv_file, arcname=csv_file.name)
        zf.write(md_file,  arcname=md_file.name)
        zf.write(LOG_FILE, arcname=LOG_FILE.name)
        # Include training run weights (runs/<model>/weights/best.pt & last.pt)
        runs_dir = output_dir / "runs"
        if runs_dir.exists():
            for f in runs_dir.rglob("weights/*.pt"):
                zf.write(f, arcname=str(f.relative_to(output_dir)))

    log.info("\n" + "=" * 70)
    log.info("STAGE 1 SINGLE-CLASS MODEL SIZE COMPARISON SUMMARY (Kaggle T4 GPU):")
    log.info("=" * 70)
    log.info("\n" + df.to_string(index=False))
    log.info("=" * 70)
    log.info(f"SUCCESS! ZIP archive created at      : {Path(zip_filename).resolve()}")
    log.info(f"Training log saved to               : {LOG_FILE.resolve()}")
    log.info("=" * 70)

if __name__ == "__main__":
    main()
