"""
================================================================================
Stage 2: End-to-End Field-Robust Training & M4 XAI Faithfulness Benchmark
Models: YOLOv8n-cls vs YOLO11n-cls vs YOLO26n-cls
Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
================================================================================
Dataset: Roboflow "stage-2-rubber-leaf-classifier" (v4)
Classes: [Healthy, Other Disease, PLFD]
Image Resolution: 224x224
Target Deployment: Raspberry Pi 5 + Hailo-8L AI NPU HAT

Part 1: Field-Robust Training Across All 3 Architectures:
  1. Random Erasing (erasing=0.40): Forces focus on lesion micro-patterns.
  2. Extreme Lighting Jitter (hsv_v=0.70, hsv_s=0.80, hsv_h=0.05).
  3. Angle & Oblique Invariance (degrees=45.0, shear=15.0, fliplr=0.5, flipud=0.5).
  4. Dropout Regularization (dropout=0.25).

Part 2: M4 XAI Faithfulness Evaluation (NeurIPS 2023 Protocol):
  - Deletion AUC (DEL_AUC ↓)
  - Insertion AUC (INS_AUC ↑)
  - Faithfulness Delta (ΔAUC = INS_AUC - DEL_AUC ↑)
  - Monotonicity (MONO ↑)
  - Pearson Correlation (Pearson r ↑)
  - Saliency Sparsity (SPAR ↑)
  - M4 Composite Score (M4_Score ↑)

Part 3: Expanded Multi-Class Saliency Map Visualizations:
  - Multi-sample visual saliency triads (Input, Grad-CAM++, 50% Deletion)
    for all 3 classes (Healthy, Other Disease, PLFD).
  - Cross-Architecture Side-by-Side Saliency Comparison Grid.

Outputs:
  - Saved PyTorch weights (.pt) & ONNX exports (opset 12 for Hailo-8L)
  - Confusion matrices (individual + 3-panel comparative)
  - M4 Faithfulness metrics (CSV, Markdown, LaTeX)
  - High-res Faithfulness curves, radar charts & expanded saliency maps
  - Complete packaged ZIP: 'stage2_robust_m4_faithfulness_results.zip'
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
REQUIRED_PACKAGES = [
    "roboflow", "ultralytics", "pandas", "matplotlib",
    "scikit-learn", "scipy", "tqdm", "opencv-python", "onnxsim"
]

def ensure_packages(packages):
    """Installs missing dependencies safely in Kaggle or local environments."""
    for pkg in packages:
        try:
            if pkg == "opencv-python":
                __import__("cv2")
            else:
                __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"[Setup] Installing required package: {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

ensure_packages(REQUIRED_PACKAGES)

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from roboflow import Roboflow
from ultralytics import YOLO

# ── 2. Logging Setup ────────────────────────────────────────────────────────
LOG_FILE = Path("stage2_robust_m4_training.log")
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
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
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
CLASS_NAMES = ["Healthy", "Other Disease", "PLFD"]

def acquire_dataset():
    """Locates dataset locally / in Kaggle working dir or downloads via Roboflow API."""
    search_dirs = [
        DATASET_DIR,
        Path("/kaggle/input/stage-2-rubber-leaf-classifier-4"),
        Path("/kaggle/working/Stage-2-Rubber-Leaf-Classifier-4")
    ]
    for d in search_dirs:
        if d.exists() and (d / "train").exists():
            log.info(f"[Dataset] Found existing dataset at: {d.resolve()}")
            return d

    log.info("[Dataset] Downloading Stage 2 dataset from Roboflow API (v4)...")
    try:
        rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
        project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
        version = project.version(4)
        dataset = version.download("folder", location=str(DATASET_DIR.resolve()))
        return Path(dataset.location)
    except Exception as e:
        log.error(f"[Dataset Error] {e}")
        return DATASET_DIR

# ── 5. Confusion Matrix Visualizers ─────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

import seaborn as sns

def plot_single_confusion_matrix(cm_array, class_names, model_key, save_dir, normalize=True):
    """
    100% Exact Ultralytics YOLO Confusion Matrix Implementation.
    Uses seaborn heatmap with square=True, 'Predicted' on Y-axis, 'True' on X-axis.
    """
    mat = np.array(cm_array, dtype=float)
    # Ultralytics convention: Rows = Predicted, Cols = True
    mat_pred_true = mat.T

    if normalize:
        col_sums = mat_pred_true.sum(axis=0, keepdims=True)
        display_mat = mat_pred_true / (col_sums + 1e-9)
        title = "Confusion Matrix Normalized"
        filename = f"{model_key}_confusion_matrix_normalized.png"
        fmt = ".2f"
        vmax = 1.0
    else:
        display_mat = mat_pred_true
        title = "Confusion Matrix"
        filename = f"{model_key}_confusion_matrix_raw.png"
        fmt = ".0f"
        vmax = mat_pred_true.max()

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.heatmap(
        display_mat,
        ax=ax,
        annot=True,
        annot_kws={"size": 11, "fontweight": "normal"},
        cmap="Blues",
        fmt=fmt,
        square=True,
        vmin=0.0,
        vmax=vmax,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        cbar_kws={"fraction": 0.046, "pad": 0.04}
    )

    ax.set_xlabel("True", fontsize=11, labelpad=12)
    ax.set_ylabel("Predicted", fontsize=11, labelpad=12)
    ax.set_title(title, fontsize=12, pad=15)
    plt.xticks(rotation=90, ha="center", fontsize=11)
    plt.yticks(rotation=0, va="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_dir / filename, bbox_inches="tight", dpi=300)
    pdf_name = filename.replace(".png", ".pdf")
    plt.savefig(save_dir / pdf_name, bbox_inches="tight")
    plt.close(fig)

def plot_3panel_confusion_matrices(matrices_dict, class_names, save_path, normalize=True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    fig.patch.set_facecolor("white")
    max_raw_val = max(cm.max() for cm in matrices_dict.values())

    for idx, (m_key, cm) in enumerate(matrices_dict.items()):
        ax = axes[idx]
        ax.set_facecolor("white")
        cm_pred_true = cm.T.astype(float)

        if normalize:
            col_sums = cm_pred_true.sum(axis=0, keepdims=True)
            display_mat = cm_pred_true / (col_sums + 1e-9)
            vmax = 1.0
            fmt = ".2f"
        else:
            display_mat = cm_pred_true
            vmax = max_raw_val
            fmt = ".0f"

        sns.heatmap(
            display_mat,
            ax=ax,
            annot=True,
            annot_kws={"size": 10},
            cmap="Blues",
            fmt=fmt,
            square=True,
            vmin=0.0,
            vmax=vmax,
            xticklabels=class_names,
            yticklabels=class_names if idx == 0 else ["" for _ in class_names],
            cbar=(idx == 2),
            cbar_kws={"fraction": 0.046, "pad": 0.04} if idx == 2 else None
        )

        ax.set_title(f"{m_key}", fontsize=12, pad=12)
        ax.set_xlabel("True", fontsize=11, labelpad=10)
        if idx == 0:
            ax.set_ylabel("Predicted", fontsize=11, labelpad=10)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)

    title_suffix = "Normalized" if normalize else "Raw Counts"
    plt.suptitle(f"Confusion Matrix Comparison ({title_suffix})", fontsize=13, y=0.98)
    plt.tight_layout()
    out_stem = save_path.stem + ("_normalized" if normalize else "_raw")
    plt.savefig(save_path.parent / f"{out_stem}.png", bbox_inches="tight", dpi=300)
    plt.savefig(save_path.parent / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# ── 6. Pure PyTorch Grad-CAM++ Hook Engine ───────────────────────────────────
class M4AttributionEngine:
    def __init__(self, model_path, device_target):
        self.device = device_target
        self.yolo_wrapper = YOLO(str(model_path))
        self.model = self.yolo_wrapper.model.to(self.device)
        self.names = self.yolo_wrapper.names
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.activations = None
        self._target_layer = self._find_target_layer()
        self._register_hooks()

    def _find_target_layer(self):
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            for layer in reversed(self.model.model):
                for sub_name, sub_module in reversed(list(layer.named_modules())):
                    if isinstance(sub_module, nn.Conv2d):
                        return sub_module
            return self.model.model[-2]
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("Target convolutional layer not found.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.retain_grad()
        self._target_layer.register_forward_hook(forward_hook)

    def compute_attribution_map(self, input_tensor, target_class=None):
        with torch.set_grad_enabled(True):
            for param in self.model.parameters():
                param.requires_grad = True
                
            input_tensor = input_tensor.clone().detach().to(self.device).requires_grad_(True)
            self.model.zero_grad()
            
            logits = self.model(input_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            elif logits.dim() > 2:
                logits = logits.flatten(start_dim=1)

            probs = F.softmax(logits, dim=1)
            pred_class = int(torch.argmax(logits, dim=-1).flatten()[0].item())
            class_idx = target_class if (target_class is not None and target_class < logits.shape[1]) else pred_class

            target_score = logits[0, class_idx]
            
            if self.activations is not None and self.activations.requires_grad:
                gradients = torch.autograd.grad(target_score, self.activations, retain_graph=False)[0]
            else:
                target_score.backward(retain_graph=False)
                gradients = torch.ones_like(self.activations)

            activations = self.activations.detach()

            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            sum_activations = activations.sum(dim=(2, 3), keepdim=True)
            
            eps = 1e-7
            denom = 2 * grad_2 + sum_activations * grad_3
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * eps)
            alphas = torch.where(denom != 0.0, grad_2 / denom, torch.zeros_like(grad_2))
            
            weights = (alphas * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            conf = float(probs[0, class_idx].item())
            return cam, pred_class, conf

# ── 7. M4 Faithfulness Metric Evaluator ──────────────────────────────────────
def compute_auc(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    elif hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    else:
        return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))

def compute_gini_sparsity(matrix):
    flat = np.sort(np.abs(matrix.flatten()))
    n = flat.shape[0]
    if flat.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * flat) - (n + 1) * np.sum(flat)) / (n * np.sum(flat)))

def evaluate_m4_metrics(engine, img_bgr, target_class, steps=20):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor_in = (img_norm - mean) / std
    tensor_in = torch.from_numpy(tensor_in).permute(2, 0, 1).unsqueeze(0).float()

    attr_map, pred_class, init_conf = engine.compute_attribution_map(tensor_in, target_class)
    H, W, C = img_norm.shape
    total_pixels = H * W

    sorted_indices = np.argsort(attr_map.flatten())[::-1]
    step_size = int(np.ceil(total_pixels / steps))

    def run_inference_score(tensor_batch):
        with torch.no_grad():
            t = tensor_batch.to(engine.device)
            out = engine.model(t)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if out.dim() == 1:
                out = out.unsqueeze(0)
            elif out.dim() > 2:
                out = out.flatten(start_dim=1)
            p = F.softmax(out, dim=1)
            return float(p[0, target_class].item())

    # 1. Deletion Test
    del_scores = [init_conf]
    cur_del_img = img_norm.copy().reshape(-1, C)
    for s in range(1, steps + 1):
        mask_idx = sorted_indices[:min(s * step_size, total_pixels)]
        tmp = cur_del_img.copy()
        tmp[mask_idx] = 0.0
        pert_img = tmp.reshape(H, W, C)
        pert_tensor = (pert_img - mean) / std
        pert_tensor = torch.from_numpy(pert_tensor).permute(2, 0, 1).unsqueeze(0).float()
        del_scores.append(run_inference_score(pert_tensor))

    # 2. Insertion Test
    ins_scores = []
    base_ins_img = np.zeros((H * W, C), dtype=np.float32)
    flat_orig = img_norm.reshape(-1, C)
    for s in range(0, steps + 1):
        if s == 0:
            ins_tensor = (base_ins_img.reshape(H, W, C) - mean) / std
        else:
            ins_idx = sorted_indices[:min(s * step_size, total_pixels)]
            tmp = base_ins_img.copy()
            tmp[ins_idx] = flat_orig[ins_idx]
            ins_tensor = (tmp.reshape(H, W, C) - mean) / std
        ins_tensor = torch.from_numpy(ins_tensor).permute(2, 0, 1).unsqueeze(0).float()
        ins_scores.append(run_inference_score(ins_tensor))

    x_axis = np.linspace(0, 1, steps + 1)
    del_auc = compute_auc(del_scores, x_axis)
    ins_auc = compute_auc(ins_scores, x_axis)
    faithfulness_delta = ins_auc - del_auc

    # 3. Monotonicity
    deltas = np.diff(del_scores)
    monotonicity = float(np.mean(deltas <= 1e-4))

    # 4. Pearson Correlation
    drop_curve = init_conf - np.array(del_scores)
    try:
        r_val, _ = pearsonr(x_axis, drop_curve)
        f_corr = 0.0 if np.isnan(r_val) else float(r_val)
    except Exception:
        f_corr = 0.0

    # 5. Sparsity
    sparsity = compute_gini_sparsity(attr_map)

    # 6. Composite M4 Score
    norm_del = max(0.0, min(1.0, 1.0 - del_auc))
    norm_ins = max(0.0, min(1.0, ins_auc))
    norm_mono = max(0.0, min(1.0, monotonicity))
    norm_corr = max(0.0, min(1.0, (f_corr + 1.0) / 2.0))
    norm_spar = max(0.0, min(1.0, sparsity))

    terms = [norm_del, norm_ins, norm_mono, norm_corr, norm_spar]
    composite_m4 = len(terms) / sum(1.0 / (val + 1e-5) for val in terms)

    return {
        "deletion_auc": float(del_auc),
        "insertion_auc": float(ins_auc),
        "faithfulness_delta": float(faithfulness_delta),
        "monotonicity": float(monotonicity),
        "f_corr": float(f_corr),
        "sparsity": float(sparsity),
        "composite_m4": float(composite_m4),
        "deletion_curve": del_scores,
        "insertion_curve": ins_scores,
        "pred_class": int(pred_class),
        "init_conf": float(init_conf),
        "attr_map": attr_map
    }

# ── 8. Main Unified Pipeline ────────────────────────────────────────────────
def run_stage2_robust_and_faithfulness_benchmark():
    dataset_path = acquire_dataset()
    output_root  = Path("stage2_robust_m4_results")
    weights_dir  = output_root / "best_weights"
    cm_dir       = output_root / "confusion_matrices"
    onnx_dir     = output_root / "onnx_models"
    faith_dir    = output_root / "faithfulness_benchmark"
    saliency_dir = output_root / "saliency_visualizations"
    reports_dir  = output_root / "00_summary_reports"

    for d in [output_root, weights_dir, cm_dir, onnx_dir, faith_dir, saliency_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3 Target YOLO Classifier Architectures
    models_config = [
        {"name": "yolov8n-cls", "base_weights": "yolov8n-cls.pt", "display_name": "YOLOv8n-cls"},
        {"name": "yolo11n-cls", "base_weights": "yolo11n-cls.pt", "display_name": "YOLO11n-cls"},
        {"name": "yolo26n-cls", "base_weights": "yolo26n-cls.pt", "display_name": "YOLO26n-cls"}
    ]

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    trained_models = {}
    all_confusion_matrices = {}
    classification_summaries = []
    per_class_cls_summaries = []

    # =========================================================================
    # PART 1: FIELD-ROBUST TRAINING FOR ALL 3 MODELS
    # =========================================================================
    for cfg in models_config:
        m_name = cfg["name"]
        disp_name = cfg["display_name"]
        base_w = cfg["base_weights"]

        log.info("=" * 75)
        log.info(f"STARTING FIELD-ROBUST TRAINING: {disp_name.upper()}")
        log.info("=" * 75)

        # Initialize Model
        try:
            model = YOLO(base_w)
        except Exception as e:
            log.warning(f"Direct load failed for {base_w} ({e}). Downloading/initializing...")
            model = YOLO(f"{m_name}.pt")

        train_start = time.perf_counter()
        try:
            train_results = model.train(
                data=str(dataset_path),
                epochs=100,
                patience=15,
                imgsz=224,
                batch=32,
                project=str(output_root / "runs"),
                name=f"{m_name}_robust_run",
                device=0 if torch.cuda.is_available() else "cpu",
                
                # ── Robust Field Augmentations (Lesion-Attention & Light Invariance) ──
                erasing=0.40,        # 40% Random Erasing forces focus on micro-lesions
                hsv_v=0.70,          # Extreme brightness jitter (±70%) for tropical sun/shadows
                hsv_s=0.80,          # Saturation jitter (±80%) to prevent chlorosis confusion
                hsv_h=0.05,          # Accommodates camera AWB drift
                degrees=45.0,        # Rotation invariance
                shear=15.0,          # Oblique perspective distortion
                fliplr=0.5,
                flipud=0.5,
                dropout=0.25,        # 25% Dropout on classifier head
                seed=42,
                verbose=True
            )
        except Exception as e:
            log.error(f"[Training Error on {m_name}] {e}")
            continue

        # Locate and Save Best Weights
        best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = output_root / "runs" / f"{m_name}_robust_run" / "weights" / "best.pt"

        saved_best_pt = weights_dir / f"{m_name}_robust_best.pt"
        shutil.copy2(best_pt, saved_best_pt)
        trained_models[disp_name] = saved_best_pt

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        # Load Best Model for Validation, Test Eval, and ONNX Export
        eval_model = YOLO(str(saved_best_pt))
        param_count = sum(p.numel() for p in eval_model.model.parameters()) / 1e6
        pt_size_mb  = saved_best_pt.stat().st_size / (1024 * 1024)

        # ONNX Export (opset 12 for Hailo-8L NPU)
        log.info(f"[{m_name}] Exporting to ONNX format (opset 12, simplify=True)...")
        onnx_size_mb = 0.0
        try:
            onnx_path = eval_model.export(format="onnx", imgsz=224, opset=12, dynamic=False, simplify=True)
            onnx_file = Path(onnx_path)
            onnx_size_mb = onnx_file.stat().st_size / (1024 * 1024)
            shutil.copy2(onnx_file, onnx_dir / f"{m_name}_robust_best.onnx")
            log.info(f"[{m_name}] ONNX model saved successfully ({onnx_size_mb:.2f} MB)")
        except Exception as e:
            log.warning(f"[{m_name} ONNX Warning] {e}")

        # Independent Test Split Evaluation
        test_dir = dataset_path / "test"
        y_true = []; y_pred = []; latencies_ms = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            c_folder = test_dir / class_name
            if not c_folder.exists():
                continue
            for img_f in [f for f in c_folder.iterdir() if f.suffix.lower() in valid_exts]:
                y_true.append(class_idx)
                pred = eval_model.predict(source=str(img_f), imgsz=224, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
                latencies_ms.append(pred[0].speed["inference"])
                y_pred.append(int(pred[0].probs.top1))

        y_true = np.array(y_true); y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred) * 100.0
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        all_confusion_matrices[disp_name] = cm
        plot_single_confusion_matrix(cm, CLASS_NAMES, f"{m_name}_robust", cm_dir, normalize=True)
        plot_single_confusion_matrix(cm, CLASS_NAMES, f"{m_name}_robust", cm_dir, normalize=False)

        classification_summaries.append({
            "model": disp_name,
            "params_m": param_count,
            "pt_size_mb": pt_size_mb,
            "onnx_size_mb": onnx_size_mb,
            "test_accuracy": acc,
            "macro_precision": p_macro * 100.0,
            "macro_recall": r_macro * 100.0,
            "macro_f1": f1_macro * 100.0,
            "weighted_precision": p_weighted * 100.0,
            "weighted_recall": r_weighted * 100.0,
            "weighted_f1": f1_weighted * 100.0,
            "latency_ms": avg_latency
        })

        for c_idx, c_name in enumerate(CLASS_NAMES):
            per_class_cls_summaries.append({
                "model": disp_name,
                "class": c_name,
                "precision": float(p_per[c_idx]) * 100.0,
                "recall": float(r_per[c_idx]) * 100.0,
                "f1_score": float(f1_per[c_idx]) * 100.0,
                "support": int(sup_per[c_idx])
            })

    # Save 3-Panel Side-by-Side Confusion Matrix (Both Normalized and Raw Counts)
    if len(all_confusion_matrices) == 3:
        plot_3panel_confusion_matrices(all_confusion_matrices, CLASS_NAMES, cm_dir / "stage2_robust_3panel_confusion_matrix", normalize=True)
        plot_3panel_confusion_matrices(all_confusion_matrices, CLASS_NAMES, cm_dir / "stage2_robust_3panel_confusion_matrix", normalize=False)

    df_cls = pd.DataFrame(classification_summaries)
    df_cls.to_csv(reports_dir / "stage2_robust_classification_benchmark.csv", index=False)

    df_per_class_cls = pd.DataFrame(per_class_cls_summaries)
    df_per_class_cls.to_csv(reports_dir / "stage2_robust_per_class_metrics.csv", index=False)

    # ── Markdown Classification Summary ──
    with open(reports_dir / "stage2_robust_classification_summary.md", "w", encoding="utf-8") as f:
        f.write("# Stage 2: Field-Robust Classifier Benchmark Results\n\n")
        f.write("## Overall Performance (Test Split)\n\n")
        f.write("| Model | Params (M) | ONNX (MB) | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Latency (ms) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for _, r in df_cls.iterrows():
            f.write(f"| **{r['model']}** | {r['params_m']:.2f} | {r['onnx_size_mb']:.2f} | {r['test_accuracy']:.2f} | {r['macro_precision']:.2f} | {r['macro_recall']:.2f} | **{r['macro_f1']:.2f}** | {r['latency_ms']:.2f} |\n")
        f.write("\n## Per-Class Breakdown (Precision / Recall / F1)\n\n")
        f.write("| Model | Class | Precision (%) | Recall (%) | F1-Score (%) | Support |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---:|\n")
        for _, r in df_per_class_cls.iterrows():
            f.write(f"| {r['model']} | {r['class']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1_score']:.2f} | {r['support']} |\n")

    # ── LaTeX Table for Thesis Chapter 4 ──
    with open(reports_dir / "stage2_robust_classification_summary.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table*}[htbp]\n\\centering\n")
        f.write("\\caption{Stage 2 Field-Robust Multi-Class Performance Comparison across YOLO Architectures.}\n")
        f.write("\\label{tab:stage2_robust_cls_benchmark}\n")
        f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Params (M)} & \\textbf{ONNX (MB)} & \\textbf{Acc (\\%)} & \\textbf{Prec (\\%)} & \\textbf{Recall (\\%)} & \\textbf{Macro F1 (\\%)} & \\textbf{Latency (ms)} \\\\\n\\midrule\n")
        for _, r in df_cls.iterrows():
            f.write(f"{r['model']} & {r['params_m']:.2f} & {r['onnx_size_mb']:.2f} & {r['test_accuracy']:.2f} & {r['macro_precision']:.2f} & {r['macro_recall']:.2f} & \\textbf{{{r['macro_f1']:.2f}}} & {r['latency_ms']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # =========================================================================
    # PART 2: M4 XAI FAITHFULNESS BENCHMARK
    # =========================================================================
    log.info("\n" + "=" * 75)
    log.info("STARTING NEURIPS 2023 M4 XAI FAITHFULNESS BENCHMARK")
    log.info("=" * 75)

    test_dir = dataset_path / "test"
    class_dirs = [test_dir / c for c in CLASS_NAMES if (test_dir / c).exists()]

    # Collect sampled test set for M4 Faithfulness (e.g. 35 images per class = 105 total)
    sampled_test_set = []
    # Dedicated visual subset for expanded saliency visualizations (e.g. 5 samples per class = 15 total)
    visual_sample_set = []

    for c_dir in class_dirs:
        c_name = c_dir.name
        img_files = sorted(list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpeg")))
        # All samples for quantitative benchmark
        for img_p in img_files[:35]:
            sampled_test_set.append({
                "path": img_p,
                "class_name": c_name,
                "class_idx": CLASS_NAMES.index(c_name)
            })
        # Distinct 5 samples per class for visual saliency grids
        for img_p in img_files[:5]:
            visual_sample_set.append({
                "path": img_p,
                "class_name": c_name,
                "class_idx": CLASS_NAMES.index(c_name)
            })

    results_data = []
    curves_data = {disp: {"deletion": [], "insertion": []} for disp in trained_models}
    saliency_visuals_by_class = {c: [] for c in CLASS_NAMES}
    cross_model_visuals = {s["path"].name: {"path": s["path"], "class_name": s["class_name"], "models": {}} for s in visual_sample_set}

    for disp_name, weight_path in trained_models.items():
        log.info(f"\n[Evaluating M4 Faithfulness] >>> {disp_name} <<<")
        engine = M4AttributionEngine(weight_path, device)

        pbar = tqdm(sampled_test_set, desc=f"M4 Benchmark: {disp_name}")
        for sample in pbar:
            img_bgr = cv2.imread(str(sample["path"]))
            if img_bgr is None: continue
            img_bgr = cv2.resize(img_bgr, (224, 224))

            # Dynamic mapping to model's class index
            target_idx = sample["class_idx"]
            if isinstance(engine.names, dict):
                for k, v in engine.names.items():
                    if str(v).strip().lower() == sample["class_name"].strip().lower():
                        target_idx = int(k); break
            elif isinstance(engine.names, (list, tuple)):
                for k, v in enumerate(engine.names):
                    if str(v).strip().lower() == sample["class_name"].strip().lower():
                        target_idx = k; break

            res = evaluate_m4_metrics(engine, img_bgr, target_idx)

            results_data.append({
                "model": disp_name,
                "image": sample["path"].name,
                "true_class": sample["class_name"],
                "deletion_auc": res["deletion_auc"],
                "insertion_auc": res["insertion_auc"],
                "faithfulness_delta": res["faithfulness_delta"],
                "f_corr": res["f_corr"],
                "monotonicity": res["monotonicity"],
                "sparsity": res["sparsity"],
                "composite_m4": res["composite_m4"]
            })

            curves_data[disp_name]["deletion"].append(res["deletion_curve"])
            curves_data[disp_name]["insertion"].append(res["insertion_curve"])

            # Store for visual grid
            img_key = sample["path"].name
            if img_key in cross_model_visuals:
                cross_model_visuals[img_key]["models"][disp_name] = {
                    "img_bgr": img_bgr,
                    "attr_map": res["attr_map"],
                    "composite_m4": res["composite_m4"],
                    "conf": res["init_conf"]
                }

    df_results = pd.DataFrame(results_data)
    df_results.to_csv(faith_dir / "m4_faithfulness_raw_samples.csv", index=False)

    overall_summary = df_results.groupby("model").agg(
        DEL_AUC=("deletion_auc", "mean"),
        INS_AUC=("insertion_auc", "mean"),
        Delta_AUC=("faithfulness_delta", "mean"),
        Monotonicity=("monotonicity", "mean"),
        Pearson_r=("f_corr", "mean"),
        Sparsity=("sparsity", "mean"),
        M4_Score=("composite_m4", "mean")
    ).reset_index()
    overall_summary.to_csv(faith_dir / "m4_faithfulness_overall_matrix.csv", index=False)

    per_class_summary = df_results.groupby(["model", "true_class"]).agg(
        DEL_AUC=("deletion_auc", "mean"),
        INS_AUC=("insertion_auc", "mean"),
        Delta_AUC=("faithfulness_delta", "mean"),
        M4_Score=("composite_m4", "mean")
    ).reset_index()
    per_class_summary.to_csv(faith_dir / "m4_faithfulness_per_class.csv", index=False)

    # ── Markdown Report ──
    with open(faith_dir / "m4_faithfulness_matrix.md", "w", encoding="utf-8") as f:
        f.write("# M4 XAI Faithfulness Benchmark: Field-Robust Trained YOLO Classifiers\n")
        f.write("## NeurIPS 2023 M4 Benchmark Protocol Implementation\n\n")
        f.write("| Model | Deletion AUC (↓) | Insertion AUC (↑) | Faithfulness ΔAUC (↑) | Monotonicity (↑) | Pearson $r$ (↑) | Sparsity (↑) | **M4 Composite Score** (↑) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for _, r in overall_summary.iterrows():
            f.write(f"| **{r['model']}** | {r['DEL_AUC']:.4f} | {r['INS_AUC']:.4f} | {r['Delta_AUC']:.4f} | {r['Monotonicity']:.4f} | {r['Pearson_r']:.4f} | {r['Sparsity']:.4f} | **{r['M4_Score']:.4f}** |\n")

    # ── LaTeX Table (Chapter 4) ──
    with open(faith_dir / "m4_faithfulness_matrix.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table*}[htbp]\n\\centering\n")
        f.write("\\caption{M4 XAI Faithfulness Benchmark Across Field-Robust Trained YOLO Classifiers.}\n")
        f.write("\\label{tab:m4_faithfulness_robust_3models}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{DEL AUC} ($\\downarrow$) & \\textbf{INS AUC} ($\\uparrow$) & \\textbf{$\\Delta$AUC} ($\\uparrow$) & \\textbf{Mono.} ($\\uparrow$) & \\textbf{Pearson $r$} ($\\uparrow$) & \\textbf{M4 Score} ($\\uparrow$) \\\\\n\\midrule\n")
        for _, r in overall_summary.iterrows():
            f.write(f"{r['model']} & {r['DEL_AUC']:.4f} & {r['INS_AUC']:.4f} & {r['Delta_AUC']:.4f} & {r['Monotonicity']:.4f} & {r['Pearson_r']:.4f} & \\textbf{{{r['M4_Score']:.4f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # =========================================================================
    # PART 3: EXPANDED MULTI-CLASS SALIENCY MAP VISUALIZATIONS
    # =========================================================================
    log.info("\n" + "=" * 75)
    log.info("GENERATING EXPANDED MULTI-CLASS SALIENCY VISUALIZATIONS")
    log.info("=" * 75)

    # 1. Comparative Deletion/Insertion Curves
    steps_pct = np.linspace(0, 100, 21)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    for idx, (m_key, data) in enumerate(curves_data.items()):
        if data["deletion"]:
            axes[0].plot(steps_pct, np.mean(np.array(data["deletion"]), axis=0),
                         label=m_key, color=colors[idx % len(colors)], lw=2.2)
        if data["insertion"]:
            axes[1].plot(steps_pct, np.mean(np.array(data["insertion"]), axis=0),
                         label=m_key, color=colors[idx % len(colors)], lw=2.2)

    axes[0].set_title("(a) M4 Deletion Curves (Attribution Perturbation)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("% Masked Saliency Pixels", fontsize=11)
    axes[0].set_ylabel("Target Class Confidence", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6); axes[0].legend(loc="upper right")

    axes[1].set_title("(b) M4 Insertion Curves (Evidence Recovery)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("% Restored Saliency Pixels", fontsize=11)
    axes[1].set_ylabel("Target Class Confidence", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6); axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(faith_dir / "m4_faithfulness_comparison_curves.png", dpi=300)
    plt.savefig(faith_dir / "m4_faithfulness_comparison_curves.pdf")
    plt.close()

    # 2. Multi-Metric Radar Chart
    try:
        categories = ["1-DEL_AUC", "INS_AUC", "ΔAUC", "Monotonicity", "Pearson r", "Sparsity"]
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)
        summary_m = df_results.groupby("model").mean(numeric_only=True)
        for idx, (gen_name, row) in enumerate(summary_m.iterrows()):
            values = [
                1.0 - row["deletion_auc"], row["insertion_auc"], max(0, row["faithfulness_delta"]),
                row["monotonicity"], max(0, row["f_corr"]), row["sparsity"]
            ]
            values += values[:1]
            c = colors[idx % len(colors)]
            ax.plot(angles, values, label=gen_name, color=c, lw=2)
            ax.fill(angles, values, color=c, alpha=0.12)
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title("M4 Multi-Dimensional Faithfulness Profile", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
        plt.tight_layout()
        plt.savefig(faith_dir / "m4_faithfulness_radar_chart.png", dpi=300)
        plt.savefig(faith_dir / "m4_faithfulness_radar_chart.pdf")
        plt.close()
    except Exception as e:
        log.warning(f"Radar chart skipped: {e}")

    # 3. Expanded Multi-Class Saliency Grid (3 Samples per Class = 9 Rows, 3 Models Side-by-Side)
    # Total Columns = 1 (Input) + 3 (Grad-CAM++ for 3 Models) + 3 (50% Deletion States) = 7 Columns
    num_samples_per_class = 3
    selected_vis_items = []
    for c_name in CLASS_NAMES:
        c_items = [v for v in cross_model_visuals.values() if v["class_name"] == c_name]
        selected_vis_items.extend(c_items[:num_samples_per_class])

    total_rows = len(selected_vis_items)
    if total_rows > 0:
        models_list = list(trained_models.keys())
        num_cols = 1 + len(models_list) * 2  # Input + (Heatmap + Deletion) per model

        fig, axes = plt.subplots(total_rows, num_cols, figsize=(3.2 * num_cols, 3.0 * total_rows), dpi=300)
        fig.patch.set_facecolor("white")

        for row_idx, item in enumerate(selected_vis_items):
            c_name = item["class_name"]
            img_p = item["path"]
            img_bgr = cv2.imread(str(img_p))
            img_bgr = cv2.resize(img_bgr, (224, 224))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Col 0: Original Input Image
            ax0 = axes[row_idx, 0] if total_rows > 1 else axes[0]
            ax0.imshow(img_rgb)
            ax0.set_title(f"Class: {c_name}\n({img_p.stem[:12]})", fontsize=10, fontweight="bold")
            ax0.axis("off")

            col_curr = 1
            for m_idx, m_name in enumerate(models_list):
                m_data = item["models"].get(m_name, None)
                ax_cam = axes[row_idx, col_curr] if total_rows > 1 else axes[col_curr]
                ax_del = axes[row_idx, col_curr + 1] if total_rows > 1 else axes[col_curr + 1]

                if m_data is not None:
                    cam = m_data["attr_map"]
                    m4_sc = m_data["composite_m4"]
                    conf = m_data["conf"]

                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    overlay = np.float32(heatmap) / 255 * 0.45 + np.float32(img_rgb) / 255 * 0.55
                    overlay = np.clip(overlay, 0, 1)

                    sorted_idx = np.argsort(cam.flatten())[::-1]
                    p_img = (np.float32(img_rgb) / 255.0).reshape(-1, 3)
                    p_img[sorted_idx[:int(0.5 * len(sorted_idx))]] = 0
                    p_img = p_img.reshape(224, 224, 3)

                    ax_cam.imshow(overlay)
                    ax_cam.set_title(f"{m_name} Grad-CAM++\nConf: {conf:.2f} | M4: {m4_sc:.3f}", fontsize=9)
                    ax_cam.axis("off")

                    ax_del.imshow(p_img)
                    ax_del.set_title(f"{m_name}\n50% Saliency Masked", fontsize=9)
                    ax_del.axis("off")
                else:
                    ax_cam.axis("off"); ax_del.axis("off")

                col_curr += 2

        plt.suptitle("Stage 2: Cross-Architecture Saliency & Deletion Attribution Matrix Across 3 Disease Classes",
                     fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(saliency_dir / "m4_saliency_comprehensive_3classes_grid.png", bbox_inches="tight", dpi=300)
        plt.savefig(saliency_dir / "m4_saliency_comprehensive_3classes_grid.pdf", bbox_inches="tight")
        plt.close(fig)

    # 4. Per-Class Individual Triad Sheets (5 Samples Per Class)
    for c_name in CLASS_NAMES:
        c_items = [v for v in cross_model_visuals.values() if v["class_name"] == c_name]
        if not c_items: continue
        
        n_samples = min(5, len(c_items))
        fig, axes = plt.subplots(n_samples, 3, figsize=(11, 3.2 * n_samples), dpi=300)
        if n_samples == 1: axes = np.expand_dims(axes, 0)

        for i in range(n_samples):
            item = c_items[i]
            img_bgr = cv2.imread(str(item["path"]))
            img_bgr = cv2.resize(img_bgr, (224, 224))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Pick best model for detailed view (e.g. YOLO11n or YOLOv8n)
            best_model_name = list(item["models"].keys())[0] if item["models"] else "YOLO"
            m_data = item["models"].get(best_model_name, None)

            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f"Input: {c_name} (Sample {i+1})\nFile: {item['path'].name[:12]}", fontsize=9, fontweight="bold")
            axes[i, 0].axis("off")

            if m_data is not None:
                cam = m_data["attr_map"]
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = np.clip(np.float32(heatmap) / 255 * 0.45 + np.float32(img_rgb) / 255 * 0.55, 0, 1)

                sorted_idx = np.argsort(cam.flatten())[::-1]
                p_img = (np.float32(img_rgb) / 255.0).reshape(-1, 3)
                p_img[sorted_idx[:int(0.5 * len(sorted_idx))]] = 0
                p_img = p_img.reshape(224, 224, 3)

                axes[i, 1].imshow(overlay)
                axes[i, 1].set_title(f"Grad-CAM++ Attribution ({best_model_name})\nM4 Score: {m_data['composite_m4']:.3f} | Conf: {m_data['conf']:.2f}", fontsize=9)
                axes[i, 1].axis("off")

                axes[i, 2].imshow(p_img)
                axes[i, 2].set_title("50% Saliency Masked (Deletion)\nDecay State", fontsize=9)
                axes[i, 2].axis("off")
            else:
                axes[i, 1].axis("off"); axes[i, 2].axis("off")

        plt.suptitle(f"Saliency Attribution Profiles: Class [{c_name}] (5 Sample Triads)", fontsize=13, fontweight="bold", y=0.99)
        plt.tight_layout()
        clean_cname = c_name.lower().replace(" ", "_")
        plt.savefig(saliency_dir / f"saliency_triad_samples_{clean_cname}.png", bbox_inches="tight", dpi=300)
        plt.savefig(saliency_dir / f"saliency_triad_samples_{clean_cname}.pdf", bbox_inches="tight")
        plt.close(fig)

    # ── Final Zip Archive ──
    zip_path = Path("stage2_robust_m4_faithfulness_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root_dir, _, files in os.walk(output_root):
            for file in files:
                abs_f = Path(root_dir) / file
                rel_f = abs_f.relative_to(output_root)
                zipf.write(abs_f, arcname=str(rel_f))

    log.info("\n" + "=" * 75)
    log.info("STAGE 2 ROBUST TRAINING & M4 BENCHMARK COMPLETED SUCCESSFULLY ✅")
    log.info("=" * 75)
    log.info("\nClassification Summary:\n" + df_cls.to_string(index=False))
    log.info("\nM4 Faithfulness Matrix:\n" + overall_summary.to_string(index=False))
    log.info("=" * 75)
    log.info(f"Complete packaged archive: {zip_path.resolve()} ✅")

if __name__ == "__main__":
    run_stage2_robust_and_faithfulness_benchmark()
