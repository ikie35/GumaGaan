"""
================================================================================
Kaggle GPU Notebook: M4 XAI Faithfulness Evaluation across 3 Trained YOLO Models
Reference: "M4: A Unified XAI Benchmark for Faithfulness Evaluation of Feature
Attribution Methods across Metrics, Modalities and Models" (NeurIPS 2023)
GitHub: https://github.com/holyseven/M4_XAI_Benchmark
================================================================================
Target 3 Models Evaluated:
  1. YOLOv8n-cls (Baseline)
  2. YOLO11n-cls (Standard / Field-Robust)
  3. YOLO26n-cls (Next-Gen)

Evaluated M4 Faithfulness Metrics:
  - Deletion AUC (DEL_AUC ↓): Score decay under progressive feature masking.
  - Insertion AUC (INS_AUC ↑): Score recovery under progressive feature insertion.
  - Faithfulness Delta (ΔAUC = INS_AUC - DEL_AUC ↑): Net attribution fidelity.
  - Monotonicity (MONO ↑): Consistency of attribution ranking vs incremental drop.
  - Pearson Correlation (Pearson r ↑): Correlation of attribution vs score drop.
  - Saliency Sparsity (SPAR ↑): Gini concentration on pathological lesions.
  - M4 Composite Score (M4_Score ↑): Unified multi-dimensional harmonic index.

Outputs Packaged into 'stage2_m4_faithfulness_results.zip':
  - 'm4_faithfulness_overall_matrix.csv' & 'm4_faithfulness_per_class.csv'
  - 'm4_faithfulness_matrix.md' & 'm4_faithfulness_matrix.tex' (LaTeX for Ch4)
  - 'm4_faithfulness_comparison_curves.png' / '.pdf' (High-res 300 DPI)
  - 'm4_faithfulness_radar_chart.png' / '.pdf' (Multi-metric architectural profile)
  - 'm4_saliency_triad_visualizations.png' / '.pdf'
================================================================================
"""

import os
import sys
import time
import shutil
import zipfile
import logging
import subprocess
from pathlib import Path

# ── 1. Automated Dependency Installation ────────────────────────────────────
REQUIRED_PACKAGES = ["roboflow", "ultralytics", "pandas", "matplotlib", "scikit-learn", "scipy", "tqdm", "opencv-python"]

for pkg in REQUIRED_PACKAGES:
    try:
        if pkg == "opencv-python":
            __import__("cv2")
        else:
            __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"[Setup] Installing required package: {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

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
from tqdm import tqdm
from roboflow import Roboflow
from ultralytics import YOLO

# ── 2. Logging Setup ────────────────────────────────────────────────────────
LOG_FILE = Path("kaggle_m4_benchmark.log")
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

# ── 3. Device Setup ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    log.info("=" * 75)
    log.info(f"[GPU Accelerator Active] : {torch.cuda.get_device_name(0)}")
    log.info(f"[VRAM Available]         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    log.info("=" * 75)
else:
    log.warning("[WARNING] No CUDA GPU found. Executing on CPU.")

# ── 4. Dataset Acquisition ──────────────────────────────────────────────────
DATASET_DIR = Path("Stage-2-Rubber-Leaf-Classifier-4")
CLASS_NAMES = ["Healthy", "Other Disease", "PLFD"]

def acquire_dataset():
    """Locates dataset in Kaggle input/working directory or downloads via Roboflow API."""
    search_dirs = [
        DATASET_DIR,
        Path("/kaggle/input/stage-2-rubber-leaf-classifier-4"),
        Path("/kaggle/working/Stage-2-Rubber-Leaf-Classifier-4")
    ]
    for d in search_dirs:
        if d.exists() and (d / "test").exists():
            log.info(f"[Dataset] Found existing dataset at: {d.resolve()}")
            return d

    log.info("[Dataset] Downloading Stage 2 dataset from Roboflow API (v4)...")
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
    version = project.version(4)
    dataset = version.download("folder", location=str(DATASET_DIR.resolve()))
    return Path(dataset.location)

# ── 5. Intelligent Model Discovery for 3 Trained Models ─────────────────────
def discover_trained_models():
    """
    Discovers the 3 trained classification model weights from Kaggle dataset 'Cls best pts'
    or local directory: yolov8n-cls_best.pt, yolo11n-cls_best.pt, yolo26n-cls_best.pt.
    """
    models = {}
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path(".")]

    for root in search_roots:
        if not root.exists():
            continue

        # 1. YOLOv8n-cls
        if "YOLOv8n-cls" not in models:
            v8_candidates = list(root.glob("**/yolov8n-cls_best.pt")) + list(root.glob("**/*v8*cls*.pt"))
            if v8_candidates:
                models["YOLOv8n-cls"] = v8_candidates[0].resolve()

        # 2. YOLO11n-cls
        if "YOLO11n-cls" not in models:
            v11_candidates = list(root.glob("**/yolo11n-cls_best.pt")) + list(root.glob("**/yolo11n_robust_best.pt")) + list(root.glob("**/*11*cls*.pt"))
            if v11_candidates:
                models["YOLO11n-cls"] = v11_candidates[0].resolve()

        # 3. YOLO26n-cls
        if "YOLO26n-cls" not in models:
            v26_candidates = list(root.glob("**/yolo26n-cls_best.pt")) + list(root.glob("**/*26*cls*.pt"))
            if v26_candidates:
                models["YOLO26n-cls"] = v26_candidates[0].resolve()

    log.info("=" * 75)
    log.info(f"[Model Discovery] Found {len(models)} classification models in Kaggle:")
    for name, path in models.items():
        log.info(f"  -> {name}: {path}")
    log.info("=" * 75)
    
    if not models:
        raise FileNotFoundError("Could not locate weights in /kaggle/input/cls-best-pts. Please check dataset attachment.")

    return models

# ── 6. Pure PyTorch Grad-CAM++ Hook Engine ───────────────────────────────────
class M4AttributionEngine:
    def __init__(self, model_path, device):
        self.device = device
        self.yolo_wrapper = YOLO(str(model_path))
        self.model = self.yolo_wrapper.model.to(self.device)
        self.names = self.yolo_wrapper.names
        
        # Ensure all parameters track gradients for XAI feature attribution
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
        raise ValueError("Target layer not found.")

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
            
            # Compute gradients w.r.t activation map
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
    """Universal trapezoidal integration compatible with NumPy 1.x, NumPy 2.x, and SciPy."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    elif hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    else:
        from scipy.integrate import trapezoid
        return float(trapezoid(y, x))

def evaluate_m4_metrics(engine, image_bgr, target_class, num_steps=21):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0
    input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(engine.device)
    
    attr_map, pred_class, init_conf = engine.compute_attribution_map(input_tensor, target_class=target_class)
    
    h, w = attr_map.shape
    num_pixels = h * w
    flat_attr = attr_map.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1]
    
    steps = np.linspace(0, 1, num_steps)
    deletion_scores = []
    insertion_scores = []
    
    with torch.no_grad():
        # Deletion
        for step in steps:
            k = int(step * num_pixels)
            p_img = img_float.copy().reshape(-1, 3)
            if k > 0:
                p_img[sorted_indices[:k]] = 0.0
            p_img = p_img.reshape(h, w, 3)
            p_tensor = torch.from_numpy(p_img).permute(2, 0, 1).unsqueeze(0).to(engine.device)
            out = engine.model(p_tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if out.dim() == 1:
                out = out.unsqueeze(0)
            elif out.dim() > 2:
                out = out.flatten(start_dim=1)
            c_idx = target_class if target_class < out.shape[1] else 0
            deletion_scores.append(float(F.softmax(out, dim=1)[0, c_idx].item()))

        # Insertion
        for step in steps:
            k = int(step * num_pixels)
            p_img = np.zeros_like(img_float).reshape(-1, 3)
            if k > 0:
                p_img[sorted_indices[:k]] = img_float.reshape(-1, 3)[sorted_indices[:k]]
            p_img = p_img.reshape(h, w, 3)
            p_tensor = torch.from_numpy(p_img).permute(2, 0, 1).unsqueeze(0).to(engine.device)
            out = engine.model(p_tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if out.dim() == 1:
                out = out.unsqueeze(0)
            elif out.dim() > 2:
                out = out.flatten(start_dim=1)
            c_idx = target_class if target_class < out.shape[1] else 0
            insertion_scores.append(float(F.softmax(out, dim=1)[0, c_idx].item()))

    del_auc = compute_auc(deletion_scores, steps)
    ins_auc = compute_auc(insertion_scores, steps)
    faithfulness_delta = ins_auc - del_auc

    step_drops = np.maximum(0, init_conf - np.array(deletion_scores))
    f_corr = float(pearsonr(steps, step_drops)[0]) if np.std(step_drops) > 1e-6 and np.std(steps) > 1e-6 else 0.0
    mono_score = float(np.mean(np.diff(deletion_scores) <= 1e-4))
    
    sorted_vals = np.sort(flat_attr)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    gini_sparsity = float((np.sum((2 * index - n - 1) * sorted_vals)) / (n * np.sum(sorted_vals) + 1e-8))
    composite_m4 = float(0.35 * (1.0 - del_auc) + 0.35 * ins_auc + 0.15 * mono_score + 0.15 * max(0, f_corr))

    return {
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        "faithfulness_delta": faithfulness_delta,
        "f_corr": f_corr,
        "monotonicity": mono_score,
        "sparsity": gini_sparsity,
        "composite_m4": composite_m4,
        "deletion_curve": deletion_scores,
        "insertion_curve": insertion_scores,
        "attr_map": attr_map,
        "init_conf": init_conf,
        "pred_class": pred_class
    }

# ── 8. Main Benchmark Runner ────────────────────────────────────────────────
def main():
    output_dir = Path("stage2_m4_faithfulness_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = acquire_dataset()
    test_dir = dataset_dir / "test" if (dataset_dir / "test").exists() else dataset_dir / "valid"
    
    models_registry = discover_trained_models()
    
    # Collect held-out test specimens (35 samples per class = 105 images for high statistical power)
    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    sampled_test_set = []
    for c_dir in class_dirs:
        c_name = c_dir.name
        img_files = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpeg"))
        for img_path in img_files[:35]:
            sampled_test_set.append({
                "path": img_path,
                "class_name": c_name,
                "class_idx": CLASS_NAMES.index(c_name) if c_name in CLASS_NAMES else 0
            })

    log.info(f"\n[Benchmark Execution] Evaluating {len(sampled_test_set)} test images across {len(models_registry)} models...")

    results_data = []
    curves_data = {gen: {"deletion": [], "insertion": []} for gen in models_registry}
    sample_visuals = []

    for gen_name, weight_path in models_registry.items():
        log.info(f"\n[Evaluating Model] >>> {gen_name} <<<")
        engine = M4AttributionEngine(weight_path, device)
        
        pbar = tqdm(sampled_test_set, desc=f"M4 XAI Faithfulness: {gen_name}")
        for idx, sample in enumerate(pbar):
            img_bgr = cv2.imread(str(sample["path"]))
            if img_bgr is None:
                continue
            img_bgr = cv2.resize(img_bgr, (224, 224))
            
            # Dynamic mapping of dataset folder name to model's class index
            target_class_idx = None
            if isinstance(engine.names, dict):
                for k, v in engine.names.items():
                    if str(v).strip().lower() == sample["class_name"].strip().lower():
                        target_class_idx = int(k)
                        break
            elif isinstance(engine.names, (list, tuple)):
                for k, v in enumerate(engine.names):
                    if str(v).strip().lower() == sample["class_name"].strip().lower():
                        target_class_idx = k
                        break
                        
            if target_class_idx is None:
                target_class_idx = sample["class_idx"] if sample["class_idx"] < len(engine.names) else 0

            res = evaluate_m4_metrics(engine, img_bgr, target_class_idx)
            
            # Safe extraction of predicted class name
            pred_idx = res["pred_class"]
            if isinstance(engine.names, dict):
                pred_label = engine.names.get(pred_idx, str(pred_idx))
            elif isinstance(engine.names, (list, tuple)) and pred_idx < len(engine.names):
                pred_label = engine.names[pred_idx]
            elif pred_idx < len(CLASS_NAMES):
                pred_label = CLASS_NAMES[pred_idx]
            else:
                pred_label = f"Class_{pred_idx}"

            results_data.append({
                "model": gen_name,
                "image": sample["path"].name,
                "true_class": sample["class_name"],
                "pred_class": pred_label,
                "deletion_auc": res["deletion_auc"],
                "insertion_auc": res["insertion_auc"],
                "faithfulness_delta": res["faithfulness_delta"],
                "f_corr": res["f_corr"],
                "monotonicity": res["monotonicity"],
                "sparsity": res["sparsity"],
                "composite_m4": res["composite_m4"]
            })
            
            curves_data[gen_name]["deletion"].append(res["deletion_curve"])
            curves_data[gen_name]["insertion"].append(res["insertion_curve"])

            if len(sample_visuals) < len(models_registry) * 3 and (idx % (max(1, len(sampled_test_set) // 3)) == 0):
                sample_visuals.append({
                    "model": gen_name,
                    "img_bgr": img_bgr,
                    "attr_map": res["attr_map"],
                    "class_name": sample["class_name"],
                    "del_auc": res["deletion_auc"],
                    "ins_auc": res["insertion_auc"],
                    "composite_m4": res["composite_m4"]
                })

    df_results = pd.DataFrame(results_data)
    
    # ── Compile Reports ──
    df_results.to_csv(output_dir / "m4_faithfulness_raw_samples.csv", index=False)
    
    overall_summary = df_results.groupby("model").agg(
        DEL_AUC=("deletion_auc", "mean"),
        INS_AUC=("insertion_auc", "mean"),
        Delta_AUC=("faithfulness_delta", "mean"),
        Monotonicity=("monotonicity", "mean"),
        Pearson_r=("f_corr", "mean"),
        Sparsity=("sparsity", "mean"),
        M4_Score=("composite_m4", "mean")
    ).reset_index()
    overall_summary.to_csv(output_dir / "m4_faithfulness_overall_matrix.csv", index=False)

    per_class_summary = df_results.groupby(["model", "true_class"]).agg(
        DEL_AUC=("deletion_auc", "mean"),
        INS_AUC=("insertion_auc", "mean"),
        Delta_AUC=("faithfulness_delta", "mean"),
        M4_Score=("composite_m4", "mean")
    ).reset_index()
    per_class_summary.to_csv(output_dir / "m4_faithfulness_per_class.csv", index=False)

    # ── Markdown Report ──
    with open(output_dir / "m4_faithfulness_matrix.md", "w", encoding="utf-8") as f:
        f.write("# M4 XAI Faithfulness Benchmark: 3 Trained YOLO Classifiers\n")
        f.write("## NeurIPS 2023 M4 Benchmark Protocol Implementation\n\n")
        f.write("| Model | Deletion AUC (↓) | Insertion AUC (↑) | Faithfulness ΔAUC (↑) | Monotonicity (↑) | Pearson $r$ (↑) | Sparsity (↑) | **M4 Composite Score** (↑) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for _, r in overall_summary.iterrows():
            f.write(f"| **{r['model']}** | {r['DEL_AUC']:.4f} | {r['INS_AUC']:.4f} | {r['Delta_AUC']:.4f} | {r['Monotonicity']:.4f} | {r['Pearson_r']:.4f} | {r['Sparsity']:.4f} | **{r['M4_Score']:.4f}** |\n")

    # ── LaTeX Table (Chapter 4) ──
    with open(output_dir / "m4_faithfulness_matrix.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{M4 XAI Faithfulness Evaluation across the 3 Trained YOLO Classifier Models.}\n")
        f.write("\\label{tab:m4_faithfulness_3models}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model Architecture} & \\textbf{DEL AUC} ($\\downarrow$) & \\textbf{INS AUC} ($\\uparrow$) & \\textbf{$\\Delta$AUC} ($\\uparrow$) & \\textbf{Mono.} ($\\uparrow$) & \\textbf{Pearson $r$} ($\\uparrow$) & \\textbf{M4 Score} ($\\uparrow$) \\\\\n")
        f.write("\\midrule\n")
        for _, r in overall_summary.iterrows():
            f.write(f"{r['model']} & {r['DEL_AUC']:.4f} & {r['INS_AUC']:.4f} & {r['Delta_AUC']:.4f} & {r['Monotonicity']:.4f} & {r['Pearson_r']:.4f} & \\textbf{{{r['M4_Score']:.4f}}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    # ── Publication Figures ──
    steps_pct = np.linspace(0, 100, 21)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # 1. Comparative Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    for idx, (gen_name, data) in enumerate(curves_data.items()):
        if not data["deletion"]:
            continue
        del_mat = np.array(data["deletion"])
        mean_del = np.mean(del_mat, axis=0)
        c = colors[idx % len(colors)]
        axes[0].plot(steps_pct, mean_del, label=f"{gen_name}", color=c, lw=2.2)
        
    axes[0].set_title("(a) M4 Deletion Curves (Attribution Perturbation)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("% Masked Saliency Pixels", fontsize=11)
    axes[0].set_ylabel("Target Class Confidence", fontsize=11)
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    for idx, (gen_name, data) in enumerate(curves_data.items()):
        if not data["insertion"]:
            continue
        ins_mat = np.array(data["insertion"])
        mean_ins = np.mean(ins_mat, axis=0)
        c = colors[idx % len(colors)]
        axes[1].plot(steps_pct, mean_ins, label=f"{gen_name}", color=c, lw=2.2)

    axes[1].set_title("(b) M4 Insertion Curves (Evidence Recovery)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("% Restored Saliency Pixels", fontsize=11)
    axes[1].set_ylabel("Target Class Confidence", fontsize=11)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_dir / "m4_faithfulness_comparison_curves.png", dpi=300)
    plt.savefig(output_dir / "m4_faithfulness_comparison_curves.pdf")
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
                1.0 - row["deletion_auc"],
                row["insertion_auc"],
                max(0, row["faithfulness_delta"]),
                row["monotonicity"],
                max(0, row["f_corr"]),
                row["sparsity"]
            ]
            values += values[:1]
            c = colors[idx % len(colors)]
            ax.plot(angles, values, label=gen_name, color=c, lw=2)
            ax.fill(angles, values, color=c, alpha=0.12)
            
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title("M4 Multi-Dimensional Faithfulness Profile", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
        plt.tight_layout()
        plt.savefig(output_dir / "m4_faithfulness_radar_chart.png", dpi=300)
        plt.savefig(output_dir / "m4_faithfulness_radar_chart.pdf")
        plt.close()
    except Exception as e:
        log.warning(f"Radar chart skipped: {e}")

    # 3. Visual Saliency Triads
    if sample_visuals:
        num_vis = min(6, len(sample_visuals))
        fig, axes = plt.subplots(num_vis, 3, figsize=(10, 3 * num_vis), dpi=300)
        if num_vis == 1:
            axes = np.expand_dims(axes, 0)
            
        for i in range(num_vis):
            vis = sample_visuals[i]
            img_rgb = cv2.cvtColor(vis["img_bgr"], cv2.COLOR_BGR2RGB)
            cam = vis["attr_map"]
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.float32(heatmap) / 255 * 0.4 + np.float32(img_rgb) / 255 * 0.6
            overlay = np.clip(overlay, 0, 1)

            sorted_idx = np.argsort(cam.flatten())[::-1]
            p_img = (np.float32(img_rgb) / 255.0).reshape(-1, 3)
            p_img[sorted_idx[:int(0.5 * len(sorted_idx))]] = 0
            p_img = p_img.reshape(224, 224, 3)

            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f"Input: {vis['class_name']}\n[{vis['model']}]", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"Grad-CAM++ Saliency\nScore: {vis['composite_m4']:.3f}", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(p_img)
            axes[i, 2].set_title("50% Saliency Masked\n(Deletion State)", fontsize=9)
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "m4_saliency_triad_visualizations.png", dpi=300)
        plt.savefig(output_dir / "m4_saliency_triad_visualizations.pdf")
        plt.close()

    # ── Zip Archive ──
    zip_path = Path("stage2_m4_faithfulness_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.glob("*"):
            zipf.write(file, arcname=file.name)

    log.info("\n" + "=" * 75)
    log.info("M4 FAITHFULNESS BENCHMARK RESULTS (NeurIPS 2023)")
    log.info("=" * 75)
    log.info("\n" + overall_summary.to_string(index=False))
    log.info("=" * 75)
    log.info(f"Complete packaged archive created at: {zip_path.resolve()} ✅")

if __name__ == "__main__":
    main()
