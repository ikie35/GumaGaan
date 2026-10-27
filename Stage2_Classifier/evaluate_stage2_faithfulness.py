"""
================================================================================
Stage 2: XAI Faithfulness Matrix & Perturbation Benchmark Script
Cross-Generation Evaluation: YOLOv8-cls vs YOLO11-cls vs YOLO11n-Robust vs YOLO26-cls
Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
================================================================================
Dataset: Roboflow "stage-2-rubber-leaf-classifier" (v4)
Target Classes: [Healthy, Other Disease, PLFD]
Input Resolution: 224x224

Faithfulness Evaluation Metrics:
  1. Deletion Curve & Deletion AUC (Lower is better, indicates faithful saliency)
  2. Insertion Curve & Insertion AUC (Higher is better, indicates salient feature capture)
  3. Faithfulness Delta (ΔAUC = Insertion AUC - Deletion AUC; Higher is better)
  4. Pearson Faithfulness Correlation (Correlation between saliency and score drop)
  5. Per-Class & Overall Faithfulness Matrix (CSV, Markdown, LaTeX)

Outputs:
  - 'stage2_faithfulness_matrix.csv' (Raw metrics table)
  - 'stage2_faithfulness_matrix.md' (Publication Markdown report)
  - 'stage2_faithfulness_matrix.tex' (LaTeX table ready for thesis Chapter 4)
  - 'faithfulness_curves_comparison.png' / '.pdf' (Cross-generation curves)
  - 'faithfulness_visual_triads.png' / '.pdf' (Grad-CAM++ & perturbation visuals)
  - Packaged ZIP archive: 'stage2_faithfulness_results.zip'
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

# ── 1. Automated Dependency Management ──────────────────────────────────────
REQUIRED_PACKAGES = ["roboflow", "ultralytics", "pandas", "matplotlib", "scikit-learn", "scipy", "tqdm", "opencv-python"]

def ensure_packages(packages):
    """Installs required packages if missing."""
    for pkg in packages:
        try:
            if pkg == "opencv-python":
                __import__("cv2")
            else:
                __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"[Setup] Installing missing package: {pkg}...")
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
from tqdm import tqdm
from roboflow import Roboflow
from ultralytics import YOLO

# ── 2. Logging Setup ────────────────────────────────────────────────────────
LOG_FILE = Path("stage2_faithfulness_benchmark.log")
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

# ── 3. Device & Hardware Configuration ──────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("=" * 75)
    log.info(f"[GPU Accelerator Active] : {gpu_name}")
    log.info(f"[VRAM Available]         : {vram_gb:.2f} GB")
    log.info("=" * 75)
else:
    log.warning("=" * 75)
    log.warning("[WARNING] No CUDA GPU detected. Running on CPU.")
    log.warning("=" * 75)

# ── 4. Dataset Acquisition & Loading ────────────────────────────────────────
DATASET_DIR = Path("Stage-2-Rubber-Leaf-Classifier-4")
CLASS_NAMES = ["Healthy", "Other Disease", "PLFD"]

def acquire_dataset():
    """Locates or downloads the Stage 2 classification dataset."""
    search_paths = [
        DATASET_DIR,
        Path("../Stage-2-Rubber-Leaf-Classifier-4"),
        Path("d:/school/Masters/Research/Training/Two stage Rubber Disease/Stage-2-Rubber-Leaf-Classifier-4"),
        Path("Stage2_Classifier/Stage-2-Rubber-Leaf-Classifier-4")
    ]
    for p in search_paths:
        if p.exists() and (p / "test").exists():
            log.info(f"[Dataset] Found local dataset at: {p.resolve()}")
            return p

    log.info("[Dataset] Downloading Stage 2 dataset from Roboflow API (v4)...")
    try:
        rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
        project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
        version = project.version(4)
        dataset = version.download("folder", location=str(DATASET_DIR.resolve()))
        log.info(f"[Dataset] Download complete: {dataset.location}")
        return Path(dataset.location)
    except Exception as e:
        log.error(f"[Dataset Error] Roboflow download failed: {e}")
        raise

# ── 5. Model Generations Registry ───────────────────────────────────────────
def discover_model_generations(base_dir=Path(".")):
    """Discovers available trained YOLO classification model weights."""
    models_registry = {}

    candidates = {
        "YOLOv8n-cls (Baseline)": [
            base_dir / "stage2_generation_benchmark_results" / "best_weights" / "yolov8n-cls_best.pt",
            base_dir / "Training Result on v4 dataset" / "weights" / "best.pt",
            base_dir / "stage2_size_comparison_results" / "best_weights" / "yolov8n_best.pt",
            Path("Stage2_Classifier/stage2_generation_benchmark_results/best_weights/yolov8n-cls_best.pt"),
            Path("Stage2_Classifier/Training Result on v4 dataset/weights/best.pt")
        ],
        "YOLO11n-cls (Standard)": [
            base_dir / "stage2_generation_benchmark_results" / "best_weights" / "yolo11n-cls_best.pt",
            base_dir / "stage2_yolo11_size_benchmark_results" / "best_weights" / "yolo11n_best.pt",
            Path("Stage2_Classifier/stage2_generation_benchmark_results/best_weights/yolo11n-cls_best.pt"),
            Path("Stage2_Classifier/stage2_yolo11_size_benchmark_results/best_weights/yolo11n_best.pt")
        ],
        "YOLO11n-cls (Field-Robust)": [
            base_dir / "stage2_yolo11n_robust_results" / "best_weights" / "yolo11n_robust_best.pt",
            Path("Stage2_Classifier/stage2_yolo11n_robust_results/best_weights/yolo11n_robust_best.pt")
        ],
        "YOLO26n-cls (Next-Gen)": [
            base_dir / "stage2_generation_benchmark_results" / "best_weights" / "yolo26n-cls_best.pt",
            Path("Stage2_Classifier/stage2_generation_benchmark_results/best_weights/yolo26n-cls_best.pt")
        ]
    }

    for gen_name, path_list in candidates.items():
        for p in path_list:
            if p.exists():
                models_registry[gen_name] = p.resolve()
                log.info(f"[Model Registry] Discovered '{gen_name}' at: {p}")
                break

    if not models_registry:
        raise FileNotFoundError("No trained Stage 2 YOLO classifier weights (.pt) found!")
    
    return models_registry

# ── 6. Pure PyTorch Grad-CAM++ Engine ────────────────────────────────────────
class GradCAMPlusPlusEngine:
    """
    Self-contained, robust Grad-CAM++ implementation for Ultralytics YOLO models.
    Hooks into the final convolutional/spatial feature representation layer.
    """
    def __init__(self, model_path, device):
        self.device = device
        self.yolo_wrapper = YOLO(str(model_path))
        self.model = self.yolo_wrapper.model.to(self.device)
        self.names = self.yolo_wrapper.names
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.gradients = None
        self.activations = None
        self._target_layer = self._find_target_layer()
        self._register_hooks()

    def _find_target_layer(self):
        """Finds the optimal target feature layer before the global pooling head."""
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            for layer in reversed(self.model.model):
                for sub_name, sub_module in reversed(list(layer.named_modules())):
                    if isinstance(sub_module, nn.Conv2d):
                        return sub_module
            return self.model.model[-2]
        
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("Could not automatically locate a target Conv2D layer in model.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.retain_grad()
        self._target_layer.register_forward_hook(forward_hook)

    def generate_saliency_map(self, input_tensor, target_class=None):
        """
        Generates normalized (224x224) Grad-CAM++ heatmap for input tensor.
        """
        with torch.set_grad_enabled(True):
            for param in self.model.parameters():
                param.requires_grad = True

            input_tensor = input_tensor.clone().detach().to(self.device).requires_grad_(True)
            self.model.zero_grad()
            
            logits = self.model(input_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            probs = F.softmax(logits, dim=1)
            pred_class = int(logits.argmax(dim=1).item())
            class_idx = target_class if target_class is not None else pred_class

            target_score = logits[0, class_idx]
            
            if self.activations is not None and self.activations.requires_grad:
                gradients = torch.autograd.grad(target_score, self.activations, retain_graph=False)[0]
            else:
                target_score.backward(retain_graph=False)
                gradients = self.gradients if self.gradients is not None else torch.ones_like(self.activations)

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
            conf = float(probs[0, class_idx].item())
            return cam, pred_class, conf

# ── 7. Perturbation Analysis & Faithfulness Metrics ─────────────────────────
def compute_auc(y, x):
    """Universal trapezoidal integration compatible with NumPy 1.x, NumPy 2.x, and SciPy."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    elif hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    else:
        from scipy.integrate import trapezoid
        return float(trapezoid(y, x))

def compute_faithfulness_metrics(engine, image_bgr, target_class, num_steps=21):
    """
    Computes Deletion & Insertion Curves and calculates Faithfulness AUC metrics.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0
    
    input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(engine.device)
    
    cam_map, pred_class, init_conf = engine.generate_saliency_map(input_tensor, target_class=target_class)
    
    h, w = cam_map.shape
    num_pixels = h * w
    flat_cam = cam_map.flatten()
    sorted_indices = np.argsort(flat_cam)[::-1]
    
    steps = np.linspace(0, 1, num_steps)
    deletion_scores = []
    insertion_scores = []
    
    with torch.no_grad():
        # 1. Deletion Curve (Mask most salient pixels gradually)
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
            score = float(F.softmax(out, dim=1)[0, target_class].item())
            deletion_scores.append(score)
            
        # 2. Insertion Curve (Introduce most salient pixels onto empty canvas)
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
            score = float(F.softmax(out, dim=1)[0, target_class].item())
            insertion_scores.append(score)
            
    del_auc = compute_auc(deletion_scores, steps)
    ins_auc = compute_auc(insertion_scores, steps)
    faithfulness_delta = ins_auc - del_auc
    
    step_drops = np.maximum(0, init_conf - np.array(deletion_scores))
    if np.std(step_drops) > 1e-6 and np.std(steps) > 1e-6:
        corr, _ = pearsonr(steps, step_drops)
    else:
        corr = 0.0

    return {
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        "faithfulness_delta": faithfulness_delta,
        "pearson_corr": float(corr),
        "deletion_curve": deletion_scores,
        "insertion_curve": insertion_scores,
        "steps": steps,
        "cam_map": cam_map,
        "init_conf": init_conf,
        "pred_class": pred_class
    }

# ── 8. Benchmark Execution Engine ───────────────────────────────────────────
def run_faithfulness_benchmark(dataset_dir, models_registry, output_dir, samples_per_class=30):
    """
    Executes cross-generation XAI faithfulness benchmark across the held-out test split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        test_dir = dataset_dir / "valid"

    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    sampled_test_set = []
    
    for c_dir in class_dirs:
        c_name = c_dir.name
        img_files = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpeg"))
        selected = img_files[:samples_per_class] if samples_per_class > 0 else img_files
        for img_path in selected:
            sampled_test_set.append({
                "path": img_path,
                "class_name": c_name,
                "class_idx": CLASS_NAMES.index(c_name) if c_name in CLASS_NAMES else 0
            })
            
    log.info(f"[Benchmark] Evaluating {len(sampled_test_set)} test images across {len(models_registry)} model architectures...")

    results_data = []
    curves_data = {gen: {"deletion": [], "insertion": []} for gen in models_registry}
    sample_visuals = []

    for gen_name, weight_path in models_registry.items():
        log.info(f"\n[Evaluating Architecture] >>> {gen_name} <<<")
        engine = GradCAMPlusPlusEngine(weight_path, device)
        
        gen_del_aucs = []
        gen_ins_aucs = []
        gen_deltas = []
        gen_corrs = []
        
        pbar = tqdm(sampled_test_set, desc=f"XAI Faithfulness: {gen_name}")
        for idx, sample in enumerate(pbar):
            img_bgr = cv2.imread(str(sample["path"]))
            if img_bgr is None:
                continue
            img_bgr = cv2.resize(img_bgr, (224, 224))
            
            res = compute_faithfulness_metrics(engine, img_bgr, sample["class_idx"])
            
            results_data.append({
                "generation": gen_name,
                "image_name": sample["path"].name,
                "true_class": sample["class_name"],
                "pred_class": CLASS_NAMES[res["pred_class"]],
                "initial_confidence": res["init_conf"],
                "deletion_auc": res["deletion_auc"],
                "insertion_auc": res["insertion_auc"],
                "faithfulness_delta": res["faithfulness_delta"],
                "pearson_corr": res["pearson_corr"]
            })
            
            gen_del_aucs.append(res["deletion_auc"])
            gen_ins_aucs.append(res["insertion_auc"])
            gen_deltas.append(res["faithfulness_delta"])
            gen_corrs.append(res["pearson_corr"])
            curves_data[gen_name]["deletion"].append(res["deletion_curve"])
            curves_data[gen_name]["insertion"].append(res["insertion_curve"])

            if len(sample_visuals) < len(models_registry) * 3 and (idx % (max(1, len(sampled_test_set) // 3)) == 0):
                sample_visuals.append({
                    "gen_name": gen_name,
                    "img_bgr": img_bgr,
                    "cam_map": res["cam_map"],
                    "class_name": sample["class_name"],
                    "del_auc": res["deletion_auc"],
                    "ins_auc": res["insertion_auc"]
                })

        log.info(f"[{gen_name} Summary] Deletion AUC: {np.mean(gen_del_aucs):.4f} (↓) | Insertion AUC: {np.mean(gen_ins_aucs):.4f} (↑) | ΔAUC: {np.mean(gen_deltas):.4f} (↑)")

    df_results = pd.DataFrame(results_data)
    
    # ── 9. Compile Reports ──
    compile_matrix_reports(df_results, output_dir)
    
    # ── 10. Generate Visual Plots ──
    generate_faithfulness_plots(curves_data, sample_visuals, output_dir)
    
    # ── 11. Archive Results to ZIP ──
    zip_path = output_dir / "stage2_faithfulness_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.glob("*"):
            if file.name != zip_path.name:
                zipf.write(file, arcname=file.name)
    log.info(f"\n[Packaging] All XAI Faithfulness artifacts packaged into: {zip_path.resolve()}")

# ── 11. Report Compilation (CSV, Markdown, LaTeX) ───────────────────────────
def compile_matrix_reports(df, output_dir):
    """Compiles publication-grade comparative tables across architectures and classes."""
    csv_raw = output_dir / "stage2_faithfulness_raw_samples.csv"
    df.to_csv(csv_raw, index=False)
    
    # Overall Architecture Summary
    overall_summary = df.groupby("generation").agg(
        Mean_Deletion_AUC=("deletion_auc", "mean"),
        Std_Deletion_AUC=("deletion_auc", "std"),
        Mean_Insertion_AUC=("insertion_auc", "mean"),
        Std_Insertion_AUC=("insertion_auc", "std"),
        Mean_Faithfulness_Delta=("faithfulness_delta", "mean"),
        Mean_Pearson_Corr=("pearson_corr", "mean"),
        Sample_Count=("deletion_auc", "count")
    ).reset_index()
    
    overall_csv = output_dir / "stage2_faithfulness_overall_matrix.csv"
    overall_summary.to_csv(overall_csv, index=False)
    
    # Per-Class Faithfulness Matrix
    per_class_summary = df.groupby(["generation", "true_class"]).agg(
        Mean_Deletion_AUC=("deletion_auc", "mean"),
        Mean_Insertion_AUC=("insertion_auc", "mean"),
        Mean_Delta=("faithfulness_delta", "mean")
    ).reset_index()
    
    class_csv = output_dir / "stage2_faithfulness_per_class_matrix.csv"
    per_class_summary.to_csv(class_csv, index=False)

    # ── Markdown Report ──
    md_path = output_dir / "stage2_faithfulness_matrix.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stage 2 YOLO Classifier: Explainable AI (XAI) Faithfulness Matrix\n")
        f.write("## Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection\n\n")
        f.write("This document summarizes the **Perturbation Faithfulness Evaluation** comparing visual saliency maps across lightweight YOLO classification generations.\n\n")
        f.write("> **Interpretation Guide**:\n")
        f.write("> - **Deletion AUC (↓)**: Lower indicates the model relies strictly on the highlighted pixels (rapid drop in confidence when salient pixels are masked).\n")
        f.write("> - **Insertion AUC (↑)**: Higher indicates salient pixels contain sufficient discriminatory visual evidence.\n")
        f.write("> - **Faithfulness Delta $\\Delta\\text{AUC}$ (↑)**: $\\Delta\\text{AUC} = \\text{Insertion AUC} - \\text{Deletion AUC}$. Higher indicates superior overall explanation faithfulness.\n\n")
        
        f.write("### 1. Overall Architectural Faithfulness Benchmark\n\n")
        f.write("| Architecture / Generation | Deletion AUC (↓) | Insertion AUC (↑) | Faithfulness ΔAUC (↑) | Pearson Correlation ($r$) | Tested Samples |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for _, row in overall_summary.iterrows():
            f.write(f"| **{row['generation']}** | {row['Mean_Deletion_AUC']:.4f} ± {row['Std_Deletion_AUC']:.3f} | {row['Mean_Insertion_AUC']:.4f} ± {row['Std_Insertion_AUC']:.3f} | **{row['Mean_Faithfulness_Delta']:.4f}** | {row['Mean_Pearson_Corr']:.4f} | {row['Sample_Count']} |\n")
            
        f.write("\n### 2. Per-Class Faithfulness Matrix\n\n")
        f.write("| Architecture | Disease Category | Deletion AUC (↓) | Insertion AUC (↑) | Faithfulness ΔAUC (↑) |\n")
        f.write("|:---|:---|:---:|:---:|:---:|\n")
        for _, row in per_class_summary.iterrows():
            f.write(f"| {row['generation']} | {row['true_class']} | {row['Mean_Deletion_AUC']:.4f} | {row['Mean_Insertion_AUC']:.4f} | **{row['Mean_Delta']:.4f}** |\n")

    # ── LaTeX Report (Chapter 4 Ready) ──
    tex_path = output_dir / "stage2_faithfulness_matrix.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% ==========================================================================\n")
        f.write("% Stage 2 YOLO Classifier Faithfulness Matrix (Perturbation Analysis)\n")
        f.write("% Chapter 4: Explainability and Architectural Validation\n")
        f.write("% ==========================================================================\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Perturbation Faithfulness Evaluation across YOLO Classifier Generations on Held-Out Test Data.}\n")
        f.write("\\label{tab:stage2_faithfulness_matrix}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model Generation} & \\textbf{Deletion AUC} ($\\downarrow$) & \\textbf{Insertion AUC} ($\\uparrow$) & \\textbf{Faithfulness $\\Delta$AUC} ($\\uparrow$) & \\textbf{Pearson $r$} \\\\\n")
        f.write("\\midrule\n")
        for _, row in overall_summary.iterrows():
            f.write(f"{row['generation']} & {row['Mean_Deletion_AUC']:.4f} & {row['Mean_Insertion_AUC']:.4f} & \\textbf{{{row['Mean_Faithfulness_Delta']:.4f}}} & {row['Mean_Pearson_Corr']:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    log.info(f"[Reports] Generated Markdown matrix: {md_path.resolve()}")
    log.info(f"[Reports] Generated LaTeX matrix: {tex_path.resolve()}")

# ── 12. Visualization Generator ─────────────────────────────────────────────
def generate_faithfulness_plots(curves_data, sample_visuals, output_dir):
    """Generates publication-quality Deletion/Insertion curves and Grad-CAM visual panels."""
    steps_pct = np.linspace(0, 100, 21)
    
    # ── Figure 1: Cross-Generation Perturbation Curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    colors = {"YOLOv8n-cls (Baseline)": "#1f77b4", "YOLO11n-cls (Standard)": "#ff7f0e", "YOLO11n-cls (Field-Robust)": "#2ca02c", "YOLO26n-cls (Next-Gen)": "#d62728"}
    
    for gen_name, data in curves_data.items():
        if not data["deletion"]:
            continue
        del_mat = np.array(data["deletion"])
        mean_del = np.mean(del_mat, axis=0)
        std_del = np.std(del_mat, axis=0) / np.sqrt(len(del_mat))
        c = colors.get(gen_name, "#333333")
        axes[0].plot(steps_pct, mean_del, label=f"{gen_name}", color=c, lw=2.2)
        axes[0].fill_between(steps_pct, mean_del - std_del, mean_del + std_del, color=c, alpha=0.15)
        
    axes[0].set_title("(a) Deletion Curve (Saliency Perturbation)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("% Most Salient Pixels Masked (Black Baseline)", fontsize=11)
    axes[0].set_ylabel("Mean Target Class Confidence", fontsize=11)
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    for gen_name, data in curves_data.items():
        if not data["insertion"]:
            continue
        ins_mat = np.array(data["insertion"])
        mean_ins = np.mean(ins_mat, axis=0)
        std_ins = np.std(ins_mat, axis=0) / np.sqrt(len(ins_mat))
        c = colors.get(gen_name, "#333333")
        axes[1].plot(steps_pct, mean_ins, label=f"{gen_name}", color=c, lw=2.2)
        axes[1].fill_between(steps_pct, mean_ins - std_ins, mean_ins + std_ins, color=c, alpha=0.15)

    axes[1].set_title("(b) Insertion Curve (Evidence Restoration)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("% Most Salient Pixels Restored from Blank Canvas", fontsize=11)
    axes[1].set_ylabel("Mean Target Class Confidence", fontsize=11)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    curve_png = output_dir / "faithfulness_curves_comparison.png"
    curve_pdf = output_dir / "faithfulness_curves_comparison.pdf"
    plt.savefig(curve_png, dpi=300)
    plt.savefig(curve_pdf)
    plt.close()
    log.info(f"[Visuals] Saved perturbation curves: {curve_png.resolve()}")

    # ── Figure 2: Visual Explanation Triads ──
    if sample_visuals:
        num_vis = min(6, len(sample_visuals))
        fig, axes = plt.subplots(num_vis, 3, figsize=(10, 3 * num_vis), dpi=300)
        if num_vis == 1:
            axes = np.expand_dims(axes, 0)
            
        for i in range(num_vis):
            vis = sample_visuals[i]
            img_rgb = cv2.cvtColor(vis["img_bgr"], cv2.COLOR_BGR2RGB)
            cam = vis["cam_map"]
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.float32(heatmap) / 255 * 0.4 + np.float32(img_rgb) / 255 * 0.6
            overlay = np.clip(overlay, 0, 1)

            sorted_idx = np.argsort(cam.flatten())[::-1]
            p_img = (np.float32(img_rgb) / 255.0).reshape(-1, 3)
            p_img[sorted_idx[:int(0.5 * len(sorted_idx))]] = 0
            p_img = p_img.reshape(224, 224, 3)

            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f"Input: {vis['class_name']}\n[{vis['gen_name']}]", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"Grad-CAM++\nΔAUC: {vis['ins_auc'] - vis['del_auc']:.3f}", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(p_img)
            axes[i, 2].set_title("50% Saliency Masked\n(Deletion State)", fontsize=9)
            axes[i, 2].axis("off")

        plt.tight_layout()
        vis_png = output_dir / "faithfulness_visual_triads.png"
        vis_pdf = output_dir / "faithfulness_visual_triads.pdf"
        plt.savefig(vis_png, dpi=300)
        plt.savefig(vis_pdf)
        plt.close()
        log.info(f"[Visuals] Saved saliency triads: {vis_png.resolve()}")

# ── 13. Main Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "stage2_faithfulness_results"
    
    log.info("=" * 75)
    log.info("Starting Stage 2 XAI Faithfulness Evaluation Benchmark")
    log.info(f"Target Output Directory: {output_dir.resolve()}")
    log.info("=" * 75)
    
    dataset_path = acquire_dataset()
    models = discover_model_generations(base_dir=script_dir)
    
    run_faithfulness_benchmark(
        dataset_dir=dataset_path,
        models_registry=models,
        output_dir=output_dir,
        samples_per_class=30
    )
    
    log.info("=" * 75)
    log.info("XAI Faithfulness Benchmark completed successfully! ✅")
    log.info(f"Results Archive: {output_dir / 'stage2_faithfulness_results.zip'}")
    log.info("=" * 75)
