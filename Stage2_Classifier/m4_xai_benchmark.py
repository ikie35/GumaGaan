"""
================================================================================
M4 Unified XAI Faithfulness Benchmark for Stage 2 YOLO Classifiers
Reference Paper:
  Xuhong Li, Mengnan Du, Jiamin Chen, Yekun Chai, Himabindu Lakkaraju, Haoyi Xiong.
  "M4: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution
  Methods across Metrics, Modalities and Models." NeurIPS 2023 (Datasets & Benchmarks).
  GitHub: https://github.com/holyseven/M4_XAI_Benchmark
================================================================================
Architectures Evaluated:
  - YOLOv8n-cls (Baseline)
  - YOLO11n-cls (Standard)
  - YOLO11n-cls (Field-Robust)
  - YOLO26n-cls (Next-Gen)

Evaluated M4 Faithfulness Metric Dimensions:
  1. Deletion AUC (DEL_AUC ↓): Measures score drop rate under feature masking.
  2. Insertion AUC (INS_AUC ↑): Measures score recovery rate under feature restoration.
  3. Faithfulness Delta (ΔAUC = INS_AUC - DEL_AUC ↑): Net attribution fidelity.
  4. Faithfulness Correlation (F-Corr ↑): Pearson correlation of attribution vs score drop.
  5. Monotonicity Metric (MONO ↑): Consistency of attribution ranking vs incremental drop.
  6. Saliency Sparsity / Gini Index (SPAR ↑): Energy concentration on pathological lesions.
  7. M4 Composite Score (M4_Score ↑): Standardized multi-dimensional faithfulness index.

Outputs:
  - 'm4_faithfulness_matrix.csv' (Raw per-sample & aggregated metrics)
  - 'm4_faithfulness_matrix.md' (Publication Markdown report)
  - 'm4_faithfulness_matrix.tex' (LaTeX table ready for thesis Chapter 4)
  - 'm4_benchmark_curves.png' / '.pdf' (Deletion & Insertion curve comparisons)
  - 'm4_faithfulness_radar.png' / '.pdf' (Multi-metric architectural radar chart)
  - 'm4_attribution_visuals.png' / '.pdf' (Attribution heatmaps across classes)
  - Packaged ZIP archive: 'm4_xai_benchmark_results.zip'
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

# ── 1. Dependency Management ────────────────────────────────────────────────
REQUIRED_PACKAGES = ["roboflow", "ultralytics", "pandas", "matplotlib", "scikit-learn", "scipy", "tqdm", "opencv-python"]

def ensure_packages(packages):
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
LOG_FILE = Path("m4_xai_benchmark.log")
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

# ── 3. Device Configuration ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    log.info("=" * 75)
    log.info(f"[M4 Benchmark] GPU Active: {torch.cuda.get_device_name(0)}")
    log.info(f"[M4 Benchmark] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    log.info("=" * 75)
else:
    log.warning("=" * 75)
    log.warning("[M4 Benchmark] No CUDA GPU found. Executing on CPU.")
    log.warning("=" * 75)

# ── 4. Dataset Acquisition ──────────────────────────────────────────────────
DATASET_DIR = Path("Stage-2-Rubber-Leaf-Classifier-4")
CLASS_NAMES = ["Healthy", "Other Disease", "PLFD"]

def acquire_dataset():
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

    log.info("[Dataset] Fetching Stage 2 dataset from Roboflow API (v4)...")
    try:
        rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
        project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
        version = project.version(4)
        dataset = version.download("folder", location=str(DATASET_DIR.resolve()))
        return Path(dataset.location)
    except Exception as e:
        log.error(f"[Dataset Error] Failed: {e}")
        raise

# ── 5. Model Discovery ──────────────────────────────────────────────────────
def discover_model_generations(base_dir=Path(".")):
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
                log.info(f"[Model Registry] Found '{gen_name}' at: {p}")
                break

    if not models_registry:
        log.warning("[Model Registry] No trained weights found locally, falling back to Ultralytics base weights.")
        models_registry = {"YOLOv8n-cls": "yolov8n-cls.pt", "YOLO11n-cls": "yolo11n-cls.pt"}
        
    return models_registry

# ── 6. M4 Attribution Generator (Grad-CAM++ Implementation) ─────────────────
class M4AttributionEngine:
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
        """Generates normalized 2D feature attribution saliency map."""
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
            else:
                cam = np.zeros_like(cam)

            conf = float(probs[0, class_idx].item())
            return cam, pred_class, conf

# ── 7. M4 Faithfulness Metric Evaluator (NeurIPS 2023 Protocol) ──────────────
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
    """
    Evaluates all M4 Benchmark dimensions for a single image specimen:
      - Deletion AUC (DEL_AUC)
      - Insertion AUC (INS_AUC)
      - Faithfulness Delta (ΔAUC)
      - Faithfulness Correlation (F-Corr)
      - Monotonicity (MONO)
      - Sparsity / Gini Concentration (SPAR)
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0
    input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(engine.device)
    
    attr_map, pred_class, init_conf = engine.compute_attribution_map(input_tensor, target_class=target_class)
    
    h, w = attr_map.shape
    num_pixels = h * w
    flat_attr = attr_map.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1] # Order from highest to lowest attribution
    
    steps = np.linspace(0, 1, num_steps)
    deletion_scores = []
    insertion_scores = []
    
    with torch.no_grad():
        # 1. M4 Deletion Protocol
        for step in steps:
            k = int(step * num_pixels)
            p_img = img_float.copy().reshape(-1, 3)
            if k > 0:
                p_img[sorted_indices[:k]] = 0.0 # Zero mask
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

        # 2. M4 Insertion Protocol
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

    # AUC Calculations
    del_auc = compute_auc(deletion_scores, steps)
    ins_auc = compute_auc(insertion_scores, steps)
    faithfulness_delta = ins_auc - del_auc

    # 3. M4 Faithfulness Correlation (Pearson r between attribution step and confidence drop)
    step_drops = np.maximum(0, init_conf - np.array(deletion_scores))
    if np.std(step_drops) > 1e-6 and np.std(steps) > 1e-6:
        f_corr = float(pearsonr(steps, step_drops)[0])
    else:
        f_corr = 0.0

    # 4. M4 Monotonicity Metric (fraction of non-increasing consecutive steps in deletion)
    diffs = np.diff(deletion_scores)
    mono_score = float(np.mean(diffs <= 1e-4)) # Ideal deletion is strictly non-increasing

    # 5. M4 Saliency Sparsity (Gini index of attribution distribution)
    sorted_vals = np.sort(flat_attr)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    gini_sparsity = float((np.sum((2 * index - n - 1) * sorted_vals)) / (n * np.sum(sorted_vals) + 1e-8))

    # 6. M4 Composite Faithfulness Score (Normalized index)
    # Combines normalized (1 - DEL_AUC), INS_AUC, MONO, and F_Corr into single score [0, 1]
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

# ── 8. M4 Benchmark Execution Suite ─────────────────────────────────────────
def run_m4_benchmark(dataset_dir, models_registry, output_dir, samples_per_class=30):
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
            
    log.info(f"[M4 Benchmark] Evaluating {len(sampled_test_set)} test images across {len(models_registry)} YOLO architectures...")

    results_data = []
    curves_data = {gen: {"deletion": [], "insertion": []} for gen in models_registry}
    sample_visuals = []

    for gen_name, weight_path in models_registry.items():
        log.info(f"\n[Evaluating Architecture via M4 Suite] >>> {gen_name} <<<")
        engine = M4AttributionEngine(weight_path, device)
        
        pbar = tqdm(sampled_test_set, desc=f"M4 XAI Suite: {gen_name}")
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
                "generation": gen_name,
                "image_name": sample["path"].name,
                "true_class": sample["class_name"],
                "pred_class": pred_label,
                "initial_confidence": res["init_conf"],
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
                    "gen_name": gen_name,
                    "img_bgr": img_bgr,
                    "attr_map": res["attr_map"],
                    "class_name": sample["class_name"],
                    "del_auc": res["deletion_auc"],
                    "ins_auc": res["insertion_auc"],
                    "composite_m4": res["composite_m4"]
                })

    df_results = pd.DataFrame(results_data)
    
    # ── Generate Reports & Visuals ──
    compile_m4_reports(df_results, output_dir)
    generate_m4_visualizations(curves_data, df_results, sample_visuals, output_dir)
    
    # ── Zip Archive ──
    zip_path = output_dir / "m4_xai_benchmark_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.glob("*"):
            if file.name != zip_path.name:
                zipf.write(file, arcname=file.name)
    log.info(f"\n[Packaging] Packaged M4 XAI Benchmark archive: {zip_path.resolve()}")

# ── 9. M4 Reports Generation (Markdown, LaTeX, CSV) ─────────────────────────
def compile_m4_reports(df, output_dir):
    csv_raw = output_dir / "m4_faithfulness_raw_samples.csv"
    df.to_csv(csv_raw, index=False)
    
    # Overall Architecture Summary
    overall_summary = df.groupby("generation").agg(
        Mean_DEL_AUC=("deletion_auc", "mean"),
        Std_DEL_AUC=("deletion_auc", "std"),
        Mean_INS_AUC=("insertion_auc", "mean"),
        Std_INS_AUC=("insertion_auc", "std"),
        Mean_Delta_AUC=("faithfulness_delta", "mean"),
        Mean_F_Corr=("f_corr", "mean"),
        Mean_Monotonicity=("monotonicity", "mean"),
        Mean_Sparsity=("sparsity", "mean"),
        Mean_Composite_M4=("composite_m4", "mean"),
        Samples=("deletion_auc", "count")
    ).reset_index()
    
    overall_csv = output_dir / "m4_faithfulness_matrix.csv"
    overall_summary.to_csv(overall_csv, index=False)
    
    # Per-Class Summary
    per_class_summary = df.groupby(["generation", "true_class"]).agg(
        DEL_AUC=("deletion_auc", "mean"),
        INS_AUC=("insertion_auc", "mean"),
        Delta_AUC=("faithfulness_delta", "mean"),
        F_Corr=("f_corr", "mean"),
        Composite_M4=("composite_m4", "mean")
    ).reset_index()

    # ── Markdown Report ──
    md_path = output_dir / "m4_faithfulness_matrix.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# M4 XAI Faithfulness Benchmark: Stage 2 YOLO Classifiers\n")
        f.write("## NeurIPS 2023 M4 Benchmark Protocol Implementation\n\n")
        f.write("Evaluation based on the **M4 Unified XAI Benchmark** (*Li et al., NeurIPS 2023*, https://github.com/holyseven/M4_XAI_Benchmark).\n\n")
        
        f.write("### 1. Overall Cross-Generation M4 Faithfulness Matrix\n\n")
        f.write("| Architecture / Generation | Deletion AUC (↓) | Insertion AUC (↑) | Faithfulness ΔAUC (↑) | Monotonicity (↑) | Pearson $r$ (↑) | Sparsity (↑) | **M4 Composite Score** (↑) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for _, row in overall_summary.iterrows():
            f.write(f"| **{row['generation']}** | {row['Mean_DEL_AUC']:.4f} | {row['Mean_INS_AUC']:.4f} | {row['Mean_Delta_AUC']:.4f} | {row['Mean_Monotonicity']:.4f} | {row['Mean_F_Corr']:.4f} | {row['Mean_Sparsity']:.4f} | **{row['Mean_Composite_M4']:.4f}** |\n")

        f.write("\n### 2. Per-Class M4 Breakdown Matrix\n\n")
        f.write("| Architecture | Disease Class | Deletion AUC (↓) | Insertion AUC (↑) | ΔAUC (↑) | Pearson $r$ (↑) | **M4 Score** (↑) |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for _, row in per_class_summary.iterrows():
            f.write(f"| {row['generation']} | {row['true_class']} | {row['DEL_AUC']:.4f} | {row['INS_AUC']:.4f} | {row['Delta_AUC']:.4f} | {row['F_Corr']:.4f} | **{row['Composite_M4']:.4f}** |\n")

    # ── LaTeX Table (Chapter 4 Ready) ──
    tex_path = output_dir / "m4_faithfulness_matrix.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% ==========================================================================\n")
        f.write("% M4 Unified XAI Benchmark Faithfulness Matrix (NeurIPS 2023 Protocol)\n")
        f.write("% Chapter 4: Explainability & Model Faithfulness Validation\n")
        f.write("% ==========================================================================\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Unified M4 XAI Faithfulness Benchmark Across YOLO Classification Generations.}\n")
        f.write("\\label{tab:m4_xai_faithfulness_benchmark}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model Generation} & \\textbf{DEL AUC} ($\\downarrow$) & \\textbf{INS AUC} ($\\uparrow$) & \\textbf{$\\Delta$AUC} ($\\uparrow$) & \\textbf{Mono.} ($\\uparrow$) & \\textbf{Pearson $r$} ($\\uparrow$) & \\textbf{M4 Score} ($\\uparrow$) \\\\\n")
        f.write("\\midrule\n")
        for _, row in overall_summary.iterrows():
            f.write(f"{row['generation']} & {row['Mean_DEL_AUC']:.4f} & {row['Mean_INS_AUC']:.4f} & {row['Mean_Delta_AUC']:.4f} & {row['Mean_Monotonicity']:.4f} & {row['Mean_F_Corr']:.4f} & \\textbf{{{row['Mean_Composite_M4']:.4f}}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    log.info(f"[M4 Reports] Generated Markdown: {md_path.resolve()}")
    log.info(f"[M4 Reports] Generated LaTeX: {tex_path.resolve()}")

# ── 10. M4 Visualizations (Curves, Radar Chart, Visual Triads) ───────────────
def generate_m4_visualizations(curves_data, df_results, sample_visuals, output_dir):
    steps_pct = np.linspace(0, 100, 21)
    colors = {"YOLOv8n-cls (Baseline)": "#1f77b4", "YOLO11n-cls (Standard)": "#ff7f0e", "YOLO11n-cls (Field-Robust)": "#2ca02c", "YOLO26n-cls (Next-Gen)": "#d62728"}
    
    # ── 1. Deletion & Insertion Curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    for gen_name, data in curves_data.items():
        if not data["deletion"]:
            continue
        del_mat = np.array(data["deletion"])
        mean_del = np.mean(del_mat, axis=0)
        std_del = np.std(del_mat, axis=0) / np.sqrt(len(del_mat))
        c = colors.get(gen_name, "#333333")
        axes[0].plot(steps_pct, mean_del, label=f"{gen_name}", color=c, lw=2.2)
        axes[0].fill_between(steps_pct, mean_del - std_del, mean_del + std_del, color=c, alpha=0.15)
        
    axes[0].set_title("(a) M4 Deletion Curve (Attribution Perturbation)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("% Masked High-Attribution Pixels", fontsize=11)
    axes[0].set_ylabel("Target Class Confidence", fontsize=11)
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

    axes[1].set_title("(b) M4 Insertion Curve (Feature Evidence Recovery)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("% Restored High-Attribution Pixels", fontsize=11)
    axes[1].set_ylabel("Target Class Confidence", fontsize=11)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_dir / "m4_benchmark_curves.png", dpi=300)
    plt.savefig(output_dir / "m4_benchmark_curves.pdf")
    plt.close()

    # ── 2. Multi-Metric Radar Chart ──
    try:
        categories = ["1-DEL_AUC", "INS_AUC", "ΔAUC", "Monotonicity", "Pearson r", "Sparsity"]
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)
        summary = df_results.groupby("generation").mean(numeric_only=True)
        
        for gen_name, row in summary.iterrows():
            values = [
                1.0 - row["deletion_auc"],
                row["insertion_auc"],
                max(0, row["faithfulness_delta"]),
                row["monotonicity"],
                max(0, row["f_corr"]),
                row["sparsity"]
            ]
            values += values[:1]
            c = colors.get(gen_name, "#333333")
            ax.plot(angles, values, label=gen_name, color=c, lw=2)
            ax.fill(angles, values, color=c, alpha=0.10)
            
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title("M4 XAI Multi-Dimensional Faithfulness Profile", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
        plt.tight_layout()
        plt.savefig(output_dir / "m4_faithfulness_radar.png", dpi=300)
        plt.savefig(output_dir / "m4_faithfulness_radar.pdf")
        plt.close()
    except Exception as e:
        log.warning(f"[Visuals] Radar chart skipped: {e}")

    # ── 3. Visual Saliency Triads ──
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
            axes[i, 0].set_title(f"Input: {vis['class_name']}\n[{vis['gen_name']}]", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"M4 Attribution\nScore: {vis['composite_m4']:.3f}", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(p_img)
            axes[i, 2].set_title("50% Saliency Masked\n(M4 Deletion State)", fontsize=9)
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "m4_attribution_visuals.png", dpi=300)
        plt.savefig(output_dir / "m4_attribution_visuals.pdf")
        plt.close()

# ── 11. Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "m4_xai_benchmark_results"
    
    log.info("=" * 75)
    log.info("Starting M4 Unified XAI Faithfulness Benchmark Suite")
    log.info(f"Target Output Directory: {output_dir.resolve()}")
    log.info("=" * 75)
    
    dataset_path = acquire_dataset()
    models = discover_model_generations(base_dir=script_dir)
    
    run_m4_benchmark(
        dataset_dir=dataset_path,
        models_registry=models,
        output_dir=output_dir,
        samples_per_class=30
    )
    
    log.info("=" * 75)
    log.info("M4 XAI Benchmark completed successfully! ✅")
    log.info(f"Archive: {output_dir / 'm4_xai_benchmark_results.zip'}")
    log.info("=" * 75)
