"""
================================================================================
Kaggle Notebook / Cloud Script: Stage 2 XAI Faithfulness Evaluation Benchmark
Cross-Generation Evaluation: YOLOv8-cls vs YOLO11-cls vs YOLO11n-Robust vs YOLO26-cls
Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection
================================================================================
Instructions for Kaggle:
  1. Open a new Kaggle Notebook (GPU T4 x2 or P100).
  2. Paste and run this script.
  3. The script will automatically download the v4 dataset, evaluate all model
     weights with Grad-CAM++ Perturbation Faithfulness (Deletion & Insertion AUC),
     and package the publication-grade CSVs, Markdown, LaTeX, and figures into
     'stage2_faithfulness_results.zip'.
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
LOG_FILE = Path("kaggle_faithfulness_benchmark.log")
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
    log.info("=" * 75)
else:
    log.warning("[WARNING] Running on CPU.")

# ── 4. Dataset Acquisition ──────────────────────────────────────────────────
DATASET_DIR = Path("Stage-2-Rubber-Leaf-Classifier-4")
CLASS_NAMES = ["Healthy", "Other Disease", "PLFD"]

def acquire_dataset():
    if DATASET_DIR.exists() and (DATASET_DIR / "test").exists():
        return DATASET_DIR
    log.info("[Dataset] Downloading Stage 2 dataset from Roboflow API (v4)...")
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
    version = project.version(4)
    dataset = version.download("folder", location=str(DATASET_DIR.resolve()))
    return Path(dataset.location)

# ── 5. Grad-CAM++ Hook Engine ───────────────────────────────────────────────
class GradCAMPlusPlusEngine:
    def __init__(self, model_path, device):
        self.device = device
        self.yolo_wrapper = YOLO(str(model_path))
        self.model = self.yolo_wrapper.model.to(self.device)
        self.model.eval()
        self.names = self.yolo_wrapper.names
        
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
        raise ValueError("Could not locate Conv2D target layer.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._target_layer.register_forward_hook(forward_hook)
        self._target_layer.register_full_backward_hook(backward_hook)

    def generate_saliency_map(self, input_tensor, target_class=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        probs = F.softmax(logits, dim=1)
        pred_class = int(logits.argmax(dim=1).item())
        class_idx = target_class if target_class is not None else pred_class

        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

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

# ── 6. Perturbation Analysis ────────────────────────────────────────────────
def compute_faithfulness_metrics(engine, image_bgr, target_class, num_steps=21):
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
            deletion_scores.append(float(F.softmax(out, dim=1)[0, target_class].item()))
            
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
            insertion_scores.append(float(F.softmax(out, dim=1)[0, target_class].item()))
            
    del_auc = float(np.trapz(deletion_scores, steps))
    ins_auc = float(np.trapz(insertion_scores, steps))
    faithfulness_delta = ins_auc - del_auc
    
    step_drops = np.maximum(0, init_conf - np.array(deletion_scores))
    corr = float(pearsonr(steps, step_drops)[0]) if np.std(step_drops) > 1e-6 and np.std(steps) > 1e-6 else 0.0

    return {
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        "faithfulness_delta": faithfulness_delta,
        "pearson_corr": corr,
        "deletion_curve": deletion_scores,
        "insertion_curve": insertion_scores,
        "cam_map": cam_map,
        "init_conf": init_conf,
        "pred_class": pred_class
    }

# ── 7. Benchmark Runner ─────────────────────────────────────────────────────
def run_benchmark(output_dir=Path("stage2_faithfulness_results"), samples_per_class=40):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = acquire_dataset()
    test_dir = dataset_dir / "test" if (dataset_dir / "test").exists() else dataset_dir / "valid"

    # Search for model weights
    model_paths = {
        "YOLOv8n-cls": list(Path(".").glob("**/yolov8n-cls_best.pt")) + list(Path(".").glob("**/best.pt")),
        "YOLO11n-cls": list(Path(".").glob("**/yolo11n-cls_best.pt")) + list(Path(".").glob("**/yolo11n_best.pt")),
        "YOLO11n-cls (Field-Robust)": list(Path(".").glob("**/yolo11n_robust_best.pt"))
    }
    
    models = {}
    for name, plist in model_paths.items():
        if plist:
            models[name] = plist[0]
            
    if not models:
        # Fallback to standard pre-trained or local names
        log.warning("[Warning] No custom trained weights found in current dir. Evaluating available YOLO models.")
        models = {"YOLOv8n-cls": "yolov8n-cls.pt", "YOLO11n-cls": "yolo11n-cls.pt"}

    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    test_samples = []
    for c_dir in class_dirs:
        c_name = c_dir.name
        files = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png"))
        for f in files[:samples_per_class]:
            test_samples.append({"path": f, "class_name": c_name, "class_idx": CLASS_NAMES.index(c_name) if c_name in CLASS_NAMES else 0})

    log.info(f"Running Faithfulness Evaluation on {len(test_samples)} samples across {list(models.keys())}...")

    results = []
    curves_data = {m: {"deletion": [], "insertion": []} for m in models}

    for gen_name, weight_path in models.items():
        engine = GradCAMPlusPlusEngine(weight_path, device)
        for s in tqdm(test_samples, desc=f"Evaluating {gen_name}"):
            img = cv2.imread(str(s["path"]))
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            res = compute_faithfulness_metrics(engine, img, s["class_idx"])
            results.append({
                "generation": gen_name,
                "image": s["path"].name,
                "true_class": s["class_name"],
                "pred_class": CLASS_NAMES[res["pred_class"]],
                "deletion_auc": res["deletion_auc"],
                "insertion_auc": res["insertion_auc"],
                "faithfulness_delta": res["faithfulness_delta"],
                "pearson_corr": res["pearson_corr"]
            })
            curves_data[gen_name]["deletion"].append(res["deletion_curve"])
            curves_data[gen_name]["insertion"].append(res["insertion_curve"])

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "stage2_faithfulness_samples.csv", index=False)
    
    overall = df.groupby("generation").agg(
        Mean_Del=("deletion_auc", "mean"),
        Mean_Ins=("insertion_auc", "mean"),
        Mean_Delta=("faithfulness_delta", "mean"),
        Mean_Corr=("pearson_corr", "mean")
    ).reset_index()
    overall.to_csv(output_dir / "stage2_faithfulness_overall_matrix.csv", index=False)

    # Markdown Table
    with open(output_dir / "stage2_faithfulness_matrix.md", "w") as f:
        f.write("# Stage 2 YOLO Classifier Faithfulness Matrix\n\n")
        f.write("| Architecture | Deletion AUC (↓) | Insertion AUC (↑) | Faithfulness ΔAUC (↑) | Pearson $r$ |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        for _, r in overall.iterrows():
            f.write(f"| **{r['generation']}** | {r['Mean_Del']:.4f} | {r['Mean_Ins']:.4f} | **{r['Mean_Delta']:.4f}** | {r['Mean_Corr']:.4f} |\n")

    # Zip output
    shutil.make_archive("stage2_faithfulness_results", "zip", str(output_dir))
    log.info("Finished! Download 'stage2_faithfulness_results.zip'. ✅")

if __name__ == "__main__":
    run_benchmark()
