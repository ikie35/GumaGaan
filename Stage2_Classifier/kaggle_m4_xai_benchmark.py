"""
================================================================================
Kaggle Notebook Script: M4 Unified XAI Faithfulness Benchmark
Reference:
  "M4: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution
  Methods across Metrics, Modalities and Models" (NeurIPS 2023)
  Repository: https://github.com/holyseven/M4_XAI_Benchmark
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

REQUIRED_PACKAGES = ["roboflow", "ultralytics", "pandas", "matplotlib", "scikit-learn", "scipy", "tqdm", "opencv-python"]
for pkg in REQUIRED_PACKAGES:
    try:
        if pkg == "opencv-python":
            __import__("cv2")
        else:
            __import__(pkg.replace("-", "_"))
    except ImportError:
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[M4 Setup] Device: {device} | CUDA Available: {torch.cuda.is_available()}")

# Download Dataset
DATASET_DIR = Path("Stage-2-Rubber-Leaf-Classifier-4")
CLASS_NAMES = ["Healthy", "Other Disease", "PLFD"]

def acquire_dataset():
    if DATASET_DIR.exists() and (DATASET_DIR / "test").exists():
        return DATASET_DIR
    print("[M4 Dataset] Downloading dataset v4 from Roboflow...")
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
    version = project.version(4)
    dataset = version.download("folder", location=str(DATASET_DIR.resolve()))
    return Path(dataset.location)

# Attribution Engine
class M4AttributionEngine:
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
        raise ValueError("Target layer not found.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._target_layer.register_forward_hook(forward_hook)
        self._target_layer.register_full_backward_hook(backward_hook)

    def compute_attribution_map(self, input_tensor, target_class=None):
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
        "insertion_curve": insertion_scores
    }

def main():
    output_dir = Path("m4_xai_benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = acquire_dataset()
    test_dir = dataset_dir / "test" if (dataset_dir / "test").exists() else dataset_dir / "valid"

    # Find model weights in Kaggle working dir
    models = {}
    for pt in Path(".").rglob("*.pt"):
        if "best" in pt.name:
            models[pt.stem] = pt
            
    if not models:
        models = {"YOLOv8n-cls": "yolov8n-cls.pt", "YOLO11n-cls": "yolo11n-cls.pt"}

    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    test_samples = []
    for c_dir in class_dirs:
        c_name = c_dir.name
        files = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png"))
        for f in files[:35]:
            test_samples.append({"path": f, "class_name": c_name, "class_idx": CLASS_NAMES.index(c_name) if c_name in CLASS_NAMES else 0})

    print(f"Benchmarking {len(test_samples)} samples across {list(models.keys())}...")
    results = []

    for name, path in models.items():
        engine = M4AttributionEngine(path, device)
        for s in tqdm(test_samples, desc=f"Evaluating {name}"):
            img = cv2.imread(str(s["path"]))
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            res = evaluate_m4_metrics(engine, img, s["class_idx"])
            results.append({
                "model": name,
                "class": s["class_name"],
                "deletion_auc": res["deletion_auc"],
                "insertion_auc": res["insertion_auc"],
                "delta_auc": res["faithfulness_delta"],
                "monotonicity": res["monotonicity"],
                "pearson_r": res["f_corr"],
                "sparsity": res["sparsity"],
                "m4_score": res["composite_m4"]
            })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "m4_raw_samples.csv", index=False)
    summary = df.groupby("model").mean(numeric_only=True).reset_index()
    summary.to_csv(output_dir / "m4_faithfulness_matrix.csv", index=False)

    print("\n" + "=" * 70)
    print("M4 FAITHFULNESS BENCHMARK RESULTS (NeurIPS 2023 PROTOCOL)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70)

    shutil.make_archive("m4_xai_benchmark_results", "zip", str(output_dir))
    print("\nCompleted! Download 'm4_xai_benchmark_results.zip'.")

if __name__ == "__main__":
    main()
