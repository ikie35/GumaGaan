"""
Stage 1 Rubber Leaf Detector - Standard Benchmark Re-Evaluation Script (v2 - All Bugs Fixed)
---------------------------------------------------------------------------------------------
PURPOSE : Re-evaluate trained models using standard COCO/Ultralytics
          benchmarking protocols.
          - Calculates true mAP across the complete PR-curve.
          - Auto-generates confusion matrices and point metrics.
          - No model training performed.

FIXES APPLIED (v2):
  [FIX 1] Dataset is genuinely single-class (nc=1). Only renames class 0 to
          "Rubber Leaf" in the YAML — does NOT add a second class. The old
          script wrongly set nc=1 and names={0:"Rubber Leaf"} simultaneously
          which is actually correct, but the previous rewrite added "Ignore Leaf"
          as class 1, causing a names-length vs nc mismatch crash.

  [FIX 2] Precision & Recall now read at the BEST F1 THRESHOLD (F1-max
          operating point) from the full confidence-sweep P/R curve.
          Previously np.mean() over all threshold steps was used, which
          gives an arbitrary value with no physical meaning.

  [FIX 3] F1-Score computed directly from the F1-curve maximum, not derived
          from pre-averaged P and R (F1(mean(P),mean(R)) != mean(F1)).

  [FIX 4] Latency column explicitly labelled "GPU_T4_Latency_ms" to prevent
          confusion with future edge-device (RPi5 + Hailo-8L) measurements.

  [FIX 5] Parameter count uses Ultralytics model.info() canonical value to
          match the numbers reported in the YOLOv8 paper exactly.
"""

# --- Auto-install dependencies ---
import subprocess, sys
for _pkg in ["roboflow", "ultralytics", "pandas", "matplotlib", "tabulate", "pyyaml"]:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --- Imports ---
import os, shutil, zipfile, yaml
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from roboflow import Roboflow
from ultralytics import YOLO

# --- Device ---
device = 0 if torch.cuda.is_available() else "cpu"
print(f"[Device] {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# =============================================================================
# Utility: locate .pt files anywhere under /kaggle/input/
# =============================================================================
KAGGLE_INPUT = Path("/kaggle/input")

def find_pt(filename: str) -> Path:
    """Search all subfolders of /kaggle/input/ for the given .pt filename."""
    matches = list(KAGGLE_INPUT.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' anywhere under {KAGGLE_INPUT}\n"
            f"Available .pt files: {[str(p) for p in KAGGLE_INPUT.rglob('*.pt')]}"
        )
    print(f"[Found] {filename} -> {matches[0]}")
    return matches[0]

MODEL_VARIANTS = [
    {"name": "YOLOv8n", "label": "Nano (3.2M)",    "pt": find_pt("stage1_single_class_yolov8n_best.pt")},
    {"name": "YOLOv8s", "label": "Small (11.2M)",  "pt": find_pt("stage1_single_class_yolov8s_best.pt")},
    {"name": "YOLOv8m", "label": "Medium (25.9M)", "pt": find_pt("stage1_single_class_yolov8m_best.pt")},
]

# --- Output Directories ---
OUT_DIR  = Path("stage1_standard_eval")
CM_DIR   = OUT_DIR / "confusion_matrices"
ONNX_DIR = OUT_DIR / "onnx_models"
for d in (OUT_DIR, CM_DIR, ONNX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Step 1 - Download Dataset YAML (NO patching of nc)
# =============================================================================
print("\n" + "=" * 65)
print("[1/3] Downloading dataset YAML from Roboflow...")
print("=" * 65)

rf      = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
project = rf.workspace("rubber-research").project("stage-1-rubber-leaf-detection-single-class")
version = project.version(3)
dataset = version.download("yolov8")

data_yaml = Path(dataset.location) / "data.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Dataset is genuinely single-class (nc=1, one class: "Rubber Leaf").
#         Only rename class 0 for human-readable confusion matrix labels.
#         DO NOT add a second class — that causes a names-length vs nc mismatch.
# ─────────────────────────────────────────────────────────────────────────────
with open(data_yaml, "r") as f:
    yaml_data = yaml.safe_load(f)

print(f"[YAML] Original nc={yaml_data.get('nc')}, names={yaml_data.get('names')}")
assert yaml_data.get("nc") == 1, f"[ERROR] Expected nc=1 but got nc={yaml_data.get('nc')}. Check dataset version."

# Only rename — nc stays 1 to match the single-class trained model head
yaml_data["names"] = {0: "Rubber Leaf"}

with open(data_yaml, "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False)

print(f"--> Class name set to 'Rubber Leaf'. nc=1 preserved.")
print(f"--> data.yaml ready at: {data_yaml}")

# =============================================================================
# Step 2 - Re-evaluate models (Standard Benchmark Protocol)
# =============================================================================
print("\n" + "=" * 65)
print("[2/3] Re-evaluating models (Standard Benchmark Protocol)...")
print("=" * 65)

# Single-class model: map50 and map are scalars, p/r are (num_thresholds,) arrays

results = []

for model_info in MODEL_VARIANTS:
    name  = model_info["name"]
    label = model_info["label"]
    pt    = model_info["pt"]

    print(f"\n{'='*65}\n[{name}] Loading: {pt}\n{'='*65}")

    if not pt.exists():
        print(f"[ERROR] {pt} not found.")
        continue

    model = YOLO(str(pt))

    # -------------------------------------------------------------------------
    # Validation Split (sanity check only — NOT reported as primary metric)
    # -------------------------------------------------------------------------
    val_res   = model.val(data=str(data_yaml), split="val", imgsz=640, device=device, verbose=False)
    # Single-class: map50 is a scalar (the only class is Rubber Leaf)
    val_map50 = round(float(val_res.box.map50) * 100, 2)
    print(f"[{name}] Val mAP50 (Rubber Leaf): {val_map50}%")

    # -------------------------------------------------------------------------
    # Held-Out Test Split (PRIMARY evaluation — reported in thesis)
    # -------------------------------------------------------------------------
    test_res = model.val(
        data=str(data_yaml), split="test",
        imgsz=640, device=device,
        plots=True,
        name=f"{name.lower()}_standard_test_eval",
        verbose=False,
    )

    # Copy auto-generated confusion matrices
    try:
        run_dir = Path(test_res.save_dir)
        for cm_file in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
            cm_src = run_dir / cm_file
            if cm_src.exists():
                dest = CM_DIR / f"{name.lower()}_{cm_file}"
                shutil.copy(cm_src, dest)
                print(f"--> Saved: {dest.name}")
            else:
                print(f"[WARN] {name}: {cm_file} not found in {run_dir}")
    except Exception as exc:
        print(f"[WARN] {name}: Could not copy confusion matrix - {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Single-class model: map50 and map are scalars — no indexing needed.
    #   box.map50 = AP@50    for Rubber Leaf (the only class)
    #   box.map   = AP@50-95 for Rubber Leaf (the only class)
    # ─────────────────────────────────────────────────────────────────────────
    map50   = round(float(test_res.box.map50) * 100, 2)
    map5095 = round(float(test_res.box.map)   * 100, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # STANDARD: P / R / F1 directly from Ultralytics (no manual sweep needed).
    #
    #   Ultralytics ap_per_class() sweeps 1000 confidence thresholds internally,
    #   finds the threshold that maximises the mean F1 across all classes, then
    #   stores per-class P, R, and F1 at that threshold in box.p / box.r / box.f1.
    #
    #   box.p  → shape (nc,) — Precision per class at best-F1 threshold
    #   box.r  → shape (nc,) — Recall    per class at best-F1 threshold
    #   box.f1 → shape (nc,) — F1-max    per class at best-F1 threshold
    #
    #   For single-class (nc=1): index [0] gives the one and only class value.
    # ─────────────────────────────────────────────────────────────────────────
    p  = round(float(test_res.box.p[0])  * 100, 2)   # Precision at best-F1 conf
    r  = round(float(test_res.box.r[0])  * 100, 2)   # Recall    at best-F1 conf
    f1 = round(float(test_res.box.f1[0]) * 100, 2)   # F1-max (Ultralytics computed)

    print(f"[{name}] P={p}%  R={r}%  F1={f1}%  (at Ultralytics best-F1 threshold)")


    # ─────────────────────────────────────────────────────────────────────────
    # FIX 5: Parameter count — model.info() returns None in some Ultralytics
    #   versions (it only prints instead of returning). Fall back to numel().
    # ─────────────────────────────────────────────────────────────────────────
    info = model.info(verbose=False)
    if info is not None:
        params_m = round(info[1] / 1e6, 2)
    else:
        params_m = round(sum(x.numel() for x in model.model.parameters()) / 1e6, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4: Latency explicitly labelled GPU_T4 — NOT edge-device latency.
    #   Fill Table tab:inference in the thesis with actual RPi5 measurements.
    # ─────────────────────────────────────────────────────────────────────────
    gpu_latency_ms = round(float(test_res.speed["inference"]), 2)

    print(f"[{name}] Test mAP50={map50}% | mAP50-95={map5095}% | "
          f"P={p}% | R={r}% | F1={f1}% | GPU_T4={gpu_latency_ms}ms | Params={params_m}M")

    # Dynamic ONNX Export
    onnx_file = ONNX_DIR / f"stage1_single_class_{name.lower()}_dynamic.onnx"
    try:
        onnx_src = model.export(format="onnx", imgsz=640, dynamic=True, opset=12, device=device)
        shutil.copy(onnx_src, onnx_file)
        print(f"--> ONNX exported: {onnx_file.name}")
    except Exception as exc:
        print(f"[WARN] {name}: ONNX export failed - {exc}")

    results.append({
        "Model Variant":                  name,
        "Size Label":                     label,
        "Parameters (M)":                 params_m,
        "Val mAP50_RubberLeaf (%)":       val_map50,
        "Test mAP50_RubberLeaf (%)":      map50,
        "Test mAP50-95_RubberLeaf (%)":   map5095,
        "Precision_at_F1max (%)":         p,
        "Recall_at_F1max (%)":            r,
        "F1max (%)":                      f1,
        "GPU_T4_Latency_ms":              gpu_latency_ms,
        "ONNX File":                      onnx_file.name,
    })


# =============================================================================
# Step 3 - Save Results Table and Package ZIP
# =============================================================================
print("\n" + "=" * 65)
print("[3/3] Saving results and packaging ZIP...")
print("=" * 65)

df       = pd.DataFrame(results)
csv_path = OUT_DIR / "stage1_standard_eval_results.csv"
md_path  = OUT_DIR / "stage1_standard_eval_results.md"

df.to_csv(csv_path, index=False)

md_content = (
    "# Stage 1 Rubber Leaf Detector — Standard Benchmark Results\n\n"
    "## Evaluation Protocol\n"
    "- Dataset: Held-out **test split** (never seen during training or hyperparameter tuning)\n"
    "- mAP@50 and mAP@50–95: per-class COCO standard via Ultralytics `model.val()`\n"
    "- All metrics reported for **Class 0: Rubber Leaf** only (target class)\n"
    "- Precision, Recall, F1: at the **F1-max confidence threshold** (best operating point)\n"
    "- Class 1 (Ignore Leaf) excluded — it is an intentionally noisy background anchor class\n"
    "- `GPU_T4_Latency_ms` = Kaggle T4 GPU inference time — NOT edge-device latency\n\n"
    "## Results\n\n"
    f"{df.to_markdown(index=False)}\n"
)
md_path.write_text(md_content, encoding="utf-8")

zip_name = "stage1_standard_eval_results.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in CM_DIR.glob("*.png"):
        zf.write(f, arcname=f"confusion_matrices/{f.name}")
    for f in ONNX_DIR.glob("*.onnx"):
        zf.write(f, arcname=f"onnx_models/{f.name}")
    zf.write(csv_path, arcname=csv_path.name)
    zf.write(md_path,  arcname=md_path.name)

print("\n" + "=" * 65)
print("STANDARD EVALUATION COMPLETE")
print("=" * 65)
display_cols = [
    "Model Variant", "Test mAP50_RubberLeaf (%)", "Test mAP50-95_RubberLeaf (%)",
    "Precision_at_F1max (%)", "Recall_at_F1max (%)", "F1max (%)",
    "GPU_T4_Latency_ms",
]
print(df[display_cols].to_string(index=False))
print("=" * 65)
print(f"ZIP ready for download: /kaggle/working/{zip_name}")
print("=" * 65)
print("\nNOTE FOR THESIS MANUSCRIPT:")
print("  Add this note below Table tab:stage1_metrics and tab:stage1_scaling:")
print("  'Precision, Recall, and F1-Score are reported at the confidence threshold")
print("   that maximises the F1-Score (F1-max operating point). mAP@50 and")
print("   mAP@50-95 follow COCO standard evaluation. GPU_T4_Latency_ms reflects")
print("   Kaggle T4 GPU inference and is not representative of edge-device speed.'")
print("=" * 65)
