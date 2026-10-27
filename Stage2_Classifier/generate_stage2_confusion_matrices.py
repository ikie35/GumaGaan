"""
Exact Ultralytics Native Style Confusion Matrix Generator for Stage 2
======================================================================
Produces the exact layout from Stage 1 (3000 x 2250 px, True on X-axis rotated 90 deg, Predicted on Y-axis).
Generates:
  - yolov8n_confusion_matrix.png & yolov8n_confusion_matrix_normalized.png
  - yolov8s_confusion_matrix.png & yolov8s_confusion_matrix_normalized.png
  - yolov8m_confusion_matrix.png & yolov8m_confusion_matrix_normalized.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'sans-serif']

out_dir = Path("stage2_publication_figures")
out_dir.mkdir(parents=True, exist_ok=True)

classes = ["Healthy", "Other Disease", "PLFD"]

# Confusion matrices where row = Predicted, col = True (matching Ultralytics convention)
# [Predicted, True]
confusion_matrices = {
    "yolov8n": np.array([
        [178,   3,   7],    # Predicted Healthy
        [  0, 107,   2],    # Predicted Other Disease
        [  3,  18, 255]     # Predicted PLFD
    ]),
    "yolov8s": np.array([
        [177,   2,   3],    # Predicted Healthy
        [  2, 114,  13],    # Predicted Other Disease
        [  2,  12, 248]     # Predicted PLFD
    ]),
    "yolov8m": np.array([
        [179,   1,   2],    # Predicted Healthy
        [  0, 112,   6],    # Predicted Other Disease
        [  2,  15, 256]     # Predicted PLFD
    ])
}

def plot_ultralytics_style_cm(matrix: np.ndarray, names: list, model_prefix: str, normalize: bool = False):
    """
    Exact 1:1 replica of Ultralytics official confusion matrix layout:
    - Dimensions: 12x9 inches @ 250 DPI -> 3000 x 2250 px
    - Y-Axis: 'Predicted'
    - X-Axis: 'True' with vertical 90-degree class names
    - Title: 'Confusion Matrix' or 'Confusion Matrix Normalized'
    """
    mat = np.array(matrix, dtype=float)
    
    if normalize:
        # Normalize along True (columns)
        norm_mat = mat / (mat.sum(axis=0, keepdims=True) + 1e-9)
        display_mat = norm_mat
        title = "Confusion Matrix Normalized"
        filename = f"{model_prefix}_confusion_matrix_normalized.png"
        vmax = 1.0
    else:
        display_mat = mat
        title = "Confusion Matrix"
        filename = f"{model_prefix}_confusion_matrix.png"
        vmax = mat.max()

    fig, ax = plt.subplots(figsize=(12, 9), dpi=250)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    im = ax.imshow(display_mat, cmap='Blues', interpolation='nearest', vmin=0, vmax=vmax)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    nc = len(names)
    ax.set_xticks(range(nc))
    ax.set_yticks(range(nc))
    ax.set_xticklabels(names, rotation=90, fontsize=11, ha='center')
    ax.set_yticklabels(names, fontsize=11)

    ax.set_xlabel('True', fontsize=12, labelpad=12)
    ax.set_ylabel('Predicted', fontsize=12, labelpad=12)
    ax.set_title(title, fontsize=14, pad=15)

    thresh = vmax / 2.0
    for i in range(nc):
        for j in range(nc):
            val = display_mat[i, j]
            text_color = 'white' if val > thresh else 'black'
            val_str = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, val_str, ha='center', va='center', color=text_color, fontsize=11)

    # Remove extra border spines to match clean Ultralytics look
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    save_path = out_dir / filename
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {save_path.name}")

# Generate for all 3 models
for model_key in ["yolov8n", "yolov8s", "yolov8m"]:
    plot_ultralytics_style_cm(confusion_matrices[model_key], classes, model_key, normalize=False)
    plot_ultralytics_style_cm(confusion_matrices[model_key], classes, model_key, normalize=True)

print(f"[Success] All Stage 2 native Ultralytics confusion matrices generated in: {out_dir.resolve()}")
