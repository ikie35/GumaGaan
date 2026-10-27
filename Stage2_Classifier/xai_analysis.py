import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Ensure pytorch-grad-cam is installed
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("[WARNING] pytorch-grad-cam not installed. Please install it using: pip install grad-cam")

def preprocess_image(image_path):
    """Preprocess image to match YOLOv8-cls input structure (224x224, float32, normalized)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0
    
    # Shape: (1, 3, 224, 224)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    return img_float, tensor

def run_gradcam(model_path, image_path, output_dir, target_class=None):
    """Generates Grad-CAM++ visualization for a leaf image and saves the result."""
    model = YOLO(str(model_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.eval()
    
    img_float, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Find target layer - model.model.model[-2] is the last conv layer before classification head
    target_layers = [model.model.model[-2]]
    
    # Run prediction to get default target class if not provided
    with torch.no_grad():
        preds = model.model(input_tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        predicted_class = int(preds.argmax(dim=1).item())
        predicted_conf = float(preds.softmax(dim=1)[0, predicted_class].item())
        
    class_idx = target_class if target_class is not None else predicted_class
    class_name = model.names[class_idx]
    
    # Setup Grad-CAM++
    cam = GradCAMPlusPlus(model=model.model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]
    
    # Saliency map
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Overlay CAM on image
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    # Save the output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"gradcam_{Path(image_path).stem}_{class_name}.png"
    out_path = output_dir / out_name
    
    # Save as BGR for cv2
    cv2.imwrite(str(out_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM++ saved to: {out_path} (Class: {class_name}, Conf: {predicted_conf:.2%})")
    
    return grayscale_cam, img_float, class_idx

def compute_perturbation_metrics(model_path, image_path, cam_map, target_class, output_dir):
    """
    Computes Deletion and Insertion Curves to evaluate the faithfulness of explanations.
    Saves the curve visualization plot.
    """
    model = YOLO(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.eval()
    
    img_float, input_tensor = preprocess_image(image_path)
    
    # Flatten cam_map and get indices sorted by importance (descending)
    h, w = cam_map.shape
    num_pixels = h * w
    flat_cam = cam_map.flatten()
    sorted_indices = np.argsort(flat_cam)[::-1]
    
    # Steps for perturbation (e.g. 5% increments)
    steps = np.linspace(0, 1, 21) # 21 steps from 0% to 100%
    deletion_scores = []
    insertion_scores = []
    
    # Run predictions
    with torch.no_grad():
        # Deletion: Start with original, gradually set top pixels to zero
        for step in steps:
            num_to_mask = int(step * num_pixels)
            perturbed_img = img_float.copy().reshape(-1, 3)
            if num_to_mask > 0:
                perturbed_img[sorted_indices[:num_to_mask]] = 0 # set to black
            perturbed_img = perturbed_img.reshape(h, w, 3)
            
            p_tensor = torch.from_numpy(perturbed_img).permute(2, 0, 1).unsqueeze(0).to(device)
            preds = model.model(p_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = preds.softmax(dim=1)
            score = float(preds[0, target_class].item())
            deletion_scores.append(score)
            
        # Insertion: Start with black canvas, gradually insert top pixels
        for step in steps:
            num_to_insert = int(step * num_pixels)
            perturbed_img = np.zeros_like(img_float).reshape(-1, 3)
            if num_to_insert > 0:
                perturbed_img[sorted_indices[:num_to_insert]] = img_float.copy().reshape(-1, 3)[sorted_indices[:num_to_insert]]
            perturbed_img = perturbed_img.reshape(h, w, 3)
            
            p_tensor = torch.from_numpy(perturbed_img).permute(2, 0, 1).unsqueeze(0).to(device)
            preds = model.model(p_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = preds.softmax(dim=1)
            score = float(preds[0, target_class].item())
            insertion_scores.append(score)
            
    # Calculate AUCs
    del_auc = np.trapz(deletion_scores, steps)
    ins_auc = np.trapz(insertion_scores, steps)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(steps * 100, deletion_scores, '-o', label=f'Deletion Curve (AUC: {del_auc:.3f})', color='red')
    plt.plot(steps * 100, insertion_scores, '-o', label=f'Insertion Curve (AUC: {ins_auc:.3f})', color='green')
    plt.xlabel('% Pixels Perturbed')
    plt.ylabel(f'Model Confidence for class {model.names[target_class]}')
    plt.title(f'Explanation Faithfulness Metrics (Perturbation Analysis)\nImage: {Path(image_path).name}')
    plt.legend()
    plt.grid(True)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"perturbation_{Path(image_path).stem}.png"
    plt.savefig(str(plot_path))
    plt.close()
    
    print(f"Perturbation plot saved to: {plot_path}")
    print(f"Faithfulness Metrics: Deletion AUC = {del_auc:.3f} (lower is better), Insertion AUC = {ins_auc:.3f} (higher is better)")
    
    return del_auc, ins_auc
