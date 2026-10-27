# Stage 2: Rubber Leaf Disease Classifier Training (Kaggle Notebook Script)
# Copy and paste these cells into your Kaggle Notebook.

# --- CELL 1: Install Dependencies ---
# !pip install roboflow ultralytics

# --- CELL 2: Download the Stage 2 Dataset from Roboflow ---
import os
import shutil
from roboflow import Roboflow

print("Downloading dataset from Roboflow...")
rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
project = rf.workspace("rubber-research").project("stage-2-rubber-leaf-classifier")
version = project.version(4)
dataset = version.download("folder")

# Print the folder structure to confirm download
print(f"Dataset downloaded to: {dataset.location}")
print("Split and classes found:")
for split in ["train", "valid", "test"]:
    split_path = os.path.join(dataset.location, split)
    if os.path.exists(split_path) and os.path.isdir(split_path):
        print(f"\n* Split: {split}")
        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                print(f"  - {class_folder}: {num_images} images")

# --- CELL 3: Train the YOLOv8-cls Classification Model ---
from ultralytics import YOLO

# Choose and Load YOLOv8 classification model size
# Options: 'nano' (yolov8n-cls.pt), 'small' (yolov8s-cls.pt), 'medium' (yolov8m-cls.pt)
MODEL_SIZE = 'nano' # Change to 'nano', 'small', or 'medium'

model_map = {
    'nano': 'yolov8n-cls.pt',
    'small': 'yolov8s-cls.pt',
    'medium': 'yolov8m-cls.pt'
}
model_name = model_map.get(MODEL_SIZE.lower(), 'yolov8n-cls.pt')
print(f"Loading YOLOv8 classification {MODEL_SIZE} model ({model_name})...")
model = YOLO(model_name)

# Train the classification model
# It automatically uses the folder structure for classes (Healthy, Other Disease, PLFD)
# YOLO will auto-detect GPU (CUDA) on Kaggle.
print("Starting training...")
results = model.train(
    data=dataset.location,
    epochs=100,            # Increased to 100; early stopping (patience) will stop it if needed
    patience=10,           # Stop training if validation loss does not improve for 10 epochs
    imgsz=224,             # Standard input resolution for classification
    batch=32,              # Adjust based on GPU memory limits
    project="stage2_disease_classification",
    name="yolov8n_classifier",
    device=0,              # Runs on GPU 0 in Kaggle
    # Strategy 3: Strong augmentations & regularization for heterogeneous "Other Disease" class:
    dropout=0.2,           # Increase dropout to prevent memorizing specific disease spots
    hsv_h=0.02,            # Hue adjustments to simulate leaf color shifts
    hsv_s=0.7,             # Saturation adjustments
    hsv_v=0.5,             # Brightness adjustments to handle shadows and lighting changes
    degrees=30.0,          # Random rotation to handle arbitrary camera angles
    flipud=0.5,            # Random vertical flip for orientation invariance
    shear=10.0,            # Shearing to handle perspective skew
)


# --- CELL 4: Evaluate on Validation and Test Sets ---
print("\nEvaluating model performance on Validation and Test splits...")
try:
    from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
    import numpy as np
    from pathlib import Path
    import torch
    
    # Evaluate both 'valid' and 'test' splits
    splits_to_eval = ["valid", "test"]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    class_names = model.names
    class_name_to_id = {name.lower(): idx for idx, name in class_names.items()}
    
    # Setup save directory
    save_dir = Path(model.trainer.save_dir) if (hasattr(model, 'trainer') and model.trainer is not None) else Path("runs/detect/two_stage")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the classification report to a text file
    report_file_path = save_dir / "classification_report.txt"
    with open(report_file_path, "w") as rf_report:
        rf_report.write("DETAILED CLASSIFICATION METRICS REPORT\n")
        rf_report.write("="*60 + "\n\n")
        
        for split in splits_to_eval:
            split_dir = Path(dataset.location) / split
            print(f"\nEvaluating detailed metrics on {split.upper()} split...")
            
            y_true = []
            y_pred = []
            
            if split_dir.exists():
                for class_folder in sorted(split_dir.iterdir()):
                    if not class_folder.is_dir():
                        continue
                    class_name_lower = class_folder.name.lower()
                    
                    # Match folder to class ID
                    matched_id = None
                    for model_class_name, model_class_id in class_name_to_id.items():
                        if model_class_name in class_name_lower or class_name_lower in model_class_name:
                            matched_id = model_class_id
                            break
                    
                    if matched_id is not None:
                        for img_file in class_folder.iterdir():
                            if img_file.suffix.lower() in valid_extensions:
                                # Run prediction (without gradients for speed)
                                with torch.no_grad():
                                    results_test = model.predict(source=str(img_file), verbose=False)[0]
                                    pred_id = int(results_test.probs.top1)
                                y_true.append(matched_id)
                                y_pred.append(pred_id)
                                
                if y_true:
                    y_true = np.array(y_true)
                    y_pred = np.array(y_pred)
                    sorted_keys = sorted(list(class_names.keys()))
                    target_names = [class_names[i] for i in sorted_keys]
                    
                    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
                    accuracy = accuracy_score(y_true, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
                    
                    print("\n" + "="*60)
                    print(f"          DETAILED CLASSIFICATION REPORT ({split.upper()} SPLIT)")
                    print("="*60)
                    print(report)
                    print("-"*60)
                    print(f"Accuracy:        {accuracy:.4%}")
                    print(f"Macro Precision: {precision:.4%}")
                    print(f"Macro Recall:    {recall:.4%}")
                    print(f"Macro F1-Score:  {f1:.4%}")
                    print("="*60 + "\n")
                    
                    # Write to file
                    rf_report.write(f"SPLIT: {split.upper()}\n")
                    rf_report.write("-"*60 + "\n")
                    rf_report.write(report)
                    rf_report.write("-"*60 + "\n")
                    rf_report.write(f"Accuracy:        {accuracy:.4%}\n")
                    rf_report.write(f"Macro Precision: {precision:.4%}\n")
                    rf_report.write(f"Macro Recall:    {recall:.4%}\n")
                    rf_report.write(f"Macro F1-Score:  {f1:.4%}\n")
                    rf_report.write("="*60 + "\n\n")
                else:
                    print(f"Warning: Could not process any images for {split} split.")
            else:
                print(f"Warning: {split} directory not found. Skipping.")
                
    print(f"Saved classification metrics report to: {report_file_path}")
except Exception as e:
    print(f"Warning: Failed to calculate classification metrics: {e}")


# --- CELL 5: Export Model to ONNX and Engine (for deployment) ---
# Having it in ONNX format makes it easy to compile for the Raspberry Pi AI Kit / Hailo.
print("Exporting model to ONNX format...")
onnx_path = model.export(format="onnx")
print(f"Model successfully exported to: {onnx_path}")

# --- CELL 6: Zip Results for Downloading ---
save_dir = str(model.trainer.save_dir)
print(f"Zipping classification training results from: {save_dir}")
shutil.make_archive("stage2_training_results", "zip", save_dir)
print("Zip archive created: 'stage2_training_results.zip' (Ready to download from Kaggle output)")


# --- CELL 7: Install Explainable AI (XAI) Libs and run Grad-CAM++ ---
# !pip install grad-cam matplotlib
try:
    import cv2
    import matplotlib.pyplot as plt
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import torch
    import numpy as np
    from pathlib import Path
    
    # Select a few test images from Roboflow dataset
    test_dir = Path(dataset.location) / "test"
    test_images = []
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(Path(root) / f)
                if len(test_images) >= 5:
                    break
        if len(test_images) >= 5:
            break
            
    print(f"Found {len(test_images)} test images for Grad-CAM++ visualization.")
    
    # Enable gradients explicitly for XAI
    torch.set_grad_enabled(True)
    for param in model.model.parameters():
        param.requires_grad = True
        
    # Setup Grad-CAM++
    # Target layer: model.model.model[-2] (the final convolution layer)
    target_layers = [model.model.model[-2]]
    cam = GradCAMPlusPlus(model=model.model, target_layers=target_layers)
    
    # Output directory for Grad-CAM
    xai_out = Path("xai_results")
    xai_out.mkdir(exist_ok=True)
    
    cam_maps = []
    img_floats = []
    target_classes = []
    
    for idx, img_path in enumerate(test_images):
        # Preprocess
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0
        input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(0)
        input_tensor.requires_grad = True
        
        # Predict to get top target class
        with torch.no_grad():
            preds = model.model(input_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            predicted_class = int(preds.argmax(dim=1).item())
            predicted_conf = float(preds.softmax(dim=1)[0, predicted_class].item())
            
        class_name = model.names[predicted_class]
        targets = [ClassifierOutputTarget(predicted_class)]
        
        # Generate map
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        # Save image
        out_name = f"gradcam_{img_path.stem}_{class_name}.png"
        cv2.imwrite(str(xai_out / out_name), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"[{idx+1}/{len(test_images)}] Generated Grad-CAM++ for {img_path.name} (Predicted: {class_name} {predicted_conf:.2%})")
        
        cam_maps.append(grayscale_cam)
        img_floats.append(img_float)
        target_classes.append(predicted_class)
        
    # --- CELL 8: Compute Perturbation Metrics (Deletion & Insertion Curves) ---
    print("\nComputing Perturbation Faithfulness Metrics...")
    steps = np.linspace(0, 1, 21) # 21 steps from 0% to 100%
    
    for idx, (img_path, cam_map, img_float, target_class) in enumerate(zip(test_images, cam_maps, img_floats, target_classes)):
        h, w = cam_map.shape
        num_pixels = h * w
        flat_cam = cam_map.flatten()
        sorted_indices = np.argsort(flat_cam)[::-1]
        
        deletion_scores = []
        insertion_scores = []
        
        # Run predictions
        with torch.no_grad():
            # Deletion
            for step in steps:
                num_to_mask = int(step * num_pixels)
                perturbed_img = img_float.copy().reshape(-1, 3)
                if num_to_mask > 0:
                    perturbed_img[sorted_indices[:num_to_mask]] = 0 # set to black
                perturbed_img = perturbed_img.reshape(h, w, 3)
                p_tensor = torch.from_numpy(perturbed_img).permute(2, 0, 1).unsqueeze(0).to(0)
                preds = model.model(p_tensor)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                preds = preds.softmax(dim=1)
                deletion_scores.append(float(preds[0, target_class].item()))
                
            # Insertion
            for step in steps:
                num_to_insert = int(step * num_pixels)
                perturbed_img = np.zeros_like(img_float).reshape(-1, 3)
                if num_to_insert > 0:
                    perturbed_img[sorted_indices[:num_to_insert]] = img_float.copy().reshape(-1, 3)[sorted_indices[:num_to_insert]]
                perturbed_img = perturbed_img.reshape(h, w, 3)
                p_tensor = torch.from_numpy(perturbed_img).permute(2, 0, 1).unsqueeze(0).to(0)
                preds = model.model(p_tensor)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                preds = preds.softmax(dim=1)
                insertion_scores.append(float(preds[0, target_class].item()))
                
        # Calculate AUC
        del_auc = np.trapz(deletion_scores, steps)
        ins_auc = np.trapz(insertion_scores, steps)
        
        # Plot curves
        plt.figure(figsize=(8, 4))
        plt.plot(steps * 100, deletion_scores, '-o', label=f'Deletion Curve (AUC: {del_auc:.3f})', color='red')
        plt.plot(steps * 100, insertion_scores, '-o', label=f'Insertion Curve (AUC: {ins_auc:.3f})', color='green')
        plt.xlabel('% Pixels Perturbed')
        plt.ylabel(f'Model Confidence ({model.names[target_class]})')
        plt.title(f'Perturbation Analysis: {img_path.name}')
        plt.legend()
        plt.grid(True)
        
        plot_name = f"perturbation_{img_path.stem}.png"
        plt.savefig(str(xai_out / plot_name))
        plt.close()
        
        print(f"[{idx+1}/{len(test_images)}] Faithfulness for {img_path.name}: Deletion AUC = {del_auc:.3f}, Insertion AUC = {ins_auc:.3f}")
        
    # Zip XAI results too
    shutil.make_archive("xai_results", "zip", "xai_results")
    print("XAI results zipped as 'xai_results.zip' (Ready to download from Kaggle output)")
    
except Exception as e:
    print(f"\n[Warning] Explainable AI (XAI) generation failed: {e}")
    print("Skipping XAI step to ensure training results are zipped and saved.")
