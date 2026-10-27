import os
import cv2
import argparse
from pathlib import Path
import numpy as np

# Ensure ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    import sys
    print("Required package (ultralytics) not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

def get_color(label):
    label_lower = label.lower()
    if 'healthy' in label_lower:
        return (0, 255, 0)       # Green
    elif 'plfd' in label_lower:
        return (0, 0, 255)       # Red
    else:
        return (0, 165, 255)     # Orange / Other diseases

def process_image(img, detector, classifier, conf_det, device, imgsz_det=640):
    # Run Stage 1: Single-Class Rubber Leaf Detection
    results_det = detector.predict(
        source=img,
        conf=conf_det,
        device=device,
        verbose=False,
        imgsz=imgsz_det
    )
    
    boxes = results_det[0].boxes
    if len(boxes) == 0:
        return img, 0
        
    annotated_img = img.copy()
    h, w, _ = img.shape
    crops = []
    box_coords = []
    
    # Extract boxes and crop leaves
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Clip coordinates to stay within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
            box_coords.append((x1, y1, x2, y2))
            
    if not crops:
        return img, 0
        
    # Run Stage 2: Disease Classification in a single parallel batch call
    disease_labels = []
    disease_confs = []
    
    results_cls_list = classifier.predict(
        source=crops,
        batch=len(crops),
        device=device,
        verbose=False
    )
    
    for results_cls in results_cls_list:
        probs = results_cls.probs
        cls_idx = probs.top1
        cls_name = classifier.names[cls_idx]
        cls_conf = float(probs.top1conf)
        disease_labels.append(cls_name)
        disease_confs.append(cls_conf)
        
    # Draw annotations on the original image
    for (x1, y1, x2, y2), label, conf in zip(box_coords, disease_labels, disease_confs):
        color = get_color(label)
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        # Display label and confidence
        display_text = f"{label} ({conf:.2%})"
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(display_text, font, font_scale, thickness)
        text_w, text_h = text_size
        
        # Draw text background
        cv2.rectangle(
            annotated_img, 
            (x1, y1 - text_h - 6), 
            (x1 + text_w + 6, y1), 
            color, 
            -1
        )
        
        # Draw text shadow/text
        cv2.putText(
            annotated_img, 
            display_text, 
            (x1 + 3, y1 - 3), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness, 
            cv2.LINE_AA
        )
        
    return annotated_img, len(crops)

def main():
    parser = argparse.ArgumentParser(description="End-to-End Two-Stage Rubber Leaf Disease Detector")
    parser.add_argument("--source", type=str, required=True, help="Path to image file, video file, or directory")
    parser.add_argument("--conf-det", type=float, default=0.40, help="Confidence threshold for Stage 1 detector")
    parser.add_argument("--imgsz-det", type=int, default=640, help="Image size for Stage 1 detector (e.g. 640, 1024, 1280)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on ('cpu', 'cuda', 0)")
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    
    det_onnx = script_dir / "Stage 1 Rubber Leaf Detector" / "stage1_training_results" / "weights" / "best.onnx"
    det_top_onnx = script_dir / "Stage 1 Rubber Leaf Detector" / "stage1_leaf_detector.onnx"
    det_pt = script_dir / "Stage 1 Rubber Leaf Detector" / "stage1_training_results" / "weights" / "best.pt"
    detector_path = det_onnx if det_onnx.exists() else (det_top_onnx if det_top_onnx.exists() else det_pt)

    cls_onnx = script_dir / "Stage2_Classifier" / "Training Result on v4 dataset" / "weights" / "best.onnx"
    cls_pt1 = script_dir / "Stage2_Classifier" / "Training Result on v4 dataset" / "weights" / "best.pt"
    cls_pt2 = script_dir / "Stage2_Classifier" / "Training Result v2" / "weights" / "best.pt"
    cls_pt3 = script_dir / "Stage2_Classifier" / "stage2_training_results" / "weights" / "best.pt"
    
    if cls_onnx.exists():
        classifier_path = cls_onnx
    elif cls_pt1.exists():
        classifier_path = cls_pt1
    elif cls_pt2.exists():
        classifier_path = cls_pt2
    else:
        classifier_path = cls_pt3
        
    # Verify model files exist
    if not detector_path.exists():
        print(f"Error: Stage 1 detector weights not found at: {detector_path}")
        print("Please train Stage 1 and ensure the weights folder exists.")
        return
        
    if not classifier_path.exists():
        print(f"Error: Stage 2 classifier weights not found at: {classifier_path}")
        print("Please train Stage 2 or download stage2_training_results weights.")
        return
        
    print(f"Loading Stage 1 Leaf Detector: {detector_path}")
    detector = YOLO(str(detector_path))
    
    print(f"Loading Stage 2 Disease Classifier: {classifier_path}")
    classifier = YOLO(str(classifier_path))
    
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source path does not exist: {args.source}")
        return
        
    # Setup runs directory
    output_dir = script_dir / "runs" / "detect" / "two_stage"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    valid_video_ext = {'.mp4', '.avi', '.mov', '.mkv'}
    
    if source_path.is_file():
        ext = source_path.suffix.lower()
        if ext in valid_image_ext:
            # Single image inference
            img = cv2.imread(str(source_path))
            if img is None:
                print(f"Error: Failed to read image: {source_path}")
                return
                
            print(f"Processing image: {source_path.name}...")
            annotated_img, num_leaves = process_image(img, detector, classifier, args.conf_det, args.device, args.imgsz_det)
            
            output_file = output_dir / f"annotated_{source_path.name}"
            cv2.imwrite(str(output_file), annotated_img)
            print("\nTwo-stage inference complete!")
            print(f"Detected & classified {num_leaves} rubber leaves.")
            print(f"Saved annotated image to: {output_file}")
            
        elif ext in valid_video_ext:
            # Video inference frame-by-frame
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                print(f"Error: Failed to open video: {source_path}")
                return
                
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_file = output_dir / f"annotated_{source_path.name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (w, h))
            
            print(f"Processing video: {source_path.name} ({total_frames} frames)...")
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                annotated_frame, _ = process_image(frame, detector, classifier, args.conf_det, args.device, args.imgsz_det)
                out.write(annotated_frame)
                
                frame_idx += 1
                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    print(f"Processed frame {frame_idx}/{total_frames}...")
                    
            cap.release()
            out.release()
            print("\nTwo-stage video inference complete!")
            print(f"Saved annotated video to: {output_file}")
            
        else:
            print(f"Unsupported file format: {ext}")
            
    elif source_path.is_dir():
        # Folder directory of images inference
        print(f"Processing image directory: {source_path}")
        image_files = [f for f in source_path.iterdir() if f.suffix.lower() in valid_image_ext]
        print(f"Found {len(image_files)} image files.")
        
        for idx, img_file in enumerate(image_files):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            annotated_img, num_leaves = process_image(img, detector, classifier, args.conf_det, args.device, args.imgsz_det)
            output_file = output_dir / f"annotated_{img_file.name}"
            cv2.imwrite(str(output_file), annotated_img)
            print(f"[{idx+1}/{len(image_files)}] Processed {img_file.name} - found {num_leaves} leaves.")
            
        print(f"\nDirectory processing complete! Annotated images saved to: {output_dir}")

if __name__ == "__main__":
    main()
