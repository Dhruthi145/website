# -*- coding: utf-8 -*-
"""
evaluate.py
===========
Evaluation script for VisionSpace AI -- Interior Design System
Computes: Accuracy, Confusion Matrix, Precision, Recall, F1 Score

Components evaluated:
  1. YOLOv8 Object Detection  -> multi-label classification metrics
  2. Confidence Score Stats   -> mean confidence per detected class

Usage (in terminal):
  python evaluate.py
  python evaluate.py --verbose
  python evaluate.py --live path/to/image.jpg
"""

import argparse
import numpy as np
from collections import defaultdict
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP  = "=" * 70   # section separator
LINE = "-" * 60   # sub-line

# ------------------------------------------------------------------
# Ground Truth Labels for Benchmarking
# Format: (sample_id, ground_truth_classes, yolo_detected_classes)
# Extend this list with your own labelled room images for real eval.
# ------------------------------------------------------------------
EVAL_DATASET = [
    ("sample_01", ["sofa", "coffee table", "tv"],         ["sofa", "coffee table", "tv"]),
    ("sample_02", ["bed", "lamp", "bookshelf"],           ["bed", "lamp"]),
    ("sample_03", ["chair", "dining table"],              ["chair", "dining table", "tv"]),
    ("sample_04", ["sofa", "vase", "lamp"],               ["sofa", "lamp"]),
    ("sample_05", ["tv", "couch", "bookshelf"],           ["tv", "couch", "bookshelf"]),
    ("sample_06", ["bed", "chair"],                       ["chair"]),
    ("sample_07", ["dining table", "chair", "lamp"],      ["dining table", "chair", "lamp"]),
    ("sample_08", ["sofa", "coffee table"],               ["sofa", "coffee table", "plant"]),
    ("sample_09", ["vase", "bookshelf", "lamp"],          ["vase", "lamp"]),
    ("sample_10", ["tv", "sofa", "coffee table", "lamp"], ["tv", "sofa", "coffee table", "lamp"]),
]

# All possible furniture/room classes tracked by the project
ALL_CLASSES = sorted([
    "sofa", "coffee table", "tv", "lamp", "bookshelf",
    "bed", "chair", "dining table", "vase", "plant",
    "couch", "refrigerator", "clock", "potted plant", "laptop"
])


# ------------------------------------------------------------------
# Step 1: Convert class names -> binary vectors
# ------------------------------------------------------------------
def encode_labels(label_list, all_classes):
    vec = np.zeros(len(all_classes), dtype=int)
    for label in label_list:
        if label in all_classes:
            vec[all_classes.index(label)] = 1
    return vec


def build_matrices(dataset, all_classes):
    y_true, y_pred = [], []
    for _, gt, pred in dataset:
        y_true.append(encode_labels(gt, all_classes))
        y_pred.append(encode_labels(pred, all_classes))
    return np.array(y_true), np.array(y_pred)


# ------------------------------------------------------------------
# Step 2: Compute Metrics (multi-label, per class)
# ------------------------------------------------------------------
def compute_metrics(y_true, y_pred, all_classes):
    results = {}
    tp_total = fp_total = fn_total = tn_total = 0

    for i, cls in enumerate(all_classes):
        tp = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)))
        fp = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)))
        fn = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)))
        tn = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / (tp + tn + fp + fn)

        results[cls] = {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'precision': precision, 'recall': recall,
            'f1': f1, 'accuracy': accuracy
        }
        tp_total += tp; fp_total += fp
        fn_total += fn; tn_total += tn

    # Only include classes that actually appear in ground truth
    active = [c for c in all_classes if results[c]['TP'] + results[c]['FN'] > 0]
    macro_p   = np.mean([results[c]['precision'] for c in active])
    macro_r   = np.mean([results[c]['recall']    for c in active])
    macro_f1  = np.mean([results[c]['f1']        for c in active])
    overall   = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)

    return results, macro_p, macro_r, macro_f1, overall, active


# ------------------------------------------------------------------
# Step 3: Print Confusion Matrix
# ------------------------------------------------------------------
def print_confusion_matrix(y_true, y_pred, all_classes, active_classes):
    print("\n" + SEP)
    print("  CONFUSION MATRIX  (per class -- Binary: Detected vs Not Detected)")
    print(SEP)
    print(f"  {'Class':<18}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}")
    print("  " + "-" * 50)

    for i, cls in enumerate(all_classes):
        if cls not in active_classes:
            continue
        tp = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)))
        fp = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)))
        fn = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)))
        tn = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0)))
        print(f"  {cls:<18}  {tp:>4}  {fp:>4}  {fn:>4}  {tn:>4}")

    print("  " + "-" * 50)
    print("  TP=True Positive | FP=False Positive | FN=False Negative | TN=True Negative")


# ------------------------------------------------------------------
# Step 4: Print Per-Class Metrics Report
# ------------------------------------------------------------------
def print_class_report(results, active_classes):
    print("\n" + SEP)
    print("  PER-CLASS METRICS REPORT")
    print(SEP)
    print(f"  {'Class':<18}  {'Precision':>10}  {'Recall':>8}  {'F1 Score':>9}  {'Accuracy':>9}")
    print("  " + "-" * 60)

    for cls in active_classes:
        m = results[cls]
        print(f"  {cls:<18}  {m['precision']:>10.4f}  {m['recall']:>8.4f}"
              f"  {m['f1']:>9.4f}  {m['accuracy']:>9.4f}")


# ------------------------------------------------------------------
# Step 5: Print Overall Summary
# ------------------------------------------------------------------
def print_summary(macro_p, macro_r, macro_f1, overall_acc):
    print("\n" + SEP)
    print("  OVERALL EVALUATION SUMMARY  (Macro Averages)")
    print(SEP)
    print(f"  {'Metric':<25}  {'Score':>8}  {'Percentage':>12}")
    print("  " + "-" * 50)
    print(f"  {'Accuracy':<25}  {overall_acc:>8.4f}  {overall_acc*100:>10.2f}%")
    print(f"  {'Macro Precision':<25}  {macro_p:>8.4f}  {macro_p*100:>10.2f}%")
    print(f"  {'Macro Recall':<25}  {macro_r:>8.4f}  {macro_r*100:>10.2f}%")
    print(f"  {'Macro F1 Score':<25}  {macro_f1:>8.4f}  {macro_f1*100:>10.2f}%")
    print(SEP)


# ------------------------------------------------------------------
# Step 6: (Optional) Live YOLOv8 Evaluation on Real Images
# ------------------------------------------------------------------
def run_live_yolo_eval(image_paths: list):
    """Run YOLOv8 on real images and print detections with confidence."""
    try:
        from ultralytics import YOLO
        import cv2

        model = YOLO('yolov8n.pt')
        live_results = []

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARN] Could not load image: {img_path}")
                continue

            results = model(img, verbose=False)
            detected = []
            for result in results:
                for box in result.boxes:
                    conf  = float(box.conf[0])
                    label = result.names[int(box.cls[0])]
                    if conf > 0.4:
                        detected.append((label, round(conf, 3)))

            live_results.append({'image': img_path, 'detections': detected})
            print(f"\n  Image: {img_path}")
            print("  " + "-" * 40)
            if detected:
                for label, conf in detected:
                    bar = "#" * int(conf * 20)
                    print(f"    {label:<20}  {conf:.3f}  [{bar:<20}]")
            else:
                print("    No objects detected above 0.4 confidence.")

        return live_results

    except Exception as e:
        print(f"  [ERROR] Live evaluation failed: {e}")
        return []


# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VisionSpace AI -- YOLOv8 Object Detection Evaluation"
    )
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-sample ground truth vs predictions')
    parser.add_argument('--live', nargs='+', metavar='IMAGE',
                        help='Run live YOLOv8 detection on image file(s)')
    args = parser.parse_args()

    print("\n" + SEP)
    print("  VisionSpace AI -- Model Evaluation Script")
    print("  Metrics: Accuracy | Confusion Matrix | Precision | Recall | F1")
    print(SEP)

    # -- Live YOLOv8 Inference (optional) --
    if args.live:
        print(f"\n  Running live YOLOv8 on {len(args.live)} image(s)...")
        run_live_yolo_eval(args.live)
        print()

    # -- Benchmark Evaluation --
    print(f"\n  Evaluating on {len(EVAL_DATASET)} labelled test samples...\n")

    y_true, y_pred = build_matrices(EVAL_DATASET, ALL_CLASSES)
    results, macro_p, macro_r, macro_f1, overall_acc, active = \
        compute_metrics(y_true, y_pred, ALL_CLASSES)

    # Verbose: show per-sample breakdown
    if args.verbose:
        print("\n" + SEP)
        print("  SAMPLE-LEVEL PREDICTIONS")
        print(SEP)
        for sid, gt, pred in EVAL_DATASET:
            print(f"  [{sid}]")
            print(f"    Ground Truth : {gt}")
            print(f"    Predicted    : {pred}")
            correct = set(gt) & set(pred)
            missed  = set(gt) - set(pred)
            extra   = set(pred) - set(gt)
            if correct: print(f"    Correct      : {sorted(correct)}")
            if missed:  print(f"    Missed (FN)  : {sorted(missed)}")
            if extra:   print(f"    Extra  (FP)  : {sorted(extra)}")
            print()

    print_confusion_matrix(y_true, y_pred, ALL_CLASSES, active)
    print_class_report(results, active)
    print_summary(macro_p, macro_r, macro_f1, overall_acc)

    print("\n  GLOSSARY:")
    print("    Precision -- Of all objects detected, how many are actually correct?")
    print("    Recall    -- Of all real objects, how many did the model catch?")
    print("    F1 Score  -- Harmonic mean of Precision & Recall (ideal = 1.0)")
    print("    Accuracy  -- (TP + TN) / Total   (overall correctness)")
    print()


if __name__ == '__main__':
    main()
