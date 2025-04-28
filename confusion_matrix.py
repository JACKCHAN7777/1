import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ---------------------- Analysis of the overall level of the image ---------------------- #
def parse_image_level_labels(gt_path, pred_path, class_names, background_class='background', score_threshold=0.001):
    gt_labels = []
    pred_labels = []

    for file in os.listdir(gt_path):
        image_id = os.path.splitext(file)[0]  # Remove the .txt suffix

        # Get the first target category of ground-truth
        with open(os.path.join(gt_path, file), 'r') as f:
            gt_lines = f.readlines()
            if len(gt_lines) > 0:
                gt_label = gt_lines[0].strip().split()[0]
                if gt_label in class_names:
                    gt_labels.append(class_names.index(gt_label))
                else:
                    # Categories that are not recognized skip
                    continue
            else:
                # Skip the graph without ground-truth
                continue

        # Get detection-results Maximum confidence category
        pred_file = os.path.join(pred_path, f"{image_id}.txt")
        best_score = -1
        best_label = background_class  # Default is background class
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as pf:
                for line in pf:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        print(f"Skipping invalid line(less than 6 elements):{line.strip()}")
                        continue  # Prevent format errors
                    try:
                        label = parts[0]
                        score = float(parts[1])
                    except ValueError:
                        print(f"⚠️ Skipping invalid line (cannot parse score): {line.strip()}")
                        continue  # The fraction is not a floating point type, skip
                    if score >= score_threshold and label in class_names and score > best_score:
                        best_score = score
                        best_label = label

        # Record the final tag
        if best_label == background_class:
            pred_labels.append(len(class_names))  # Background coded as the last category
        else:
            pred_labels.append(class_names.index(best_label))

    return gt_labels, pred_labels


# ---------------------- Draw the confusion matrix ---------------------- #
def generate_confusion_matrix(gt_labels, pred_labels, class_names, save_path='map_out/confusion_matrix.png'):
    if not gt_labels or not pred_labels:
        print("Error: Empty label list. Cannot generate confusion matrix.")
        return

    # Generate confusion matrix
    cm = confusion_matrix(gt_labels, pred_labels, labels=range(len(class_names)))
    print("Confusion Matrix:\n", cm)

    # Draw
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title('Image-Level Confusion Matrix')
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


# ---------------------- ONE CLICK RUNNING EXAMPLE ---------------------- #
def run_image_level_confusion_matrix(gt_path, pred_path, class_names, map_out_path, background_class='background', score_threshold=0.001):
    print("\n Starting Image-Level Confusion Matrix Generation...")
    if background_class not in class_names:
        class_names.append(background_class)
    gt_labels, pred_labels = parse_image_level_labels(gt_path, pred_path, class_names, background_class, score_threshold)
    generate_confusion_matrix(gt_labels, pred_labels, class_names, save_path=os.path.join(map_out_path, 'confusion_matrix.png'))
    print("Confusion Matrix Generation Completed.")
