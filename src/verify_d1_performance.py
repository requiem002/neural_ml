"""Verify training performance on D1 and D2 with confusion matrices."""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pickle

# Paths
data_dir = Path(__file__).parent.parent / 'datasets'
model_dir = Path(__file__).parent.parent / 'models'
analysis_dir = Path(__file__).parent.parent / 'analysis'
analysis_dir.mkdir(exist_ok=True)

# Load trained pipeline
print("Loading trained pipeline...")
with open(model_dir / 'spike_sorter.pkl', 'rb') as f:
    state = pickle.load(f)

# We need to import the pipeline class to use it
from pipeline import SpikeSortingPipeline
pipeline = SpikeSortingPipeline()
pipeline.load(model_dir / 'spike_sorter.pkl')


def match_spikes(pred_indices, pred_classes, true_indices, true_classes, tolerance=50):
    """
    Match predicted spikes to ground truth spikes within tolerance.
    Returns matched true and predicted classes for confusion matrix.
    """
    matched_true = []
    matched_pred = []
    used_true = set()

    for pred_i, pred_idx in enumerate(pred_indices):
        best_match = None
        best_dist = tolerance + 1

        for true_i, true_idx in enumerate(true_indices):
            if true_i in used_true:
                continue
            dist = abs(pred_idx - true_idx)
            if dist <= tolerance and dist < best_dist:
                best_match = true_i
                best_dist = dist

        if best_match is not None:
            used_true.add(best_match)
            matched_true.append(true_classes[best_match])
            matched_pred.append(pred_classes[pred_i])

    return np.array(matched_true), np.array(matched_pred), len(used_true)


def evaluate_dataset(dataset_name, has_ground_truth=True):
    """Evaluate pipeline on a dataset and generate confusion matrix."""
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name}")
    print('='*60)

    # Load data
    data = sio.loadmat(data_dir / f'{dataset_name}.mat')
    d = data['d'].flatten()

    if has_ground_truth:
        true_index = data['Index'].flatten()
        true_class = data['Class'].flatten()
        print(f"Ground truth spikes: {len(true_index)}")
    else:
        true_index = None
        true_class = None

    # Run prediction
    print("Running prediction...")
    pred_index, pred_class = pipeline.predict(d, threshold_factor=5.0)
    print(f"Detected spikes: {len(pred_index)}")

    if not has_ground_truth:
        print("No ground truth available for this dataset")
        return None

    # Match spikes
    print(f"Matching spikes (tolerance=±50 samples)...")
    matched_true, matched_pred, num_matched = match_spikes(
        pred_index, pred_class, true_index, true_class, tolerance=50
    )

    print(f"Matched spikes: {num_matched}")
    print(f"Unmatched predictions (false positives): {len(pred_index) - num_matched}")
    print(f"Missed ground truth (false negatives): {len(true_index) - num_matched}")

    # Detection metrics
    precision = num_matched / len(pred_index) if len(pred_index) > 0 else 0
    recall = num_matched / len(true_index) if len(true_index) > 0 else 0
    det_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- Detection Performance ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Detection F1: {det_f1:.4f}")

    # Classification metrics (on matched spikes only)
    if len(matched_true) > 0:
        print(f"\n--- Classification Performance (matched spikes) ---")
        print(classification_report(matched_true, matched_pred, digits=4))

        class_f1 = f1_score(matched_true, matched_pred, average='weighted')
        print(f"Weighted Classification F1: {class_f1:.4f}")

        # Combined F1 as per coursework
        combined_f1 = 0.3 * det_f1 + 0.7 * class_f1
        print(f"\n--- Combined Score ---")
        print(f"F_dataset = 0.3 × {det_f1:.4f} + 0.7 × {class_f1:.4f} = {combined_f1:.4f}")

        # Generate confusion matrix
        cm = confusion_matrix(matched_true, matched_pred, labels=[1, 2, 3, 4, 5])

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'],
                    yticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'],
                    ax=ax)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title(f'{dataset_name} Confusion Matrix\n(Matched Spikes: {num_matched}, Combined F1: {combined_f1:.4f})')

        plt.tight_layout()
        plt.savefig(analysis_dir / f'{dataset_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {analysis_dir / f'{dataset_name}_confusion_matrix.png'}")

        return {
            'detected': len(pred_index),
            'ground_truth': len(true_index),
            'matched': num_matched,
            'detection_f1': det_f1,
            'classification_f1': class_f1,
            'combined_f1': combined_f1,
            'confusion_matrix': cm
        }
    else:
        print("No matched spikes for classification evaluation")
        return None


# Evaluate D1 (training data with ground truth)
results_d1 = evaluate_dataset('D1', has_ground_truth=True)

# Evaluate D2 (closest to training, but no ground truth in the file)
# Let me check if D2 has ground truth
data_d2 = sio.loadmat(data_dir / 'D2.mat')
if 'Index' in data_d2 and 'Class' in data_d2:
    results_d2 = evaluate_dataset('D2', has_ground_truth=True)
else:
    print(f"\n{'='*60}")
    print("D2 does not have ground truth labels (Index/Class)")
    print("Cannot generate confusion matrix for D2")
    print('='*60)
    results_d2 = None

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)

if results_d1:
    print(f"\nD1 Results:")
    print(f"  Spikes detected: {results_d1['detected']}")
    print(f"  Ground truth spikes: {results_d1['ground_truth']}")
    print(f"  Matched spikes: {results_d1['matched']}")
    print(f"  Detection F1: {results_d1['detection_f1']:.4f}")
    print(f"  Classification F1: {results_d1['classification_f1']:.4f}")
    print(f"  Combined F1: {results_d1['combined_f1']:.4f}")

if results_d2:
    print(f"\nD2 Results:")
    print(f"  Spikes detected: {results_d2['detected']}")
    print(f"  Ground truth spikes: {results_d2['ground_truth']}")
    print(f"  Matched spikes: {results_d2['matched']}")
    print(f"  Detection F1: {results_d2['detection_f1']:.4f}")
    print(f"  Classification F1: {results_d2['classification_f1']:.4f}")
    print(f"  Combined F1: {results_d2['combined_f1']:.4f}")

plt.show()
