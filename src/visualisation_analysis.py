#!/usr/bin/env python3
"""
Comprehensive Visualization and Analysis for Spike Classification

This script generates:
1. D1 Ground Truth Comparison (confusion matrix, accuracy plots)
2. Waveform Quality Analysis (average waveforms per class)
3. Detection Performance Analysis
4. Class Distribution Comparison (your results vs friend's)
5. Training Data Analysis (what's being lost during alignment)
"""

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # For non-GUI environments
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import Counter
from scipy import signal
import seaborn as sns

# Try to import from fixed modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from cnn_experiment_fixed import CNNExperimentFixed, detect_spikes_aligned, ALIGN_PEAK_AT, WINDOW_SIZE
except ImportError:
    print("Warning: Could not import from cnn_experiment_fixed.py")
    ALIGN_PEAK_AT = 30
    WINDOW_SIZE = 60

# Setup
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / 'analysis' / 'visualizations'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_d1_ground_truth():
    """Load D1 with ground truth labels."""
    data = sio.loadmat(BASE_DIR / 'datasets' / 'D1.mat')
    return {
        'd': data['d'].flatten(),
        'Index': data['Index'].flatten(),
        'Class': data['Class'].flatten()
    }


def analyze_training_data_loss():
    """
    Analyze what's being lost during aligned extraction.
    This explains why we only get 1796/2176 spikes.
    """
    print("\n" + "="*70)
    print("TRAINING DATA LOSS ANALYSIS")
    print("="*70)
    
    d1 = load_d1_ground_truth()
    d = d1['d']
    indices = d1['Index']
    classes = d1['Class']
    
    # Track what happens to each spike
    results = {c: {'kept': 0, 'rejected_misaligned': 0, 'rejected_boundary': 0} 
               for c in range(1, 6)}
    
    peak_offsets_by_class = {c: [] for c in range(1, 6)}
    
    for idx, cls in zip(indices, classes):
        idx_0 = int(idx) - 1
        
        # Find actual peak
        search_start = max(0, idx_0 - 15)
        search_end = min(len(d), idx_0 + 15)
        
        if search_end - search_start < 5:
            results[cls]['rejected_boundary'] += 1
            continue
        
        local_region = d[search_start:search_end]
        local_peak_offset = np.argmax(local_region)
        actual_peak = search_start + local_peak_offset
        
        # Check if extraction would be in bounds
        start = actual_peak - ALIGN_PEAK_AT
        end = start + WINDOW_SIZE
        
        if start < 0 or end > len(d):
            results[cls]['rejected_boundary'] += 1
            continue
        
        # Extract and check alignment
        waveform = d[start:end]
        peak_in_wf = np.argmax(waveform)
        peak_offsets_by_class[cls].append(peak_in_wf)
        
        if abs(peak_in_wf - ALIGN_PEAK_AT) > 3:
            results[cls]['rejected_misaligned'] += 1
        else:
            results[cls]['kept'] += 1
    
    # Print results
    print(f"\n{'Class':<8} {'Total':<8} {'Kept':<8} {'Lost':<8} {'Loss %':<10} {'Reason'}")
    print("-" * 60)
    
    total_original = 0
    total_kept = 0
    
    for c in range(1, 6):
        original = np.sum(classes == c)
        kept = results[c]['kept']
        lost = original - kept
        loss_pct = 100 * lost / original if original > 0 else 0
        
        reason = f"misaligned={results[c]['rejected_misaligned']}, boundary={results[c]['rejected_boundary']}"
        print(f"Class {c:<3} {original:<8} {kept:<8} {lost:<8} {loss_pct:<10.1f} {reason}")
        
        total_original += original
        total_kept += kept
    
    print("-" * 60)
    print(f"{'TOTAL':<8} {total_original:<8} {total_kept:<8} {total_original-total_kept:<8} "
          f"{100*(total_original-total_kept)/total_original:.1f}%")
    
    # Plot peak position distributions by class
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for c in range(1, 6):
        ax = axes.flat[c-1]
        offsets = peak_offsets_by_class[c]
        
        if offsets:
            ax.hist(offsets, bins=range(0, 61), color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(ALIGN_PEAK_AT, color='red', linestyle='--', linewidth=2, label=f'Target ({ALIGN_PEAK_AT})')
            ax.axvline(ALIGN_PEAK_AT - 3, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(ALIGN_PEAK_AT + 3, color='orange', linestyle=':', alpha=0.7, label='±3 tolerance')
            
            # Count in tolerance
            in_tol = sum(1 for p in offsets if abs(p - ALIGN_PEAK_AT) <= 3)
            
            ax.set_title(f'Class {c}\n{in_tol}/{len(offsets)} ({100*in_tol/len(offsets):.0f}%) in tolerance')
            ax.set_xlabel('Peak Position (sample)')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
    
    # Summary in 6th panel
    ax = axes.flat[5]
    ax.axis('off')
    
    summary_text = "TRAINING DATA LOSS SUMMARY\n" + "="*30 + "\n\n"
    for c in range(1, 6):
        original = np.sum(classes == c)
        kept = results[c]['kept']
        loss_pct = 100 * (original - kept) / original
        summary_text += f"Class {c}: {kept}/{original} kept ({loss_pct:.0f}% lost)\n"
    
    summary_text += f"\nTotal: {total_kept}/{total_original} kept"
    summary_text += f"\n({100*(total_original-total_kept)/total_original:.1f}% lost)"
    summary_text += f"\n\n⚠ Classes 4 & 5 lose most data!"
    summary_text += f"\nConsider relaxing tolerance."
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    fig.suptitle('Peak Position Distribution by Class\n(Spikes outside ±3 of target are rejected during training)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = ANALYSIS_DIR / 'training_data_loss.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")
    
    return results


def create_d1_validation_visualization(cnn=None):
    """
    Create comprehensive D1 validation visualization.
    Shows confusion matrix, per-class accuracy, and sample waveforms.
    """
    print("\n" + "="*70)
    print("D1 VALIDATION VISUALIZATION")
    print("="*70)
    
    d1 = load_d1_ground_truth()
    d = d1['d']
    gt_indices = d1['Index']
    gt_classes = d1['Class']
    
    # Load CNN if not provided
    if cnn is None:
        try:
            cnn = CNNExperimentFixed()
            cnn.load_model()
        except Exception as e:
            print(f"Could not load CNN: {e}")
            return
    
    # Detect spikes
    detected_indices, detected_waveforms = detect_spikes_aligned(
        d, threshold_factor=4.5, align_peak_at=ALIGN_PEAK_AT
    )
    
    print(f"Ground truth: {len(gt_indices)} spikes")
    print(f"Detected: {len(detected_indices)} spikes")
    
    # Classify
    import torch
    wf_norm, amp_features = cnn.prepare_data(detected_waveforms)
    cnn.model.eval()
    
    with torch.no_grad():
        X_wf = torch.FloatTensor(wf_norm).to(cnn.device)
        X_amp = torch.FloatTensor(amp_features).to(cnn.device)
        outputs = cnn.model(X_wf, X_amp)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = outputs.max(1)
        pred_classes = predicted.cpu().numpy() + 1
        confidences = probs.max(axis=1)
    
    # Match to ground truth
    tolerance = 50
    matches = []
    gt_used = set()
    unmatched_detections = []
    
    for i, (det_idx, pred_cls, conf) in enumerate(zip(detected_indices, pred_classes, confidences)):
        distances = np.abs(gt_indices - det_idx)
        closest = np.argmin(distances)
        
        if distances[closest] <= tolerance and closest not in gt_used:
            matches.append({
                'det_idx': det_idx,
                'gt_idx': gt_indices[closest],
                'gt_class': gt_classes[closest],
                'pred_class': pred_cls,
                'confidence': conf,
                'waveform': detected_waveforms[i]
            })
            gt_used.add(closest)
        else:
            unmatched_detections.append({
                'det_idx': det_idx,
                'pred_class': pred_cls,
                'confidence': conf
            })
    
    # Find missed ground truth spikes
    missed_gt = [i for i in range(len(gt_indices)) if i not in gt_used]
    
    print(f"Matched: {len(matches)}")
    print(f"False positives: {len(unmatched_detections)}")
    print(f"Missed: {len(missed_gt)}")
    
    # Compute metrics
    gt_matched = np.array([m['gt_class'] for m in matches])
    pred_matched = np.array([m['pred_class'] for m in matches])
    
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    cm = confusion_matrix(gt_matched, pred_matched, labels=[1, 2, 3, 4, 5])
    precision, recall, f1, support = precision_recall_fscore_support(
        gt_matched, pred_matched, labels=[1, 2, 3, 4, 5]
    )
    
    # ========== CREATE FIGURE ==========
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Confusion Matrix (large, top-left)
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    
    # Normalize for display
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    im = ax1.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(5):
        for j in range(5):
            color = 'white' if cm_normalized[i, j] > 50 else 'black'
            ax1.text(j, i, f'{cm[i,j]}\n({cm_normalized[i,j]:.1f}%)',
                    ha='center', va='center', color=color, fontsize=10)
    
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels([1, 2, 3, 4, 5])
    ax1.set_yticklabels([1, 2, 3, 4, 5])
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('True Class', fontsize=12)
    
    overall_acc = 100 * np.trace(cm) / cm.sum()
    ax1.set_title(f'D1 Classification Confusion Matrix\nOverall Accuracy: {overall_acc:.1f}%',
                  fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax1, label='Percentage')
    
    # 2. Per-Class Metrics (top-right)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    
    x = np.arange(5)
    width = 0.25
    
    bars1 = ax2.bar(x - width, precision * 100, width, label='Precision', color='#2ecc71')
    bars2 = ax2.bar(x, recall * 100, width, label='Recall', color='#3498db')
    bars3 = ax2.bar(x + width, f1 * 100, width, label='F1-Score', color='#9b59b6')
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Score (%)')
    ax2.set_title('Per-Class Performance Metrics', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([1, 2, 3, 4, 5])
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% target')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 3. Detection Performance (middle-right)
    ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    
    det_precision = len(matches) / len(detected_indices) * 100
    det_recall = len(matches) / len(gt_indices) * 100
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall)
    
    categories = ['Precision\n(of detections)', 'Recall\n(of ground truth)', 'F1-Score']
    values = [det_precision, det_recall, det_f1]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    bars = ax3.bar(categories, values, color=colors)
    ax3.set_ylabel('Score (%)')
    ax3.set_title(f'Spike Detection Performance\n(Matched: {len(matches)}, FP: {len(unmatched_detections)}, Missed: {len(missed_gt)})',
                  fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 105)
    
    for bar, val in zip(bars, values):
        ax3.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Average Waveforms per Class (bottom row)
    for c in range(1, 6):
        ax = plt.subplot2grid((3, 4), (2, c-1 if c <= 4 else 3), colspan=1)
        
        # Get correctly classified waveforms for this class
        class_waveforms = [m['waveform'] for m in matches 
                         if m['gt_class'] == c and m['pred_class'] == c]
        
        # Get misclassified waveforms
        misclass_waveforms = [m['waveform'] for m in matches 
                            if m['gt_class'] == c and m['pred_class'] != c]
        
        if class_waveforms:
            wf_array = np.array(class_waveforms)
            mean_wf = np.mean(wf_array, axis=0)
            std_wf = np.std(wf_array, axis=0)
            
            x = np.arange(len(mean_wf))
            ax.fill_between(x, mean_wf - std_wf, mean_wf + std_wf, alpha=0.3, color='blue')
            ax.plot(x, mean_wf, 'b-', linewidth=2, label=f'Correct (n={len(class_waveforms)})')
        
        if misclass_waveforms:
            wf_array = np.array(misclass_waveforms)
            mean_wf = np.mean(wf_array, axis=0)
            ax.plot(x, mean_wf, 'r--', linewidth=2, label=f'Misclassified (n={len(misclass_waveforms)})')
        
        ax.axvline(ALIGN_PEAK_AT, color='green', linestyle=':', alpha=0.5)
        ax.set_title(f'Class {c}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Sample')
        if c == 1:
            ax.set_ylabel('Amplitude (V)')
        ax.legend(fontsize=7, loc='upper right')
    
    # Add 5th class in remaining space
    if True:  # Class 5
        ax = plt.subplot2grid((3, 4), (2, 3), colspan=1)
        c = 5
        class_waveforms = [m['waveform'] for m in matches 
                         if m['gt_class'] == c and m['pred_class'] == c]
        misclass_waveforms = [m['waveform'] for m in matches 
                            if m['gt_class'] == c and m['pred_class'] != c]
        
        if class_waveforms:
            wf_array = np.array(class_waveforms)
            mean_wf = np.mean(wf_array, axis=0)
            std_wf = np.std(wf_array, axis=0)
            x = np.arange(len(mean_wf))
            ax.fill_between(x, mean_wf - std_wf, mean_wf + std_wf, alpha=0.3, color='blue')
            ax.plot(x, mean_wf, 'b-', linewidth=2, label=f'Correct (n={len(class_waveforms)})')
        
        if misclass_waveforms:
            wf_array = np.array(misclass_waveforms)
            mean_wf = np.mean(wf_array, axis=0)
            ax.plot(x, mean_wf, 'r--', linewidth=2, label=f'Misclassified (n={len(misclass_waveforms)})')
        
        ax.axvline(ALIGN_PEAK_AT, color='green', linestyle=':', alpha=0.5)
        ax.set_title(f'Class {c}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Sample')
        ax.legend(fontsize=7, loc='upper right')
    
    plt.suptitle('D1 Ground Truth Validation - Complete Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = ANALYSIS_DIR / 'd1_validation_complete.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")
    
    # Also save the confusion matrix separately for clarity
    create_simple_confusion_matrix(cm, output_path=ANALYSIS_DIR / 'd1_confusion_matrix.png')
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'overall_accuracy': overall_acc,
        'detection_precision': det_precision,
        'detection_recall': det_recall
    }


def create_simple_confusion_matrix(cm, output_path):
    """Create a clean, simple confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    # Create heatmap
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    
    # Add annotations
    for i in range(5):
        for j in range(5):
            color = 'white' if cm_pct[i, j] > 50 else 'black'
            text = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=12)
    
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
    ax.set_yticklabels(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
    ax.set_xlabel('Predicted Class', fontsize=14)
    ax.set_ylabel('True Class', fontsize=14)
    
    # Per-class accuracy on the right
    for i in range(5):
        acc = cm[i, i] / cm[i].sum() * 100
        ax.text(5.5, i, f'{acc:.1f}%', ha='left', va='center', fontsize=11,
               color='green' if acc >= 90 else 'orange' if acc >= 80 else 'red')
    
    ax.text(5.5, -0.7, 'Accuracy', ha='left', va='center', fontsize=11, fontweight='bold')
    
    overall_acc = np.trace(cm) / cm.sum() * 100
    ax.set_title(f'D1 Classification Results\nOverall Accuracy: {overall_acc:.1f}%',
                fontsize=16, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Percentage', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compare_with_friend():
    """
    Compare your results with your friend's results.
    """
    print("\n" + "="*70)
    print("COMPARISON WITH FRIEND'S RESULTS")
    print("="*70)
    
    # Friend's results (from feedback PDF)
    friend_results = {
        'D2': {'count': 3728, 'precision': 0.982, 'recall': 0.919},
        'D3': {'count': 3016, 'precision': 0.979, 'recall': 0.887},
        'D4': {'count': 2598, 'precision': 0.953, 'recall': 0.817},
        'D5': {'count': 1898, 'precision': 0.918, 'recall': 0.675},
        'D6': {'count': 2582, 'precision': 0.849, 'recall': 0.560},
    }
    
    # Your results (from your output)
    your_results = {
        'D2': {'count': 3758},
        'D3': {'count': 2846},
        'D4': {'count': 2423},
        'D5': {'count': 2087},
        'D6': {'count': 2937},
    }
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = ['D2', 'D3', 'D4', 'D5', 'D6']
    x = np.arange(len(datasets))
    width = 0.35
    
    # 1. Spike Count Comparison
    ax = axes[0]
    friend_counts = [friend_results[d]['count'] for d in datasets]
    your_counts = [your_results[d]['count'] for d in datasets]
    
    bars1 = ax.bar(x - width/2, friend_counts, width, label="Friend's", color='#3498db')
    bars2 = ax.bar(x + width/2, your_counts, width, label='Yours', color='#e74c3c')
    
    ax.set_ylabel('Spike Count')
    ax.set_title('Detected Spike Counts', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    # 2. Friend's Precision/Recall
    ax = axes[1]
    friend_precision = [friend_results[d]['precision'] * 100 for d in datasets]
    friend_recall = [friend_results[d]['recall'] * 100 for d in datasets]
    
    ax.bar(x - width/2, friend_precision, width, label='Precision', color='#2ecc71')
    ax.bar(x + width/2, friend_recall, width, label='Recall', color='#9b59b6')
    
    ax.set_ylabel('Score (%)')
    ax.set_title("Friend's Detection Performance", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # 3. Key Observations
    ax = axes[2]
    ax.axis('off')
    
    observations = """
    KEY OBSERVATIONS
    ================
    
    Friend's approach:
    • Higher precision (fewer false positives)
    • Lower recall on D5/D6 (misses more spikes)
    • More conservative detection
    
    Your approach:
    • More spikes detected overall
    • Need to verify precision after submission
    
    D1 Validation shows:
    • Your CNN: 95.2% accuracy ✓
    • Class 5 weakest at 83.6%
    
    Areas to improve:
    1. Class 5 accuracy (widen tolerance?)
    2. D5/D6 may have too many false positives
    3. Consider hybrid detection for noisy data
    """
    
    ax.text(0.05, 0.95, observations, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax.transAxes)
    
    plt.suptitle('Your Results vs Friend\'s Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = ANALYSIS_DIR / 'comparison_with_friend.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_class5_errors(cnn=None):
    """
    Deep dive into Class 5 errors (lowest accuracy at 83.6%).
    """
    print("\n" + "="*70)
    print("CLASS 5 ERROR ANALYSIS")
    print("="*70)
    
    d1 = load_d1_ground_truth()
    d = d1['d']
    gt_indices = d1['Index']
    gt_classes = d1['Class']
    
    if cnn is None:
        cnn = CNNExperimentFixed()
        cnn.load_model()
    
    # Detect and classify
    detected_indices, detected_waveforms = detect_spikes_aligned(d, threshold_factor=4.5)
    
    import torch
    wf_norm, amp_features = cnn.prepare_data(detected_waveforms)
    raw_amp_features = cnn.extract_amplitude_features(detected_waveforms)
    
    cnn.model.eval()
    with torch.no_grad():
        X_wf = torch.FloatTensor(wf_norm).to(cnn.device)
        X_amp = torch.FloatTensor(amp_features).to(cnn.device)
        outputs = cnn.model(X_wf, X_amp)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = outputs.max(1)
        pred_classes = predicted.cpu().numpy() + 1
        confidences = probs.max(axis=1)
    
    # Match to ground truth
    tolerance = 50
    class5_correct = []
    class5_errors = []  # (true_class, pred_class, waveform, features)
    
    gt_used = set()
    for i, (det_idx, pred_cls) in enumerate(zip(detected_indices, pred_classes)):
        distances = np.abs(gt_indices - det_idx)
        closest = np.argmin(distances)
        
        if distances[closest] <= tolerance and closest not in gt_used:
            gt_cls = gt_classes[closest]
            gt_used.add(closest)
            
            if gt_cls == 5:
                if pred_cls == 5:
                    class5_correct.append({
                        'waveform': detected_waveforms[i],
                        'amp': raw_amp_features[i, 0],
                        'fwhm': raw_amp_features[i, 3],
                        'confidence': confidences[i]
                    })
                else:
                    class5_errors.append({
                        'pred_class': pred_cls,
                        'waveform': detected_waveforms[i],
                        'amp': raw_amp_features[i, 0],
                        'fwhm': raw_amp_features[i, 3],
                        'confidence': confidences[i]
                    })
    
    print(f"Class 5: {len(class5_correct)} correct, {len(class5_errors)} errors")
    
    # Error breakdown
    error_breakdown = Counter([e['pred_class'] for e in class5_errors])
    print(f"Class 5 misclassified as: {dict(error_breakdown)}")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # 1. Error breakdown pie chart
    ax = axes[0, 0]
    labels = [f'→Class {c}\n(n={cnt})' for c, cnt in sorted(error_breakdown.items())]
    sizes = [error_breakdown[c] for c in sorted(error_breakdown.keys())]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(sizes)]
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Class 5 Misclassifications', fontweight='bold')
    
    # 2. Amplitude comparison
    ax = axes[0, 1]
    correct_amps = [e['amp'] for e in class5_correct]
    error_amps = [e['amp'] for e in class5_errors]
    
    ax.hist(correct_amps, bins=20, alpha=0.5, label=f'Correct (n={len(correct_amps)})', color='green')
    ax.hist(error_amps, bins=20, alpha=0.5, label=f'Errors (n={len(error_amps)})', color='red')
    ax.set_xlabel('Peak Amplitude (V)')
    ax.set_ylabel('Count')
    ax.set_title('Amplitude Distribution', fontweight='bold')
    ax.legend()
    
    # 3. FWHM comparison
    ax = axes[0, 2]
    correct_fwhm = [e['fwhm'] for e in class5_correct]
    error_fwhm = [e['fwhm'] for e in class5_errors]
    
    ax.hist(correct_fwhm, bins=20, alpha=0.5, label='Correct', color='green')
    ax.hist(error_fwhm, bins=20, alpha=0.5, label='Errors', color='red')
    ax.set_xlabel('FWHM (ms)')
    ax.set_ylabel('Count')
    ax.set_title('FWHM Distribution', fontweight='bold')
    ax.legend()
    
    # 4-6. Sample waveforms for most common error
    most_common_error = error_breakdown.most_common(1)[0][0] if error_breakdown else 4
    
    # Correct Class 5 waveforms
    ax = axes[1, 0]
    if class5_correct:
        for e in class5_correct[:15]:
            ax.plot(e['waveform'], 'g-', alpha=0.3)
        mean_wf = np.mean([e['waveform'] for e in class5_correct], axis=0)
        ax.plot(mean_wf, 'g-', linewidth=3, label='Mean')
    ax.axvline(30, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Correctly Classified Class 5', fontweight='bold')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude (V)')
    
    # Misclassified as Class 4
    ax = axes[1, 1]
    c4_errors = [e for e in class5_errors if e['pred_class'] == 4]
    if c4_errors:
        for e in c4_errors[:15]:
            ax.plot(e['waveform'], 'r-', alpha=0.3)
        mean_wf = np.mean([e['waveform'] for e in c4_errors], axis=0)
        ax.plot(mean_wf, 'r-', linewidth=3, label='Mean')
    ax.axvline(30, color='blue', linestyle='--', alpha=0.5)
    ax.set_title(f'Class 5 → Class 4 (n={len(c4_errors)})', fontweight='bold')
    ax.set_xlabel('Sample')
    
    # Misclassified as Class 3
    ax = axes[1, 2]
    c3_errors = [e for e in class5_errors if e['pred_class'] == 3]
    if c3_errors:
        for e in c3_errors[:15]:
            ax.plot(e['waveform'], 'orange', alpha=0.3)
        mean_wf = np.mean([e['waveform'] for e in c3_errors], axis=0)
        ax.plot(mean_wf, 'orange', linewidth=3, label='Mean')
    ax.axvline(30, color='blue', linestyle='--', alpha=0.5)
    ax.set_title(f'Class 5 → Class 3 (n={len(c3_errors)})', fontweight='bold')
    ax.set_xlabel('Sample')
    
    plt.suptitle('Class 5 Error Analysis\n(Why 16.4% of Class 5 spikes are misclassified)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = ANALYSIS_DIR / 'class5_error_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run all analysis and visualization."""
    print("="*70)
    print("SPIKE CLASSIFICATION PIPELINE ANALYSIS")
    print("="*70)
    
    # 1. Analyze training data loss
    analyze_training_data_loss()
    
    # 2. D1 Validation visualization
    try:
        cnn = CNNExperimentFixed()
        cnn.load_model()
        create_d1_validation_visualization(cnn)
        
        # 3. Class 5 error analysis
        analyze_class5_errors(cnn)
    except Exception as e:
        print(f"Could not load CNN for visualization: {e}")
        print("Run with --train first: python cnn_experiment_fixed.py --train")
    
    # 4. Compare with friend
    compare_with_friend()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"All visualizations saved to: {ANALYSIS_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()