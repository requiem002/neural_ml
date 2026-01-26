#!/usr/bin/env python3
"""
V2 Pipeline Verification Script

Generates visual verification of the improved pipeline:
1. Average waveform plots per predicted class (D2, D6)
2. D1 confusion matrix (ground truth available)
3. Predicted class distributions for D2-D6
4. Spike temporal distribution plots
"""

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cnn_experiment_fixed import CNNExperiment
from generate_submissions_fixed import detect_spikes_mad

# Setup paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / 'analysis' / 'v2_verification'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def plot_average_waveforms(dataset_name, waveforms, classes, output_path):
    """
    Plot average waveform for each predicted class.
    Sharp, well-defined peaks indicate good alignment.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # D1 reference stats for comparison
    d1_stats = {
        1: {'amp': 4.89, 'fwhm': 0.57},
        2: {'amp': 5.51, 'fwhm': 0.75},
        3: {'amp': 1.49, 'fwhm': 0.49},
        4: {'amp': 2.86, 'fwhm': 0.63},
        5: {'amp': 4.20, 'fwhm': 0.87},
    }

    for c in range(1, 6):
        ax = axes.flat[c-1]
        mask = classes == c
        count = np.sum(mask)

        if count > 0:
            class_wf = waveforms[mask]
            avg_wf = np.mean(class_wf, axis=0)
            std_wf = np.std(class_wf, axis=0)

            x = np.arange(len(avg_wf))
            ax.plot(x, avg_wf, 'b-', linewidth=2, label='Mean')
            ax.fill_between(x, avg_wf - std_wf, avg_wf + std_wf, alpha=0.3, color='blue')

            # Mark the peak
            peak_idx = np.argmax(avg_wf)
            peak_val = avg_wf[peak_idx]
            ax.axvline(peak_idx, color='red', linestyle='--', alpha=0.5)
            ax.scatter([peak_idx], [peak_val], color='red', s=50, zorder=5)

            # Reference annotation
            ref_amp = d1_stats[c]['amp']
            ax.axhline(ref_amp, color='green', linestyle=':', alpha=0.5, label=f'D1 ref: {ref_amp:.1f}V')

            ax.set_title(f'Class {c}: n={count} ({100*count/len(classes):.1f}%)\n'
                        f'Peak={peak_val:.2f}V at sample {peak_idx}')
        else:
            ax.set_title(f'Class {c}: n=0')
            ax.text(0.5, 0.5, 'No spikes', ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude (V)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 8)

    # Summary in 6th panel
    ax = axes.flat[5]
    ax.axis('off')
    summary = f"{dataset_name} Summary\n" + "="*25 + "\n"
    for c in range(1, 6):
        mask = classes == c
        count = np.sum(mask)
        pct = 100 * count / len(classes)
        summary += f"Class {c}: {count:>4} ({pct:>5.1f}%)\n"
    summary += f"\nTotal: {len(classes)}"
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    fig.suptitle(f'{dataset_name} - Average Waveforms by Predicted Class\n'
                 f'(Peak should be at ~sample 41, matching D1 training alignment)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_d1_confusion_matrix(cnn, output_path):
    """
    Test the pipeline on D1 where we have ground truth.
    This validates that our detection + classification works.
    """
    print("\n" + "="*60)
    print("D1 CONFUSION MATRIX (Ground Truth Validation)")
    print("="*60)

    # Load D1 with ground truth
    data = sio.loadmat(BASE_DIR / 'datasets' / 'D1.mat')
    d = data['d'].flatten()
    gt_indices = data['Index'].flatten()
    gt_classes = data['Class'].flatten()

    print(f"D1 Ground Truth: {len(gt_indices)} spikes")

    # Now detect spikes using our V2 pipeline (pretend we don't have labels)
    # Use similar threshold as D2 (D1 is clean data)
    detected_indices, detected_waveforms = detect_spikes_mad(
        d, threshold_factor=4.5, align_peak_at=41  # Conservative for clean data
    )

    print(f"V2 Pipeline Detected: {len(detected_indices)} spikes")

    # Classify with CNN
    wf_norm, amp_features = cnn.prepare_data(detected_waveforms)

    import torch
    cnn.model.eval()
    X_wf = torch.FloatTensor(wf_norm).to(cnn.device)
    X_amp = torch.FloatTensor(amp_features).to(cnn.device)

    with torch.no_grad():
        outputs = cnn.model(X_wf, X_amp)
        _, predicted = outputs.max(1)
        pred_classes = predicted.cpu().numpy() + 1

    # Match detected spikes to ground truth (within 50 samples tolerance)
    tolerance = 50
    matched_gt = []
    matched_pred = []

    gt_used = set()
    for det_idx, pred_cls in zip(detected_indices, pred_classes):
        # Find closest ground truth spike
        distances = np.abs(gt_indices - det_idx)
        closest_idx = np.argmin(distances)

        if distances[closest_idx] <= tolerance and closest_idx not in gt_used:
            matched_gt.append(gt_classes[closest_idx])
            matched_pred.append(pred_cls)
            gt_used.add(closest_idx)

    matched_gt = np.array(matched_gt)
    matched_pred = np.array(matched_pred)

    print(f"Matched spikes: {len(matched_gt)} / {len(gt_indices)} ({100*len(matched_gt)/len(gt_indices):.1f}%)")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(matched_gt, matched_pred, labels=[1, 2, 3, 4, 5])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix heatmap
    ax = axes[0]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'D1 Confusion Matrix\n(Detection Recall: {100*len(matched_gt)/len(gt_indices):.1f}%)')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_yticklabels([1, 2, 3, 4, 5])

    # Add numbers to cells
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, cm[i, j], ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.colorbar(im, ax=ax)

    # Per-class accuracy bar chart
    ax = axes[1]
    class_acc = []
    for c in range(5):
        total = cm[c].sum()
        correct = cm[c, c]
        acc = 100 * correct / total if total > 0 else 0
        class_acc.append(acc)

    bars = ax.bar(range(1, 6), class_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Classification Accuracy')
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 100)
    ax.axhline(90, color='green', linestyle='--', alpha=0.5, label='90% target')
    ax.legend()

    for bar, acc in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    # Overall accuracy
    overall_acc = 100 * np.trace(cm) / cm.sum()
    fig.suptitle(f'D1 Validation - Overall Classification Accuracy: {overall_acc:.1f}%',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(matched_gt, matched_pred, labels=[1,2,3,4,5], digits=3))

    return overall_acc


def plot_predicted_distributions(output_path):
    """
    Plot predicted class distributions for all datasets.
    Key check: Class 2 should be >15% if the "Class 2 destruction" is fixed.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # D1 reference distribution
    d1_data = sio.loadmat(BASE_DIR / 'datasets' / 'D1.mat')
    d1_classes = d1_data['Class'].flatten()
    d1_dist = [np.sum(d1_classes == c) / len(d1_classes) * 100 for c in range(1, 6)]

    datasets = ['D2', 'D3', 'D4', 'D5', 'D6']

    for idx, ds in enumerate(datasets):
        ax = axes.flat[idx]

        # Load submission
        sub_data = sio.loadmat(BASE_DIR / 'submissions' / f'{ds}.mat')
        classes = sub_data['Class'].flatten()

        dist = [np.sum(classes == c) / len(classes) * 100 for c in range(1, 6)]
        counts = [np.sum(classes == c) for c in range(1, 6)]

        x = np.arange(5)
        width = 0.35

        bars1 = ax.bar(x - width/2, d1_dist, width, label='D1 Reference', alpha=0.7, color='gray')
        bars2 = ax.bar(x + width/2, dist, width, label=f'{ds} Predicted', color='steelblue')

        ax.set_xlabel('Class')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{ds}: n={len(classes)} spikes')
        ax.set_xticks(x)
        ax.set_xticklabels([1, 2, 3, 4, 5])
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 50)

        # Add count labels
        for bar, cnt in zip(bars2, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{cnt}', ha='center', va='bottom', fontsize=8)

        # Highlight Class 2
        c2_pct = dist[1]
        c2_count = counts[1]
        if c2_pct > 15:
            bars2[1].set_color('green')
            ax.text(1 + width/2, c2_pct + 3, 'OK!', ha='center', color='green', fontweight='bold')
        else:
            bars2[1].set_color('red')
            ax.text(1 + width/2, c2_pct + 3, 'LOW', ha='center', color='red', fontweight='bold')

    # Summary panel
    ax = axes.flat[5]
    ax.axis('off')

    summary = "CLASS 2 VERIFICATION\n" + "="*30 + "\n\n"
    summary += "Target: >15% (~600+ spikes)\n\n"

    for ds in datasets:
        sub_data = sio.loadmat(BASE_DIR / 'submissions' / f'{ds}.mat')
        classes = sub_data['Class'].flatten()
        c2_count = np.sum(classes == 2)
        c2_pct = 100 * c2_count / len(classes)
        status = "PASS" if c2_pct > 15 else "FAIL"
        summary += f"{ds}: {c2_count:>4} ({c2_pct:>5.1f}%) - {status}\n"

    ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    fig.suptitle('Predicted Class Distributions vs D1 Reference\n'
                 '(Class 2 should be >15% to confirm fix)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_spike_temporal_distribution(dataset_name, output_path):
    """
    Plot temporal distribution of detected spikes.
    Should be relatively uniform across the recording.
    """
    # Load submission
    sub_data = sio.loadmat(BASE_DIR / 'submissions' / f'{dataset_name}.mat')
    indices = sub_data['Index'].flatten()
    classes = sub_data['Class'].flatten()

    # Recording parameters
    sample_rate = 25000
    total_samples = 1440000
    duration_sec = total_samples / sample_rate

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Spike raster plot (time vs class)
    ax = axes[0, 0]
    times = indices / sample_rate
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for c in range(1, 6):
        mask = classes == c
        ax.scatter(times[mask], classes[mask], c=colors[c-1], s=1, alpha=0.5, label=f'C{c}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Class')
    ax.set_title('Spike Raster Plot')
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc='upper right', markerscale=5)

    # 2. Spike rate histogram (binned over time)
    ax = axes[0, 1]
    bin_width = 1.0  # 1 second bins
    bins = np.arange(0, duration_sec + bin_width, bin_width)
    ax.hist(times, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spike Count')
    ax.set_title(f'Spike Rate (1s bins)\nMean: {len(indices)/duration_sec:.1f} spikes/s')
    ax.axhline(len(indices)/len(bins), color='red', linestyle='--', label='Mean')
    ax.legend()

    # 3. Inter-spike interval histogram
    ax = axes[1, 0]
    if len(indices) > 1:
        isi = np.diff(sorted(indices)) / sample_rate * 1000  # Convert to ms
        isi = isi[isi < 100]  # Focus on ISIs < 100ms
        ax.hist(isi, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Inter-Spike Interval (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'ISI Distribution\nMedian: {np.median(isi):.2f} ms')
        ax.axvline(1.2, color='red', linestyle='--', label='Refractory (1.2ms)')
        ax.legend()

    # 4. Class distribution pie chart
    ax = axes[1, 1]
    counts = [np.sum(classes == c) for c in range(1, 6)]
    labels = [f'C{c}\n{cnt}\n({100*cnt/len(classes):.1f}%)' for c, cnt in enumerate(counts, 1)]
    ax.pie(counts, labels=labels, colors=colors, autopct='', startangle=90)
    ax.set_title(f'Class Distribution (n={len(classes)})')

    fig.suptitle(f'{dataset_name} Spike Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_waveform_quality_check(dataset_name, cnn, output_path):
    """
    Detailed waveform quality check:
    - Shows individual waveforms for each class
    - Compares to D1 templates
    """
    # Load dataset
    data = sio.loadmat(BASE_DIR / 'datasets' / f'{dataset_name}.mat')
    d = data['d'].flatten()

    # Load submission
    sub_data = sio.loadmat(BASE_DIR / 'submissions' / f'{dataset_name}.mat')
    indices = sub_data['Index'].flatten()
    classes = sub_data['Class'].flatten()

    # Extract waveforms with proper alignment
    window_before, window_after = 30, 30
    align_peak_at = 41
    waveform_size = window_before + window_after

    waveforms = []
    for idx in indices:
        peak = idx - 1  # Convert to 0-indexed
        start = peak - align_peak_at
        end = start + waveform_size
        if start >= 0 and end <= len(d):
            waveforms.append(d[start:end])
        else:
            waveforms.append(np.zeros(waveform_size))
    waveforms = np.array(waveforms)

    # Load D1 templates
    templates = np.load(BASE_DIR / 'models' / 'd1_templates.npy', allow_pickle=True).item()

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    for c in range(1, 6):
        mask = classes == c
        class_wf = waveforms[mask]

        # Top row: Sample waveforms (max 20)
        ax = axes[0, c-1]
        n_samples = min(20, len(class_wf))
        if n_samples > 0:
            for i in range(n_samples):
                ax.plot(class_wf[i], alpha=0.3, color='blue')
            ax.plot(np.mean(class_wf, axis=0), 'r-', linewidth=2, label='Mean')
        ax.set_title(f'Class {c} Waveforms (n={len(class_wf)})')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude (V)')
        ax.axvline(41, color='green', linestyle='--', alpha=0.5)
        ax.set_ylim(-3, 9)

        # Bottom row: Comparison with D1 template
        ax = axes[1, c-1]
        if c in templates:
            template = templates[c]
            ax.plot(template, 'g-', linewidth=2, label='D1 Template')
        if len(class_wf) > 0:
            avg_wf = np.mean(class_wf, axis=0)
            ax.plot(avg_wf, 'b-', linewidth=2, label=f'{dataset_name} Mean')
        ax.set_title(f'Class {c}: Template Comparison')
        ax.set_xlabel('Sample')
        ax.legend(fontsize=8)
        ax.axvline(41, color='red', linestyle='--', alpha=0.5, label='Expected Peak')
        ax.set_ylim(-3, 9)

    fig.suptitle(f'{dataset_name} Waveform Quality Check\n'
                 f'(Green dashed line = expected peak position at sample 41)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("="*70)
    print("V2 PIPELINE VERIFICATION")
    print("="*70)

    # Load CNN model
    print("\nLoading CNN model...")
    cnn = CNNExperiment()
    cnn.load_model()

    # 1. D1 Confusion Matrix (Ground Truth Validation)
    print("\n" + "="*70)
    print("1. D1 CONFUSION MATRIX")
    print("="*70)
    d1_acc = generate_d1_confusion_matrix(cnn, ANALYSIS_DIR / 'd1_confusion_matrix.png')

    # 2. Average Waveforms for D2 and D6
    print("\n" + "="*70)
    print("2. AVERAGE WAVEFORM PLOTS")
    print("="*70)

    for ds in ['D2', 'D6']:
        print(f"\nProcessing {ds}...")
        data = sio.loadmat(BASE_DIR / 'datasets' / f'{ds}.mat')
        d = data['d'].flatten()

        sub_data = sio.loadmat(BASE_DIR / 'submissions' / f'{ds}.mat')
        indices = sub_data['Index'].flatten()
        classes = sub_data['Class'].flatten()

        # Extract waveforms with proper alignment
        waveforms = []
        for idx in indices:
            peak = idx - 1  # Convert to 0-indexed
            start = peak - 41  # Align peak at sample 41
            end = start + 60
            if start >= 0 and end <= len(d):
                waveforms.append(d[start:end])
            else:
                waveforms.append(np.zeros(60))
        waveforms = np.array(waveforms)

        plot_average_waveforms(ds, waveforms, classes,
                              ANALYSIS_DIR / f'{ds.lower()}_average_waveforms.png')

    # 3. Predicted Class Distributions
    print("\n" + "="*70)
    print("3. PREDICTED CLASS DISTRIBUTIONS")
    print("="*70)
    plot_predicted_distributions(ANALYSIS_DIR / 'predicted_distributions.png')

    # 4. Spike Temporal Distribution for all datasets
    print("\n" + "="*70)
    print("4. SPIKE TEMPORAL DISTRIBUTIONS")
    print("="*70)
    for ds in ['D2', 'D3', 'D4', 'D5', 'D6']:
        print(f"\nProcessing {ds}...")
        plot_spike_temporal_distribution(ds, ANALYSIS_DIR / f'{ds.lower()}_temporal_distribution.png')

    # 5. Waveform Quality Check for D2 and D6
    print("\n" + "="*70)
    print("5. WAVEFORM QUALITY CHECK")
    print("="*70)
    for ds in ['D2', 'D6']:
        print(f"\nProcessing {ds}...")
        plot_waveform_quality_check(ds, cnn, ANALYSIS_DIR / f'{ds.lower()}_waveform_quality.png')

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print(f"\nAll plots saved to: {ANALYSIS_DIR}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nD1 Classification Accuracy: {d1_acc:.1f}%")

    print("\nClass 2 counts in submissions:")
    for ds in ['D2', 'D3', 'D4', 'D5', 'D6']:
        sub_data = sio.loadmat(BASE_DIR / 'submissions' / f'{ds}.mat')
        classes = sub_data['Class'].flatten()
        c2_count = np.sum(classes == 2)
        c2_pct = 100 * c2_count / len(classes)
        status = "PASS" if c2_pct > 15 else "FAIL"
        print(f"  {ds}: {c2_count:>4} spikes ({c2_pct:>5.1f}%) - {status}")


if __name__ == '__main__':
    main()
