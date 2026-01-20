"""
Forensic Validation of CNN Spike Classification

This script performs deep analysis to:
1. Validate CNN performance against D1 ground truth
2. Identify systematic errors and weak points
3. Analyze amplitude distributions and class separability
4. Compare different classification approaches
5. Generate diagnostic plots

Marking scheme understanding:
- F_Dataset = 0.3 × F_Ident + 0.7 × F_Class (classification is 70%!)
- F_Final = 0.1×D2 + 0.15×D3 + 0.2×D4 + 0.25×D5 + 0.3×D6
- D5 + D6 = 55% of final score!
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spike_detector import detect_spikes
from feature_extractor import extract_waveforms_at_indices


def load_d1_data():
    """Load D1 dataset with ground truth."""
    data_dir = Path(__file__).parent.parent / 'datasets'
    data = sio.loadmat(data_dir / 'D1.mat')
    d = data['d'].flatten()
    index = data['Index'].flatten()
    classes = data['Class'].flatten()
    return d, index, classes


def load_dataset(name):
    """Load any dataset."""
    data_dir = Path(__file__).parent.parent / 'datasets'
    data = sio.loadmat(data_dir / f'{name}.mat')
    d = data['d'].flatten()
    return d


def extract_amplitude_features_detailed(waveforms, sample_rate=25000):
    """Extract detailed amplitude features for analysis."""
    from scipy import signal as sig

    nyquist = sample_rate / 2
    low = 300 / nyquist
    high = 3000 / nyquist
    b, a = sig.butter(2, [low, high], btype='band')

    features = []
    for wf in waveforms:
        try:
            wf_filtered = sig.filtfilt(b, a, wf, padlen=min(15, len(wf)-1))
        except ValueError:
            wf_filtered = wf

        baseline_raw = np.mean(wf[:5])
        wf_raw = wf - baseline_raw

        baseline_filt = np.mean(wf_filtered[:5])
        wf_filt = wf_filtered - baseline_filt

        peak_idx = np.argmax(wf_filt)
        peak_amp_raw = wf_raw[peak_idx]
        peak_amp_filt = wf_filt[peak_idx]

        trough_raw = np.min(wf_raw)
        trough_filt = np.min(wf_filt)

        energy_raw = np.sum(wf_raw ** 2)
        energy_filt = np.sum(wf_filt ** 2)

        p2p_raw = np.max(wf_raw) - np.min(wf_raw)
        p2p_filt = np.max(wf_filt) - np.min(wf_filt)

        features.append({
            'peak_amp_raw': peak_amp_raw,
            'peak_amp_filt': peak_amp_filt,
            'trough_raw': trough_raw,
            'trough_filt': trough_filt,
            'energy_raw': energy_raw,
            'energy_filt': energy_filt,
            'p2p_raw': p2p_raw,
            'p2p_filt': p2p_filt,
            'peak_idx': peak_idx,
        })

    return features


def analyze_d1_ground_truth():
    """Comprehensive analysis of D1 ground truth."""
    print("=" * 80)
    print("FORENSIC ANALYSIS: D1 Ground Truth")
    print("=" * 80)

    d, index, classes = load_d1_data()

    # Extract waveforms
    waveforms, valid_indices = extract_waveforms_at_indices(d, index, window_before=30, window_after=30)
    valid_mask = np.isin(index, valid_indices)
    valid_classes = classes[valid_mask]

    print(f"\nTotal spikes: {len(valid_classes)}")
    print("\nClass distribution:")
    for c in range(1, 6):
        count = np.sum(valid_classes == c)
        print(f"  Class {c}: {count} ({100*count/len(valid_classes):.1f}%)")

    # Extract features by class
    features_by_class = {c: [] for c in range(1, 6)}
    for wf, cls in zip(waveforms, valid_classes):
        feat = extract_amplitude_features_detailed([wf])[0]
        features_by_class[cls].append(feat)

    # Analyze amplitude ranges by class
    print("\n" + "-" * 80)
    print("AMPLITUDE ANALYSIS BY CLASS")
    print("-" * 80)

    class_stats = {}
    for c in range(1, 6):
        feats = features_by_class[c]
        peak_amps = [f['peak_amp_raw'] for f in feats]
        p2p_amps = [f['p2p_raw'] for f in feats]

        class_stats[c] = {
            'peak_mean': np.mean(peak_amps),
            'peak_std': np.std(peak_amps),
            'peak_min': np.min(peak_amps),
            'peak_max': np.max(peak_amps),
            'p2p_mean': np.mean(p2p_amps),
            'p2p_std': np.std(p2p_amps),
        }

        print(f"\nClass {c}:")
        print(f"  Peak amplitude: {class_stats[c]['peak_mean']:.2f} ± {class_stats[c]['peak_std']:.2f} V")
        print(f"  Peak range: [{class_stats[c]['peak_min']:.2f}, {class_stats[c]['peak_max']:.2f}] V")
        print(f"  Peak-to-Peak: {class_stats[c]['p2p_mean']:.2f} ± {class_stats[c]['p2p_std']:.2f} V")

    # Analyze class separability
    print("\n" + "-" * 80)
    print("CLASS SEPARABILITY ANALYSIS")
    print("-" * 80)

    # Find overlapping amplitude ranges
    print("\nAmplitude overlap analysis:")
    for c1 in range(1, 6):
        for c2 in range(c1+1, 6):
            range1 = (class_stats[c1]['peak_mean'] - 2*class_stats[c1]['peak_std'],
                     class_stats[c1]['peak_mean'] + 2*class_stats[c1]['peak_std'])
            range2 = (class_stats[c2]['peak_mean'] - 2*class_stats[c2]['peak_std'],
                     class_stats[c2]['peak_mean'] + 2*class_stats[c2]['peak_std'])

            overlap = max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))
            total = max(range1[1], range2[1]) - min(range1[0], range2[0])
            overlap_pct = 100 * overlap / total if total > 0 else 0

            if overlap_pct > 10:
                print(f"  Class {c1} vs {c2}: {overlap_pct:.1f}% overlap")

    return class_stats, features_by_class, waveforms, valid_classes


def analyze_test_dataset_amplitudes(dataset_name, voltage_threshold):
    """Analyze amplitude distribution in test dataset."""
    print(f"\n{'='*80}")
    print(f"AMPLITUDE ANALYSIS: {dataset_name}")
    print(f"{'='*80}")

    d = load_dataset(dataset_name)

    # Detect spikes
    indices, waveforms = detect_spikes(
        d, voltage_threshold=voltage_threshold,
        window_before=30, window_after=30
    )

    print(f"Detected {len(indices)} spikes (threshold={voltage_threshold}V)")

    if len(indices) == 0:
        return None

    # Extract features
    features = extract_amplitude_features_detailed(waveforms)
    peak_amps = [f['peak_amp_raw'] for f in features]

    print(f"\nAmplitude statistics:")
    print(f"  Mean: {np.mean(peak_amps):.2f}V")
    print(f"  Std: {np.std(peak_amps):.2f}V")
    print(f"  Min: {np.min(peak_amps):.2f}V")
    print(f"  Max: {np.max(peak_amps):.2f}V")
    print(f"  Median: {np.median(peak_amps):.2f}V")

    # Bin by expected class ranges
    # Class 3: ~2.3V, Class 4: ~3.6V, Class 5: ~5.0V, Class 1: ~5.7V, Class 2: ~6.1V
    bins = [0, 3.0, 4.3, 5.4, 5.9, 100]
    labels = ['Low (<3V)', 'Class3-4 (3-4.3V)', 'Class5 (4.3-5.4V)', 'Class1 (5.4-5.9V)', 'Class2 (>5.9V)']

    print(f"\nAmplitude distribution by expected class range:")
    for i in range(len(bins)-1):
        count = np.sum((np.array(peak_amps) >= bins[i]) & (np.array(peak_amps) < bins[i+1]))
        print(f"  {labels[i]}: {count} ({100*count/len(peak_amps):.1f}%)")

    return {'indices': indices, 'waveforms': waveforms, 'features': features}


def test_cnn_on_d1():
    """Test CNN performance on D1 ground truth."""
    print("\n" + "=" * 80)
    print("CNN VALIDATION ON D1 GROUND TRUTH")
    print("=" * 80)

    import pickle
    import torch

    # Load D1
    d, index, classes = load_d1_data()
    waveforms, valid_indices = extract_waveforms_at_indices(d, index, window_before=30, window_after=30)
    valid_mask = np.isin(index, valid_indices)
    valid_classes = classes[valid_mask]

    # Load CNN model
    model_path = Path(__file__).parent.parent / 'models' / 'cnn_model.pkl'
    if not model_path.exists():
        print("CNN model not found!")
        return None

    with open(model_path, 'rb') as f:
        state = pickle.load(f)

    # We need to import the CNN experiment module
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from cnn_experiment import CNNExperiment

    experiment = CNNExperiment()
    experiment.load_model()

    # Prepare data
    wf_norm, amp_features = experiment.prepare_data(waveforms)

    # Predict
    experiment.model.eval()
    X_wf = torch.FloatTensor(wf_norm).to(experiment.device)
    X_amp = torch.FloatTensor(amp_features).to(experiment.device)

    with torch.no_grad():
        outputs = experiment.model(X_wf, X_amp)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        pred_classes = predicted.cpu().numpy() + 1

    # Evaluate WITHOUT amplitude correction
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    print("\n--- CNN Predictions (NO amplitude correction) ---")
    print(f"Accuracy: {100*np.mean(pred_classes == valid_classes):.2f}%")
    print(f"F1 (weighted): {f1_score(valid_classes, pred_classes, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(valid_classes, pred_classes))

    # Now apply amplitude correction
    raw_amp = experiment.extract_amplitude_features(waveforms)[:, 0]
    class_amp_centers = {1: 5.69, 2: 6.13, 3: 2.30, 4: 3.57, 5: 4.96}

    corrected_classes = pred_classes.copy()
    for i, (pred_class, amp) in enumerate(zip(pred_classes, raw_amp)):
        if pred_class == 3 and amp > 5.5:
            best_class = min([1, 2, 5], key=lambda c: abs(class_amp_centers[c] - amp))
            corrected_classes[i] = best_class
        elif pred_class == 4 and amp > 7.0:
            best_class = min([1, 2], key=lambda c: abs(class_amp_centers[c] - amp))
            corrected_classes[i] = best_class

    print("\n--- CNN Predictions (WITH amplitude correction) ---")
    print(f"Accuracy: {100*np.mean(corrected_classes == valid_classes):.2f}%")
    print(f"F1 (weighted): {f1_score(valid_classes, corrected_classes, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(valid_classes, corrected_classes))

    # Per-class analysis
    print("\nPer-class F1 scores:")
    f1_per_class = f1_score(valid_classes, corrected_classes, average=None)
    for c, f1 in enumerate(f1_per_class, 1):
        print(f"  Class {c}: {f1:.4f}")

    # Analyze misclassifications
    print("\n--- MISCLASSIFICATION ANALYSIS ---")
    for true_class in range(1, 6):
        mask = valid_classes == true_class
        preds = corrected_classes[mask]
        misclass = preds != true_class
        if np.any(misclass):
            print(f"\nClass {true_class} misclassified as:")
            for pred_class in range(1, 6):
                if pred_class != true_class:
                    count = np.sum(preds == pred_class)
                    if count > 0:
                        print(f"  Class {pred_class}: {count} ({100*count/len(preds):.1f}%)")

    return {
        'true_classes': valid_classes,
        'pred_classes': pred_classes,
        'corrected_classes': corrected_classes,
        'probs': probs.cpu().numpy(),
        'raw_amp': raw_amp
    }


def generate_diagnostic_plots(class_stats, features_by_class, waveforms, valid_classes):
    """Generate comprehensive diagnostic plots."""
    output_dir = Path(__file__).parent

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}

    # Plot 1: Amplitude distribution by class
    ax = axes[0, 0]
    for c in range(1, 6):
        peak_amps = [f['peak_amp_raw'] for f in features_by_class[c]]
        ax.hist(peak_amps, bins=30, alpha=0.5, color=colors[c], label=f'Class {c}', density=True)
    ax.set_xlabel('Peak Amplitude (V)')
    ax.set_ylabel('Density')
    ax.set_title('D1 Ground Truth: Amplitude Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Class templates (mean waveforms)
    ax = axes[0, 1]
    time_ms = np.arange(60) / 25  # 60 samples at 25kHz
    for c in range(1, 6):
        class_wfs = waveforms[valid_classes == c]
        template = np.mean(class_wfs, axis=0)
        baseline = np.mean(template[:5])
        ax.plot(time_ms, template - baseline, color=colors[c], linewidth=2, label=f'Class {c}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Class Templates (Mean Waveforms)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Normalized templates (shape comparison)
    ax = axes[0, 2]
    for c in range(1, 6):
        class_wfs = waveforms[valid_classes == c]
        template = np.mean(class_wfs, axis=0)
        baseline = np.mean(template[:5])
        template_centered = template - baseline
        peak = np.max(np.abs(template_centered))
        template_norm = template_centered / peak
        ax.plot(time_ms, template_norm, color=colors[c], linewidth=2, label=f'Class {c}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Voltage')
    ax.set_title('Normalized Templates (Shape Only)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Amplitude vs Energy scatter
    ax = axes[1, 0]
    for c in range(1, 6):
        peak_amps = [f['peak_amp_raw'] for f in features_by_class[c]]
        energies = [f['energy_raw'] for f in features_by_class[c]]
        ax.scatter(peak_amps, energies, c=colors[c], alpha=0.5, label=f'Class {c}', s=10)
    ax.set_xlabel('Peak Amplitude (V)')
    ax.set_ylabel('Energy')
    ax.set_title('Amplitude vs Energy by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Class amplitude boxplot
    ax = axes[1, 1]
    data = [[f['peak_amp_raw'] for f in features_by_class[c]] for c in range(1, 6)]
    bp = ax.boxplot(data, labels=['C1', 'C2', 'C3', 'C4', 'C5'], patch_artist=True)
    for patch, c in zip(bp['boxes'], range(1, 6)):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.7)
    ax.set_ylabel('Peak Amplitude (V)')
    ax.set_title('Amplitude Range by Class')
    ax.grid(True, alpha=0.3)

    # Plot 6: Decision boundaries visualization
    ax = axes[1, 2]
    # Show amplitude thresholds for classification
    all_amps = []
    all_classes = []
    for c in range(1, 6):
        amps = [f['peak_amp_raw'] for f in features_by_class[c]]
        all_amps.extend(amps)
        all_classes.extend([c] * len(amps))

    # Sort by amplitude
    sorted_idx = np.argsort(all_amps)
    sorted_amps = np.array(all_amps)[sorted_idx]
    sorted_classes = np.array(all_classes)[sorted_idx]

    ax.scatter(range(len(sorted_amps)), sorted_amps, c=[colors[c] for c in sorted_classes], s=2)
    ax.axhline(y=3.0, color='gray', linestyle='--', label='Threshold 3V')
    ax.axhline(y=4.5, color='gray', linestyle='--', label='Threshold 4.5V')
    ax.axhline(y=5.5, color='gray', linestyle='--', label='Threshold 5.5V')
    ax.set_xlabel('Spike Index (sorted by amplitude)')
    ax.set_ylabel('Peak Amplitude (V)')
    ax.set_title('Amplitude-sorted Spikes with Decision Thresholds')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'forensic_validation_plots.png', dpi=150)
    plt.close()
    print(f"\nDiagnostic plots saved to: {output_dir / 'forensic_validation_plots.png'}")


def analyze_noise_characteristics():
    """Analyze noise levels in each dataset."""
    print("\n" + "=" * 80)
    print("NOISE CHARACTERISTICS ANALYSIS")
    print("=" * 80)

    datasets = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

    for name in datasets:
        d = load_dataset(name)

        # Estimate noise using MAD
        mad = np.median(np.abs(d - np.median(d)))
        sigma = mad / 0.6745

        # Also compute std
        std = np.std(d)

        print(f"\n{name}:")
        print(f"  MAD-based sigma: {sigma:.4f}V")
        print(f"  Standard deviation: {std:.4f}V")
        print(f"  Estimated SNR: {20*np.log10(np.max(np.abs(d))/sigma):.1f} dB")


def main():
    print("=" * 80)
    print("FORENSIC VALIDATION OF CNN SPIKE CLASSIFICATION")
    print("=" * 80)

    # 1. Analyze D1 ground truth
    class_stats, features_by_class, waveforms, valid_classes = analyze_d1_ground_truth()

    # 2. Generate diagnostic plots
    generate_diagnostic_plots(class_stats, features_by_class, waveforms, valid_classes)

    # 3. Test CNN on D1
    cnn_results = test_cnn_on_d1()

    # 4. Analyze noise characteristics
    analyze_noise_characteristics()

    # 5. Analyze test dataset amplitudes
    thresholds = {'D2': 0.8, 'D3': 0.95, 'D4': 1.5, 'D5': 2.8, 'D6': 4.0}
    for name, thresh in thresholds.items():
        analyze_test_dataset_amplitudes(name, thresh)

    print("\n" + "=" * 80)
    print("FORENSIC VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
