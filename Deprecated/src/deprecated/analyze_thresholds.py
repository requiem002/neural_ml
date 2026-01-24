"""Analyze threshold mismatch between D1 and D2."""

import numpy as np
import scipy.io as sio
from scipy import signal
from pathlib import Path

data_dir = Path(__file__).parent.parent / 'datasets'

print("="*70)
print("THRESHOLD ANALYSIS: D1 vs D2")
print("="*70)

# Load datasets
d1_data = sio.loadmat(data_dir / 'D1.mat')
d2_data = sio.loadmat(data_dir / 'D2.mat')

d1 = d1_data['d'].flatten()
d2 = d2_data['d'].flatten()
d1_index = d1_data['Index'].flatten()
d1_class = d1_data['Class'].flatten()

print(f"\nD1 Ground Truth Spikes: {len(d1_index)}")

# ============================================================
# STEP 1: Calculate noise levels using MAD estimator
# ============================================================
print("\n" + "-"*70)
print("STEP 1: Noise Level Analysis (MAD Estimator)")
print("-"*70)

# Apply bandpass filter (same as our detector)
sample_rate = 25000
nyquist = sample_rate / 2
low = 300 / nyquist
high = 3000 / nyquist
b, a = signal.butter(3, [low, high], btype='band')

d1_filtered = signal.filtfilt(b, a, d1)
d2_filtered = signal.filtfilt(b, a, d2)

# MAD-based noise estimation
d1_mad = np.median(np.abs(d1_filtered - np.median(d1_filtered)))
d1_sigma = d1_mad / 0.6745

d2_mad = np.median(np.abs(d2_filtered - np.median(d2_filtered)))
d2_sigma = d2_mad / 0.6745

print(f"\nD1 (80dB SNR):")
print(f"  MAD: {d1_mad:.6f}")
print(f"  Sigma (noise estimate): {d1_sigma:.6f}")

print(f"\nD2 (60dB SNR):")
print(f"  MAD: {d2_mad:.6f}")
print(f"  Sigma (noise estimate): {d2_sigma:.6f}")

print(f"\nNoise ratio D2/D1: {d2_sigma/d1_sigma:.2f}x")

# ============================================================
# STEP 2: Calculate effective voltage thresholds
# ============================================================
print("\n" + "-"*70)
print("STEP 2: Effective Voltage Thresholds")
print("-"*70)

# Current thresholds from our code
d1_thresh_factor = 5.0  # Used during training detection
d2_thresh_factor = 5.0  # Used in predict.py

d1_voltage_threshold = d1_thresh_factor * d1_sigma
d2_voltage_threshold = d2_thresh_factor * d2_sigma

print(f"\nD1 Threshold = {d1_thresh_factor} × {d1_sigma:.6f} = {d1_voltage_threshold:.6f}")
print(f"D2 Threshold = {d2_thresh_factor} × {d2_sigma:.6f} = {d2_voltage_threshold:.6f}")

# ============================================================
# STEP 3: Analyze ground truth spike amplitudes in D1
# ============================================================
print("\n" + "-"*70)
print("STEP 3: Ground Truth Spike Amplitudes (D1)")
print("-"*70)

# Get amplitudes of real spikes
gt_amplitudes = []
for idx in d1_index:
    idx_0 = int(idx) - 1  # Convert to 0-indexed
    # Get local maximum around spike location
    start = max(0, idx_0 - 10)
    end = min(len(d1_filtered), idx_0 + 10)
    local_max = np.max(d1_filtered[start:end])
    gt_amplitudes.append(local_max)

gt_amplitudes = np.array(gt_amplitudes)

print(f"\nGround Truth Spike Amplitudes (filtered signal):")
print(f"  Min: {np.min(gt_amplitudes):.4f}")
print(f"  Max: {np.max(gt_amplitudes):.4f}")
print(f"  Mean: {np.mean(gt_amplitudes):.4f}")
print(f"  Median: {np.median(gt_amplitudes):.4f}")
print(f"  Std: {np.std(gt_amplitudes):.4f}")

# How many ground truth spikes are BELOW our threshold?
below_threshold = np.sum(gt_amplitudes < d1_voltage_threshold)
print(f"\n  Spikes BELOW threshold ({d1_voltage_threshold:.4f}): {below_threshold} ({100*below_threshold/len(gt_amplitudes):.1f}%)")

# ============================================================
# STEP 4: Detect spikes and identify false positives
# ============================================================
print("\n" + "-"*70)
print("STEP 4: False Positive Analysis")
print("-"*70)

# Detect spikes with our current method
peaks, properties = signal.find_peaks(
    d1_filtered,
    height=d1_voltage_threshold,
    distance=30,
    prominence=d1_voltage_threshold * 0.5
)

print(f"\nDetected peaks (threshold={d1_thresh_factor}): {len(peaks)}")
print(f"Ground truth spikes: {len(d1_index)}")
print(f"Excess detections: {len(peaks) - len(d1_index)}")

# Match detected to ground truth
tolerance = 50
matched_det = set()
matched_gt = set()

for det_idx in peaks:
    det_idx_1 = det_idx + 1  # Convert to 1-indexed
    for gt_i, gt_idx in enumerate(d1_index):
        if gt_i not in matched_gt and abs(det_idx_1 - gt_idx) <= tolerance:
            matched_det.add(det_idx)
            matched_gt.add(gt_i)
            break

# False positives = detected but not matched
false_positive_indices = [p for p in peaks if p not in matched_det]
print(f"\nTrue Positives (matched): {len(matched_det)}")
print(f"False Positives: {len(false_positive_indices)}")
print(f"False Negatives (missed GT): {len(d1_index) - len(matched_gt)}")

# Analyze false positive amplitudes
if false_positive_indices:
    fp_amplitudes = d1_filtered[false_positive_indices]
    print(f"\nFalse Positive Amplitudes:")
    print(f"  Min: {np.min(fp_amplitudes):.4f}")
    print(f"  Max: {np.max(fp_amplitudes):.4f}")
    print(f"  Mean: {np.mean(fp_amplitudes):.4f}")
    print(f"  Median: {np.median(fp_amplitudes):.4f}")

    # Compare to real spike amplitudes
    print(f"\nComparison:")
    print(f"  Mean FP amplitude: {np.mean(fp_amplitudes):.4f}")
    print(f"  Mean GT amplitude: {np.mean(gt_amplitudes):.4f}")
    print(f"  Ratio FP/GT: {np.mean(fp_amplitudes)/np.mean(gt_amplitudes):.2f}")

# ============================================================
# STEP 5: What threshold would eliminate false positives?
# ============================================================
print("\n" + "-"*70)
print("STEP 5: Optimal Threshold Analysis")
print("-"*70)

# Find minimum GT amplitude
min_gt_amplitude = np.min(gt_amplitudes)
print(f"\nMinimum ground truth spike amplitude: {min_gt_amplitude:.4f}")

# What threshold factor would this be?
optimal_factor = min_gt_amplitude / d1_sigma
print(f"This corresponds to threshold factor: {optimal_factor:.2f}")

# Test different thresholds
print("\nThreshold sweep:")
print(f"{'Factor':<10} {'Threshold':<12} {'Detected':<12} {'FP':<8} {'FN':<8} {'Det F1':<10}")
print("-"*60)

for factor in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]:
    thresh = factor * d1_sigma

    peaks_test, _ = signal.find_peaks(
        d1_filtered,
        height=thresh,
        distance=30,
        prominence=thresh * 0.5
    )

    # Match
    matched_test = set()
    matched_gt_test = set()
    for det_idx in peaks_test:
        det_idx_1 = det_idx + 1
        for gt_i, gt_idx in enumerate(d1_index):
            if gt_i not in matched_gt_test and abs(det_idx_1 - gt_idx) <= tolerance:
                matched_test.add(det_idx)
                matched_gt_test.add(gt_i)
                break

    tp = len(matched_test)
    fp = len(peaks_test) - tp
    fn = len(d1_index) - len(matched_gt_test)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{factor:<10.1f} {thresh:<12.4f} {len(peaks_test):<12} {fp:<8} {fn:<8} {f1:<10.4f}")

# ============================================================
# STEP 6: Check D2 threshold vs D1 real spikes
# ============================================================
print("\n" + "-"*70)
print("STEP 6: D2 Threshold vs D1 Real Spike Amplitudes")
print("-"*70)

print(f"\nD2 voltage threshold: {d2_voltage_threshold:.4f}")
print(f"D1 min GT spike amplitude: {min_gt_amplitude:.4f}")
print(f"D1 mean GT spike amplitude: {np.mean(gt_amplitudes):.4f}")

# What percentile of D1 spikes would be missed by D2's threshold?
# (if applied to D1's filtered signal)
would_miss = np.sum(gt_amplitudes < d2_voltage_threshold)
print(f"\nIf we applied D2's threshold to D1:")
print(f"  Would miss {would_miss} spikes ({100*would_miss/len(gt_amplitudes):.1f}%)")

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)
print("""
1. The current threshold (5.0σ) is detecting noise peaks as spikes
2. False positives have lower amplitude than real spikes
3. A higher threshold (6.5-7.0σ) would reduce FPs while keeping most TPs
4. D2 has ~{:.1f}x more noise than D1, so its effective threshold is different
""".format(d2_sigma/d1_sigma))
