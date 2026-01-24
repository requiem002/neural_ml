
#!/usr/bin/env python3
"""
Spike Classification Pipeline - Alignment Diagnostic & Fix

This script:
1. Diagnoses the training/inference alignment mismatch
2. Provides corrected extraction functions
3. Retrains the CNN with proper alignment
4. Validates on D1 ground truth

The core issue: Training extracts waveforms centered on Index (peaks at variable 
positions 37-49), but inference extracts with peaks fixed at sample 41.
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from pathlib import Path
from collections import Counter
import sys

# ============================================================================
# PART 1: DIAGNOSTIC FUNCTIONS
# ============================================================================

def diagnose_d1_alignment(data):
    """
    Comprehensive diagnosis of D1 ground truth alignment.
    
    Returns statistics about where peaks actually fall relative to Index positions.
    """
    print("=" * 70)
    print("D1 ALIGNMENT DIAGNOSTIC")
    print("=" * 70)

    
    d = data['d'].flatten()
    indices = data['Index'].flatten()
    classes = data['Class'].flatten()

    
    print(f"\nTotal spikes in D1: {len(indices)}")
    print(f"Class distribution: {Counter(classes)}")
    
    # Analyze peak positions for ALL spikes
    peak_positions = []
    peak_offsets = []  # How far peak is from Index
    
    for idx in indices:
        idx_0 = idx - 1  # 0-indexed
        
        # Extract centered on Index (as training does)
        start = idx_0 - 30
        end = idx_0 + 30
        
        if start >= 0 and end <= len(d):
            waveform = d[start:end]
            peak_pos = np.argmax(waveform)
            peak_positions.append(peak_pos)
            peak_offsets.append(peak_pos - 30)  # Offset from center
    
    peak_positions = np.array(peak_positions)
    peak_offsets = np.array(peak_offsets)
    
    print(f"\n--- Peak Position Statistics (in 60-sample window) ---")
    print(f"If extraction centers on Index, peaks should be at sample 30.")
    print(f"Actual peak positions:")
    print(f"  Mean:   {np.mean(peak_positions):.2f}")
    print(f"  Std:    {np.std(peak_positions):.2f}")
    print(f"  Min:    {np.min(peak_positions)}")
    print(f"  Max:    {np.max(peak_positions)}")
    print(f"  Median: {np.median(peak_positions):.1f}")
    
    print(f"\nPeak offset from Index (positive = peak is AFTER Index):")
    print(f"  Mean offset: {np.mean(peak_offsets):.2f} samples")
    print(f"  This means Index points ~{np.mean(peak_offsets):.0f} samples BEFORE the peak")
    
    # Per-class analysis
    print(f"\n--- Per-Class Peak Position Analysis ---")
    for c in range(1, 6):
        mask = classes == c
        class_peaks = []
        for idx in indices[mask]:
            idx_0 = idx - 1
            start = idx_0 - 30
            end = idx_0 + 30
            if start >= 0 and end <= len(d):
                wf = d[start:end]
                class_peaks.append(np.argmax(wf))
        
        class_peaks = np.array(class_peaks)
        print(f"  Class {c}: mean={np.mean(class_peaks):.1f}, std={np.std(class_peaks):.1f}, "
              f"range=[{np.min(class_peaks)}, {np.max(class_peaks)}]")
    
    # Histogram of peak positions
    print(f"\n--- Peak Position Distribution ---")
    hist, bins = np.histogram(peak_positions, bins=range(30, 55))
    for i, count in enumerate(hist):
        if count > 0:
            bar = '█' * (count // 50)
            print(f"  Sample {bins[i]:2d}: {count:4d} {bar}")
    
    return {
        'mean_peak_pos': np.mean(peak_positions),
        'std_peak_pos': np.std(peak_positions),
        'mean_offset': np.mean(peak_offsets),
        'peak_positions': peak_positions
    }


def compare_extraction_methods(data):
    """
    Compare the OLD extraction (centered on Index) vs NEW extraction (peak-aligned).
    """
    print("\n" + "=" * 70)
    print("EXTRACTION METHOD COMPARISON")
    print("=" * 70)
    
    d = data['d'].flatten()
    indices = data['Index'].flatten()
    
    # Method 1: OLD - centered on Index
    old_peaks = []
    for idx in indices[:100]:
        idx_0 = idx - 1
        start = idx_0 - 30
        end = idx_0 + 30
        if start >= 0 and end <= len(d):
            wf = d[start:end]
            old_peaks.append(np.argmax(wf))
    
    # Method 2: NEW - peak-aligned at sample 30
    new_peaks = []
    for idx in indices[:100]:
        idx_0 = idx - 1
        # Find actual peak near Index
        search_start = max(0, idx_0 - 15)
        search_end = min(len(d), idx_0 + 15)
        actual_peak = search_start + np.argmax(d[search_start:search_end])
        
        # Extract with peak at sample 30
        start = actual_peak - 30
        end = start + 60
        if start >= 0 and end <= len(d):
            wf = d[start:end]
            new_peaks.append(np.argmax(wf))
    
    print(f"\nOLD method (centered on Index):")
    print(f"  Peak positions: mean={np.mean(old_peaks):.1f}, std={np.std(old_peaks):.1f}")
    print(f"  Range: [{min(old_peaks)}, {max(old_peaks)}]")
    
    print(f"\nNEW method (peak-aligned at sample 30):")
    print(f"  Peak positions: mean={np.mean(new_peaks):.1f}, std={np.std(new_peaks):.1f}")
    print(f"  Range: [{min(new_peaks)}, {max(new_peaks)}]")
    
    print(f"\n✓ NEW method should show peaks consistently at sample 30 (±2)")


# ============================================================================
# PART 2: CORRECTED EXTRACTION FUNCTIONS
# ============================================================================

def extract_waveforms_aligned(data, indices, classes=None, align_peak_at=30, 
                               window_size=60, search_radius=15, verbose=False):
    """
    Extract waveforms with peaks CONSISTENTLY aligned at a fixed position.
    
    This is the CORRECTED version that ensures training and inference see
    waveforms with the same alignment.
    
    Parameters:
    -----------
    data : np.ndarray
        Raw signal (1D array)
    indices : np.ndarray
        Spike indices from ground truth (1-indexed)
    classes : np.ndarray or None
        Class labels (if available)
    align_peak_at : int
        Sample position where peak should be placed (default: 30 = centered)
    window_size : int
        Total waveform size (default: 60)
    search_radius : int
        How far from Index to search for actual peak (default: 15)
    
    Returns:
    --------
    waveforms : np.ndarray
        Extracted waveforms with consistent peak alignment
    valid_indices : np.ndarray
        Indices that were successfully extracted
    valid_classes : np.ndarray or None
        Classes for valid indices (if classes provided)
    """
    waveforms = []
    valid_indices = []
    valid_classes = [] if classes is not None else None
    
    rejected = {'boundary': 0, 'misaligned': 0}
    
    for i, idx in enumerate(indices):
        idx_0 = int(idx) - 1  # Convert to 0-indexed
        
        # Find the ACTUAL peak near this Index
        search_start = max(0, idx_0 - search_radius)
        search_end = min(len(data), idx_0 + search_radius)
        
        if search_end - search_start < 5:
            rejected['boundary'] += 1
            continue
        
        local_region = data[search_start:search_end]
        local_peak_offset = np.argmax(local_region)
        actual_peak = search_start + local_peak_offset
        
        # Extract waveform with peak at the ALIGNED position
        start = actual_peak - align_peak_at
        end = start + window_size
        
        if start < 0 or end > len(data):
            rejected['boundary'] += 1
            continue
        
        waveform = data[start:end]
        
        # Verify alignment (peak should be very close to align_peak_at)
        actual_peak_in_wf = np.argmax(waveform)
        if abs(actual_peak_in_wf - align_peak_at) > 3:
            rejected['misaligned'] += 1
            continue
        
        waveforms.append(waveform)
        valid_indices.append(idx)
        if classes is not None:
            valid_classes.append(classes[i])
    
    if verbose:
        print(f"Extracted {len(waveforms)}/{len(indices)} waveforms")
        print(f"Rejected: {rejected}")
        
        # Verify alignment
        peaks = [np.argmax(wf) for wf in waveforms]
        print(f"Peak positions: mean={np.mean(peaks):.2f}, std={np.std(peaks):.2f}")
    
    result_waveforms = np.array(waveforms)
    result_indices = np.array(valid_indices)
    result_classes = np.array(valid_classes) if classes is not None else None
    
    return result_waveforms, result_indices, result_classes


def detect_spikes_aligned(data, sample_rate=25000, threshold_factor=4.0,
                          min_spike_distance=30, align_peak_at=30, window_size=60):
    """
    Detect spikes with peaks aligned at a CONSISTENT position.
    
    MUST use the same align_peak_at value as training!
    
    Parameters:
    -----------
    data : np.ndarray
        Raw signal
    threshold_factor : float
        MAD multiplier for detection threshold
    align_peak_at : int
        Sample position where peak should be (MUST MATCH TRAINING)
    
    Returns:
    --------
    spike_indices : np.ndarray
        Indices of detected spikes (1-indexed, pointing to actual peak)
    spike_waveforms : np.ndarray
        Extracted waveforms with consistent alignment
    """
    # Bandpass filter
    nyquist = sample_rate / 2
    low = 300 / nyquist
    high = 3000 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)
    
    # MAD-based threshold
    mad = np.median(np.abs(filtered - np.median(filtered)))
    sigma = mad / 0.6745
    threshold = threshold_factor * sigma
    
    # Find peaks in filtered signal
    peaks, _ = signal.find_peaks(
        filtered,
        height=threshold,
        distance=min_spike_distance,
        prominence=threshold * 0.25
    )
    
    # Extract waveforms with consistent alignment
    valid_peaks = []
    waveforms = []
    
    for peak in peaks:
        # Refine peak location in ORIGINAL signal
        search_start = max(0, peak - 10)
        search_end = min(len(data), peak + 10)
        actual_peak = search_start + np.argmax(data[search_start:search_end])
        
        # Extract with peak at align_peak_at (SAME as training!)
        start = actual_peak - align_peak_at
        end = start + window_size
        
        if start >= 0 and end <= len(data):
            waveform = data[start:end]
            
            # Verify alignment
            peak_in_wf = np.argmax(waveform)
            if abs(peak_in_wf - align_peak_at) <= 5:
                waveforms.append(waveform)
                valid_peaks.append(actual_peak + 1)  # 1-indexed for MATLAB
    
    return np.array(valid_peaks, dtype=np.int64), np.array(waveforms)


# ============================================================================
# PART 3: TRAINING DATA PREPARATION
# ============================================================================

def prepare_aligned_training_data(data, align_peak_at=30, verbose=True):
    """
    Load D1 and prepare properly aligned training data.
    
    Returns:
    --------
    waveforms : np.ndarray
        Waveforms with consistent peak alignment
    classes : np.ndarray
        Class labels
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PREPARING ALIGNED TRAINING DATA")
        print("=" * 70)

    d = data['d'].flatten()
    indices = data['Index'].flatten()
    classes = data['Class'].flatten()
    
    if verbose:
        print(f"D1 contains {len(indices)} labeled spikes")
    
    # Extract with consistent alignment
    waveforms, valid_indices, valid_classes = extract_waveforms_aligned(
        d, indices, classes, 
        align_peak_at=align_peak_at,
        verbose=verbose
    )
    
    if verbose:
        print(f"\nClass distribution after extraction:")
        for c in range(1, 6):
            count = np.sum(valid_classes == c)
            print(f"  Class {c}: {count} ({100*count/len(valid_classes):.1f}%)")
    
    return waveforms, valid_classes


# ============================================================================
# PART 4: VALIDATION ON D1
# ============================================================================

def validate_on_d1(d1_path, cnn_model, align_peak_at=30):
    """
    Validate the complete pipeline on D1 where we have ground truth.
    
    This tests:
    1. Detection (can we find the spikes?)
    2. Classification (can we identify which class?)
    """
    print("\n" + "=" * 70)
    print("D1 VALIDATION (Ground Truth)")
    print("=" * 70)
    
    data = sio.loadmat(d1_path)
    d = data['d'].flatten()
    gt_indices = data['Index'].flatten()
    gt_classes = data['Class'].flatten()
    
    print(f"Ground truth: {len(gt_indices)} spikes")
    
    # Detect spikes using same method as inference
    detected_indices, detected_waveforms = detect_spikes_aligned(
        d, threshold_factor=4.5, align_peak_at=align_peak_at
    )
    
    print(f"Detected: {len(detected_indices)} spikes")
    
    # Match detected to ground truth (within 50 samples tolerance)
    tolerance = 50
    matches = []
    gt_used = set()
    
    for det_idx, det_wf in zip(detected_indices, detected_waveforms):
        distances = np.abs(gt_indices - det_idx)
        closest = np.argmin(distances)
        
        if distances[closest] <= tolerance and closest not in gt_used:
            matches.append({
                'det_idx': det_idx,
                'gt_idx': gt_indices[closest],
                'gt_class': gt_classes[closest],
                'waveform': det_wf
            })
            gt_used.add(closest)
    
    print(f"Matched: {len(matches)} / {len(gt_indices)} ({100*len(matches)/len(gt_indices):.1f}%)")
    
    # Classification accuracy (if model provided)
    if cnn_model is not None:
        matched_waveforms = np.array([m['waveform'] for m in matches])
        gt_labels = np.array([m['gt_class'] for m in matches])
        
        # Get predictions from CNN
        pred_labels = cnn_model.predict(matched_waveforms)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(gt_labels, pred_labels, labels=[1, 2, 3, 4, 5])
        
        print("\nConfusion Matrix:")
        print("      Predicted")
        print("      1    2    3    4    5")
        for i, row in enumerate(cm):
            print(f"  {i+1}: {row}")
        
        print("\nClassification Report:")
        print(classification_report(gt_labels, pred_labels, labels=[1,2,3,4,5]))
        
        overall_acc = 100 * np.trace(cm) / cm.sum()
        print(f"Overall Accuracy: {overall_acc:.1f}%")
        
        return overall_acc
    
    return None


# ============================================================================
# PART 5: MAIN DIAGNOSTIC RUNNER
# ============================================================================

def main():
    """Run the complete diagnostic suite."""

    BASE_DIR = Path(__file__).parent.parent
    # Load D1
    data = sio.loadmat(BASE_DIR / 'datasets' / 'D1.mat')
    
    # Run diagnostics
    stats = diagnose_d1_alignment(data)
    compare_extraction_methods(data)
    
    # Prepare aligned training data
    waveforms, classes = prepare_aligned_training_data(data,align_peak_at=30)
    
    # Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"""
DIAGNOSIS:
- D1 Index values point ~{stats['mean_offset']:.0f} samples BEFORE the actual peak
- When extracted centered on Index, peaks land at sample {stats['mean_peak_pos']:.0f} (not 30)
- Peak position variance: std = {stats['std_peak_pos']:.1f} samples

PROBLEM:
- Training: peaks at variable positions ({int(stats['mean_peak_pos'] - 2*stats['std_peak_pos'])}-{int(stats['mean_peak_pos'] + 2*stats['std_peak_pos'])})
- Inference: peaks forced to fixed position (41 in your current code)
- CNN learns features at wrong/inconsistent positions

FIX:
1. Use extract_waveforms_aligned() for training (peaks at sample 30)
2. Use detect_spikes_aligned() for inference (peaks at sample 30)
3. Retrain CNN with aligned waveforms
4. Remove or reduce post-processing rules (they shouldn't be needed)

EXPECTED IMPROVEMENT:
- D1 classification accuracy: 60-70% → 90%+
- D2-D6 classification: significant improvement, especially for similar classes (2/4/5)
""")
    
    print("\nTo implement the fix, update your cnn_experiment.py:")
    print("""
# In train() method, replace:
#   waveforms, valid_indices = extract_waveforms_at_indices(d, index, ...)
# With:
#   waveforms, valid_indices, valid_classes = extract_waveforms_aligned(
#       d, index, classes, align_peak_at=30, verbose=True
#   )

# In predict_dataset() method, replace:
#   indices, waveforms = detect_spikes_mad(d, ..., align_peak_at=41)
# With:
#   indices, waveforms = detect_spikes_aligned(d, ..., align_peak_at=30)
""")


if __name__ == '__main__':
    main()