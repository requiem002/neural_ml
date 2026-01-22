#!/usr/bin/env python3
"""
FIXED Submission Generator for EE40098 Coursework C

CRITICAL FIX: This version ensures waveform alignment is CONSISTENT between
the CNN training and inference. Both use align_peak_at=30.

Detection Methods Explained:
----------------------------
1. MAD (Median Absolute Deviation):
   - Finds peaks above threshold = k × estimated_noise_level
   - Good for clean data where spikes are clearly above noise
   - Use for D2, D3, D4

2. Matched Filter:
   - Correlates signal with known spike templates from D1
   - Finds spikes that "look like" the training examples
   - Better for noisy data where amplitude alone isn't reliable
   - Use for D5, D6

3. Hybrid:
   - Combines both methods (takes union of detected spikes)
   - Maximum recall but may have more false positives
   - Use when you want to catch every possible spike

Usage:
------
1. First, train the fixed CNN:
   python cnn_experiment_fixed.py --train

2. Then generate submissions:
   python generate_submissions_fixed.py
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from collections import Counter
import torch

# =============================================================================
# CRITICAL: ALIGNMENT CONSTANT - MUST MATCH CNN TRAINING
# =============================================================================
ALIGN_PEAK_AT = 30  # Peak will be at sample 30 (centered in 60-sample window)
WINDOW_SIZE = 60


# =============================================================================
# DETECTION FUNCTIONS (all use consistent alignment)
# =============================================================================

def detect_spikes_mad(data, sample_rate=25000, threshold_factor=4.0,
                      min_spike_distance=30):
    """
    Detect spikes using MAD-based adaptive thresholding.
    
    HOW IT WORKS:
    1. Bandpass filter the signal (300-3000 Hz) to isolate spike frequencies
    2. Estimate noise level using MAD (robust to outliers from spikes)
    3. Set threshold = threshold_factor × noise_level
    4. Find all peaks above threshold
    5. Extract waveforms with peak aligned at sample ALIGN_PEAK_AT
    
    WHEN TO USE:
    - Clean data (high SNR): D2, D3
    - Moderate noise: D4
    - Use lower threshold_factor for noisier data (more aggressive detection)
    
    Parameters:
    -----------
    threshold_factor : float
        - 4.0-4.5: Conservative (fewer false positives, may miss weak spikes)
        - 3.5-4.0: Moderate
        - 2.5-3.5: Aggressive (catches more spikes but more false positives)
    """
    # Step 1: Bandpass filter (spike frequencies are 300-3000 Hz)
    nyquist = sample_rate / 2
    b, a = signal.butter(3, [300/nyquist, 3000/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, data)
    
    # Step 2: Estimate noise using MAD (Median Absolute Deviation)
    # MAD is robust to outliers (the spikes themselves)
    mad = np.median(np.abs(filtered - np.median(filtered)))
    sigma = mad / 0.6745  # Convert MAD to standard deviation
    threshold = threshold_factor * sigma
    
    print(f"  MAD threshold: {threshold:.3f}V (factor={threshold_factor}, σ={sigma:.3f})")
    
    # Step 3: Find peaks above threshold
    peaks, _ = signal.find_peaks(
        filtered,
        height=threshold,
        distance=min_spike_distance,  # Refractory period
        prominence=threshold * 0.25   # Must be a real peak, not noise
    )
    
    # Step 4: Extract waveforms with CONSISTENT alignment
    valid_peaks = []
    waveforms = []
    
    for peak in peaks:
        # Refine peak location in ORIGINAL (unfiltered) signal
        search_start = max(0, peak - 10)
        search_end = min(len(data), peak + 10)
        actual_peak = search_start + np.argmax(data[search_start:search_end])
        
        # Extract with peak at ALIGN_PEAK_AT (MUST match training!)
        start = actual_peak - ALIGN_PEAK_AT
        end = start + WINDOW_SIZE
        
        if start >= 0 and end <= len(data):
            waveform = data[start:end]
            
            # Verify alignment (reject if peak isn't where expected)
            peak_in_wf = np.argmax(waveform)
            if abs(peak_in_wf - ALIGN_PEAK_AT) <= 5:
                waveforms.append(waveform)
                valid_peaks.append(actual_peak + 1)  # 1-indexed for MATLAB
    
    print(f"  Found {len(valid_peaks)} spikes")
    return np.array(valid_peaks, dtype=np.int64), np.array(waveforms) if waveforms else np.empty((0, WINDOW_SIZE))


def detect_spikes_matched_filter(data, templates, sample_rate=25000,
                                  correlation_threshold=0.4,
                                  min_spike_distance=30):
    """
    Detect spikes using matched filtering with D1 templates.
    
    HOW IT WORKS:
    1. For each known spike template, compute correlation with signal
    2. At each point, take the maximum correlation across all templates
    3. Find peaks in the correlation signal above threshold
    4. Extract waveforms at those locations
    
    WHY IT'S BETTER FOR NOISY DATA:
    - Amplitude thresholding fails when noise amplitude ≈ spike amplitude
    - But spikes still have a SHAPE that noise doesn't
    - Correlation measures shape similarity, not just amplitude
    
    WHEN TO USE:
    - Very noisy data: D5, D6
    - When MAD method misses too many spikes
    
    Parameters:
    -----------
    templates : dict
        {class_number: average_waveform} from D1 training data
    correlation_threshold : float
        Minimum correlation (0-1). Lower = more aggressive.
        - 0.5-0.6: Conservative
        - 0.3-0.4: Moderate
        - 0.2-0.3: Aggressive
    """
    # Bandpass filter
    nyquist = sample_rate / 2
    b, a = signal.butter(3, [300/nyquist, 3000/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, data)
    
    # Compute correlation with each template, keep maximum
    max_corr = np.zeros(len(data))
    
    for cls, template in templates.items():
        # Filter template the same way as signal
        template_filtered = signal.filtfilt(b, a, template)
        
        # Normalize template (zero-mean, unit-norm)
        template_zm = template_filtered - np.mean(template_filtered)
        template_norm = template_zm / (np.linalg.norm(template_zm) + 1e-10)
        
        # Compute correlation
        raw_corr = signal.correlate(filtered, template_norm, mode='same')
        
        # Normalize by local signal energy (so correlation is in 0-1 range)
        n = len(template)
        local_mean = uniform_filter1d(filtered, size=n, mode='reflect')
        local_sq_mean = uniform_filter1d(filtered**2, size=n, mode='reflect')
        local_var = np.maximum(local_sq_mean - local_mean**2, 1e-10)
        local_std = np.sqrt(local_var)
        
        corr_normalized = raw_corr / (n * local_std + 1e-10)
        max_corr = np.maximum(max_corr, corr_normalized)
    
    print(f"  Correlation: mean={np.mean(max_corr):.3f}, max={np.max(max_corr):.3f}")
    
    # Find peaks in correlation
    peaks, _ = signal.find_peaks(
        max_corr,
        height=correlation_threshold,
        distance=min_spike_distance
    )
    
    # Extract waveforms with consistent alignment
    valid_peaks = []
    waveforms = []
    
    for peak in peaks:
        # Refine peak location in original signal
        search_start = max(0, peak - 10)
        search_end = min(len(data), peak + 10)
        actual_peak = search_start + np.argmax(data[search_start:search_end])
        
        # Extract with consistent alignment
        start = actual_peak - ALIGN_PEAK_AT
        end = start + WINDOW_SIZE
        
        if start >= 0 and end <= len(data):
            waveform = data[start:end]
            waveforms.append(waveform)
            valid_peaks.append(actual_peak + 1)
    
    print(f"  Found {len(valid_peaks)} spikes")
    return np.array(valid_peaks, dtype=np.int64), np.array(waveforms) if waveforms else np.empty((0, WINDOW_SIZE))


def detect_spikes_hybrid(data, templates, sample_rate=25000,
                         mad_factor=3.5, corr_threshold=0.35,
                         min_spike_distance=30):
    """
    Hybrid detection: combine MAD and matched filter for maximum recall.
    
    Takes the UNION of both methods - if either method finds a spike, keep it.
    
    WHEN TO USE:
    - When you want to catch every possible spike
    - Noisy data where different spikes respond better to different methods
    - D5, D6 where recall is more important than precision
    """
    print("  Running MAD detection...")
    indices_mad, waveforms_mad = detect_spikes_mad(
        data, sample_rate, mad_factor, min_spike_distance
    )
    
    print("  Running matched filter detection...")
    indices_mf, waveforms_mf = detect_spikes_matched_filter(
        data, templates, sample_rate, corr_threshold, min_spike_distance
    )
    
    # Merge results (union, avoiding duplicates within refractory period)
    all_spikes = {}
    
    for idx, wf in zip(indices_mad, waveforms_mad):
        all_spikes[idx] = wf
    
    for idx, wf in zip(indices_mf, waveforms_mf):
        if idx not in all_spikes:
            # Check for nearby existing spike
            close = [i for i in all_spikes.keys() if abs(i - idx) < min_spike_distance]
            if not close:
                all_spikes[idx] = wf
    
    # Sort by index
    sorted_indices = sorted(all_spikes.keys())
    final_waveforms = [all_spikes[i] for i in sorted_indices]
    
    print(f"  Hybrid total: {len(sorted_indices)} spikes")
    return np.array(sorted_indices, dtype=np.int64), np.array(final_waveforms) if final_waveforms else np.empty((0, WINDOW_SIZE))


# =============================================================================
# TEMPLATE EXTRACTION (FIXED to use aligned waveforms)
# =============================================================================

def extract_templates_aligned(d1_path):
    """
    Extract average waveform templates from D1 ground truth.
    
    FIXED: Uses peak-aligned extraction so templates match inference waveforms.
    """
    print("Extracting aligned templates from D1...")
    
    data = sio.loadmat(d1_path)
    d = data['d'].flatten()
    indices = data['Index'].flatten()
    classes = data['Class'].flatten()
    
    templates = {}
    
    for cls in range(1, 6):
        cls_indices = indices[classes == cls]
        waveforms = []
        
        for idx in cls_indices:
            idx_0 = int(idx) - 1
            
            # Find actual peak near Index
            search_start = max(0, idx_0 - 15)
            search_end = min(len(d), idx_0 + 15)
            actual_peak = search_start + np.argmax(d[search_start:search_end])
            
            # Extract with peak at ALIGN_PEAK_AT
            start = actual_peak - ALIGN_PEAK_AT
            end = start + WINDOW_SIZE
            
            if start >= 0 and end <= len(d):
                wf = d[start:end]
                # Verify alignment
                if abs(np.argmax(wf) - ALIGN_PEAK_AT) <= 3:
                    waveforms.append(wf)
        
        waveforms = np.array(waveforms)
        templates[cls] = np.mean(waveforms, axis=0)
        print(f"  Class {cls}: {len(waveforms)} waveforms, peak at sample {np.argmax(templates[cls])}")
    
    return templates


# =============================================================================
# CNN WRAPPER (imports from fixed version)
# =============================================================================

def load_fixed_cnn(model_dir):
    """Load the fixed CNN model."""
    from cnn_experiment_fixed import CNNExperimentFixed, DualBranchSpikeNet, ALIGN_PEAK_AT as CNN_ALIGN
    
    # Verify alignment matches
    if CNN_ALIGN != ALIGN_PEAK_AT:
        raise ValueError(f"Alignment mismatch! CNN uses {CNN_ALIGN}, submission uses {ALIGN_PEAK_AT}")
    
    cnn = CNNExperimentFixed()
    cnn.load_model()
    return cnn


# =============================================================================
# POST-PROCESSING (minimal - shouldn't need much with fixed alignment)
# =============================================================================

def apply_minimal_corrections(classes, raw_amp, fwhm_values, confidences):
    """
    Apply only VERY conservative physics-based corrections.
    
    With proper alignment, the CNN should be much more accurate,
    so we only correct obvious physical impossibilities.
    """
    corrected = classes.copy()
    corrections = Counter()
    
    # D1 amplitude statistics (mean ± 2*std ranges):
    # C1: 3.03 - 6.75V    C2: 4.05 - 6.97V    C3: 0.0 - 3.31V
    # C4: 0.78 - 4.94V    C5: 2.26 - 6.14V
    
    for i, (pred, amp, fwhm, conf) in enumerate(zip(classes, raw_amp, fwhm_values, confidences)):
        
        # Rule 1: Class 3 CANNOT have amplitude > 4.5V (physically impossible)
        if pred == 3 and amp > 4.5:
            # Reassign based on FWHM
            if fwhm > 0.80:
                corrected[i] = 5
            elif fwhm > 0.65:
                corrected[i] = 2
            else:
                corrected[i] = 1
            corrections['c3_amp_impossible'] += 1
        
        # Rule 2: Class 4 CANNOT have amplitude > 5.5V (physically impossible)
        elif pred == 4 and amp > 5.5:
            if fwhm > 0.80:
                corrected[i] = 5
            else:
                corrected[i] = 2
            corrections['c4_amp_impossible'] += 1
    
    if sum(corrections.values()) > 0:
        print(f"  Physics corrections: {dict(corrections)}")
    
    return corrected


# =============================================================================
# MAIN SUBMISSION GENERATOR
# =============================================================================

def generate_submissions():
    """Generate submissions for D2-D6."""
    print("=" * 70)
    print("GENERATING SUBMISSIONS (FIXED ALIGNMENT)")
    print(f"Waveforms aligned with peak at sample {ALIGN_PEAK_AT}")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'submissions'
    output_dir.mkdir(exist_ok=True)
    
    # Extract aligned templates for matched filter
    templates = extract_templates_aligned(base_dir / 'datasets' / 'D1.mat')
    
    # Save templates for future use
    np.save(base_dir / 'models' / 'd1_templates_aligned.npy', templates)
    
    # Load fixed CNN
    print("\nLoading fixed CNN model...")
    try:
        cnn = load_fixed_cnn(base_dir / 'models')
    except FileNotFoundError:
        print("ERROR: Fixed CNN model not found!")
        print("Please train first: python cnn_experiment_fixed.py --train")
        return
    
    # Dataset configurations
    # Threshold factors tuned for each SNR level
    datasets = {
        'D2': {'method': 'mad', 'mad_factor': 3.8},   # High SNR, conservative
        'D3': {'method': 'mad', 'mad_factor': 3.4},   # Good SNR
        'D4': {'method': 'mad', 'mad_factor': 3.0},   # Moderate SNR
        'D5': {'method': 'mad', 'mad_factor': 2.8},   # Low SNR, aggressive
        'D6': {'method': 'mad', 'mad_factor': 2.6},   # Very low SNR, very aggressive
    }
    
    all_results = {}
    
    for dataset_name, config in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {dataset_name}")
        print(f"{'=' * 60}")
        
        # Load data
        data = sio.loadmat(base_dir / 'datasets' / f'{dataset_name}.mat')
        d = data['d'].flatten()
        
        # Detect spikes
        method = config['method']
        print(f"Detection method: {method}")
        
        if method == 'mad':
            indices, waveforms = detect_spikes_mad(d, threshold_factor=config['mad_factor'])
        elif method == 'hybrid':
            indices, waveforms = detect_spikes_hybrid(
                d, templates, 
                mad_factor=config.get('mad_factor', 3.5),
                corr_threshold=config.get('corr_threshold', 0.35)
            )
        elif method == 'matched':
            indices, waveforms = detect_spikes_matched_filter(
                d, templates,
                correlation_threshold=config.get('corr_threshold', 0.4)
            )
        
        if len(indices) == 0:
            print(f"WARNING: No spikes detected!")
            all_results[dataset_name] = {'count': 0, 'distribution': {}}
            continue
        
        # Verify alignment
        peak_positions = [np.argmax(wf) for wf in waveforms[:100]]
        print(f"Peak positions (sample): mean={np.mean(peak_positions):.1f}, std={np.std(peak_positions):.1f}")
        
        # Classify with CNN
        print("Classifying...")
        wf_norm, amp_features = cnn.prepare_data(waveforms)
        raw_amp_features = cnn.extract_amplitude_features(waveforms)
        raw_amp = raw_amp_features[:, 0]
        fwhm_values = raw_amp_features[:, 3]
        
        cnn.model.eval()
        X_wf = torch.FloatTensor(wf_norm).to(cnn.device)
        X_amp = torch.FloatTensor(amp_features).to(cnn.device)
        
        with torch.no_grad():
            outputs = cnn.model(X_wf, X_amp)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = outputs.max(1)
            classes = predicted.cpu().numpy() + 1
            confidences = probs.max(axis=1)
        
        # Apply minimal corrections (should be few with fixed alignment)
        classes = apply_minimal_corrections(classes, raw_amp, fwhm_values, confidences)
        
        # Print distribution
        class_dist = Counter(classes)
        print("Class distribution:")
        for c in sorted(class_dist.keys()):
            print(f"  Class {c}: {class_dist[c]} ({100*class_dist[c]/len(classes):.1f}%)")
        print(f"Average confidence: {np.mean(confidences):.1%}")
        
        # Save
        output_path = output_dir / f'{dataset_name}.mat'
        sio.savemat(str(output_path), {
            'Index': indices.reshape(1, -1).astype(np.int32),
            'Class': classes.reshape(1, -1).astype(np.uint8)
        })
        print(f"Saved: {output_path}")
        
        all_results[dataset_name] = {
            'count': len(indices),
            'distribution': class_dist
        }
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<8} {'Count':<8} {'C1':<8} {'C2':<8} {'C3':<8} {'C4':<8} {'C5':<8}")
    print("-" * 60)
    
    for ds in ['D2', 'D3', 'D4', 'D5', 'D6']:
        if ds in all_results and all_results[ds]['count'] > 0:
            count = all_results[ds]['count']
            dist = all_results[ds]['distribution']
            row = f"{ds:<8} {count:<8}"
            for c in range(1, 6):
                pct = 100 * dist.get(c, 0) / count
                row += f"{pct:.1f}%{'':<3}"
            print(row)
    
    print("\nSubmissions saved to:", output_dir)


if __name__ == '__main__':
    generate_submissions()