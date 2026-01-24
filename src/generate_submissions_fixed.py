#!/usr/bin/env python3
"""
OPTIMIZED Submission Generator - V5

KEY INSIGHT: The matched filter already computes correlation with ALL 5 templates!
We were throwing away this classification information and using CNN instead.

For D5/D6 (low SNR):
- CNN was trained on clean D1 data → doesn't generalize to noisy data
- Matched filter ALREADY knows which template matched best at each spike
- Use template-based classification instead of CNN!

Strategy:
- D2/D3/D4: MAD detection + CNN classification (high SNR, works well)
- D5/D6: Matched filter detection + TEMPLATE-BASED classification
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from collections import Counter
import torch

# =============================================================================
# ALIGNMENT CONSTANT - MUST MATCH CNN TRAINING
# =============================================================================
ALIGN_PEAK_AT = 30
WINDOW_SIZE = 60


# =============================================================================
# MAD DETECTION (for clean data: D2, D3, D4)
# =============================================================================

def detect_spikes_mad(data, sample_rate=25000, threshold_factor=4.0,
                      min_spike_distance=30):
    """MAD-based spike detection - good for high SNR data."""
    nyquist = sample_rate / 2
    b, a = signal.butter(3, [300/nyquist, 3000/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, data)
    
    mad = np.median(np.abs(filtered - np.median(filtered)))
    sigma = mad / 0.6745
    threshold = threshold_factor * sigma
    
    print(f"  MAD threshold: {threshold:.3f}V (factor={threshold_factor}, σ={sigma:.3f})")
    
    peaks, _ = signal.find_peaks(
        filtered,
        height=threshold,
        distance=min_spike_distance,
        prominence=threshold * 0.25
    )
    
    valid_peaks = []
    waveforms = []
    
    for peak in peaks:
        search_start = max(0, peak - 10)
        search_end = min(len(data), peak + 10)
        actual_peak = search_start + np.argmax(data[search_start:search_end])
        
        start = actual_peak - ALIGN_PEAK_AT
        end = start + WINDOW_SIZE
        
        if start >= 0 and end <= len(data):
            waveform = data[start:end]
            peak_in_wf = np.argmax(waveform)
            if abs(peak_in_wf - ALIGN_PEAK_AT) <= 5:
                waveforms.append(waveform)
                valid_peaks.append(actual_peak + 1)
    
    print(f"  MAD found {len(valid_peaks)} spikes")
    return np.array(valid_peaks, dtype=np.int64), np.array(waveforms) if waveforms else np.empty((0, WINDOW_SIZE)), None


# =============================================================================
# MATCHED FILTER WITH TEMPLATE-BASED CLASSIFICATION
# =============================================================================

def detect_and_classify_matched_filter(data, templates, sample_rate=25000,
                                        correlation_threshold=0.5,
                                        min_spike_distance=30):
    """
    Matched filter that ALSO returns which template matched best.
    
    This is the key insight: we're already computing correlations with all 5 
    templates to detect spikes. We should USE this information for classification
    instead of throwing it away and using a CNN trained on clean data!
    
    Returns:
        indices: spike locations (1-indexed for MATLAB)
        waveforms: extracted waveforms
        template_classes: which template (1-5) matched best at each spike
        correlations: the correlation value for each spike (confidence measure)
    """
    nyquist = sample_rate / 2
    b, a = signal.butter(3, [300/nyquist, 3000/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, data).astype(np.float64)
    
    n = WINDOW_SIZE
    
    # Precompute running statistics
    data_mean = uniform_filter1d(filtered, size=n, mode='reflect')
    data_sq_mean = uniform_filter1d(filtered**2, size=n, mode='reflect')
    data_var = np.maximum(data_sq_mean - data_mean**2, 1e-10)
    data_std = np.sqrt(data_var)
    
    # Store NCC for EACH template separately
    ncc_per_template = {}
    
    print(f"  Computing correlations for each template...")
    for cls, template in templates.items():
        template_filtered = signal.filtfilt(b, a, template).astype(np.float64)
        t_mean = np.mean(template_filtered)
        t_std = np.std(template_filtered)
        
        if t_std < 1e-10:
            print(f"    Warning: Template {cls} has zero variance")
            ncc_per_template[cls] = np.zeros(len(data))
            continue
        
        t_zm = template_filtered - t_mean
        xcorr = signal.correlate(filtered, t_zm, mode='same')
        ncc = xcorr / (n * data_std * t_std + 1e-10)
        ncc_per_template[cls] = ncc
        
        print(f"    Template {cls}: max_corr={np.max(ncc):.3f}, "
              f"mean={np.mean(ncc):.3f}")
    
    # Compute max NCC and WHICH template gave it
    max_ncc = np.zeros(len(data))
    best_template = np.zeros(len(data), dtype=int)
    
    for cls, ncc in ncc_per_template.items():
        better_mask = ncc > max_ncc
        best_template[better_mask] = cls
        max_ncc = np.maximum(max_ncc, ncc)
    
    # Report statistics
    print(f"  NCC stats: mean={np.mean(max_ncc):.3f}, max={np.max(max_ncc):.3f}, "
          f"threshold={correlation_threshold}")
    
    for thresh in [0.5, 0.6, 0.7, 0.8]:
        count = np.sum(max_ncc > thresh)
        print(f"    Points > {thresh}: {count}")
    
    # Find peaks in max NCC
    peaks, _ = signal.find_peaks(
        max_ncc,
        height=correlation_threshold,
        distance=min_spike_distance,
        prominence=0.05
    )
    
    print(f"  Peaks found above threshold: {len(peaks)}")
    
    # Extract waveforms AND the best template class at each peak
    valid_peaks = []
    waveforms = []
    template_classes = []
    correlations = []
    
    for peak in peaks:
        # Refine peak location
        #search_start = max(0, peak - 15)
        #search_end = min(len(data), peak + 15)
        #actual_peak = search_start + np.argmax(data[search_start:search_end])

        actual_peak = peak  # Use detected peak directly
        
        start = actual_peak - ALIGN_PEAK_AT
        end = start + WINDOW_SIZE
        
        if start >= 0 and end <= len(data):
            waveform = data[start:end]
            waveforms.append(waveform)
            valid_peaks.append(actual_peak + 1)
            
            # Get the best template CLASS at this location
            # Use the correlation at the original peak location (where we detected it)
            best_cls = best_template[peak]
            template_classes.append(best_cls)
            correlations.append(max_ncc[peak])
    
    # Report template-based classification distribution
    if template_classes:
        cls_dist = Counter(template_classes)
        print(f"  Template-based classification:")
        for c in sorted(cls_dist.keys()):
            print(f"    Class {c}: {cls_dist[c]} ({100*cls_dist[c]/len(template_classes):.1f}%)")
    
    print(f"  Matched filter found {len(valid_peaks)} valid spikes")
    
    return (np.array(valid_peaks, dtype=np.int64), 
            np.array(waveforms) if waveforms else np.empty((0, WINDOW_SIZE)),
            np.array(template_classes, dtype=int) if template_classes else np.array([], dtype=int),
            np.array(correlations) if correlations else np.array([]))


# =============================================================================
# TEMPLATE EXTRACTION
# =============================================================================

def extract_templates_aligned(d1_path):
    """Extract aligned templates from D1 ground truth."""
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
            search_start = max(0, idx_0 - 15)
            search_end = min(len(d), idx_0 + 15)
            actual_peak = search_start + np.argmax(d[search_start:search_end])
            
            start = actual_peak - ALIGN_PEAK_AT
            end = start + WINDOW_SIZE
            
            if start >= 0 and end <= len(d):
                wf = d[start:end]
                if abs(np.argmax(wf) - ALIGN_PEAK_AT) <= 3:
                    waveforms.append(wf)
        
        if waveforms:
            waveforms = np.array(waveforms)
            templates[cls] = np.mean(waveforms, axis=0)
            print(f"  Class {cls}: {len(waveforms)} waveforms, "
                  f"peak amp={np.max(templates[cls]) - np.mean(templates[cls][:5]):.2f}V")
    
    return templates


# =============================================================================
# CNN LOADER
# =============================================================================

def load_cnn(model_dir):
    """Load the trained CNN model."""
    from cnn_experiment_fixed import CNNExperimentFixed
    cnn = CNNExperimentFixed()
    cnn.load_model()
    return cnn




# =============================================================================
# HYBRID CLASSIFICATION (for D5/D6)
# =============================================================================

def hybrid_classification(waveforms, template_classes, template_correlations, 
                          cnn, templates, confidence_threshold=0.75):
    """
    Hybrid classification strategy for noisy data:
    
    1. If template correlation is HIGH (>0.75): trust the template match
    2. If template correlation is MODERATE: use CNN but verify with template
    3. Optionally weight by correlation confidence
    
    This combines the best of both:
    - Template matching: direct shape comparison, works even in noise
    - CNN: learned features that may capture subtleties
    """
    n_spikes = len(waveforms)
    final_classes = np.zeros(n_spikes, dtype=int)
    classification_source = []  # Track where each classification came from
    
    # Get CNN predictions
    wf_norm, amp_features = cnn.prepare_data(waveforms)
    cnn.model.eval()
    
    with torch.no_grad():
        X_wf = torch.FloatTensor(wf_norm).to(cnn.device)
        X_amp = torch.FloatTensor(amp_features).to(cnn.device)
        outputs = cnn.model(X_wf, X_amp)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = outputs.max(1)
        cnn_classes = predicted.cpu().numpy() + 1
        cnn_confidences = probs.max(axis=1)
    
    # Hybrid decision for each spike
    for i in range(n_spikes):
        template_cls = template_classes[i]
        template_corr = template_correlations[i]
        cnn_cls = cnn_classes[i]
        cnn_conf = cnn_confidences[i]
        
        # Decision logic
        if template_corr >= confidence_threshold:
            # High correlation: trust template matching
            final_classes[i] = template_cls
            classification_source.append('template')
        elif template_cls == cnn_cls:
            # Both agree: use either (they're the same)
            final_classes[i] = template_cls
            classification_source.append('both_agree')
        else:
            # Disagreement with moderate correlation
            # Weight by confidence
            if cnn_conf > 0.9:
                # CNN very confident
                final_classes[i] = cnn_cls
                classification_source.append('cnn_confident')
            elif template_corr > 0.6:
                # Template correlation decent
                final_classes[i] = template_cls
                classification_source.append('template_moderate')
            else:
                # Low confidence all around - use template (more reliable for shape)
                final_classes[i] = template_cls
                classification_source.append('template_default')
    
    # Report statistics
    source_counts = Counter(classification_source)
    print(f"  Hybrid classification sources:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source}: {count} ({100*count/n_spikes:.1f}%)")
    
    return final_classes


# =============================================================================
# MAIN
# =============================================================================

def generate_submissions():
    """Generate submissions with optimized strategy per dataset."""
    print("=" * 70)
    print("GENERATING SUBMISSIONS - V5 (TEMPLATE-BASED CLASSIFICATION)")
    print("=" * 70)
    print("\nStrategy:")
    print("  D2/D3/D4: MAD detection + CNN classification (works well)")
    print("  D5/D6: Matched filter + TEMPLATE-BASED classification")
    print("         (CNN trained on clean data doesn't generalize to noisy data)")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'submissions'
    output_dir.mkdir(exist_ok=True)
    
    # Extract templates
    templates = extract_templates_aligned(base_dir / 'datasets' / 'D1.mat')
    
    # Load CNN (still used for D2/D3/D4)
    print("\nLoading CNN model...")
    try:
        cnn = load_cnn(base_dir / 'models')
    except Exception as e:
        print(f"ERROR loading CNN: {e}")
        return
    
    # ==========================================================================
    # DATASET CONFIGURATIONS
    # ==========================================================================
    
    datasets = {
        'D2': {
            'method': 'mad',
            'mad_factor': 3.5,
            'classifier': 'cnn',
            'apply_corrections': True,
        },
        'D3': {
            'method': 'mad',
            'mad_factor': 3.0,
            'classifier': 'cnn',
            'apply_corrections': True,
        },
        'D4': {
            'method': 'mad',
            'mad_factor': 3.0,
            'classifier': 'cnn',
            'apply_corrections': False,
        },
        'D5': {
            'method': 'matched',
            'corr_threshold': 0.73,      # Adjusted threshold
            'classifier': 'cnn',     
            'apply_corrections': False,
        },
        'D6': {
            'method': 'matched',
            'corr_threshold': 0.69,      # Lower for very noisy data
            'classifier': 'cnn',     
            'apply_corrections': False,
        },
    }
    
    # Alternative: try hybrid classification
    # Change 'classifier': 'template' to 'classifier': 'hybrid' to test
    
    all_results = {}
    
    for dataset_name, config in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {dataset_name}")
        print(f"{'=' * 60}")
        
        # Load data
        data = sio.loadmat(base_dir / 'datasets' / f'{dataset_name}.mat')
        d = data['d'].flatten()
        
        method = config['method']
        classifier = config['classifier']
        print(f"Detection: {method}, Classification: {classifier}")
        
        # ===== DETECTION =====
        if method == 'mad':
            indices, waveforms, _ = detect_spikes_mad(
                d, threshold_factor=config['mad_factor']
            )
            template_classes = None
            template_correlations = None
            
        elif method == 'matched':
            indices, waveforms, template_classes, template_correlations = \
                detect_and_classify_matched_filter(
                    d, templates, 
                    correlation_threshold=config['corr_threshold']
                )
        
        if len(indices) == 0:
            print(f"WARNING: No spikes detected!")
            all_results[dataset_name] = {'count': 0}
            continue
        
        # Verify alignment
        peak_positions = [np.argmax(wf) for wf in waveforms[:100]]
        print(f"Peak alignment: mean={np.mean(peak_positions):.1f}, std={np.std(peak_positions):.1f}")
        
        # ===== CLASSIFICATION =====
        print(f"Classifying with {classifier}...")
        
        if classifier == 'cnn':
            # Use Shape-Only CNN
            # We NO LONGER calculate amp_features
            
            
            # Prepare ONLY waveform data
            wf_norm, _ = cnn.prepare_data(waveforms) # Ignore the second return value
            
            cnn.model.eval()
            with torch.no_grad():
                X_wf = torch.FloatTensor(wf_norm).to(cnn.device)
                
                # Model now only takes X_wf
                outputs = cnn.model(X_wf)
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                _, predicted = outputs.max(1)
                classes = predicted.cpu().numpy() + 1
                confidences = probs.max(axis=1)
            


        elif classifier == 'template':
            # Use template-based classification (for D5/D6)
            classes = template_classes
            confidences = template_correlations
            
        elif classifier == 'hybrid':
            # Combine template and CNN
            classes = hybrid_classification(
                waveforms, template_classes, template_correlations,
                cnn, templates, confidence_threshold=0.70
            )
            confidences = template_correlations
        
        # Print results
        class_dist = Counter(classes)
        print("Class distribution:")
        for c in sorted(class_dist.keys()):
            print(f"  Class {c}: {class_dist[c]} ({100*class_dist[c]/len(classes):.1f}%)")
        
        if confidences is not None:
            print(f"Average confidence: {np.mean(confidences):.3f}")
        
        # Save
        output_path = output_dir / f'{dataset_name}.mat'
        sio.savemat(str(output_path), {
            'Index': indices.reshape(1, -1).astype(np.int32),
            'Class': classes.reshape(1, -1).astype(np.uint8)
        })
        print(f"Saved: {output_path}")
        
        all_results[dataset_name] = {
            'count': len(indices),
            'distribution': class_dist,
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    friend = {'D2': 3728, 'D3': 3016, 'D4': 2598, 'D5': 1898, 'D6': 2582}
    
    print(f"\n{'Dataset':<8} {'Count':<8} {'Friend':<8} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6} {'C5':>6}")
    print("-" * 65)
    
    for ds in ['D2', 'D3', 'D4', 'D5', 'D6']:
        if ds in all_results and all_results[ds]['count'] > 0:
            count = all_results[ds]['count']
            dist = all_results[ds]['distribution']
            c1 = dist.get(1, 0)
            c2 = dist.get(2, 0)
            c3 = dist.get(3, 0)
            c4 = dist.get(4, 0)
            c5 = dist.get(5, 0)
            print(f"{ds:<8} {count:<8} {friend[ds]:<8} {c1:>6} {c2:>6} {c3:>6} {c4:>6} {c5:>6}")


if __name__ == '__main__':
    generate_submissions()