"""Spike detection module using adaptive thresholding."""

import numpy as np
from scipy import signal


def detect_spikes(data, sample_rate=25000, threshold_factor=4.0, min_spike_distance=30,
                  window_before=30, window_after=30, refine_peak=True):
    """
    Detect spikes in neural recording using adaptive threshold.

    Parameters:
    -----------
    data : np.ndarray
        Raw signal (1D array)
    sample_rate : int
        Sampling rate in Hz
    threshold_factor : float
        Multiplier for MAD-based threshold
    min_spike_distance : int
        Minimum samples between spikes (refractory period)
    window_before : int
        Samples before peak for waveform extraction
    window_after : int
        Samples after peak for waveform extraction
    refine_peak : bool
        If True, refine peak location using original signal

    Returns:
    --------
    spike_indices : np.ndarray
        Indices of detected spike peaks (1-indexed for MATLAB compatibility)
    spike_waveforms : np.ndarray
        Extracted waveforms (n_spikes x window_size)
    """
    # Bandpass filter to isolate spike frequencies (300-3000 Hz typical for spikes)
    nyquist = sample_rate / 2
    low = 300 / nyquist
    high = 3000 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)

    # Adaptive threshold using Median Absolute Deviation (robust to outliers)
    # MAD-based noise estimation: sigma = MAD / 0.6745
    mad = np.median(np.abs(filtered - np.median(filtered)))
    sigma = mad / 0.6745
    threshold = threshold_factor * sigma

    # Find peaks above threshold
    peaks, properties = signal.find_peaks(
        filtered,
        height=threshold,
        distance=min_spike_distance,
        prominence=threshold * 0.5  # Require some prominence to avoid noise
    )

    # Extract waveforms and refine peak locations
    valid_peaks = []
    waveforms = []

    for peak in peaks:
        # Refine peak location by finding true maximum in original signal
        if refine_peak:
            search_start = max(0, peak - 10)
            search_end = min(len(data), peak + 10)
            local_max_idx = search_start + np.argmax(data[search_start:search_end])
            peak = local_max_idx

        start = peak - window_before
        end = peak + window_after

        if start >= 0 and end <= len(data):
            waveform = data[start:end]

            # Additional validation: check waveform has spike-like characteristics
            peak_val = np.max(waveform)
            baseline = np.mean(waveform[:5])  # First 5 samples as baseline

            # Spike should have significant amplitude above baseline
            if peak_val - baseline > threshold * 0.3:
                waveforms.append(waveform)
                valid_peaks.append(peak + 1)  # 1-indexed for MATLAB

    spike_indices = np.array(valid_peaks, dtype=np.int64)
    spike_waveforms = np.array(waveforms) if waveforms else np.empty((0, window_before + window_after))

    return spike_indices, spike_waveforms


def detect_spikes_adaptive(data, sample_rate=25000, min_spike_distance=30,
                           window_before=30, window_after=30):
    """
    Detect spikes using multiple threshold levels and merge results.
    More robust for varying SNR levels.
    """
    all_peaks = set()
    all_waveforms = {}

    # Try multiple threshold factors to catch spikes at different amplitudes
    for thresh_factor in [3.5, 4.0, 4.5, 5.0]:
        indices, waveforms = detect_spikes(
            data, sample_rate, thresh_factor,
            min_spike_distance, window_before, window_after
        )
        for idx, wf in zip(indices, waveforms):
            if idx not in all_waveforms:
                all_waveforms[idx] = wf

    # Sort by index
    sorted_indices = sorted(all_waveforms.keys())

    # Remove duplicates within refractory period
    final_indices = []
    final_waveforms = []

    for idx in sorted_indices:
        if not final_indices or idx - final_indices[-1] >= min_spike_distance:
            final_indices.append(idx)
            final_waveforms.append(all_waveforms[idx])

    return np.array(final_indices, dtype=np.int64), np.array(final_waveforms)


def detect_spikes_matched_filter(data, templates, sample_rate=25000,
                                  threshold_factor=4.0, min_spike_distance=30,
                                  window_before=30, window_after=30):
    """
    Detect spikes using matched filtering with known templates.
    Better for low SNR conditions.

    Parameters:
    -----------
    templates : dict
        Dictionary mapping class -> average waveform template
    """
    nyquist = sample_rate / 2
    low = 300 / nyquist
    high = 3000 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)

    # Compute correlation with each template
    correlations = []
    for cls, template in templates.items():
        # Normalize template
        template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)
        # Correlate
        corr = signal.correlate(filtered, template_norm, mode='same')
        correlations.append(corr)

    # Take maximum correlation across templates
    max_corr = np.max(correlations, axis=0)

    # Threshold based on correlation statistics
    mad = np.median(np.abs(max_corr - np.median(max_corr)))
    sigma = mad / 0.6745
    threshold = threshold_factor * sigma

    peaks, _ = signal.find_peaks(max_corr, height=threshold, distance=min_spike_distance)

    # Extract waveforms
    valid_peaks = []
    waveforms = []

    for peak in peaks:
        start = peak - window_before
        end = peak + window_after

        if start >= 0 and end <= len(data):
            waveform = data[start:end]
            waveforms.append(waveform)
            valid_peaks.append(peak + 1)

    return np.array(valid_peaks, dtype=np.int64), np.array(waveforms)
