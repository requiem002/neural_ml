"""Spike detection module using adaptive thresholding."""

import numpy as np
from scipy import signal


def detect_spikes(data, sample_rate=25000, threshold_factor=None, voltage_threshold=None,
                  min_spike_distance=30, window_before=30, window_after=30, refine_peak=True):
    """
    Detect spikes in neural recording.

    Parameters:
    -----------
    data : np.ndarray
        Raw signal (1D array)
    sample_rate : int
        Sampling rate in Hz
    threshold_factor : float or None
        Multiplier for MAD-based threshold (if voltage_threshold not provided)
    voltage_threshold : float or None
        Fixed voltage threshold (overrides threshold_factor if provided)
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

    # Determine threshold
    if voltage_threshold is not None:
        # Use fixed voltage threshold (recommended for cross-dataset consistency)
        threshold = voltage_threshold
    else:
        # Fall back to MAD-based adaptive threshold
        if threshold_factor is None:
            threshold_factor = 5.0  # Default
        mad = np.median(np.abs(filtered - np.median(filtered)))
        sigma = mad / 0.6745
        threshold = threshold_factor * sigma

    # Find peaks above threshold
    peaks, properties = signal.find_peaks(
        filtered,
        height=threshold,
        distance=min_spike_distance,
        prominence=threshold * 0.3  # Lower prominence for better recall
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
                                  threshold_factor=4.0, correlation_threshold=None,
                                  min_spike_distance=30, window_before=30, window_after=30):
    """
    Detect spikes using matched filtering with known templates.
    Better for low SNR conditions.

    Parameters:
    -----------
    data : np.ndarray
        Raw signal (1D array)
    templates : dict
        Dictionary mapping class -> average waveform template
    sample_rate : int
        Sampling rate in Hz
    threshold_factor : float
        Multiplier for MAD-based threshold (fallback if correlation_threshold not set)
    correlation_threshold : float or None
        Fixed correlation threshold (0-1 range, recommended 0.3-0.6)
        If provided, overrides threshold_factor
    min_spike_distance : int
        Minimum samples between spikes
    window_before : int
        Samples before peak for waveform extraction
    window_after : int
        Samples after peak for waveform extraction
    """
    nyquist = sample_rate / 2
    low = 300 / nyquist
    high = 3000 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)

    # Compute NORMALIZED correlation with each template
    correlations = []
    for cls, template in templates.items():
        # Normalize template to unit energy
        template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)

        # Compute normalized cross-correlation using sliding window
        # For proper normalized correlation, we need to normalize each window of the signal too
        corr = compute_normalized_correlation(filtered, template_norm)
        correlations.append(corr)

    # Take maximum correlation across templates
    max_corr = np.max(correlations, axis=0)

    # Determine threshold
    if correlation_threshold is not None:
        # Use fixed correlation threshold (value between 0-1)
        threshold = correlation_threshold
    else:
        # Fall back to MAD-based threshold on correlation values
        mad = np.median(np.abs(max_corr - np.median(max_corr)))
        sigma = mad / 0.6745
        threshold = threshold_factor * sigma

    peaks, _ = signal.find_peaks(max_corr, height=threshold, distance=min_spike_distance)

    # Extract waveforms from ORIGINAL (unfiltered) data
    valid_peaks = []
    waveforms = []

    for peak in peaks:
        # Refine peak location in original signal
        search_start = max(0, peak - 10)
        search_end = min(len(data), peak + 10)
        local_max_idx = search_start + np.argmax(data[search_start:search_end])
        peak = local_max_idx

        start = peak - window_before
        end = peak + window_after

        if start >= 0 and end <= len(data):
            waveform = data[start:end]
            waveforms.append(waveform)
            valid_peaks.append(peak + 1)

    return np.array(valid_peaks, dtype=np.int64), np.array(waveforms) if waveforms else np.empty((0, window_before + window_after))


def compute_normalized_correlation(signal_data, template):
    """
    Compute normalized cross-correlation between signal and template.
    Returns correlation values in range [-1, 1].

    Parameters:
    -----------
    signal_data : np.ndarray
        The signal to search in
    template : np.ndarray
        The normalized template to match

    Returns:
    --------
    corr : np.ndarray
        Normalized correlation at each position
    """
    n = len(template)
    result = np.zeros(len(signal_data))

    # Use scipy correlate for speed, then normalize
    raw_corr = signal.correlate(signal_data, template, mode='same')

    # Compute local standard deviation for normalization
    # Use cumsum trick for efficient computation
    signal_sq = signal_data ** 2
    cumsum = np.cumsum(signal_data)
    cumsum_sq = np.cumsum(signal_sq)

    # Pad for edge handling
    half_n = n // 2

    for i in range(half_n, len(signal_data) - half_n):
        start = i - half_n
        end = i + half_n + (n % 2)  # Handle odd template lengths

        # Local mean and std
        local_sum = cumsum[end] - (cumsum[start] if start > 0 else 0)
        local_sum_sq = cumsum_sq[end] - (cumsum_sq[start] if start > 0 else 0)

        local_mean = local_sum / n
        local_var = local_sum_sq / n - local_mean ** 2
        local_std = np.sqrt(max(local_var, 1e-10))

        # Normalize
        result[i] = raw_corr[i] / (n * local_std + 1e-10)

    return result
