"""Feature extraction module for spike classification."""

import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureExtractor:
    """Extract and transform features from spike waveforms."""

    def __init__(self, n_pca_components=10, window_before=30, window_after=30):
        self.n_pca_components = n_pca_components
        self.window_before = window_before
        self.window_after = window_after
        self.pca = None
        self.scaler = None
        self.waveform_scaler = None

    def extract_handcrafted_features(self, waveforms, raw_waveforms=None):
        """
        Extract hand-crafted features from waveforms.

        Parameters:
        -----------
        waveforms : np.ndarray
            Filtered waveforms for shape-based features
        raw_waveforms : np.ndarray or None
            Unfiltered waveforms for amplitude features (critical for Class 2)

        Features (from filtered):
        - Peak amplitude, location, trough, peak-to-trough
        - Width at half max, rise/decay times
        - Area, energy, slopes, asymmetry

        Features (from raw - if provided):
        - Raw peak amplitude (critical for distinguishing Class 2)
        - Raw energy
        """
        features = []

        # Use raw waveforms for amplitude if available, else use filtered
        amp_waveforms = raw_waveforms if raw_waveforms is not None else waveforms

        for i, wf in enumerate(waveforms):
            # Get corresponding raw waveform for amplitude
            wf_raw = amp_waveforms[i] if raw_waveforms is not None else wf

            # Baseline subtraction
            wf_centered = wf - np.mean(wf[:5])
            wf_raw_centered = wf_raw - np.mean(wf_raw[:5])

            # Peak amplitude and location (from filtered - more accurate location)
            peak_idx = np.argmax(wf_centered)
            peak_amp = wf_centered[peak_idx]

            # RAW peak amplitude (critical for Class 2 distinction!)
            raw_peak_amp = np.max(wf_raw_centered)

            # Trough (minimum)
            trough_idx = np.argmin(wf_centered)
            trough_amp = wf_centered[trough_idx]

            # Peak to trough
            peak_to_trough = peak_amp - trough_amp

            # Width at half maximum
            half_max = peak_amp / 2
            above_half = wf_centered > half_max
            if np.any(above_half):
                first_cross = np.argmax(above_half)
                last_cross = len(above_half) - np.argmax(above_half[::-1]) - 1
                width_half_max = last_cross - first_cross
            else:
                width_half_max = 0

            # Rise time (from 10% to 90% of peak on rising edge)
            threshold_10 = 0.1 * peak_amp
            threshold_90 = 0.9 * peak_amp
            rising_edge = wf_centered[:peak_idx + 1]

            rise_start = np.argmax(rising_edge > threshold_10) if np.any(rising_edge > threshold_10) else 0
            rise_end = np.argmax(rising_edge > threshold_90) if np.any(rising_edge > threshold_90) else peak_idx
            rise_time = rise_end - rise_start

            # Decay time (from 90% to 10% after peak)
            falling_edge = wf_centered[peak_idx:]
            if len(falling_edge) > 1:
                decay_start = np.argmax(falling_edge < threshold_90) if np.any(falling_edge < threshold_90) else 0
                decay_end = np.argmax(falling_edge < threshold_10) if np.any(falling_edge < threshold_10) else len(falling_edge) - 1
                decay_time = decay_end - decay_start
            else:
                decay_time = 0

            # Area under curve (absolute)
            area = np.trapz(np.abs(wf_centered))

            # Waveform energy (filtered)
            energy = np.sum(wf_centered ** 2)

            # RAW waveform energy (important for amplitude-based classification)
            raw_energy = np.sum(wf_raw_centered ** 2)

            # Slope features
            if peak_idx > 0:
                max_rise_slope = np.max(np.diff(wf_centered[:peak_idx + 1]))
            else:
                max_rise_slope = 0

            if peak_idx < len(wf_centered) - 1:
                max_fall_slope = np.min(np.diff(wf_centered[peak_idx:]))
            else:
                max_fall_slope = 0

            # Asymmetry (difference between pre-peak and post-peak areas)
            pre_peak_area = np.trapz(np.abs(wf_centered[:peak_idx + 1])) if peak_idx > 0 else 0
            post_peak_area = np.trapz(np.abs(wf_centered[peak_idx:])) if peak_idx < len(wf_centered) else 0
            asymmetry = (post_peak_area - pre_peak_area) / (post_peak_area + pre_peak_area + 1e-10)

            # Amplitude ratio (raw/filtered) - indicates noise level
            amp_ratio = raw_peak_amp / (peak_amp + 1e-10) if peak_amp > 0 else 1.0

            features.append([
                peak_amp,
                peak_idx - self.window_before,  # Relative to center
                trough_amp,
                peak_to_trough,
                width_half_max,
                rise_time,
                decay_time,
                area,
                energy,
                max_rise_slope,
                max_fall_slope,
                asymmetry,
                raw_peak_amp,      # NEW: Raw amplitude (critical for Class 2)
                raw_energy,        # NEW: Raw energy
                amp_ratio,         # NEW: Noise indicator
            ])

        return np.array(features)

    def fit(self, waveforms):
        """Fit the feature extractor on training waveforms."""
        # Filter waveforms to reduce noise while preserving amplitude
        waveforms_filtered = self.filter_waveforms(waveforms)

        # IMPORTANT: Do NOT normalize - amplitude is critical for distinguishing classes!
        # Analysis showed Class 2, 4, 5 have similar shapes but different amplitudes.
        # Normalization destroys this information and causes Class 2 collapse.

        # Fit waveform scaler on filtered (but NOT normalized) waveforms
        self.waveform_scaler = StandardScaler()
        waveforms_scaled = self.waveform_scaler.fit_transform(waveforms_filtered)

        # Fit PCA on filtered waveforms (captures both shape AND amplitude)
        self.pca = PCA(n_components=self.n_pca_components)
        self.pca.fit(waveforms_scaled)

        # Extract handcrafted features using BOTH filtered (for shape) and raw (for amplitude)
        # The raw amplitude is critical for distinguishing Class 2 from other classes
        handcrafted = self.extract_handcrafted_features(waveforms_filtered, raw_waveforms=waveforms)
        self.scaler = StandardScaler()
        self.scaler.fit(handcrafted)

        return self

    def filter_waveforms(self, waveforms, sample_rate=25000):
        """
        Apply bandpass filter to waveforms to reduce noise.
        Spike frequencies are typically 300-3000 Hz.
        """
        # Design bandpass filter for spike frequencies
        nyquist = sample_rate / 2
        low = 300 / nyquist
        high = 3000 / nyquist

        # Use gentle filtering to preserve spike shape
        b, a = signal.butter(2, [low, high], btype='band')

        filtered = []
        for wf in waveforms:
            # Apply filter with padding to avoid edge effects
            try:
                wf_filt = signal.filtfilt(b, a, wf, padlen=min(15, len(wf)-1))
            except ValueError:
                wf_filt = wf  # Fall back to unfiltered if too short
            filtered.append(wf_filt)

        return np.array(filtered)

    def normalize_waveforms(self, waveforms, filter_first=True):
        """
        Normalize waveforms to unit peak amplitude.
        This helps the classifier focus on shape rather than absolute amplitude.
        """
        # Optionally filter to reduce noise before normalization
        if filter_first:
            waveforms_clean = self.filter_waveforms(waveforms)
        else:
            waveforms_clean = waveforms

        normalized = []
        for wf, wf_clean in zip(waveforms, waveforms_clean):
            # Use filtered version for peak finding (more robust to noise)
            baseline = np.mean(wf_clean[:5])
            wf_centered = wf - baseline  # Center original waveform

            # Find peak in filtered version
            peak = np.max(np.abs(wf_clean - baseline))
            if peak > 0.1:  # Only normalize if there's a clear signal
                wf_norm = wf_centered / peak
            else:
                wf_norm = wf_centered

            normalized.append(wf_norm)

        return np.array(normalized)

    def transform(self, waveforms):
        """Transform waveforms to feature vectors."""
        if self.pca is None or self.scaler is None:
            raise ValueError("FeatureExtractor must be fit before transform")

        # Filter waveforms to reduce noise while preserving amplitude
        waveforms_filtered = self.filter_waveforms(waveforms)

        # PCA features on filtered waveforms (captures shape AND amplitude)
        waveforms_scaled = self.waveform_scaler.transform(waveforms_filtered)
        pca_features = self.pca.transform(waveforms_scaled)

        # Handcrafted features: filtered for shape, raw for amplitude
        # The raw amplitude is critical for distinguishing Class 2 from other classes
        handcrafted = self.extract_handcrafted_features(waveforms_filtered, raw_waveforms=waveforms)
        handcrafted_scaled = self.scaler.transform(handcrafted)

        # Combine all features
        combined = np.hstack([pca_features, handcrafted_scaled])

        return combined

    def fit_transform(self, waveforms):
        """Fit and transform in one step."""
        self.fit(waveforms)
        return self.transform(waveforms)


def extract_waveforms_at_indices(data, indices, window_before=30, window_after=30):
    """
    Extract waveforms from data at given spike indices.

    Parameters:
    -----------
    data : np.ndarray
        Raw signal
    indices : np.ndarray
        Spike indices (1-indexed)
    """
    waveforms = []
    valid_indices = []

    for idx in indices:
        idx_0 = int(idx) - 1  # Convert to 0-indexed
        start = idx_0 - window_before
        end = idx_0 + window_after

        if start >= 0 and end <= len(data):
            waveforms.append(data[start:end])
            valid_indices.append(idx)

    return np.array(waveforms), np.array(valid_indices)
