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

    def extract_handcrafted_features(self, waveforms):
        """
        Extract hand-crafted features from waveforms.

        Features:
        - Peak amplitude
        - Peak location (relative to window center)
        - Trough amplitude (minimum value)
        - Peak-to-trough amplitude
        - Waveform width at half maximum
        - Rise time (10% to 90% of peak)
        - Decay time (90% to 10% after peak)
        - Area under curve
        - Waveform energy
        """
        n_spikes = len(waveforms)
        features = []

        for wf in waveforms:
            # Normalize waveform for shape features
            wf_centered = wf - np.mean(wf[:5])  # Subtract baseline (first 5 samples)

            # Peak amplitude and location
            peak_idx = np.argmax(wf_centered)
            peak_amp = wf_centered[peak_idx]

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

            # Waveform energy
            energy = np.sum(wf_centered ** 2)

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
                asymmetry
            ])

        return np.array(features)

    def fit(self, waveforms):
        """Fit the feature extractor on training waveforms."""
        # Normalize waveforms for shape-based features (uses filtering internally)
        waveforms_norm = self.normalize_waveforms(waveforms)

        # Fit waveform scaler on normalized waveforms
        self.waveform_scaler = StandardScaler()
        waveforms_scaled = self.waveform_scaler.fit_transform(waveforms_norm)

        # Fit PCA on scaled normalized waveforms (captures shape, not amplitude)
        self.pca = PCA(n_components=self.n_pca_components)
        self.pca.fit(waveforms_scaled)

        # Filter waveforms for handcrafted features (more robust to noise)
        waveforms_filtered = self.filter_waveforms(waveforms)

        # Extract and fit scaler for handcrafted features (from filtered waveforms)
        handcrafted = self.extract_handcrafted_features(waveforms_filtered)
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

        # Normalize waveforms for shape-based features (uses filtering internally)
        waveforms_norm = self.normalize_waveforms(waveforms)

        # PCA features on normalized waveforms (shape-based)
        waveforms_scaled = self.waveform_scaler.transform(waveforms_norm)
        pca_features = self.pca.transform(waveforms_scaled)

        # Filter waveforms for handcrafted features (more robust to noise)
        waveforms_filtered = self.filter_waveforms(waveforms)

        # Handcrafted features from FILTERED waveforms (noise-robust)
        handcrafted = self.extract_handcrafted_features(waveforms_filtered)
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
