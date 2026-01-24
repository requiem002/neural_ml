"""
CNN Experiment for Spike Classification - FIXED ALIGNMENT VERSION

CRITICAL FIX: This version ensures training and inference use the SAME
waveform alignment. The original code had:
- Training: waveforms centered on Index (peaks at variable positions 37-49)
- Inference: waveforms with peaks forced to sample 41

This mismatch caused the CNN to learn features at wrong positions.

FIX: Both training and inference now extract waveforms with peaks
aligned at sample 30 (centered in 60-sample window).
"""

import argparse
import numpy as np
import scipy.io as sio
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pathlib import Path
from collections import Counter
import pickle


# =============================================================================
# ALIGNMENT CONSTANT - MUST BE SAME FOR TRAINING AND INFERENCE
# =============================================================================
ALIGN_PEAK_AT = 30  # Peak will be at sample 30 (centered in 60-sample window)
WINDOW_SIZE = 60


# =============================================================================
# FIXED EXTRACTION FUNCTIONS
# =============================================================================

def extract_waveforms_aligned(data, indices, classes=None, 
                               align_peak_at=ALIGN_PEAK_AT, 
                               window_size=WINDOW_SIZE,
                               search_radius=15):
    """
    Extract waveforms with peaks CONSISTENTLY aligned at a fixed position.
    
    This is the CORRECTED version for TRAINING data extraction.
    
    Parameters:
    -----------
    data : np.ndarray
        Raw signal (1D array)
    indices : np.ndarray
        Spike indices from ground truth (1-indexed)
    classes : np.ndarray or None
        Class labels (if available)
    align_peak_at : int
        Sample position where peak should be placed
    
    Returns:
    --------
    waveforms, valid_indices, valid_classes
    """
    waveforms = []
    valid_indices = []
    valid_classes = [] if classes is not None else None
    
    for i, idx in enumerate(indices):
        idx_0 = int(idx) - 1  # Convert to 0-indexed
        
        # Find the ACTUAL peak near this Index
        search_start = max(0, idx_0 - search_radius)
        search_end = min(len(data), idx_0 + search_radius)
        
        if search_end - search_start < 5:
            continue
        
        local_region = data[search_start:search_end]
        local_peak_offset = np.argmax(local_region)
        actual_peak = search_start + local_peak_offset
        
        # Extract waveform with peak at the ALIGNED position
        start = actual_peak - align_peak_at
        end = start + window_size
        
        if start < 0 or end > len(data):
            continue
        
        waveform = data[start:end]
        
        # Verify alignment (peak should be very close to align_peak_at)
        actual_peak_in_wf = np.argmax(waveform)
        if abs(actual_peak_in_wf - align_peak_at) > 3:
            continue
        
        waveforms.append(waveform)
        valid_indices.append(idx)
        if classes is not None:
            valid_classes.append(classes[i])
    
    result_waveforms = np.array(waveforms) if waveforms else np.empty((0, window_size))
    result_indices = np.array(valid_indices)
    result_classes = np.array(valid_classes) if classes is not None else None
    
    return result_waveforms, result_indices, result_classes


def detect_spikes_aligned(data, sample_rate=25000, threshold_factor=4.0,
                          min_spike_distance=30, 
                          align_peak_at=ALIGN_PEAK_AT,
                          window_size=WINDOW_SIZE):
    """
    Detect spikes with peaks aligned at a CONSISTENT position.
    
    This is the CORRECTED version for INFERENCE.
    MUST use the same align_peak_at value as training!
    
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
    
    return np.array(valid_peaks, dtype=np.int64), np.array(waveforms) if waveforms else np.empty((0, window_size))


# =============================================================================
# CNN MODEL (unchanged from original)
# =============================================================================

class DualBranchSpikeNet(nn.Module):
    """
    Dual-branch CNN for spike classification.
    Branch 1 (Shape): Multi-scale 1D CNN for waveform shape features
    Branch 2 (Amplitude): MLP for amplitude-based features
    """

    def __init__(self, waveform_size=60, num_classes=5, num_amp_features=8, dropout_rate=0.3):
        super(DualBranchSpikeNet, self).__init__()

        self.waveform_size = waveform_size
        self.num_classes = num_classes

        # Shape Branch (1D CNN)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.shape_fc = nn.Linear(64 * (waveform_size // 4), 48)

        # Amplitude Branch
        self.amp_fc1 = nn.Linear(num_amp_features, 32)
        self.amp_bn1 = nn.BatchNorm1d(32)
        self.amp_fc2 = nn.Linear(32, 32)
        self.amp_bn2 = nn.BatchNorm1d(32)

        # Combined Classifier
        self.combined_fc1 = nn.Linear(80, 48)
        self.combined_fc2 = nn.Linear(48, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, waveform, amp_features):
        # Shape Branch
        x = waveform.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu(self.bn2(self.conv2(x))))
        x = self.pool2(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        shape_emb = self.relu(self.shape_fc(x))

        # Amplitude Branch
        amp = self.relu(self.amp_bn1(self.amp_fc1(amp_features)))
        amp_emb = self.relu(self.amp_bn2(self.amp_fc2(amp)))

        # Combined
        combined = torch.cat([shape_emb, amp_emb], dim=1)
        combined = self.dropout(combined)
        combined = self.relu(self.combined_fc1(combined))
        combined = self.dropout(combined)
        output = self.combined_fc2(combined)

        return output


# =============================================================================
# CNN EXPERIMENT CLASS
# =============================================================================

class CNNExperimentFixed:
    """Main class for CNN training and prediction with FIXED alignment."""

    def __init__(self, device=None):
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available() 
            else 'cpu'
        )
        print(f"Using device: {self.device}")

        self.model = None
        self.amp_mean = None
        self.amp_std = None

        # Paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / 'datasets'
        self.model_dir = self.base_dir / 'models'
        self.output_dir = self.base_dir / 'predictions_cnn_fixed'

        self.model_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def load_d1_data_aligned(self):
        """
        Load D1 training data with CONSISTENT peak alignment.
        
        THIS IS THE KEY FIX: ensures training waveforms have peaks at
        the same position as inference waveforms.
        """
        print(f"\nLoading D1 with ALIGNED extraction (peak at sample {ALIGN_PEAK_AT})...")
        
        data = sio.loadmat(self.data_dir / 'D1.mat')
        d = data['d'].flatten()
        index = data['Index'].flatten()
        classes = data['Class'].flatten()
        
        # Use the FIXED aligned extraction
        waveforms, valid_indices, valid_classes = extract_waveforms_aligned(
            d, index, classes, align_peak_at=ALIGN_PEAK_AT
        )
        
        # Verify alignment
        peak_positions = [np.argmax(wf) for wf in waveforms]
        print(f"Extracted {len(waveforms)} waveforms")
        print(f"Peak positions: mean={np.mean(peak_positions):.1f}, std={np.std(peak_positions):.1f}")
        print(f"  (Should be ~{ALIGN_PEAK_AT} with std < 2)")
        
        return waveforms, valid_classes

    def extract_amplitude_features(self, waveforms, sample_rate=25000):
        """Extract amplitude-based features from waveforms."""
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
            wf_raw_centered = wf - baseline_raw
            baseline_filt = np.mean(wf_filtered[:5])
            wf_filt_centered = wf_filtered - baseline_filt

            # 1. Peak amplitude
            peak_idx = np.argmax(wf_filt_centered)
            peak_amp = wf_raw_centered[peak_idx]

            # 2. Energy
            energy = np.sum(wf_raw_centered ** 2)

            # 3. Amplitude ratio
            neg_trough = np.min(wf_raw_centered)
            amp_ratio = peak_amp / (abs(neg_trough) + 1e-10) if abs(neg_trough) > 0.01 else 1.0

            # 4. FWHM
            half_max = wf_filt_centered[peak_idx] / 2
            above_half = wf_filt_centered > half_max
            if np.any(above_half):
                crossings = np.where(np.diff(above_half.astype(int)))[0]
                if len(crossings) >= 2:
                    fwhm_samples = crossings[-1] - crossings[0]
                    fwhm_ms = fwhm_samples / sample_rate * 1000
                else:
                    fwhm_ms = 0.5
            else:
                fwhm_ms = 0.5

            # 5. Repolarization slope
            if peak_idx < len(wf_filt_centered) - 5:
                post_peak = wf_filt_centered[peak_idx:min(peak_idx + 20, len(wf_filt_centered))]
                if len(post_peak) > 2:
                    trough_idx_local = np.argmin(post_peak)
                    if trough_idx_local > 0:
                        repol_slope = (post_peak[trough_idx_local] - post_peak[0]) / (trough_idx_local / sample_rate * 1000)
                    else:
                        repol_slope = 0
                else:
                    repol_slope = 0
            else:
                repol_slope = 0

            # 6. Symmetry ratio
            pre_peak_area = np.sum(np.abs(wf_filt_centered[:peak_idx + 1])) if peak_idx > 0 else 1
            post_peak_area = np.sum(np.abs(wf_filt_centered[peak_idx:])) if peak_idx < len(wf_filt_centered) else 1
            symmetry = pre_peak_area / (post_peak_area + 1e-10)

            # 7. Peak-to-trough amplitude
            peak_to_trough = peak_amp - neg_trough

            # 8. Rise slope
            if peak_idx > 2:
                rising = wf_filt_centered[:peak_idx + 1]
                peak_filt = wf_filt_centered[peak_idx]
                thresh_10 = 0.1 * peak_filt
                thresh_90 = 0.9 * peak_filt
                rise_10_idx = np.where(rising > thresh_10)[0]
                rise_90_idx = np.where(rising > thresh_90)[0]
                if len(rise_10_idx) > 0 and len(rise_90_idx) > 0:
                    rise_time = (rise_90_idx[0] - rise_10_idx[0]) / sample_rate * 1000
                    rise_slope = (thresh_90 - thresh_10) / (rise_time + 1e-10)
                else:
                    rise_slope = 0
            else:
                rise_slope = 0

            features.append([
                peak_amp, energy, amp_ratio, fwhm_ms,
                repol_slope, symmetry, peak_to_trough, rise_slope
            ])

        return np.array(features)

    def preprocess_waveform(self, waveform, normalize_for_shape=True):
        """Preprocess waveform for shape branch."""
        baseline = np.mean(waveform[:5])
        wf_centered = waveform - baseline

        if normalize_for_shape:
            peak = np.max(np.abs(wf_centered))
            if peak > 0.1:
                return wf_centered / peak
            return wf_centered
        return wf_centered

    def augment_data(self, waveforms, labels, noise_levels=[0.3, 0.5, 0.8, 1.0],
                     scale_factors=[0.85, 0.95, 1.05, 1.15], jitter_range=2):
        """Augment training data."""
        augmented_waveforms = list(waveforms)
        augmented_labels = list(labels)

        # Noise augmentation
        for noise_level in noise_levels:
            for wf, lbl in zip(waveforms, labels):
                noisy_wf = wf + np.random.randn(len(wf)) * noise_level
                augmented_waveforms.append(noisy_wf)
                augmented_labels.append(lbl)

        # Amplitude scaling
        for scale in scale_factors:
            for wf, lbl in zip(waveforms, labels):
                baseline = np.mean(wf[:5])
                wf_centered = wf - baseline
                scaled_wf = wf_centered * scale + baseline
                augmented_waveforms.append(scaled_wf)
                augmented_labels.append(lbl)

        # Time jitter
        for wf, lbl in zip(waveforms, labels):
            jitter = np.random.randint(-jitter_range, jitter_range + 1)
            if jitter != 0:
                jittered_wf = np.roll(wf, jitter)
                if jitter > 0:
                    jittered_wf[:jitter] = wf[0]
                else:
                    jittered_wf[jitter:] = wf[-1]
                augmented_waveforms.append(jittered_wf)
                augmented_labels.append(lbl)

        return np.array(augmented_waveforms), np.array(augmented_labels)

    def prepare_data(self, waveforms, labels=None, fit_scalers=False):
        """Prepare data for the CNN."""
        waveforms_normalized = np.array([
            self.preprocess_waveform(wf, normalize_for_shape=True) for wf in waveforms
        ])

        amp_features = self.extract_amplitude_features(waveforms)

        if fit_scalers:
            self.amp_mean = np.mean(amp_features, axis=0)
            self.amp_std = np.std(amp_features, axis=0) + 1e-10

        amp_features = (amp_features - self.amp_mean) / self.amp_std

        if labels is not None:
            return waveforms_normalized, amp_features, labels
        return waveforms_normalized, amp_features

    def train(self, epochs=100, batch_size=64, lr=0.001, patience=15):
        """Train the CNN classifier with ALIGNED data."""
        print("\n" + "=" * 60)
        print("CNN TRAINING (FIXED ALIGNMENT)")
        print("=" * 60)

        # Load data with ALIGNED extraction
        waveforms, classes = self.load_d1_data_aligned()

        print(f"\nTotal waveforms: {len(waveforms)}")
        print("Class distribution:")
        for c in range(1, 6):
            count = np.sum(classes == c)
            print(f"  Class {c}: {count} ({100*count/len(classes):.1f}%)")

        # Augment data
        print("\nAugmenting data...")
        aug_waveforms, aug_labels = self.augment_data(waveforms, classes)
        print(f"Augmented samples: {len(aug_waveforms)}")

        # Train/val split
        train_wf, val_wf, train_lbl, val_lbl = train_test_split(
            aug_waveforms, aug_labels, test_size=0.2, random_state=42, stratify=aug_labels
        )

        print(f"Training samples: {len(train_wf)}")
        print(f"Validation samples: {len(val_wf)}")

        # Prepare data
        print("\nPreparing data...")
        train_wf_norm, train_amp, train_lbl = self.prepare_data(train_wf, train_lbl, fit_scalers=True)
        val_wf_norm, val_amp, val_lbl = self.prepare_data(val_wf, val_lbl)

        # Convert to tensors
        X_wf_train = torch.FloatTensor(train_wf_norm).to(self.device)
        X_amp_train = torch.FloatTensor(train_amp).to(self.device)
        y_train = torch.LongTensor(train_lbl - 1).to(self.device)

        X_wf_val = torch.FloatTensor(val_wf_norm).to(self.device)
        X_amp_val = torch.FloatTensor(val_amp).to(self.device)
        y_val = torch.LongTensor(val_lbl - 1).to(self.device)

        # Class-weighted sampling
        class_counts = Counter(train_lbl)
        weights = [1.0 / class_counts[lbl] for lbl in train_lbl]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_dataset = TensorDataset(X_wf_train, X_amp_train, y_train)
        val_dataset = TensorDataset(X_wf_val, X_amp_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = DualBranchSpikeNet(waveform_size=WINDOW_SIZE, num_classes=5, num_amp_features=8).to(self.device)

        # Loss with class weights
        class_weights = torch.FloatTensor([1.0 / class_counts[c] for c in range(1, 6)]).to(self.device)
        class_weights = class_weights / class_weights.sum() * 5
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        # Training loop
        print("\nTraining...")
        best_val_f1 = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_wf, batch_amp, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_wf, batch_amp)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_acc = 100. * train_correct / train_total

            # Validation
            self.model.eval()
            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch_wf, batch_amp, batch_y in val_loader:
                    outputs = self.model(batch_wf, batch_amp)
                    _, predicted = outputs.max(1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())

            val_f1 = f1_score(val_true, val_preds, average='weighted')
            val_acc = 100. * sum(p == t for p, t in zip(val_preds, val_true)) / len(val_true)

            scheduler.step(val_f1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss={train_loss/len(train_loader):.4f}, "
                      f"Train Acc={train_acc:.2f}%, "
                      f"Val Acc={val_acc:.2f}%, "
                      f"Val F1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
        print("\n" + "=" * 60)
        print("FINAL VALIDATION RESULTS")
        print("=" * 60)

        self.model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_wf, batch_amp, batch_y in val_loader:
                outputs = self.model(batch_wf, batch_amp)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy() + 1)
                val_true.extend(batch_y.cpu().numpy() + 1)

        print("\nClassification Report:")
        print(classification_report(val_true, val_preds))
        print("\nConfusion Matrix:")
        print(confusion_matrix(val_true, val_preds))
        print(f"\nBest Validation F1: {best_val_f1:.4f}")

        self.save_model()
        return best_val_f1

    def save_model(self):
        """Save trained model and scalers."""
        state = {
            'model_state': self.model.state_dict(),
            'amp_mean': self.amp_mean,
            'amp_std': self.amp_std,
            'align_peak_at': ALIGN_PEAK_AT,  # Save alignment setting!
        }
        filepath = self.model_dir / 'cnn_model_fixed.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"\nModel saved to: {filepath}")

    def load_model(self):
        """Load trained model and scalers."""
        filepath = self.model_dir / 'cnn_model_fixed.pkl'
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.model = DualBranchSpikeNet(waveform_size=WINDOW_SIZE, num_classes=5, num_amp_features=8).to(self.device)
        self.model.load_state_dict(state['model_state'])
        self.amp_mean = state['amp_mean']
        self.amp_std = state['amp_std']
        
        saved_alignment = state.get('align_peak_at', 30)
        if saved_alignment != ALIGN_PEAK_AT:
            print(f"WARNING: Model trained with align_peak_at={saved_alignment}, "
                  f"but current setting is {ALIGN_PEAK_AT}")

        print(f"Model loaded from: {filepath}")

    def predict_dataset(self, dataset_name, threshold_factor=4.0, verbose=True):
        """Generate predictions for a single dataset using ALIGNED extraction."""
        data = sio.loadmat(self.data_dir / f'{dataset_name}.mat')
        d = data['d'].flatten()

        print(f"\n{'='*50}")
        print(f"Processing {dataset_name} (ALIGNED at sample {ALIGN_PEAK_AT})")
        print(f"{'='*50}")

        # Detect spikes with ALIGNED extraction (SAME as training!)
        indices, waveforms = detect_spikes_aligned(
            d, 
            threshold_factor=threshold_factor,
            align_peak_at=ALIGN_PEAK_AT
        )

        print(f"Detected {len(indices)} spikes")

        if len(indices) == 0:
            return np.array([]), np.array([])

        # Verify alignment
        peak_positions = [np.argmax(wf) for wf in waveforms[:100]]
        print(f"Peak positions (first 100): mean={np.mean(peak_positions):.1f}, std={np.std(peak_positions):.1f}")

        # Prepare data
        wf_norm, amp_features = self.prepare_data(waveforms)

        # Predict
        self.model.eval()
        X_wf = torch.FloatTensor(wf_norm).to(self.device)
        X_amp = torch.FloatTensor(amp_features).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_wf, X_amp)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = outputs.max(1)
            classes = predicted.cpu().numpy() + 1
            confidences = probs.max(axis=1)

        # Print class distribution
        class_dist = Counter(classes)
        print("Class distribution:")
        for c in sorted(class_dist.keys()):
            print(f"  Class {c}: {class_dist[c]} ({100*class_dist[c]/len(classes):.1f}%)")

        print(f"Average confidence: {np.mean(confidences):.2%}")

        return indices, classes

    def validate_on_d1(self):
        """
        Validate the complete pipeline on D1 where we have ground truth.
        This is the key test to verify alignment is correct.
        """
        print("\n" + "=" * 60)
        print("D1 VALIDATION (Ground Truth Test)")
        print("=" * 60)

        data = sio.loadmat(self.data_dir / 'D1.mat')
        d = data['d'].flatten()
        gt_indices = data['Index'].flatten()
        gt_classes = data['Class'].flatten()

        print(f"Ground truth: {len(gt_indices)} spikes")

        # Detect using ALIGNED method
        detected_indices, detected_waveforms = detect_spikes_aligned(
            d, threshold_factor=4.5, align_peak_at=ALIGN_PEAK_AT
        )

        print(f"Detected: {len(detected_indices)} spikes")

        # Prepare and predict
        wf_norm, amp_features = self.prepare_data(detected_waveforms)
        
        self.model.eval()
        X_wf = torch.FloatTensor(wf_norm).to(self.device)
        X_amp = torch.FloatTensor(amp_features).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_wf, X_amp)
            _, predicted = outputs.max(1)
            pred_classes = predicted.cpu().numpy() + 1

        # Match detected to ground truth
        tolerance = 50
        matches = []
        gt_used = set()

        for det_idx, pred_cls in zip(detected_indices, pred_classes):
            distances = np.abs(gt_indices - det_idx)
            closest = np.argmin(distances)

            if distances[closest] <= tolerance and closest not in gt_used:
                matches.append((gt_classes[closest], pred_cls))
                gt_used.add(closest)

        gt_matched = [m[0] for m in matches]
        pred_matched = [m[1] for m in matches]

        print(f"\nMatched: {len(matches)} / {len(gt_indices)} ({100*len(matches)/len(gt_indices):.1f}%)")

        # Compute metrics
        cm = confusion_matrix(gt_matched, pred_matched, labels=[1, 2, 3, 4, 5])
        
        print("\nConfusion Matrix:")
        print("     Predicted")
        print("     1    2    3    4    5")
        for i, row in enumerate(cm):
            print(f"  {i+1}: {row}")

        overall_acc = 100 * np.trace(cm) / cm.sum()
        print(f"\nOverall Classification Accuracy: {overall_acc:.1f}%")

        # Per-class accuracy
        print("\nPer-class accuracy:")
        for c in range(5):
            class_total = cm[c].sum()
            class_correct = cm[c, c]
            acc = 100 * class_correct / class_total if class_total > 0 else 0
            print(f"  Class {c+1}: {acc:.1f}%")

        return overall_acc


def main():
    parser = argparse.ArgumentParser(description='CNN Experiment (FIXED ALIGNMENT)')
    parser.add_argument('--train', action='store_true', help='Train the CNN model')
    parser.add_argument('--validate', action='store_true', help='Validate on D1')
    parser.add_argument('--predict', action='store_true', help='Run predictions on D2-D6')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    args = parser.parse_args()

    experiment = CNNExperimentFixed()

    if args.train:
        experiment.train(epochs=args.epochs)

    if args.validate:
        experiment.load_model()
        experiment.validate_on_d1()

    if args.predict:
        experiment.load_model()
        
        # Dataset-specific threshold factors
        thresholds = {
            'D2': 3.7,
            'D3': 3.3,
            'D4': 3.0,
            'D5': 2.8,
            'D6': 2.6,
        }
        
        for ds, thresh in thresholds.items():
            indices, classes = experiment.predict_dataset(ds, threshold_factor=thresh)

    if not any([args.train, args.validate, args.predict]):
        print("Running full pipeline: train → validate → predict")
        experiment.train(epochs=args.epochs)
        experiment.validate_on_d1()


if __name__ == '__main__':
    main()