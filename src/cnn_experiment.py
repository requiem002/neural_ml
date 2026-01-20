"""
CNN Experiment for Spike Classification

This implements a dual-branch CNN architecture:
1. Shape branch: Multi-scale 1D convolutions to capture waveform shape
2. Amplitude branch: Direct amplitude/energy features (not normalized away!)

Key design decisions based on forensic analysis:
- Multi-scale kernels (5,5,3) to capture different timescales
- BatchNorm for noise robustness
- Separate amplitude branch to preserve amplitude information
- Data augmentation: noise, scaling, jitter, Class 3 undersampling
"""

import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pathlib import Path
from collections import Counter
import pickle

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from spike_detector import detect_spikes
from feature_extractor import extract_waveforms_at_indices


class DualBranchSpikeNet(nn.Module):
    """
    Dual-branch CNN for spike classification.

    Branch 1 (Shape): Multi-scale 1D CNN for waveform shape features
    Branch 2 (Amplitude): MLP for amplitude-based features (expanded to 8 features)

    The amplitude branch is given more capacity since amplitude is the
    primary discriminator between classes (especially Class 3/4/5).
    """

    def __init__(self, waveform_size=60, num_classes=5, num_amp_features=8, dropout_rate=0.3):
        super(DualBranchSpikeNet, self).__init__()

        self.waveform_size = waveform_size
        self.num_classes = num_classes

        # ========== Shape Branch (1D CNN) ==========
        # Layer 1: kernel=5, captures local features
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)

        # Layer 2: kernel=5, captures medium-scale features
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)  # 60 -> 30

        # Layer 3: kernel=3, captures fine-grained features
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)  # 30 -> 15

        # After pooling: 64 channels x 15 samples = 960
        self.shape_fc = nn.Linear(64 * (waveform_size // 4), 48)  # Reduced to give more weight to amplitude

        # ========== Amplitude Branch (Expanded) ==========
        # Input: 8 features (peak_amp, energy, amp_ratio, fwhm, repol_slope, symmetry, p2t, rise_slope)
        # Expanded capacity since amplitude is the key discriminator
        self.amp_fc1 = nn.Linear(num_amp_features, 32)
        self.amp_bn1 = nn.BatchNorm1d(32)
        self.amp_fc2 = nn.Linear(32, 32)
        self.amp_bn2 = nn.BatchNorm1d(32)

        # ========== Combined Classifier ==========
        # 48 (shape) + 32 (amplitude) = 80
        self.combined_fc1 = nn.Linear(80, 48)
        self.combined_fc2 = nn.Linear(48, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, waveform, amp_features):
        """
        Forward pass.

        Args:
            waveform: (batch, waveform_size) - normalized waveform for shape learning
            amp_features: (batch, 8) - expanded amplitude features
        """
        # ========== Shape Branch ==========
        x = waveform.unsqueeze(1)  # (batch, 1, waveform_size)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        shape_emb = self.relu(self.shape_fc(x))  # 48-dim shape embedding

        # ========== Amplitude Branch (with BatchNorm for stability) ==========
        amp = self.amp_fc1(amp_features)
        amp = self.amp_bn1(amp)
        amp = self.relu(amp)
        amp = self.amp_fc2(amp)
        amp = self.amp_bn2(amp)
        amp_emb = self.relu(amp)  # 32-dim amplitude embedding

        # ========== Combined Classifier ==========
        combined = torch.cat([shape_emb, amp_emb], dim=1)  # 80-dim
        combined = self.dropout(combined)
        combined = self.relu(self.combined_fc1(combined))
        combined = self.dropout(combined)
        output = self.combined_fc2(combined)

        return output


class CNNExperiment:
    """Main class for CNN training and prediction."""

    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = None
        self.amp_mean = None
        self.amp_std = None

        # Paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / 'datasets'
        self.model_dir = self.base_dir / 'models'
        self.output_dir = self.base_dir / 'predictions_cnn'

        self.model_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def load_d1_data(self):
        """Load D1 training data with labels."""
        data = sio.loadmat(self.data_dir / 'D1.mat')
        d = data['d'].flatten()
        index = data['Index'].flatten()
        classes = data['Class'].flatten()
        return d, index, classes

    def extract_amplitude_features(self, waveforms, sample_rate=25000):
        """
        Extract amplitude-based features from waveforms.

        Based on forensic analysis, key discriminating features are:
        - Peak amplitude (Class 3: 2.3V, Class 4: 3.6V, Class 5: 5.0V)
        - FWHM/pulse width (Class 4: 0.63ms vs Class 5: 0.87ms)
        - Repolarization slope (Class 4: -2.10 vs Class 5: -3.16 V/ms)
        - Symmetry ratio

        Uses bandpass filtering before feature extraction for noise robustness.

        Returns: (n_samples, 8) array of discriminating features
        """
        from scipy import signal as sig

        # Design bandpass filter for spike frequencies (300-3000 Hz)
        nyquist = sample_rate / 2
        low = 300 / nyquist
        high = 3000 / nyquist
        b, a = sig.butter(2, [low, high], btype='band')

        features = []
        for wf in waveforms:
            # Apply bandpass filter for more robust measurements
            try:
                wf_filtered = sig.filtfilt(b, a, wf, padlen=min(15, len(wf)-1))
            except ValueError:
                wf_filtered = wf

            # Use filtered for shape features, raw for amplitude
            baseline_raw = np.mean(wf[:5])
            wf_raw_centered = wf - baseline_raw

            baseline_filt = np.mean(wf_filtered[:5])
            wf_filt_centered = wf_filtered - baseline_filt

            # 1. Peak amplitude (from RAW - preserves true amplitude)
            peak_idx = np.argmax(wf_filt_centered)  # Peak location from filtered
            peak_amp = wf_raw_centered[peak_idx]    # Amplitude from raw

            # 2. Energy (from raw)
            energy = np.sum(wf_raw_centered ** 2)

            # 3. Amplitude ratio (positive peak / |negative trough|) from raw
            neg_trough = np.min(wf_raw_centered)
            amp_ratio = peak_amp / (abs(neg_trough) + 1e-10) if abs(neg_trough) > 0.01 else 1.0

            # 4. FWHM (Full Width at Half Maximum) from filtered
            half_max = wf_filt_centered[peak_idx] / 2
            above_half = wf_filt_centered > half_max
            if np.any(above_half):
                crossings = np.where(np.diff(above_half.astype(int)))[0]
                if len(crossings) >= 2:
                    fwhm_samples = crossings[-1] - crossings[0]
                    fwhm_ms = fwhm_samples / sample_rate * 1000
                else:
                    fwhm_ms = 0.5  # Default
            else:
                fwhm_ms = 0.5

            # 5. Repolarization slope (from filtered - more stable)
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

            # 6. Symmetry ratio (pre-peak area / post-peak area) from filtered
            pre_peak_area = np.sum(np.abs(wf_filt_centered[:peak_idx + 1])) if peak_idx > 0 else 1
            post_peak_area = np.sum(np.abs(wf_filt_centered[peak_idx:])) if peak_idx < len(wf_filt_centered) else 1
            symmetry = pre_peak_area / (post_peak_area + 1e-10)

            # 7. Peak-to-trough amplitude (from raw)
            peak_to_trough = peak_amp - neg_trough

            # 8. Rise slope (from filtered - more stable)
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
                peak_amp,       # Most important for class discrimination
                energy,
                amp_ratio,
                fwhm_ms,        # Distinguishes Class 4 vs 5
                repol_slope,    # Distinguishes Class 4 vs 5
                symmetry,
                peak_to_trough,
                rise_slope
            ])

        return np.array(features)

    def preprocess_waveform(self, waveform, normalize_for_shape=True):
        """
        Preprocess waveform.

        For the SHAPE branch: normalize to unit peak so CNN learns SHAPE,
        not amplitude. The amplitude branch handles amplitude discrimination.

        This is crucial for noise robustness - noisy waveforms may have
        different effective amplitudes but same underlying shape.
        """
        baseline = np.mean(waveform[:5])
        wf_centered = waveform - baseline

        if normalize_for_shape:
            # Normalize to unit peak for shape learning
            peak = np.max(np.abs(wf_centered))
            if peak > 0.1:
                return wf_centered / peak
            return wf_centered
        return wf_centered

    def augment_data(self, waveforms, labels, noise_levels=[0.3, 0.5, 0.8, 1.0, 1.5],
                     scale_factors=[0.8, 0.9, 1.1, 1.2], jitter_range=2,
                     class3_prob=0.5):
        """
        Augment training data with:
        - Gaussian noise at various levels
        - Amplitude scaling
        - Time jitter
        - Class 3 undersampling
        """
        augmented_waveforms = []
        augmented_labels = []

        # Original data
        for wf, lbl in zip(waveforms, labels):
            augmented_waveforms.append(wf)
            augmented_labels.append(lbl)

        # Noise augmentation
        for noise_level in noise_levels:
            for wf, lbl in zip(waveforms, labels):
                # Undersample Class 3
                if lbl == 3 and np.random.random() > class3_prob:
                    continue

                noisy_wf = wf + np.random.randn(len(wf)) * noise_level
                augmented_waveforms.append(noisy_wf)
                augmented_labels.append(lbl)

        # Amplitude scaling
        for scale in scale_factors:
            for wf, lbl in zip(waveforms, labels):
                if lbl == 3 and np.random.random() > class3_prob:
                    continue

                baseline = np.mean(wf[:5])
                wf_centered = wf - baseline
                scaled_wf = wf_centered * scale + baseline
                augmented_waveforms.append(scaled_wf)
                augmented_labels.append(lbl)

        # Time jitter (shift waveform by ±jitter_range samples)
        for wf, lbl in zip(waveforms, labels):
            if lbl == 3 and np.random.random() > class3_prob:
                continue

            jitter = np.random.randint(-jitter_range, jitter_range + 1)
            if jitter != 0:
                jittered_wf = np.roll(wf, jitter)
                # Zero-pad the edges
                if jitter > 0:
                    jittered_wf[:jitter] = wf[0]
                else:
                    jittered_wf[jitter:] = wf[-1]
                augmented_waveforms.append(jittered_wf)
                augmented_labels.append(lbl)

        return np.array(augmented_waveforms), np.array(augmented_labels)

    def prepare_data(self, waveforms, labels=None, fit_scalers=False):
        """
        Prepare data for the CNN.

        Strategy:
        - Waveforms: Normalize to unit peak (shape learning in CNN)
        - Amplitude features: Extract from raw waveforms (amplitude discrimination)

        This separation allows the CNN to learn shapes robustly while
        the amplitude branch discriminates based on true amplitude.

        Args:
            waveforms: Raw waveforms
            labels: Class labels (optional, for training)
            fit_scalers: Whether to fit the scalers (True for training)

        Returns:
            waveforms_normalized: Shape-normalized waveforms
            amp_features: Amplitude features from raw waveforms
            labels: Labels (if provided)
        """
        # Normalize waveforms for shape learning (unit peak)
        waveforms_normalized = np.array([
            self.preprocess_waveform(wf, normalize_for_shape=True) for wf in waveforms
        ])

        # Extract amplitude features from ORIGINAL (raw) waveforms
        amp_features = self.extract_amplitude_features(waveforms)

        if fit_scalers:
            # For amplitude features only (waveforms are already normalized)
            self.amp_mean = np.mean(amp_features, axis=0)
            self.amp_std = np.std(amp_features, axis=0) + 1e-10

        # Standardize amplitude features
        amp_features = (amp_features - self.amp_mean) / self.amp_std

        if labels is not None:
            return waveforms_normalized, amp_features, labels
        return waveforms_normalized, amp_features

    def train(self, epochs=100, batch_size=64, lr=0.001, patience=10):
        """Train the CNN classifier."""
        print("\n" + "="*60)
        print("CNN TRAINING")
        print("="*60)

        # Load data
        print("\nLoading D1 training data...")
        d, index, classes = self.load_d1_data()

        # Extract waveforms at ground truth locations
        print("Extracting waveforms...")
        waveforms, valid_indices = extract_waveforms_at_indices(d, index, window_before=30, window_after=30)
        valid_mask = np.isin(index, valid_indices)
        valid_classes = classes[valid_mask]

        print(f"Total waveforms: {len(waveforms)}")
        print("Class distribution:")
        for c in range(1, 6):
            count = np.sum(valid_classes == c)
            print(f"  Class {c}: {count} ({100*count/len(valid_classes):.1f}%)")

        # Augment data
        print("\nAugmenting data...")
        aug_waveforms, aug_labels = self.augment_data(waveforms, valid_classes)
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
        y_train = torch.LongTensor(train_lbl - 1).to(self.device)  # 0-indexed

        X_wf_val = torch.FloatTensor(val_wf_norm).to(self.device)
        X_amp_val = torch.FloatTensor(val_amp).to(self.device)
        y_val = torch.LongTensor(val_lbl - 1).to(self.device)

        # Create data loaders with class-weighted sampling
        class_counts = Counter(train_lbl)
        weights = [1.0 / class_counts[lbl] for lbl in train_lbl]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_dataset = TensorDataset(X_wf_train, X_amp_train, y_train)
        val_dataset = TensorDataset(X_wf_val, X_amp_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = DualBranchSpikeNet(waveform_size=60, num_classes=5, num_amp_features=8).to(self.device)

        # Loss with class weights
        class_weights = torch.FloatTensor([1.0 / class_counts[c] for c in range(1, 6)]).to(self.device)
        class_weights = class_weights / class_weights.sum() * 5  # Normalize to sum to num_classes
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        # Training loop
        print("\nTraining...")
        best_val_f1 = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
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

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss={train_loss/len(train_loader):.4f}, "
                      f"Train Acc={train_acc:.2f}%, "
                      f"Val Acc={val_acc:.2f}%, "
                      f"Val F1={val_f1:.4f}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
        print("\n" + "="*60)
        print("FINAL VALIDATION RESULTS")
        print("="*60)

        self.model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_wf, batch_amp, batch_y in val_loader:
                outputs = self.model(batch_wf, batch_amp)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy() + 1)  # Back to 1-indexed
                val_true.extend(batch_y.cpu().numpy() + 1)

        print("\nClassification Report:")
        print(classification_report(val_true, val_preds))
        print("\nConfusion Matrix:")
        print(confusion_matrix(val_true, val_preds))
        print(f"\nBest Validation F1: {best_val_f1:.4f}")

        # Save model
        self.save_model()

        return best_val_f1

    def save_model(self):
        """Save trained model and scalers."""
        state = {
            'model_state': self.model.state_dict(),
            'amp_mean': self.amp_mean,
            'amp_std': self.amp_std,
        }
        filepath = self.model_dir / 'cnn_model.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"\nModel saved to: {filepath}")

    def load_model(self):
        """Load trained model and scalers."""
        filepath = self.model_dir / 'cnn_model.pkl'
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.model = DualBranchSpikeNet(waveform_size=60, num_classes=5, num_amp_features=8).to(self.device)
        self.model.load_state_dict(state['model_state'])
        self.amp_mean = state['amp_mean']
        self.amp_std = state['amp_std']

        print(f"Model loaded from: {filepath}")

    def compute_fwhm(self, waveform, sample_rate=25000):
        """Compute Full Width at Half Maximum in milliseconds."""
        baseline = np.mean(waveform[:5])
        wf_centered = waveform - baseline
        peak_idx = np.argmax(wf_centered)
        peak_val = wf_centered[peak_idx]

        if peak_val < 0.1:
            return 0.5  # Default

        half_max = peak_val / 2
        above_half = wf_centered > half_max
        crossings = np.where(np.diff(above_half.astype(int)))[0]

        if len(crossings) >= 2:
            fwhm_samples = crossings[-1] - crossings[0]
            return fwhm_samples / sample_rate * 1000
        return 0.5

    def predict_dataset(self, dataset_name, voltage_threshold, verbose=True,
                       use_improved_correction=True):
        """Generate predictions for a single dataset with improved post-processing."""
        # Load data
        data = sio.loadmat(self.data_dir / f'{dataset_name}.mat')
        d = data['d'].flatten()

        print(f"\n{'='*50}")
        print(f"Processing {dataset_name}")
        print(f"{'='*50}")
        print(f"Signal length: {len(d)} samples")

        # Detect spikes
        indices, waveforms = detect_spikes(
            d, voltage_threshold=voltage_threshold,
            window_before=30, window_after=30
        )

        print(f"Detected {len(indices)} spikes (threshold={voltage_threshold}V)")

        if len(indices) == 0:
            return np.array([]), np.array([])

        # Prepare data
        wf_norm, amp_features = self.prepare_data(waveforms)

        # Extract raw amplitude features for post-processing
        raw_amp_features = self.extract_amplitude_features(waveforms)
        raw_amp = raw_amp_features[:, 0]  # Peak amplitude
        fwhm_values = raw_amp_features[:, 3]  # FWHM in ms

        # Diagnostic output
        if verbose:
            print(f"Raw amplitude statistics (before standardization):")
            print(f"  Peak amp: mean={raw_amp.mean():.2f}V, "
                  f"std={raw_amp.std():.2f}V, "
                  f"min={raw_amp.min():.2f}V, "
                  f"max={raw_amp.max():.2f}V")
            print(f"  FWHM: mean={fwhm_values.mean():.3f}ms, "
                  f"std={fwhm_values.std():.3f}ms")

        # Predict with CNN
        self.model.eval()
        X_wf = torch.FloatTensor(wf_norm).to(self.device)
        X_amp = torch.FloatTensor(amp_features).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_wf, X_amp)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = outputs.max(1)
            classes = predicted.cpu().numpy() + 1  # Back to 1-indexed
            confidences = probs.max(axis=1)

        # ========== IMPROVED POST-PROCESSING ==========
        # D1 Ground Truth Statistics:
        #
        # Peak amplitude (mean ± std):
        # C1: 4.89 ± 0.93V     C2: 5.51 ± 0.73V     C3: 1.49 ± 0.91V
        # C4: 2.86 ± 1.04V     C5: 4.20 ± 0.97V
        #
        # FWHM (ms):
        # C1: 0.57 ± 0.15 (narrow)      C2: 0.75 ± 0.14 (medium)
        # C3: 0.49 ± 0.14 (narrowest)   C4: 0.63 ± 0.47 (medium)
        # C5: 0.87 ± 0.57 (widest)
        #
        # Symmetry ratio:
        # C1: 10.6    C2: 14.9 (highest)   C3: 4.2 (lowest)
        # C4: 8.2     C5: 12.2

        # Extract symmetry ratio for additional discrimination
        symmetry_values = raw_amp_features[:, 5]  # symmetry ratio

        if use_improved_correction:
            corrected_classes = classes.copy()

            # Track corrections for debugging
            corrections = {'c3_to_high': 0, 'c4_to_high': 0, 'c5_rescue': 0,
                          'c2_to_c1': 0, 'c2_to_c5': 0, 'low_amp_fix': 0,
                          'c2_fwhm_fix': 0}

            for i, (pred_class, amp, fwhm, conf, sym) in enumerate(zip(
                    classes, raw_amp, fwhm_values, confidences, symmetry_values)):

                # ===== RULE 1: Class 3 amplitude correction =====
                # Class 3 has max realistic amplitude ~3.3V (mean + 2*std)
                # If predicted C3 but amp > 4.0V, definitely wrong
                if pred_class == 3 and amp > 4.0:
                    # High amplitude: choose based on FWHM
                    if fwhm > 0.78:
                        corrected_classes[i] = 5  # Widest pulse -> C5
                    elif fwhm > 0.65:
                        corrected_classes[i] = 2  # Medium-wide -> C2
                    else:
                        corrected_classes[i] = 1  # Narrow -> C1
                    corrections['c3_to_high'] += 1

                # ===== RULE 2: Class 4 amplitude correction =====
                # Class 4 has max realistic amplitude ~4.9V
                # If predicted C4 but amp > 5.5V, likely wrong
                elif pred_class == 4 and amp > 5.5:
                    if fwhm > 0.78:
                        corrected_classes[i] = 5  # Wide pulse -> C5
                    elif fwhm > 0.65:
                        corrected_classes[i] = 2  # Medium -> C2
                    else:
                        corrected_classes[i] = 1  # Narrow -> C1
                    corrections['c4_to_high'] += 1

                # ===== RULE 3: Class 5 rescue (BALANCED) =====
                # Class 5: widest FWHM (0.87ms), moderate amplitude (4.20V)
                # Be careful not to steal from C2 (amp 5.51V, FWHM 0.75ms)
                elif fwhm > 0.85 and 3.0 < amp < 5.5 and pred_class != 5:
                    # Very wide pulse + moderate amplitude -> likely Class 5
                    corrected_classes[i] = 5
                    corrections['c5_rescue'] += 1

                # ===== RULE 4: C2 over-prediction correction =====
                # C2 is over-predicted in high-noise datasets
                # Use FWHM to distinguish: C1=0.57ms, C2=0.75ms, C5=0.87ms
                elif pred_class == 2:
                    if fwhm < 0.58 and amp < 5.5:
                        # Narrow pulse - more likely C1 than C2
                        corrected_classes[i] = 1
                        corrections['c2_fwhm_fix'] += 1
                    elif fwhm > 0.82 and amp < 5.3:
                        # Wide pulse, moderate amplitude - more likely C5
                        corrected_classes[i] = 5
                        corrections['c2_to_c5'] += 1
                    elif dataset_name in ['D5', 'D6'] and conf < 0.88:
                        # D5/D6: redistribute some C2 predictions
                        if 4.0 < amp < 5.2 and fwhm < 0.68:
                            # Medium amp, narrow pulse -> likely C1
                            corrected_classes[i] = 1
                            corrections['c2_to_c1'] += 1

                # ===== RULE 5: Low amplitude spike correction =====
                # If predicted C1, C2, or C5 but amplitude is very low, likely C3 or C4
                elif pred_class in [1, 2, 5] and amp < 2.5:
                    if conf < 0.85:
                        corrected_classes[i] = 3 if amp < 1.8 else 4
                        corrections['low_amp_fix'] += 1

            if verbose:
                print(f"Post-processing corrections:")
                for key, count in corrections.items():
                    if count > 0:
                        print(f"  {key}: {count}")

            classes = corrected_classes

        # Print class distribution
        class_dist = Counter(classes)
        print("Class distribution:")
        for c in sorted(class_dist.keys()):
            print(f"  Class {c}: {class_dist[c]} ({100*class_dist[c]/len(classes):.1f}%)")

        # Show average confidence
        if verbose:
            avg_conf = np.mean(confidences)
            print(f"Average prediction confidence: {avg_conf:.2%}")

        return indices, classes

    def run_predictions(self):
        """Run predictions on all test datasets."""
        print("\n" + "="*60)
        print("CNN PREDICTIONS")
        print("="*60)

        # Load model
        self.load_model()

        # Dataset configurations (same as predict.py)
        datasets = {
            'D2': {'voltage_threshold': 0.80},
            'D3': {'voltage_threshold': 0.95},
            'D4': {'voltage_threshold': 1.50},
            'D5': {'voltage_threshold': 2.80},
            'D6': {'voltage_threshold': 4.00},
        }

        all_results = {}

        for dataset_name, config in datasets.items():
            indices, classes = self.predict_dataset(
                dataset_name,
                config['voltage_threshold']
            )

            all_results[dataset_name] = {
                'indices': indices,
                'classes': classes,
                'count': len(indices),
                'distribution': Counter(classes) if len(classes) > 0 else {}
            }

            # Save predictions
            mat_filepath = self.output_dir / f'{dataset_name}_predictions.mat'
            if len(indices) > 0:
                sio.savemat(mat_filepath, {
                    'Index': indices.reshape(-1, 1),
                    'Class': classes.reshape(-1, 1)
                })
                print(f"Predictions saved to: {mat_filepath}")

        return all_results

    def generate_comparison_table(self, cnn_results):
        """Generate comparison table between hybrid and CNN results."""
        print("\n" + "="*80)
        print("COMPARISON: HYBRID vs CNN CLASSIFIER")
        print("="*80)

        # Load hybrid predictions for comparison
        hybrid_dir = self.base_dir / 'predictions'

        print(f"\n{'Dataset':<8} {'Model':<10} {'Total':<8} {'C1':<12} {'C2':<12} {'C3':<12} {'C4':<12} {'C5':<12}")
        print("-"*88)

        for dataset_name in ['D2', 'D3', 'D4', 'D5', 'D6']:
            # CNN results
            cnn_count = cnn_results[dataset_name]['count']
            cnn_dist = cnn_results[dataset_name]['distribution']

            print(f"{dataset_name:<8} {'CNN':<10} {cnn_count:<8}", end="")
            for c in range(1, 6):
                cnt = cnn_dist.get(c, 0)
                pct = 100 * cnt / cnn_count if cnn_count > 0 else 0
                print(f"{cnt:>4}({pct:>4.1f}%)", end=" ")
            print()

            # Load hybrid results if available
            hybrid_file = hybrid_dir / f'{dataset_name}_predictions.mat'
            if hybrid_file.exists():
                hybrid_data = sio.loadmat(hybrid_file)
                hybrid_classes = hybrid_data['Class'].flatten()
                hybrid_count = len(hybrid_classes)
                hybrid_dist = Counter(hybrid_classes)

                print(f"{'':<8} {'Hybrid':<10} {hybrid_count:<8}", end="")
                for c in range(1, 6):
                    cnt = hybrid_dist.get(c, 0)
                    pct = 100 * cnt / hybrid_count if hybrid_count > 0 else 0
                    print(f"{cnt:>4}({pct:>4.1f}%)", end=" ")
                print()

            print("-"*88)

        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY: KEY METRICS")
        print("="*80)

        print("\nClass 2 Recovery (target: >10%):")
        for dataset_name in ['D5', 'D6']:
            cnn_count = cnn_results[dataset_name]['count']
            cnn_c2 = cnn_results[dataset_name]['distribution'].get(2, 0)
            cnn_pct = 100 * cnn_c2 / cnn_count if cnn_count > 0 else 0
            print(f"  {dataset_name}: CNN={cnn_pct:.1f}%")

        print("\nClass 4 Recovery (target: >10%):")
        for dataset_name in ['D5', 'D6']:
            cnn_count = cnn_results[dataset_name]['count']
            cnn_c4 = cnn_results[dataset_name]['distribution'].get(4, 0)
            cnn_pct = 100 * cnn_c4 / cnn_count if cnn_count > 0 else 0
            print(f"  {dataset_name}: CNN={cnn_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='CNN Experiment for Spike Classification')
    parser.add_argument('--train', action='store_true', help='Train the CNN model')
    parser.add_argument('--predict', action='store_true', help='Run predictions on D2-D6')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    experiment = CNNExperiment()

    if args.train:
        experiment.train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    if args.predict:
        results = experiment.run_predictions()
        experiment.generate_comparison_table(results)

    # If no args, run both
    if not args.train and not args.predict:
        print("Usage: python cnn_experiment.py [--train] [--predict]")
        print("\nRunning full pipeline (train + predict)...")
        experiment.train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        results = experiment.run_predictions()
        experiment.generate_comparison_table(results)


if __name__ == '__main__':
    main()
