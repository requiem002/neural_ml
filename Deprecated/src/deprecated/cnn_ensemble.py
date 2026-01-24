"""
CNN Ensemble for Spike Classification

This module implements ensemble training and prediction using multiple
CNN models with different random seeds. Ensemble averaging reduces
prediction noise and improves stability on noisy datasets.

Expected benefits:
- Reduces random prediction noise
- Improves Class 4/5 separation (the "twin problem")
- More stable confidence estimates
- Typical improvement: 2-5% F1 on noisy datasets
"""

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from pathlib import Path
from collections import Counter
import pickle
import json
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from spike_detector import detect_spikes
from deprecated.feature_extractor import extract_waveforms_at_indices
from deprecated.cnn_experiment import DualBranchSpikeNet, CNNExperiment


class CNNEnsemble:
    """Ensemble of CNN models for robust spike classification."""

    def __init__(self, n_models=5, device=None):
        """
        Initialize ensemble.

        Args:
            n_models: Number of models in the ensemble
            device: PyTorch device (auto-detected if None)
        """
        self.n_models = n_models
        self.device = device if device else (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.models = []
        self.amp_mean = None
        self.amp_std = None

        # Paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / 'datasets'
        self.model_dir = self.base_dir / 'models' / 'ensemble'
        self.output_dir = self.base_dir / 'submissions_nightly'

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Seeds for each model (42, 142, 242, ...)
        self.seeds = [42 + i * 100 for i in range(n_models)]

        print(f"CNNEnsemble initialized with {n_models} models on {self.device}")
        print(f"Seeds: {self.seeds}")

    def train_ensemble(self, epochs=100, batch_size=64, lr=0.001, patience=10):
        """
        Train n_models with different random seeds.

        Each model is trained independently with its own random initialization.
        """
        print("\n" + "="*70)
        print("ENSEMBLE TRAINING")
        print("="*70)
        print(f"Training {self.n_models} models with different random seeds...")

        # Create base experiment for data loading
        base_exp = CNNExperiment(device=self.device)

        # Load D1 data once
        print("\nLoading D1 training data...")
        d, index, classes = base_exp.load_d1_data()

        # Extract waveforms
        print("Extracting waveforms...")
        waveforms, valid_indices = extract_waveforms_at_indices(
            d, index, window_before=30, window_after=30
        )
        valid_mask = np.isin(index, valid_indices)
        valid_classes = classes[valid_mask]

        print(f"Total waveforms: {len(waveforms)}")

        # Augment data
        print("\nAugmenting data...")
        aug_waveforms, aug_labels = base_exp.augment_data(waveforms, valid_classes)
        print(f"Augmented samples: {len(aug_waveforms)}")

        # Train each model
        validation_f1s = []

        for i, seed in enumerate(self.seeds):
            print(f"\n{'='*70}")
            print(f"TRAINING MODEL {i+1}/{self.n_models} (seed={seed})")
            print(f"{'='*70}")

            # Set seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Train/val split (different for each seed)
            train_wf, val_wf, train_lbl, val_lbl = train_test_split(
                aug_waveforms, aug_labels, test_size=0.2,
                random_state=seed, stratify=aug_labels
            )

            print(f"Training samples: {len(train_wf)}")
            print(f"Validation samples: {len(val_wf)}")

            # Prepare data (fit scalers on first model only)
            if i == 0:
                train_wf_norm, train_amp, train_lbl = base_exp.prepare_data(
                    train_wf, train_lbl, fit_scalers=True
                )
                self.amp_mean = base_exp.amp_mean
                self.amp_std = base_exp.amp_std
            else:
                base_exp.amp_mean = self.amp_mean
                base_exp.amp_std = self.amp_std
                train_wf_norm, train_amp, train_lbl = base_exp.prepare_data(
                    train_wf, train_lbl, fit_scalers=False
                )

            val_wf_norm, val_amp, val_lbl = base_exp.prepare_data(val_wf, val_lbl)

            # Convert to tensors
            X_wf_train = torch.FloatTensor(train_wf_norm).to(self.device)
            X_amp_train = torch.FloatTensor(train_amp).to(self.device)
            y_train = torch.LongTensor(train_lbl - 1).to(self.device)

            X_wf_val = torch.FloatTensor(val_wf_norm).to(self.device)
            X_amp_val = torch.FloatTensor(val_amp).to(self.device)
            y_val = torch.LongTensor(val_lbl - 1).to(self.device)

            # Create weighted sampler
            class_counts = Counter(train_lbl)
            weights = [1.0 / class_counts[lbl] for lbl in train_lbl]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

            train_dataset = TensorDataset(X_wf_train, X_amp_train, y_train)
            val_dataset = TensorDataset(X_wf_val, X_amp_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model with seed-specific initialization
            model = DualBranchSpikeNet(
                waveform_size=60, num_classes=5, num_amp_features=8
            ).to(self.device)

            # Loss with class weights
            class_weights = torch.FloatTensor(
                [1.0 / class_counts[c] for c in range(1, 6)]
            ).to(self.device)
            class_weights = class_weights / class_weights.sum() * 5
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=5, factor=0.5
            )

            # Training loop
            best_val_f1 = 0
            patience_counter = 0
            best_state = None

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_wf, batch_amp, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_wf, batch_amp)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()

                train_acc = 100. * train_correct / train_total

                # Validation
                model.eval()
                val_preds = []
                val_true = []

                with torch.no_grad():
                    for batch_wf, batch_amp, batch_y in val_loader:
                        outputs = model(batch_wf, batch_amp)
                        _, predicted = outputs.max(1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_true.extend(batch_y.cpu().numpy())

                val_f1 = f1_score(val_true, val_preds, average='weighted')
                scheduler.step(val_f1)

                # Print progress
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs}: "
                          f"Train Acc={train_acc:.1f}%, Val F1={val_f1:.4f}")

                # Early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # Load best state and save
            if best_state is not None:
                model.load_state_dict(best_state)

            self.models.append(model)
            validation_f1s.append(best_val_f1)

            print(f"Model {i+1} best validation F1: {best_val_f1:.4f}")

            # Save individual model
            self._save_model(model, i)

        # Save ensemble metadata
        self._save_metadata(validation_f1s)

        # Summary
        print("\n" + "="*70)
        print("ENSEMBLE TRAINING COMPLETE")
        print("="*70)
        print(f"\nValidation F1 scores: {[f'{f:.4f}' for f in validation_f1s]}")
        print(f"Mean validation F1: {np.mean(validation_f1s):.4f}")
        print(f"Std validation F1: {np.std(validation_f1s):.4f}")

        return validation_f1s

    def _save_model(self, model, index):
        """Save individual model."""
        state = {
            'model_state': model.state_dict(),
            'amp_mean': self.amp_mean,
            'amp_std': self.amp_std,
        }
        filepath = self.model_dir / f'cnn_model_{index}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"  Model saved to: {filepath}")

    def _save_metadata(self, validation_f1s):
        """Save ensemble metadata."""
        metadata = {
            'n_models': self.n_models,
            'seeds': self.seeds,
            'validation_f1s': validation_f1s,
            'mean_f1': float(np.mean(validation_f1s)),
            'std_f1': float(np.std(validation_f1s)),
            'timestamp': datetime.now().isoformat(),
        }
        filepath = self.model_dir / 'ensemble_metadata.json'
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {filepath}")

    def load_ensemble(self):
        """Load all models from disk."""
        print(f"\nLoading ensemble from {self.model_dir}...")

        self.models = []
        for i in range(self.n_models):
            filepath = self.model_dir / f'cnn_model_{i}.pkl'
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            model = DualBranchSpikeNet(
                waveform_size=60, num_classes=5, num_amp_features=8
            ).to(self.device)
            model.load_state_dict(state['model_state'])
            model.eval()

            # Load scalers from first model
            if i == 0:
                self.amp_mean = state['amp_mean']
                self.amp_std = state['amp_std']

            self.models.append(model)
            print(f"  Loaded model {i}")

        print(f"Ensemble loaded: {len(self.models)} models")

    def predict_ensemble(self, waveforms, amp_features):
        """
        Average probabilities from all models.

        Args:
            waveforms: Normalized waveforms (N x 60)
            amp_features: Standardized amplitude features (N x 8)

        Returns:
            predictions: Class predictions (1-indexed)
            avg_probs: Averaged probabilities
            confidences: Prediction confidences
        """
        X_wf = torch.FloatTensor(waveforms).to(self.device)
        X_amp = torch.FloatTensor(amp_features).to(self.device)

        all_probs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(X_wf, X_amp)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)

        # Average probabilities (reduces noise/hallucinations)
        avg_probs = torch.stack(all_probs).mean(dim=0)
        predictions = avg_probs.argmax(dim=1).cpu().numpy() + 1  # 1-indexed
        confidences = avg_probs.max(dim=1).values.cpu().numpy()

        return predictions, avg_probs.cpu().numpy(), confidences

    def predict_dataset(self, dataset_name, voltage_threshold, thresholds=None, verbose=True):
        """
        Generate predictions for a dataset using ensemble.

        Args:
            dataset_name: Dataset name (e.g., 'D2', 'D6')
            voltage_threshold: Voltage threshold for spike detection
            thresholds: Post-processing thresholds dict (uses defaults if None)
            verbose: Whether to print details
        """
        # Load defaults if no thresholds provided
        if thresholds is None:
            from deprecated.optimize_thresholds import ThresholdOptimizer
            thresholds = ThresholdOptimizer.DEFAULT_THRESHOLDS

        # Load data
        data = sio.loadmat(self.data_dir / f'{dataset_name}.mat')
        d = data['d'].flatten()

        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name}")
            print(f"{'='*50}")

        # Detect spikes
        indices, waveforms = detect_spikes(
            d, voltage_threshold=voltage_threshold,
            window_before=30, window_after=30
        )

        if verbose:
            print(f"Detected {len(indices)} spikes (threshold={voltage_threshold}V)")

        if len(indices) == 0:
            return np.array([]), np.array([])

        # Create experiment for feature extraction
        exp = CNNExperiment(device=self.device)
        exp.amp_mean = self.amp_mean
        exp.amp_std = self.amp_std

        # Prepare data
        wf_norm, amp_features = exp.prepare_data(waveforms)

        # Get raw features for post-processing
        raw_features = exp.extract_amplitude_features(waveforms)
        raw_amp = raw_features[:, 0]
        fwhm_values = raw_features[:, 3]
        symmetry_values = raw_features[:, 5]

        # Ensemble prediction
        classes, probs, confidences = self.predict_ensemble(wf_norm, amp_features)

        # Post-processing
        corrected = self._apply_post_processing(
            classes, raw_amp, fwhm_values, confidences, symmetry_values,
            thresholds, dataset_name, verbose
        )

        # Print distribution
        if verbose:
            class_dist = Counter(corrected)
            print("Class distribution:")
            for c in sorted(class_dist.keys()):
                print(f"  Class {c}: {class_dist[c]} ({100*class_dist[c]/len(corrected):.1f}%)")

        return indices, corrected

    def _apply_post_processing(self, classes, raw_amp, fwhm_values, confidences,
                               symmetry_values, thresholds, dataset_name, verbose):
        """Apply post-processing corrections."""
        corrected = classes.copy()
        corrections = {'c3_to_high': 0, 'c4_to_high': 0, 'c5_rescue': 0,
                       'c2_to_c1': 0, 'c2_to_c5': 0, 'low_amp_fix': 0,
                       'c2_fwhm_fix': 0}

        for i, (pred_class, amp, fwhm, conf, sym) in enumerate(zip(
                classes, raw_amp, fwhm_values, confidences, symmetry_values)):

            # RULE 1: Class 3 amplitude correction
            if pred_class == 3 and amp > thresholds['c3_amp_cap']:
                if fwhm > 0.78:
                    corrected[i] = 5
                elif fwhm > 0.65:
                    corrected[i] = 2
                else:
                    corrected[i] = 1
                corrections['c3_to_high'] += 1

            # RULE 2: Class 4 amplitude correction
            elif pred_class == 4 and amp > thresholds['c4_amp_cap']:
                if fwhm > 0.78:
                    corrected[i] = 5
                elif fwhm > 0.65:
                    corrected[i] = 2
                else:
                    corrected[i] = 1
                corrections['c4_to_high'] += 1

            # RULE 3: Class 5 rescue
            elif (fwhm > thresholds['c5_fwhm_thresh'] and
                  thresholds['c5_amp_low'] < amp < thresholds['c5_amp_high'] and
                  pred_class != 5):
                corrected[i] = 5
                corrections['c5_rescue'] += 1

            # RULE 4: C2 over-prediction correction
            elif pred_class == 2:
                if fwhm < thresholds['c2_narrow_fwhm'] and amp < 5.5:
                    corrected[i] = 1
                    corrections['c2_fwhm_fix'] += 1
                elif fwhm > thresholds['c2_wide_fwhm'] and amp < 5.3:
                    corrected[i] = 5
                    corrections['c2_to_c5'] += 1
                elif dataset_name in ['D5', 'D6'] and conf < thresholds['confidence_thresh']:
                    if 4.0 < amp < 5.2 and fwhm < 0.68:
                        corrected[i] = 1
                        corrections['c2_to_c1'] += 1

            # RULE 5: Low amplitude correction
            elif pred_class in [1, 2, 5] and amp < thresholds['low_amp_thresh']:
                if conf < thresholds['confidence_thresh']:
                    corrected[i] = 3 if amp < 1.8 else 4
                    corrections['low_amp_fix'] += 1

        if verbose:
            active = {k: v for k, v in corrections.items() if v > 0}
            if active:
                print(f"Post-processing corrections: {active}")

        return corrected

    def generate_predictions(self, thresholds=None):
        """Generate predictions for all test datasets."""
        print("\n" + "="*70)
        print("ENSEMBLE PREDICTIONS")
        print("="*70)

        # Dataset configurations
        datasets = {
            'D2': {'voltage_threshold': 0.80},
            'D3': {'voltage_threshold': 0.95},
            'D4': {'voltage_threshold': 1.50},
            'D5': {'voltage_threshold': 2.80},
            'D6': {'voltage_threshold': 4.00},
        }

        results = {}

        for dataset_name, config in datasets.items():
            indices, classes = self.predict_dataset(
                dataset_name,
                config['voltage_threshold'],
                thresholds=thresholds
            )

            results[dataset_name] = {
                'indices': indices,
                'classes': classes,
                'count': len(indices),
                'distribution': Counter(classes) if len(classes) > 0 else {}
            }

            # Save predictions
            if len(indices) > 0:
                mat_filepath = self.output_dir / f'{dataset_name}.mat'
                sio.savemat(mat_filepath, {
                    'Index': indices.reshape(-1, 1),
                    'Class': classes.reshape(-1, 1)
                })
                print(f"Saved: {mat_filepath}")

        return results


def train_cnn_ensemble(n_models=5, epochs=100):
    """Train a CNN ensemble."""
    ensemble = CNNEnsemble(n_models=n_models)
    ensemble.train_ensemble(epochs=epochs)
    return ensemble


def load_and_predict(thresholds=None):
    """Load ensemble and generate predictions."""
    ensemble = CNNEnsemble()
    ensemble.load_ensemble()
    results = ensemble.generate_predictions(thresholds=thresholds)
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Ensemble')
    parser.add_argument('--train', action='store_true', help='Train ensemble')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--n-models', type=int, default=5, help='Number of models')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    args = parser.parse_args()

    if args.train:
        train_cnn_ensemble(n_models=args.n_models, epochs=args.epochs)

    if args.predict:
        load_and_predict()

    if not args.train and not args.predict:
        print("Usage: python cnn_ensemble.py [--train] [--predict]")
        print("Running full pipeline (train + predict)...")
        ensemble = train_cnn_ensemble(n_models=args.n_models, epochs=args.epochs)
        ensemble.generate_predictions()
