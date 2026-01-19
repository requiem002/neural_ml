"""Main spike sorting pipeline."""

import numpy as np
import scipy.io as sio
from pathlib import Path
import pickle

from spike_detector import detect_spikes, detect_spikes_matched_filter
from feature_extractor import FeatureExtractor, extract_waveforms_at_indices
from classifier import SpikeClassifier


class SpikeSortingPipeline:
    """Complete pipeline for spike detection and classification."""

    def __init__(self, window_before=30, window_after=30, n_pca_components=10):
        self.window_before = window_before
        self.window_after = window_after
        self.feature_extractor = FeatureExtractor(
            n_pca_components=n_pca_components,
            window_before=window_before,
            window_after=window_after
        )
        self.classifier = SpikeClassifier(method='ensemble')
        self.templates = None  # Average waveforms per class for matched filtering
        self.is_trained = False

    def load_data(self, filepath):
        """Load a .mat file and return its contents."""
        data = sio.loadmat(filepath)
        d = data['d'].flatten()

        # Check if labeled data (training set)
        if 'Index' in data and 'Class' in data:
            index = data['Index'].flatten()
            classes = data['Class'].flatten()
            return d, index, classes
        else:
            return d, None, None

    def train(self, d, index, classes, voltage_threshold=0.75, tolerance=50):
        """
        Train the pipeline on labeled data.

        Uses a two-phase approach:
        1. First extracts ground truth waveforms to build templates
        2. Then detects spikes and matches them to ground truth for classifier training

        Parameters:
        -----------
        d : np.ndarray
            Raw signal
        index : np.ndarray
            Ground truth spike indices (1-indexed)
        classes : np.ndarray
            Ground truth class labels
        voltage_threshold : float
            Fixed voltage threshold for spike detection
        tolerance : int
            Maximum deviation for matching detected to ground truth spikes
        """
        # Phase 1: Extract ground truth waveforms for templates
        print("Extracting ground truth waveforms for templates...")
        gt_waveforms, gt_valid_indices = extract_waveforms_at_indices(
            d, index, self.window_before, self.window_after
        )
        valid_mask = np.isin(index, gt_valid_indices)
        gt_valid_classes = classes[valid_mask]

        print(f"Ground truth waveforms: {len(gt_waveforms)}")

        # Compute and store templates for matched filtering
        self.templates = {}
        for c in range(1, 6):
            class_mask = gt_valid_classes == c
            if np.any(class_mask):
                self.templates[c] = np.mean(gt_waveforms[class_mask], axis=0)

        # Fit feature extractor on ground truth waveforms first
        print("Fitting feature extractor on ground truth waveforms...")
        self.feature_extractor.fit(gt_waveforms)

        # Phase 2: Detect spikes and match to ground truth
        print(f"\nDetecting spikes (voltage_threshold={voltage_threshold})...")
        from spike_detector import detect_spikes
        detected_indices, detected_waveforms = detect_spikes(
            d, voltage_threshold=voltage_threshold,
            window_before=self.window_before,
            window_after=self.window_after
        )
        print(f"Detected spikes: {len(detected_indices)}")

        # Match detected spikes to ground truth
        print("Matching detected spikes to ground truth...")
        matched_waveforms = []
        matched_classes = []
        used_gt = set()

        for det_i, (det_idx, det_wf) in enumerate(zip(detected_indices, detected_waveforms)):
            best_match = None
            best_dist = tolerance + 1

            for gt_i, gt_idx in enumerate(index):
                if gt_i in used_gt:
                    continue
                dist = abs(det_idx - gt_idx)
                if dist <= tolerance and dist < best_dist:
                    best_match = gt_i
                    best_dist = dist

            if best_match is not None:
                used_gt.add(best_match)
                matched_waveforms.append(det_wf)
                matched_classes.append(classes[best_match])

        matched_waveforms = np.array(matched_waveforms)
        matched_classes = np.array(matched_classes)

        print(f"Matched spikes for training: {len(matched_waveforms)}")
        print(f"Detection recall: {len(matched_waveforms)/len(index):.2%}")

        # Also include ground truth waveforms to augment training data
        print("Augmenting with ground truth waveforms...")
        all_waveforms = np.vstack([matched_waveforms, gt_waveforms])
        all_classes = np.concatenate([matched_classes, gt_valid_classes])

        # Add noise-augmented samples to help classifier generalize
        print("Adding noise-augmented training samples...")
        augmented_waveforms = []
        augmented_classes = []

        noise_levels = [0.5, 1.0, 2.0, 3.0]  # Different noise levels to simulate varying SNR
        for noise_level in noise_levels:
            for wf, cls in zip(gt_waveforms, gt_valid_classes):
                noisy_wf = wf + np.random.randn(len(wf)) * noise_level
                augmented_waveforms.append(noisy_wf)
                augmented_classes.append(cls)

        augmented_waveforms = np.array(augmented_waveforms)
        augmented_classes = np.array(augmented_classes)

        all_waveforms = np.vstack([all_waveforms, augmented_waveforms])
        all_classes = np.concatenate([all_classes, augmented_classes])

        print(f"Total training samples (with augmentation): {len(all_waveforms)}")

        # Transform features
        print("Extracting features...")
        X = self.feature_extractor.transform(all_waveforms)

        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X, all_classes)

        # Evaluate on training data
        print("\n=== Training Performance ===")
        self.classifier.evaluate(X, all_classes)

        self.is_trained = True
        print("\nTraining complete!")

        return self

    def predict(self, d, use_matched_filter=False, threshold_factor=None, voltage_threshold=None):
        """
        Detect and classify spikes in new data.

        Parameters:
        -----------
        d : np.ndarray
            Raw signal
        use_matched_filter : bool
            Use matched filtering for detection (better for low SNR)
        threshold_factor : float or None
            Detection threshold multiplier (MAD-based)
        voltage_threshold : float or None
            Fixed voltage threshold (overrides threshold_factor)

        Returns:
        --------
        indices : np.ndarray
            Detected spike indices (1-indexed)
        classes : np.ndarray
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before prediction")

        # Detect spikes
        print("Detecting spikes...")
        if use_matched_filter and self.templates:
            indices, waveforms = detect_spikes_matched_filter(
                d, self.templates,
                threshold_factor=threshold_factor if threshold_factor else 4.0,
                window_before=self.window_before,
                window_after=self.window_after
            )
        else:
            indices, waveforms = detect_spikes(
                d,
                threshold_factor=threshold_factor,
                voltage_threshold=voltage_threshold,
                window_before=self.window_before,
                window_after=self.window_after
            )

        print(f"Detected {len(indices)} spikes")

        if len(indices) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # Extract features and classify
        print("Classifying spikes...")
        X = self.feature_extractor.transform(waveforms)
        predicted_classes = self.classifier.predict(X)

        return indices, predicted_classes

    def evaluate_detection(self, predicted_indices, true_indices, tolerance=50):
        """
        Evaluate spike detection performance.

        Parameters:
        -----------
        predicted_indices : np.ndarray
            Detected spike indices
        true_indices : np.ndarray
            Ground truth spike indices
        tolerance : int
            Allowed deviation in samples (±50 by default)
        """
        true_positives = 0
        matched_true = set()
        matched_pred = set()

        # For each predicted spike, find if there's a matching true spike
        for pred_idx in predicted_indices:
            for i, true_idx in enumerate(true_indices):
                if i not in matched_true and abs(pred_idx - true_idx) <= tolerance:
                    true_positives += 1
                    matched_true.add(i)
                    matched_pred.add(pred_idx)
                    break

        false_positives = len(predicted_indices) - true_positives
        false_negatives = len(true_indices) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n=== Detection Performance ===")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return {'precision': precision, 'recall': recall, 'f1': f1, 'matched_true': matched_true}

    def evaluate_full(self, d, true_indices, true_classes, tolerance=50, voltage_threshold=0.75):
        """
        Full evaluation: detection + classification.
        Returns combined F1 score as per coursework criteria.
        """
        # Predict
        pred_indices, pred_classes = self.predict(d, voltage_threshold=voltage_threshold)

        # Detection evaluation
        det_results = self.evaluate_detection(pred_indices, true_indices, tolerance)

        # Classification evaluation (only for matched spikes)
        # Build mapping from predicted to true indices
        true_matched_classes = []
        pred_matched_classes = []

        for pred_i, pred_idx in enumerate(pred_indices):
            for true_i, true_idx in enumerate(true_indices):
                if abs(pred_idx - true_idx) <= tolerance:
                    true_matched_classes.append(true_classes[true_i])
                    pred_matched_classes.append(pred_classes[pred_i])
                    break

        if len(true_matched_classes) > 0:
            from sklearn.metrics import f1_score
            f1_classification = f1_score(true_matched_classes, pred_matched_classes, average='weighted')
            print(f"\n=== Classification Performance (matched spikes) ===")
            print(f"F1 Score: {f1_classification:.4f}")
        else:
            f1_classification = 0
            print("\nNo matched spikes for classification evaluation")

        # Combined score as per coursework
        f_combined = 0.3 * det_results['f1'] + 0.7 * f1_classification
        print(f"\n=== Combined F1 Score ===")
        print(f"F_dataset = 0.3 × {det_results['f1']:.4f} + 0.7 × {f1_classification:.4f} = {f_combined:.4f}")

        return {
            'detection': det_results,
            'classification_f1': f1_classification,
            'combined_f1': f_combined,
            'predicted_indices': pred_indices,
            'predicted_classes': pred_classes
        }

    def save(self, filepath):
        """Save the trained pipeline."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_extractor': self.feature_extractor,
                'classifier': self.classifier,
                'templates': self.templates,
                'window_before': self.window_before,
                'window_after': self.window_after,
                'is_trained': self.is_trained
            }, f)
        print(f"Pipeline saved to {filepath}")

    def load(self, filepath):
        """Load a trained pipeline."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.feature_extractor = state['feature_extractor']
        self.classifier = state['classifier']
        self.templates = state['templates']
        self.window_before = state['window_before']
        self.window_after = state['window_after']
        self.is_trained = state['is_trained']
        print(f"Pipeline loaded from {filepath}")
        return self

    def save_predictions(self, indices, classes, filepath):
        """Save predictions to .mat file."""
        sio.savemat(filepath, {
            'Index': indices.reshape(-1, 1),
            'Class': classes.reshape(-1, 1)
        })
        print(f"Predictions saved to {filepath}")
