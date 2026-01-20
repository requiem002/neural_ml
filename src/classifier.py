"""Spike classification module."""

import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings

warnings.filterwarnings('ignore')


class SpikeClassifier:
    """Ensemble classifier for spike classification."""

    def __init__(self, method='ensemble'):
        """
        Initialize classifier.

        Parameters:
        -----------
        method : str
            'rf' - Random Forest only
            'svm' - SVM only
            'mlp' - Neural Network only
            'ensemble' - Voting ensemble of multiple classifiers
        """
        self.method = method
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build the classification model."""
        if self.method == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.method == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        elif self.method == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=64,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        elif self.method == 'ensemble':
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            svm = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

            self.model = VotingClassifier(
                estimators=[('rf', rf), ('svm', svm), ('gb', gb)],
                voting='soft',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X, y):
        """Train the classifier."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X, y, verbose=True):
        """Evaluate classifier performance."""
        y_pred = self.predict(X)

        # Calculate metrics
        f1_weighted = f1_score(y, y_pred, average='weighted')
        f1_per_class = f1_score(y, y_pred, average=None)

        if verbose:
            print("\nClassification Report:")
            print(classification_report(y, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y, y_pred))
            print(f"\nWeighted F1 Score: {f1_weighted:.4f}")

        return {
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'predictions': y_pred
        }

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        print(f"Cross-validation F1 scores: {scores}")
        print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return scores


class TemplateClassifier:
    """
    Template-based classifier using correlation matching.
    More robust to noise because it directly compares waveform shapes.
    """

    def __init__(self, sample_rate=25000):
        self.templates = {}  # class -> template waveform
        self.templates_filtered = {}
        self.sample_rate = sample_rate

        # Bandpass filter for preprocessing
        nyquist = sample_rate / 2
        low = 300 / nyquist
        high = 3000 / nyquist
        self.filter_b, self.filter_a = signal.butter(2, [low, high], btype='band')

    def _filter_waveform(self, wf):
        """Apply bandpass filter to a waveform."""
        try:
            return signal.filtfilt(self.filter_b, self.filter_a, wf, padlen=min(15, len(wf)-1))
        except ValueError:
            return wf

    def _normalize_waveform(self, wf):
        """Normalize waveform to unit peak (for shape comparison)."""
        baseline = np.mean(wf[:5])
        wf_centered = wf - baseline
        peak = np.max(np.abs(wf_centered))
        if peak > 0.1:
            return wf_centered / peak
        return wf_centered

    def fit(self, waveforms, labels):
        """Build templates from labeled waveforms."""
        # Compute mean template for each class
        for cls in np.unique(labels):
            class_waveforms = waveforms[labels == cls]

            # Raw template (with amplitude info)
            self.templates[cls] = np.mean(class_waveforms, axis=0)

            # Filtered and normalized template (for shape matching)
            filtered_wfs = np.array([self._filter_waveform(wf) for wf in class_waveforms])
            normalized_wfs = np.array([self._normalize_waveform(wf) for wf in filtered_wfs])
            self.templates_filtered[cls] = np.mean(normalized_wfs, axis=0)

        return self

    def predict(self, waveforms):
        """Predict classes using template correlation."""
        predictions = []

        for wf in waveforms:
            # Filter and normalize for shape comparison
            wf_filtered = self._filter_waveform(wf)
            wf_norm = self._normalize_waveform(wf_filtered)

            # Get raw amplitude for amplitude-based discrimination
            baseline = np.mean(wf[:5])
            raw_amplitude = np.max(wf - baseline)

            # Compute correlation with each template
            best_class = 1
            best_score = -np.inf

            for cls, template in self.templates_filtered.items():
                # Shape correlation (normalized)
                shape_corr = np.corrcoef(wf_norm, template)[0, 1]
                if np.isnan(shape_corr):
                    shape_corr = 0

                # Amplitude similarity - use log scale to reduce dominance of high-amp classes
                template_amp = np.max(self.templates[cls] - np.mean(self.templates[cls][:5]))
                # Use relative difference with log dampening
                amp_ratio = raw_amplitude / (template_amp + 1e-10)
                amp_score = 1.0 / (1.0 + abs(np.log(amp_ratio + 1e-10)))

                # Combined score: 75% shape, 25% amplitude (prioritize shape)
                combined_score = 0.75 * shape_corr + 0.25 * amp_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_class = cls

            predictions.append(best_class)

        return np.array(predictions)

    def predict_proba(self, waveforms):
        """Predict class probabilities based on correlation scores."""
        n_classes = len(self.templates)
        probs = np.zeros((len(waveforms), n_classes))

        for i, wf in enumerate(waveforms):
            wf_filtered = self._filter_waveform(wf)
            wf_norm = self._normalize_waveform(wf_filtered)

            baseline = np.mean(wf[:5])
            raw_amplitude = np.max(wf - baseline)

            scores = []
            for cls in sorted(self.templates.keys()):
                template = self.templates_filtered[cls]
                shape_corr = np.corrcoef(wf_norm, template)[0, 1]
                if np.isnan(shape_corr):
                    shape_corr = 0

                template_amp = np.max(self.templates[cls] - np.mean(self.templates[cls][:5]))
                amp_ratio = raw_amplitude / (template_amp + 1e-10)
                amp_score = 1.0 / (1.0 + abs(np.log(amp_ratio + 1e-10)))

                combined_score = 0.75 * shape_corr + 0.25 * amp_score
                scores.append(max(0, combined_score))

            # Normalize to probabilities
            total = sum(scores) + 1e-10
            probs[i] = [s / total for s in scores]

        return probs


class HybridClassifier:
    """
    Hybrid classifier that combines ML-based and template-based classification.
    Uses ML for high-confidence predictions, falls back to template matching otherwise.
    """

    def __init__(self, ml_classifier, template_classifier, confidence_threshold=0.6):
        self.ml_classifier = ml_classifier
        self.template_classifier = template_classifier
        self.confidence_threshold = confidence_threshold

    def fit(self, X_features, waveforms, labels):
        """Train both classifiers."""
        self.ml_classifier.fit(X_features, labels)
        self.template_classifier.fit(waveforms, labels)
        return self

    def predict(self, X_features, waveforms):
        """
        Predict using hybrid approach:
        - Use ML classifier if confidence > threshold
        - Fall back to template matching otherwise
        """
        ml_proba = self.ml_classifier.predict_proba(X_features)
        ml_pred = np.argmax(ml_proba, axis=1) + 1  # Classes 1-5

        template_pred = self.template_classifier.predict(waveforms)

        # Use ML prediction if confident, otherwise use template
        max_proba = np.max(ml_proba, axis=1)
        predictions = np.where(max_proba >= self.confidence_threshold, ml_pred, template_pred)

        return predictions


def tune_hyperparameters(X, y, method='rf'):
    """Tune hyperparameters using grid search."""
    if method == 'rf':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
        }
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif method == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        }
        base_model = SVC(random_state=42, probability=True)
    else:
        raise ValueError(f"Tuning not implemented for {method}")

    grid_search = GridSearchCV(
        base_model, param_grid, cv=5,
        scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
