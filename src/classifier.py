"""Spike classification module."""

import numpy as np
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
