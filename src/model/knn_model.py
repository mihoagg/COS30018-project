"""
COS30018 - Task 3: KNN Implementation with scikit-learn

This module implements a K-Nearest Neighbors (KNN) classifier
for handwritten digit classification.

KNN works by comparing a test sample with stored training samples,
selecting the closest K neighbors, and assigning the most frequent label.

Important: Input images must be reshaped into 1D vectors (28x28 → 784).
"""

import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from models.base_model import BaseModel
from config import KNN_N_NEIGHBORS


class KNNModel(BaseModel):
    """KNN-based classifier for recognizing digits."""

    def __init__(self):
        super().__init__("KNN Classifier")
        self.classifier = None
        self.normalizer = StandardScaler()

    def build(self, k=KNN_N_NEIGHBORS):
        """
        Initialize the KNN model.

        Parameters:
        - k: number of nearest neighbors
        """
        self.classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",     # closer points have higher weight
            algorithm="auto",       # let sklearn choose optimal method
            n_jobs=-1               # utilize all CPU cores
        )
        return self.classifier

    def train(self, X_train, y_train, X_val=None, y_val=None,
              limit=20000, callback=None, **kwargs):
        """
        Fit the KNN model.

        Since KNN stores data instead of learning parameters,
        we optionally reduce dataset size to improve performance.
        """

        if self.classifier is None:
            self.build()

        # Convert images to flat vectors
        X_processed = self._reshape_input(X_train)

        # Random sampling to limit dataset size
        if len(X_processed) > limit:
            selected_idx = np.random.choice(len(X_processed), limit, replace=False)
            X_processed = X_processed[selected_idx]
            y_train = np.array(y_train)[selected_idx]

        # Normalize feature values
        X_processed = self.normalizer.fit_transform(X_processed)

        print(f"Fitting KNN (k={self.classifier.n_neighbors}) with {len(X_processed)} samples...")
        self.classifier.fit(X_processed, y_train)
        self.is_trained = True

        # Estimate training accuracy on a subset
        subset_size = min(2000, len(X_processed))
        eval_idx = np.random.choice(len(X_processed), subset_size, replace=False)

        accuracy = self.classifier.score(
            X_processed[eval_idx],
            np.array(y_train)[eval_idx]
        )

        self.training_history = {"accuracy": [accuracy]}
        print(f"Estimated training accuracy: {accuracy:.4f}")

        if callback:
            callback(1, 1, self.training_history)

        return self.training_history

    def predict(self, X):
        """Return predicted class labels."""
        X_processed = self._reshape_input(X)
        X_processed = self.normalizer.transform(X_processed)
        return self.classifier.predict(X_processed)

    def predict_proba(self, X):
        """Return probability scores for each class."""
        X_processed = self._reshape_input(X)
        X_processed = self.normalizer.transform(X_processed)
        return self.classifier.predict_proba(X_processed)

    def save(self, filepath):
        """Persist model and scaler to file."""
        if self.classifier:
            joblib.dump(
                {"knn": self.classifier, "scaler": self.normalizer},
                filepath
            )

    def load(self, filepath):
        """Load model and scaler from file."""
        if os.path.exists(filepath):
            saved = joblib.load(filepath)
            self.classifier = saved["knn"]
            self.normalizer = saved["scaler"]
            self.is_trained = True

    def _reshape_input(self, X):
        """
        Convert image data into flat vectors.

        Handles:
        - Single image (28x28)
        - Batch of images (N, 28, 28)
        """
        X = np.array(X, dtype=np.float32)

        if X.ndim == 2 and X.shape == (28, 28):
            return X.reshape(1, -1)

        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)

        return X
