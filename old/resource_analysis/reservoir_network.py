"""
Echo State Network with computational cost instrumentation.
Tracks total multiplications for training and inference.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class EchoStateNetwork:
    """ESN classifier with multiplication counting."""

    def __init__(self, n_reservoir=200, spectral_radius=0.9, input_scaling=0.5,
                 leak_rate=0.3, ridge_alpha=1.0, sparsity=0.9,
                 random_state=42, n_transient=5):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.sparsity = sparsity
        self.random_state = random_state
        self.n_transient = n_transient

        self.W_in = None
        self.W = None
        self.W_out = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.classes_ = None
        self.is_fitted = False

        # Instrumentation
        self.train_multiplications = 0
        self.inference_multiplications = 0

    def _init_reservoir(self, n_input):
        rng = np.random.RandomState(self.random_state)
        self.W_in = rng.randn(self.n_reservoir, n_input) * self.input_scaling
        W = rng.randn(self.n_reservoir, self.n_reservoir)
        mask = rng.rand(self.n_reservoir, self.n_reservoir) > self.sparsity
        W *= mask
        if np.max(np.abs(W)) > 0:
            eigenvalues = np.linalg.eigvals(W)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            if max_eigenvalue > 0:
                W = W * (self.spectral_radius / max_eigenvalue)
        self.W = W

    def _reservoir_step_mults(self, n_input):
        """Count multiplications for one reservoir step."""
        # W_in @ u:  n_reservoir * n_input
        # W @ x:     n_reservoir * n_reservoir (but sparse — count non-zeros)
        w_nnz = int(np.count_nonzero(self.W)) if self.W is not None else int(self.n_reservoir ** 2 * (1 - self.sparsity))
        # leak_rate scaling: n_reservoir mults
        # tanh is transcendental — count as ~5 mults per element (approx)
        return (self.n_reservoir * n_input) + w_nnz + self.n_reservoir + (5 * self.n_reservoir)

    def _compute_reservoir_state(self, u):
        n_features = u.shape[0]
        x = np.zeros(self.n_reservoir)
        n_steps = self.n_transient + 1
        for step in range(n_steps):
            noise = np.random.RandomState(step).randn(n_features) * 0.01
            u_noisy = u if step == 0 else u + noise
            pre_activation = self.W_in @ u_noisy + self.W @ x
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(pre_activation)
        return x

    def fit(self, X, y):
        self.train_multiplications = 0
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)
        n_samples, n_features = X_scaled.shape
        self._init_reservoir(n_features)

        step_mults = self._reservoir_step_mults(n_features)
        n_steps = self.n_transient + 1

        states = np.zeros((n_samples, self.n_reservoir + 1))
        for i in range(n_samples):
            x = self._compute_reservoir_state(X_scaled[i])
            states[i, :self.n_reservoir] = x
            states[i, self.n_reservoir] = 1.0
            self.train_multiplications += step_mults * n_steps

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        Y_target = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            cls_idx = np.where(self.classes_ == label)[0][0]
            Y_target[i, cls_idx] = 1.0

        d = self.n_reservoir + 1
        # S^T @ S: d * d * n_samples mults
        self.train_multiplications += d * d * n_samples
        # S^T @ Y: d * n_classes * n_samples
        self.train_multiplications += d * n_classes * n_samples
        # solve (d x d) system: ~d^3
        self.train_multiplications += d * d * d

        I = np.eye(d)
        self.W_out = np.linalg.solve(
            states.T @ states + self.ridge_alpha * I,
            states.T @ Y_target,
        )
        self.is_fitted = True
        return self

    def _predict_raw(self, X):
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        n_samples = X_scaled.shape[0]
        n_features = X_scaled.shape[1]

        step_mults = self._reservoir_step_mults(n_features)
        n_steps = self.n_transient + 1

        outputs = np.zeros((n_samples, len(self.classes_)))
        for i in range(n_samples):
            x = self._compute_reservoir_state(X_scaled[i])
            state = np.append(x, 1.0)
            outputs[i] = state @ self.W_out
            # reservoir steps + readout
            self.inference_multiplications += step_mults * n_steps
            self.inference_multiplications += (self.n_reservoir + 1) * len(self.classes_)
        return outputs

    def predict(self, X):
        outputs = self._predict_raw(X)
        return self.classes_[np.argmax(outputs, axis=1)]

    def predict_proba(self, X):
        outputs = self._predict_raw(X)
        exp_out = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        return exp_out / exp_out.sum(axis=1, keepdims=True)

    def predict_with_confidence(self, X):
        proba = self.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        return int(self.classes_[pred_idx]), float(proba[pred_idx])

    def get_total_multiplications(self):
        return self.train_multiplications + self.inference_multiplications

    def reset_counters(self):
        self.train_multiplications = 0
        self.inference_multiplications = 0
