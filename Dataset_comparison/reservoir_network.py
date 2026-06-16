"""
================================================================================
Echo State Network (Reservoir Computing) for Arrhythmia Classification
================================================================================

A recurrent neural network approach inspired by optical reservoir computing
(Peng et al., "Coherent all-Optical Reservoir Computing for Equalization of
Impairments in Coherent Fiber Optic Communication Systems", IEEE 2024).

ARCHITECTURE
------------
  Input weights  W_in : dense random matrix, scaled by input_scaling
  Reservoir      W    : sparse random matrix, scaled to spectral_radius
  Output weights W_out: trained via ridge regression (no backpropagation)

STATE UPDATE EQUATION
---------------------
  x(n) = (1 - leak_rate) * x(n-1) + leak_rate * tanh(W_in @ u(n) + W @ x(n-1))

OUTPUT
------
  y = W_out @ [x; 1]   (reservoir state concatenated with bias term)

TRAINING
--------
  Only W_out is trained using ridge regression (closed-form solution):
    W_out = (S^T S + alpha * I)^{-1} S^T Y
  where S is the matrix of collected reservoir states and Y is the target.

  This makes training extremely fast compared to backpropagation-based RNNs
  (LSTM, GRU) since there is no gradient computation through time.

KEY PARAMETERS
--------------
  n_reservoir      : number of neurons in the reservoir (virtual nodes)
  spectral_radius  : largest eigenvalue of W; controls memory/stability
                     < 1.0 ensures echo state property (fading memory)
  input_scaling    : scales W_in; controls input drive strength
  leak_rate        : controls how fast reservoir state changes (0=no update, 1=full)
  ridge_alpha      : regularization for ridge regression
  sparsity         : fraction of zero connections in W (0.9 = 90% sparse)
  n_transient      : number of warm-up steps before reading reservoir state

USAGE
-----
  from reservoir_network import EchoStateNetwork

  esn = EchoStateNetwork(n_reservoir=200, spectral_radius=0.9)
  esn.fit(X_train, y_train)
  predictions = esn.predict(X_test)
  proba = esn.predict_proba(X_test)
  label, confidence = esn.predict_with_confidence(X_test_single)

================================================================================
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class EchoStateNetwork:
    """
    Echo State Network (ESN) classifier using reservoir computing.

    The reservoir is a fixed, randomly connected recurrent layer. Only the
    output (readout) weights are trained, via ridge regression. This gives
    RNN-like temporal processing without backpropagation through time.
    """

    def __init__(
        self,
        n_reservoir: int = 200,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leak_rate: float = 0.3,
        ridge_alpha: float = 1.0,
        sparsity: float = 0.9,
        random_state: int = 42,
        n_transient: int = 5,
    ):
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

    def _init_reservoir(self, n_input: int):
        """
        Initialize input and reservoir weight matrices.

        W_in: (n_reservoir x n_input) dense random, scaled by input_scaling.
        W:    (n_reservoir x n_reservoir) sparse random, rescaled so that
              its spectral radius equals self.spectral_radius.
        """
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

    def _compute_reservoir_state(self, u: np.ndarray) -> np.ndarray:
        """
        Drive the reservoir with a single input vector.

        For classification of static feature vectors (not time series), we
        feed the full feature vector into the reservoir repeatedly for
        n_transient + 1 steps with small perturbations. This creates temporal
        dynamics from spatial features — the reservoir's recurrent connections
        mix and transform the input non-linearly at each step.

        The state update at each step is:
          x(n) = (1 - leak_rate) * x(n-1) + leak_rate * tanh(W_in @ u + W @ x(n-1))

        Returns the final reservoir state vector of shape (n_reservoir,).
        """
        n_features = u.shape[0]
        x = np.zeros(self.n_reservoir)

        n_steps = self.n_transient + 1
        for step in range(n_steps):
            noise = np.random.RandomState(step).randn(n_features) * 0.01
            u_noisy = u if step == 0 else u + noise
            pre_activation = self.W_in @ u_noisy + self.W @ x
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(pre_activation)

        return x

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EchoStateNetwork":
        """
        Train the ESN.

        Steps:
          1. Impute missing values and scale features.
          2. Initialize reservoir weights (W_in, W).
          3. For each training sample, drive the reservoir and collect
             the final state into a state matrix S.
          4. Solve for W_out via ridge regression:
             W_out = (S^T S + alpha I)^{-1} S^T Y_target

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix
        y : (n_samples,) class labels

        Returns
        -------
        self
        """
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)
        n_samples, n_features = X_scaled.shape

        self._init_reservoir(n_features)

        # Collect reservoir states for all training samples
        states = np.zeros((n_samples, self.n_reservoir + 1))
        for i in range(n_samples):
            x = self._compute_reservoir_state(X_scaled[i])
            states[i, :self.n_reservoir] = x
            states[i, self.n_reservoir] = 1.0  # bias term

        # One-hot encode targets
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        Y_target = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            cls_idx = np.where(self.classes_ == label)[0][0]
            Y_target[i, cls_idx] = 1.0

        # Ridge regression: W_out = (S^T S + alpha I)^{-1} S^T Y
        I = np.eye(states.shape[1])
        self.W_out = np.linalg.solve(
            states.T @ states + self.ridge_alpha * I,
            states.T @ Y_target,
        )

        self.is_fitted = True
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw output scores (pre-softmax) for each class."""
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        n_samples = X_scaled.shape[0]

        outputs = np.zeros((n_samples, len(self.classes_)))
        for i in range(n_samples):
            x = self._compute_reservoir_state(X_scaled[i])
            state = np.append(x, 1.0)
            outputs[i] = state @ self.W_out

        return outputs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for one or more samples."""
        outputs = self._predict_raw(X)
        return self.classes_[np.argmax(outputs, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via softmax of raw outputs."""
        outputs = self._predict_raw(X)
        exp_out = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        return exp_out / exp_out.sum(axis=1, keepdims=True)

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[int, float]:
        """Return (predicted_class, confidence) for a single sample."""
        proba = self.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        return int(self.classes_[pred_idx]), float(proba[pred_idx])
