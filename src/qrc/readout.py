"""QR1 readout: ridge regression without an intercept, fitted in the scaled target space
(docs/qrc_spec.md, section 5). Follows the model interface in src/harness.py.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .features import TargetScale


class RidgeReadout:
    """``W = y Z^T (Z Z^T + ridge I)^-1`` on reservoir feature columns; predicts log RV."""

    def __init__(self, cols: list[str], target: TargetScale, ridge: float = 1e-8):
        self.cols = cols
        self.target = target
        self.ridge = ridge
        self.n_params = len(cols)
        self.fit_seconds = 0.0
        self.weights: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RidgeReadout":
        t0 = time.perf_counter()
        Z = X[self.cols].to_numpy()
        gram = Z.T @ Z + self.ridge * np.eye(len(self.cols))
        self.weights = np.linalg.solve(gram, Z.T @ self.target.scale(y))
        self.fit_seconds += time.perf_counter() - t0
        return self

    def predict(self, x) -> float:
        return float(self.target.unscale(np.asarray(x[self.cols], dtype=float) @ self.weights))
