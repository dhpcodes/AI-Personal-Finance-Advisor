"""Post-hoc calibration via Temperature Scaling for probabilistic classifiers."""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple


def compute_ece(y_true: np.ndarray, proba: np.ndarray, bins: int = 15) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any():
            e += k.mean() * abs((pred[k] == y_true[k]).mean() - conf[k].mean())
    return float(e)


class TemperatureScaler:
    """Post-hoc temperature scaling fitted via Negative Log Likelihood (NLL).

    Learns a single scalar temperature T > 0 on a held-out validation set.
    Dividing uncalibrated logits by T > 1 softens overconfident predictions,
    while T < 1 sharpens underconfident predictions.
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Optimize temperature T to minimize NLL on validation set."""
        def nll(t):
            t_val = max(float(t[0]), 1e-3)
            scaled = logits / t_val
            # Numerically stable softmax
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs = np.clip(probs, 1e-15, 1.0 - 1e-15)

            y_one_hot = np.zeros_like(probs)
            y_one_hot[np.arange(len(y_true)), y_true] = 1.0
            return -np.mean(np.sum(y_one_hot * np.log(probs), axis=1))

        res = minimize(nll, x0=[1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling to convert logits to calibrated probabilities."""
        scaled = logits / max(self.temperature, 1e-3)
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)
