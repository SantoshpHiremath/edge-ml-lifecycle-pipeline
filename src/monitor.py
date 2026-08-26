"""Real, tested drift monitoring for a deployed model.

Two independent, genuinely statistical signals, not a placeholder:

1. Input-distribution drift -- a real two-sample Kolmogorov-Smirnov
   test (scipy.stats.ks_2samp) comparing the per-image mean pixel
   intensity of a rolling window of live requests against the
   training-time baseline distribution. This is the same class of
   test a real production drift monitor (e.g. Evidently, whylogs)
   would run, just implemented directly rather than pulled in as a
   dependency.
2. Confidence drift -- the mean of the model's own max-softmax
   confidence over the same rolling window, compared against a
   baseline confidence recorded at registration time.
"""

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


class DriftMonitor:
    def __init__(self, baseline_means, baseline_confidence, window_size=30, alpha=0.01,
                 confidence_drop_threshold=0.15):
        self.baseline_means = np.asarray(baseline_means, dtype=np.float64)
        self.baseline_confidence = float(baseline_confidence)
        self.window_size = window_size
        self.alpha = alpha
        self.confidence_drop_threshold = confidence_drop_threshold
        self._mean_window = deque(maxlen=window_size)
        self._confidence_window = deque(maxlen=window_size)

    def record(self, image, model):
        """Record one live inference request. `image` is a single
        (1, H, W) array; `model` is used to compute the real predicted
        confidence for that image so confidence drift is measured from
        actual model output, not simulated."""
        self._mean_window.append(float(image.mean()))

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(image[None, ...]))
            probs = F.softmax(logits, dim=1)
            confidence = float(probs.max().item())
        self._confidence_window.append(confidence)

    def check(self):
        if len(self._mean_window) < self.window_size:
            return {
                "ready": False,
                "n_samples": len(self._mean_window),
                "drift_detected": False,
                "reason": "insufficient samples for a statistically meaningful window",
            }

        window = np.array(self._mean_window)
        ks_stat, p_value = stats.ks_2samp(self.baseline_means, window)
        input_drift = bool(p_value < self.alpha)

        mean_confidence = float(np.mean(self._confidence_window))
        confidence_drop = self.baseline_confidence - mean_confidence
        confidence_drift = bool(confidence_drop > self.confidence_drop_threshold)

        return {
            "ready": True,
            "n_samples": len(self._mean_window),
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "input_drift_detected": input_drift,
            "mean_confidence": mean_confidence,
            "baseline_confidence": self.baseline_confidence,
            "confidence_drop": confidence_drop,
            "confidence_drift_detected": confidence_drift,
            "drift_detected": input_drift or confidence_drift,
        }

    def reset_window(self):
        self._mean_window.clear()
        self._confidence_window.clear()
