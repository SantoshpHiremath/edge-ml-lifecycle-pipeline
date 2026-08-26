import numpy as np

from src.data import generate_dataset, split_dataset
from src.model import train_model
from src.monitor import DriftMonitor


def _setup():
    images, labels = generate_dataset(n_per_class=40, seed=0)
    (train_x, train_y), (val_x, val_y) = split_dataset(images, labels, seed=1)
    model, _ = train_model(train_x, train_y, epochs=12, seed=0)
    baseline_means = [float(img.mean()) for img in train_x]
    baseline_confidence = 0.85  # representative baseline recorded at registration
    return model, train_x, baseline_means, baseline_confidence


def test_not_ready_before_window_fills():
    model, train_x, baseline_means, baseline_conf = _setup()
    monitor = DriftMonitor(baseline_means, baseline_conf, window_size=30)
    monitor.record(train_x[0], model)
    result = monitor.check()
    assert result["ready"] is False
    assert result["drift_detected"] is False


def test_no_drift_when_live_traffic_matches_training_distribution():
    model, train_x, baseline_means, baseline_conf = _setup()
    monitor = DriftMonitor(baseline_means, baseline_conf, window_size=30, alpha=0.01)

    rng = np.random.default_rng(99)
    idx = rng.choice(len(train_x), size=30, replace=True)
    for i in idx:
        monitor.record(train_x[i], model)

    result = monitor.check()
    assert result["ready"] is True
    assert result["input_drift_detected"] is False


def test_drift_detected_on_shifted_input_distribution():
    model, train_x, baseline_means, baseline_conf = _setup()
    monitor = DriftMonitor(baseline_means, baseline_conf, window_size=30, alpha=0.01)

    # Simulate a real distribution shift: much brighter, noisier images
    # than anything the model was trained on (e.g. a camera exposed to
    # a very different lighting environment on an edge device).
    rng = np.random.default_rng(1)
    shifted = np.clip(train_x[:30] + rng.uniform(0.5, 0.9, size=train_x[:30].shape), 0, 1)
    for img in shifted:
        monitor.record(img.astype(np.float32), model)

    result = monitor.check()
    assert result["ready"] is True
    assert result["input_drift_detected"] is True
    assert result["drift_detected"] is True


def test_confidence_drift_detected_on_ambiguous_inputs():
    model, train_x, baseline_means, _ = _setup()
    # Deliberately unrealistic high baseline so the model's real
    # (lower) confidence on noisy input reads as a drop.
    monitor = DriftMonitor(baseline_means, baseline_confidence=0.99,
                            window_size=30, confidence_drop_threshold=0.05)

    rng = np.random.default_rng(2)
    noisy = np.clip(rng.normal(0.5, 0.3, size=train_x[:30].shape), 0, 1).astype(np.float32)
    for img in noisy:
        monitor.record(img, model)

    result = monitor.check()
    assert result["ready"] is True
    assert result["confidence_drift_detected"] is True


def test_reset_window_clears_state():
    model, train_x, baseline_means, baseline_conf = _setup()
    monitor = DriftMonitor(baseline_means, baseline_conf, window_size=5)
    for img in train_x[:5]:
        monitor.record(img, model)
    assert monitor.check()["ready"] is True

    monitor.reset_window()
    assert monitor.check()["ready"] is False
