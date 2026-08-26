"""Synthetic small image dataset for a lightweight edge-deployable classifier.

Four classes drawn on a 16x16 grayscale canvas: circle, square, triangle,
blank. Deliberately tiny (fast to train in seconds, small enough to be a
realistic edge-inference target) rather than a copy of the larger
steel-defect-cv-classifier project elsewhere in this portfolio -- this
project is about the lifecycle around a model, not about the model's
own domain accuracy.
"""

import numpy as np

CLASSES = ["blank", "circle", "square", "triangle"]
IMG_SIZE = 16


def _draw_circle(canvas, rng, jitter):
    cx, cy = IMG_SIZE / 2 + jitter[0], IMG_SIZE / 2 + jitter[1]
    r = IMG_SIZE / 2 - 2 + jitter[2]
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    canvas[mask] = 0.85
    return canvas


def _draw_square(canvas, rng, jitter):
    margin = int(2 + jitter[0])
    margin = max(1, min(margin, IMG_SIZE // 2 - 1))
    canvas[margin:IMG_SIZE - margin, margin:IMG_SIZE - margin] = 0.85
    return canvas


def _draw_triangle(canvas, rng, jitter):
    apex_shift = jitter[0]
    for row in range(IMG_SIZE):
        frac = row / (IMG_SIZE - 1)
        half_width = frac * (IMG_SIZE / 2 - 1)
        center = IMG_SIZE / 2 + apex_shift
        lo = max(0, int(center - half_width))
        hi = min(IMG_SIZE, int(center + half_width) + 1)
        canvas[row, lo:hi] = 0.85
    return canvas


def _generate_one(label, rng, noise_std):
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    jitter = rng.uniform(-1.0, 1.0, size=3)
    if label == "circle":
        canvas = _draw_circle(canvas, rng, jitter)
    elif label == "square":
        canvas = _draw_square(canvas, rng, jitter)
    elif label == "triangle":
        canvas = _draw_triangle(canvas, rng, jitter)
    # "blank" stays all zeros aside from noise

    noise = rng.normal(0, noise_std, size=canvas.shape).astype(np.float32)
    canvas = np.clip(canvas + noise, 0.0, 1.0)
    return canvas


def generate_dataset(n_per_class=60, seed=0, noise_std=0.05):
    """Generate a labeled synthetic image dataset.

    Returns (images, labels) where images is a float32 array of shape
    (N, 1, IMG_SIZE, IMG_SIZE) in [0, 1] and labels is an int array of
    shape (N,) indexing into CLASSES.
    """
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for label_idx, label in enumerate(CLASSES):
        for _ in range(n_per_class):
            img = _generate_one(label, rng, noise_std)
            images.append(img)
            labels.append(label_idx)

    images = np.stack(images)[:, None, :, :]  # (N, 1, H, W)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle so classes aren't grouped block-by-block.
    perm = rng.permutation(len(labels))
    return images[perm], labels[perm]


def split_dataset(images, labels, val_frac=0.2, seed=1):
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])


def baseline_input_stats(images):
    """Per-channel mean/std of a dataset -- used later as the training-time
    baseline that the drift monitor compares live traffic against."""
    return {"mean": float(images.mean()), "std": float(images.std())}
