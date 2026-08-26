import numpy as np

from src.data import CLASSES, IMG_SIZE, baseline_input_stats, generate_dataset, split_dataset


def test_dataset_shape_and_range():
    images, labels = generate_dataset(n_per_class=10, seed=0)
    assert images.shape == (40, 1, IMG_SIZE, IMG_SIZE)
    assert labels.shape == (40,)
    assert images.min() >= 0.0 and images.max() <= 1.0


def test_dataset_is_class_balanced():
    images, labels = generate_dataset(n_per_class=15, seed=1)
    counts = np.bincount(labels, minlength=len(CLASSES))
    assert all(c == 15 for c in counts)


def test_dataset_not_grouped_by_class():
    _, labels = generate_dataset(n_per_class=20, seed=2)
    # If it were grouped, the first 20 labels would all be identical.
    assert len(set(labels[:20].tolist())) > 1


def test_same_seed_reproducible():
    images_a, labels_a = generate_dataset(n_per_class=10, seed=42)
    images_b, labels_b = generate_dataset(n_per_class=10, seed=42)
    assert np.array_equal(images_a, images_b)
    assert np.array_equal(labels_a, labels_b)


def test_different_seeds_differ():
    images_a, _ = generate_dataset(n_per_class=10, seed=1)
    images_b, _ = generate_dataset(n_per_class=10, seed=2)
    assert not np.array_equal(images_a, images_b)


def test_split_dataset_proportions():
    images, labels = generate_dataset(n_per_class=25, seed=3)
    (train_x, train_y), (val_x, val_y) = split_dataset(images, labels, val_frac=0.2, seed=1)
    assert len(train_x) + len(val_x) == len(images)
    assert len(val_x) == int(len(images) * 0.2)


def test_split_dataset_no_overlap_by_content():
    images, labels = generate_dataset(n_per_class=25, seed=4)
    (train_x, _), (val_x, _) = split_dataset(images, labels, val_frac=0.3, seed=2)
    # Every val image should be bit-for-bit absent from the train set.
    train_flat = {t.tobytes() for t in train_x}
    for v in val_x:
        assert v.tobytes() not in train_flat


def test_baseline_input_stats_are_real_floats():
    images, _ = generate_dataset(n_per_class=10, seed=5)
    stats = baseline_input_stats(images)
    assert set(stats.keys()) == {"mean", "std"}
    assert 0.0 <= stats["mean"] <= 1.0
    assert stats["std"] > 0.0
