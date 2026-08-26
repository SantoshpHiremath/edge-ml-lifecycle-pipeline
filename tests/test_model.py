import numpy as np
import torch

from src.data import CLASSES, generate_dataset, split_dataset
from src.model import TinyEdgeCNN, evaluate_model, train_model


def test_forward_pass_output_shape():
    model = TinyEdgeCNN()
    x = torch.zeros(4, 1, 16, 16)
    out = model(x)
    assert out.shape == (4, len(CLASSES))


def test_training_loss_decreases():
    images, labels = generate_dataset(n_per_class=30, seed=0)
    (train_x, train_y), _ = split_dataset(images, labels, seed=1)
    _, losses = train_model(train_x, train_y, epochs=10, seed=0)
    assert losses[-1] < losses[0]


def test_trained_model_beats_random_chance():
    images, labels = generate_dataset(n_per_class=40, seed=0)
    (train_x, train_y), (val_x, val_y) = split_dataset(images, labels, seed=1)
    model, _ = train_model(train_x, train_y, epochs=15, seed=0)
    result = evaluate_model(model, val_x, val_y)
    random_chance = 1.0 / len(CLASSES)
    assert result["accuracy"] > random_chance * 1.5


def test_evaluate_model_output_structure():
    images, labels = generate_dataset(n_per_class=10, seed=0)
    model = TinyEdgeCNN()
    result = evaluate_model(model, images, labels)
    assert set(result.keys()) == {"accuracy", "n"}
    assert result["n"] == len(labels)


def test_same_seed_gives_reproducible_training():
    images, labels = generate_dataset(n_per_class=20, seed=0)
    (train_x, train_y), _ = split_dataset(images, labels, seed=1)
    model_a, losses_a = train_model(train_x, train_y, epochs=5, seed=7)
    model_b, losses_b = train_model(train_x, train_y, epochs=5, seed=7)
    assert np.allclose(losses_a, losses_b)
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b)
