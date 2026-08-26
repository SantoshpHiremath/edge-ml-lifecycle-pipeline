import torch

from src.data import generate_dataset, split_dataset
from src.model import TinyEdgeCNN, train_model
from src.registry import STAGE_ARCHIVED, STAGE_PRODUCTION, STAGE_STAGING, ModelRegistry


def _tiny_trained_model():
    images, labels = generate_dataset(n_per_class=10, seed=0)
    (train_x, train_y), _ = split_dataset(images, labels, seed=1)
    model, _ = train_model(train_x, train_y, epochs=2, seed=0)
    return model


def test_register_assigns_incrementing_versions(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    v1 = reg.register(_tiny_trained_model(), {"accuracy": 0.5})
    v2 = reg.register(_tiny_trained_model(), {"accuracy": 0.6})
    assert v1 == 1
    assert v2 == 2


def test_new_versions_start_as_staging(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    v1 = reg.register(_tiny_trained_model(), {"accuracy": 0.5})
    assert reg.get_version(v1)["stage"] == STAGE_STAGING


def test_promote_sets_production_and_archives_previous(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    v1 = reg.register(_tiny_trained_model(), {"accuracy": 0.5})
    v2 = reg.register(_tiny_trained_model(), {"accuracy": 0.7})

    reg.promote(v1)
    assert reg.get_production()["version"] == v1

    reg.promote(v2)
    assert reg.get_production()["version"] == v2
    assert reg.get_version(v1)["stage"] == STAGE_ARCHIVED


def test_get_production_returns_none_when_nothing_promoted(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    reg.register(_tiny_trained_model(), {"accuracy": 0.5})
    assert reg.get_production() is None


def test_rollback_restores_earlier_version_to_production(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    v1 = reg.register(_tiny_trained_model(), {"accuracy": 0.5})
    v2 = reg.register(_tiny_trained_model(), {"accuracy": 0.4})  # a regression
    reg.promote(v1)
    reg.promote(v2)
    assert reg.get_production()["version"] == v2

    reg.rollback_to(v1)
    assert reg.get_production()["version"] == v1
    assert reg.get_version(v2)["stage"] == STAGE_ARCHIVED


def test_manifest_persists_across_new_registry_instance(tmp_path):
    root = tmp_path / "registry"
    reg_a = ModelRegistry(root)
    v1 = reg_a.register(_tiny_trained_model(), {"accuracy": 0.5})
    reg_a.promote(v1)

    reg_b = ModelRegistry(root)  # fresh instance, same directory
    assert reg_b.get_production()["version"] == v1
    assert len(reg_b.list_versions()) == 1


def test_load_model_round_trips_identical_weights(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    original = _tiny_trained_model()
    v1 = reg.register(original, {"accuracy": 0.5})

    loaded = reg.load_model(v1)
    for p_orig, p_loaded in zip(original.parameters(), loaded.parameters()):
        assert torch.allclose(p_orig, p_loaded)


def test_list_versions_sorted_by_version_number(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    reg.register(_tiny_trained_model(), {"accuracy": 0.5})
    reg.register(_tiny_trained_model(), {"accuracy": 0.6})
    reg.register(_tiny_trained_model(), {"accuracy": 0.7})
    versions = [v["version"] for v in reg.list_versions()]
    assert versions == [1, 2, 3]
