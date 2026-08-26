from src.data import IMG_SIZE, generate_dataset, split_dataset
from src.model import train_model
from src.registry import ModelRegistry
from src.serve import create_app


def _registry_with_production(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    images, labels = generate_dataset(n_per_class=15, seed=0)
    (train_x, train_y), _ = split_dataset(images, labels, seed=1)
    model, _ = train_model(train_x, train_y, epochs=8, seed=0)
    v1 = reg.register(model, {"accuracy": 0.9})
    reg.promote(v1)
    return reg, v1


def test_health_endpoint(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    client = create_app(reg).test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_model_info_returns_503_with_no_production_model(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    client = create_app(reg).test_client()
    resp = client.get("/model-info")
    assert resp.status_code == 503


def test_model_info_returns_current_production_version(tmp_path):
    reg, v1 = _registry_with_production(tmp_path)
    client = create_app(reg).test_client()
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["version"] == v1
    assert body["stage"] == "production"


def test_predict_returns_valid_class(tmp_path):
    reg, _ = _registry_with_production(tmp_path)
    client = create_app(reg).test_client()
    image = [[0.0] * IMG_SIZE for _ in range(IMG_SIZE)]
    resp = client.post("/predict", json={"image": image})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["predicted_class"] in ["blank", "circle", "square", "triangle"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_missing_image_returns_400(tmp_path):
    reg, _ = _registry_with_production(tmp_path)
    client = create_app(reg).test_client()
    resp = client.post("/predict", json={})
    assert resp.status_code == 400


def test_predict_wrong_shape_returns_400(tmp_path):
    reg, _ = _registry_with_production(tmp_path)
    client = create_app(reg).test_client()
    resp = client.post("/predict", json={"image": [[0.0, 0.0], [0.0, 0.0]]})
    assert resp.status_code == 400


def test_predict_returns_503_with_no_production_model(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    client = create_app(reg).test_client()
    image = [[0.0] * IMG_SIZE for _ in range(IMG_SIZE)]
    resp = client.post("/predict", json={"image": image})
    assert resp.status_code == 503


def test_predict_reflects_registry_promotion_change(tmp_path):
    reg, v1 = _registry_with_production(tmp_path)
    images, labels = generate_dataset(n_per_class=15, seed=2)
    (train_x, train_y), _ = split_dataset(images, labels, seed=3)
    model2, _ = train_model(train_x, train_y, epochs=8, seed=1)
    v2 = reg.register(model2, {"accuracy": 0.95})
    reg.promote(v2)

    client = create_app(reg).test_client()
    resp = client.get("/model-info")
    assert resp.get_json()["version"] == v2
