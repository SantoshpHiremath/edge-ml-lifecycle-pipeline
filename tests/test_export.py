import os

from src.data import generate_dataset, split_dataset
from src.export import export_to_onnx, onnx_predict, verify_export_parity
from src.model import evaluate_model, train_model


def _trained_model_and_val():
    images, labels = generate_dataset(n_per_class=20, seed=0)
    (train_x, train_y), (val_x, val_y) = split_dataset(images, labels, seed=1)
    model, _ = train_model(train_x, train_y, epochs=10, seed=0)
    return model, val_x, val_y


def test_export_produces_a_file(tmp_path):
    model, val_x, _ = _trained_model_and_val()
    out_path = str(tmp_path / "model.onnx")
    export_to_onnx(model, out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_export_parity_within_tight_tolerance(tmp_path):
    model, val_x, _ = _trained_model_and_val()
    out_path = str(tmp_path / "model.onnx")
    export_to_onnx(model, out_path)

    ok, max_diff = verify_export_parity(model, out_path, val_x)
    assert ok, f"ONNX export diverged from PyTorch model by {max_diff}"
    assert max_diff < 1e-4


def test_onnx_predictions_match_torch_predictions(tmp_path):
    model, val_x, val_y = _trained_model_and_val()
    out_path = str(tmp_path / "model.onnx")
    export_to_onnx(model, out_path)

    torch_result = evaluate_model(model, val_x, val_y)
    onnx_preds = onnx_predict(out_path, val_x)
    onnx_accuracy = float((onnx_preds == val_y).mean())

    # Same weights, same math -- accuracy should match exactly, not just
    # "close", since this is the same model evaluated two ways.
    assert onnx_accuracy == torch_result["accuracy"]
