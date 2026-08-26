"""End-to-end demo of the full lifecycle loop this project builds:

    train -> register -> promote -> export (edge format) -> serve
        -> monitor (drift) -> retrain decision -> new version -> rollback

Every step below calls real code from src/ -- nothing in this script is
printed without having actually happened (models are actually trained,
the registry is actually written to disk, the ONNX export is actually
run through onnxruntime, the drift check is a real KS-test result).
"""

import shutil
import tempfile

import numpy as np

from src.data import generate_dataset, split_dataset
from src.export import export_to_onnx, verify_export_parity
from src.model import evaluate_model, train_model
from src.monitor import DriftMonitor
from src.registry import ModelRegistry
from src.retrain_trigger import decide_retrain


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    workdir = tempfile.mkdtemp(prefix="edge_ml_lifecycle_")
    try:
        registry = ModelRegistry(f"{workdir}/registry")

        # 1. TRAIN v1 -----------------------------------------------------
        section("1. TRAIN — v1")
        images, labels = generate_dataset(n_per_class=60, seed=0)
        (train_x, train_y), (val_x, val_y) = split_dataset(images, labels, seed=1)
        model_v1, losses = train_model(train_x, train_y, epochs=15, seed=0)
        metrics_v1 = evaluate_model(model_v1, val_x, val_y)
        print(f"Trained on {len(train_x)} images, validated on {len(val_x)}.")
        print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"Validation accuracy: {metrics_v1['accuracy']:.2%}")

        # 2. REGISTER + PROMOTE -------------------------------------------
        section("2. REGISTER + PROMOTE — v1")
        v1 = registry.register(model_v1, metrics_v1, metadata={"epochs": 15, "seed": 0})
        registry.promote(v1)
        print(f"Registered as version {v1}, promoted to production.")
        print(f"Current production: v{registry.get_production()['version']} "
              f"(accuracy={registry.get_production()['metrics']['accuracy']:.2%})")

        # 3. EXPORT for edge inference -------------------------------------
        section("3. EXPORT — ONNX for edge inference")
        onnx_path = f"{workdir}/v1_model.onnx"
        export_to_onnx(model_v1, onnx_path)
        parity_ok, max_diff = verify_export_parity(model_v1, onnx_path, val_x)
        print(f"Exported to {onnx_path}")
        print(f"PyTorch vs. ONNX-Runtime parity check: max abs diff = {max_diff:.2e} "
              f"({'OK' if parity_ok else 'FAILED'})")

        # 4. MONITOR — normal traffic, no drift ----------------------------
        section("4. MONITOR — simulated normal live traffic")
        baseline_means = [float(img.mean()) for img in train_x]
        baseline_confidence = float(np.mean([
            float(np.max(_softmax(model_v1, img))) for img in val_x[:20]
        ]))
        monitor = DriftMonitor(baseline_means, baseline_confidence, window_size=30, alpha=0.01)

        rng = np.random.default_rng(7)
        idx = rng.choice(len(val_x), size=30, replace=True)
        for i in idx:
            monitor.record(val_x[i], model_v1)
        normal_check = monitor.check()
        print(f"Window filled with {normal_check['n_samples']} in-distribution samples.")
        print(f"KS p-value: {normal_check['p_value']:.4f} "
              f"(drift threshold alpha=0.01) -> "
              f"drift_detected={normal_check['drift_detected']}")

        normal_decision = decide_retrain(normal_check)
        print(f"Retrain decision: {normal_decision}")

        # 5. MONITOR — shifted traffic, drift fires ------------------------
        section("5. MONITOR — simulated distribution shift on the edge device")
        monitor.reset_window()
        shifted = np.clip(
            val_x[:30] + rng.uniform(0.5, 0.9, size=val_x[:30].shape), 0, 1
        ).astype(np.float32)
        for img in shifted:
            monitor.record(img, model_v1)
        shifted_check = monitor.check()
        print(f"KS p-value: {shifted_check['p_value']:.4g} -> "
              f"input_drift_detected={shifted_check['input_drift_detected']}")
        print(f"Mean confidence: {shifted_check['mean_confidence']:.3f} "
              f"(baseline {shifted_check['baseline_confidence']:.3f}) -> "
              f"confidence_drift_detected={shifted_check['confidence_drift_detected']}")

        shifted_decision = decide_retrain(shifted_check, recent_labeled_accuracy=0.55)
        print(f"Retrain decision: {shifted_decision}")

        # 6. RETRAIN — v2 on more data --------------------------------------
        section("6. RETRAIN — v2 trained on a larger dataset")
        images2, labels2 = generate_dataset(n_per_class=120, seed=10)
        (train_x2, train_y2), (val_x2, val_y2) = split_dataset(images2, labels2, seed=11)
        model_v2, _ = train_model(train_x2, train_y2, epochs=15, seed=10)
        metrics_v2 = evaluate_model(model_v2, val_x2, val_y2)
        v2 = registry.register(model_v2, metrics_v2, metadata={"epochs": 15, "seed": 10})
        print(f"v2 validation accuracy: {metrics_v2['accuracy']:.2%} "
              f"(v1 was {metrics_v1['accuracy']:.2%})")

        if metrics_v2["accuracy"] >= metrics_v1["accuracy"]:
            registry.promote(v2)
            print(f"v2 promoted to production.")
        else:
            print("v2 did not beat v1 — left in staging, v1 remains production.")

        # 7. ROLLBACK ----------------------------------------------------
        section("7. ROLLBACK — demonstrate reverting a bad promotion")
        registry.rollback_to(v1)
        print(f"Rolled back to v{v1}. Current production: "
              f"v{registry.get_production()['version']}")

        section("SUMMARY")
        for entry in registry.list_versions():
            print(f"  v{entry['version']}: stage={entry['stage']:<10} "
                  f"accuracy={entry['metrics']['accuracy']:.2%}")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _softmax(model, image):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(image[None, ...]))
        return F.softmax(logits, dim=1)[0].numpy()


if __name__ == "__main__":
    main()
