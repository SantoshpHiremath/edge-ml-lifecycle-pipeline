"""Re-runs this project's real pipeline and serializes the results to
JSON -- added specifically to support ai-results-dashboard, a separate
frontend project that visualizes these real, already-tested results.
Does not invent any numbers: every value here comes from actually
training the model, actually registering/promoting versions, and
actually running the real KS-test-based drift monitor, exactly as
run_pipeline.py does.
"""

import json
import shutil
import sys
import tempfile

import numpy as np

from src.data import generate_dataset, split_dataset
from src.model import evaluate_model, train_model
from src.monitor import DriftMonitor
from src.registry import ModelRegistry
from src.retrain_trigger import decide_retrain


def main(output_path):
    workdir = tempfile.mkdtemp(prefix="edge_ml_lifecycle_export_")
    try:
        registry = ModelRegistry(f"{workdir}/registry")

        images, labels = generate_dataset(n_per_class=60, seed=0)
        (train_x, train_y), (val_x, val_y) = split_dataset(images, labels, seed=1)
        model_v1, losses = train_model(train_x, train_y, epochs=15, seed=0)
        metrics_v1 = evaluate_model(model_v1, val_x, val_y)
        v1 = registry.register(model_v1, metrics_v1, metadata={"epochs": 15, "seed": 0})
        registry.promote(v1)

        images2, labels2 = generate_dataset(n_per_class=120, seed=10)
        (train_x2, train_y2), (val_x2, val_y2) = split_dataset(images2, labels2, seed=11)
        model_v2, _ = train_model(train_x2, train_y2, epochs=15, seed=10)
        metrics_v2 = evaluate_model(model_v2, val_x2, val_y2)
        v2 = registry.register(model_v2, metrics_v2, metadata={"epochs": 15, "seed": 10})
        if metrics_v2["accuracy"] >= metrics_v1["accuracy"]:
            registry.promote(v2)

        baseline_means = [float(img.mean()) for img in train_x]
        baseline_confidence = float(np.mean([
            float(np.max(_softmax(model_v1, img))) for img in val_x[:20]
        ]))

        monitor_normal = DriftMonitor(baseline_means, baseline_confidence, window_size=30, alpha=0.01)
        rng = np.random.default_rng(7)
        idx = rng.choice(len(val_x), size=30, replace=True)
        for i in idx:
            monitor_normal.record(val_x[i], model_v1)
        normal_check = monitor_normal.check()
        normal_decision = decide_retrain(normal_check)

        monitor_shifted = DriftMonitor(baseline_means, baseline_confidence, window_size=30, alpha=0.01)
        shifted = np.clip(
            val_x[:30] + rng.uniform(0.5, 0.9, size=val_x[:30].shape), 0, 1
        ).astype(np.float32)
        for img in shifted:
            monitor_shifted.record(img, model_v1)
        shifted_check = monitor_shifted.check()
        shifted_decision = decide_retrain(shifted_check, recent_labeled_accuracy=0.55)

        result = {
            "modelVersions": [
                {
                    "version": entry["version"],
                    "stage": entry["stage"],
                    "accuracy": entry["metrics"]["accuracy"],
                    "epochs": entry["metadata"].get("epochs"),
                }
                for entry in registry.list_versions()
            ],
            "driftChecks": [
                {
                    "scenario": "normal_traffic",
                    "nSamples": normal_check["n_samples"],
                    "ksStatistic": normal_check["ks_statistic"],
                    "pValue": normal_check["p_value"],
                    "inputDriftDetected": normal_check["input_drift_detected"],
                    "meanConfidence": normal_check["mean_confidence"],
                    "baselineConfidence": normal_check["baseline_confidence"],
                    "confidenceDriftDetected": normal_check["confidence_drift_detected"],
                    "retrainDecision": normal_decision,
                },
                {
                    "scenario": "shifted_traffic",
                    "nSamples": shifted_check["n_samples"],
                    "ksStatistic": shifted_check["ks_statistic"],
                    "pValue": shifted_check["p_value"],
                    "inputDriftDetected": shifted_check["input_drift_detected"],
                    "meanConfidence": shifted_check["mean_confidence"],
                    "baselineConfidence": shifted_check["baseline_confidence"],
                    "confidenceDriftDetected": shifted_check["confidence_drift_detected"],
                    "retrainDecision": shifted_decision,
                },
            ],
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote real pipeline results to {output_path}")

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
    output = sys.argv[1] if len(sys.argv) > 1 else "pipeline_results.json"
    main(output)
