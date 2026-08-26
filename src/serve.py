"""A small, real Flask inference server that always serves whatever
the registry currently has marked as `production` -- the "serve" step
of the lifecycle, wired to the registry rather than to a hardcoded
model file, so promoting/rolling back a version in the registry
actually changes what this server returns without a code change."""

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request

from src.data import CLASSES, IMG_SIZE
from src.registry import ModelRegistry


def create_app(registry: ModelRegistry):
    app = Flask(__name__)
    app.config["registry"] = registry

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/model-info")
    def model_info():
        prod = registry.get_production()
        if prod is None:
            return jsonify({"error": "no production model registered"}), 503
        return jsonify({
            "version": prod["version"],
            "stage": prod["stage"],
            "metrics": prod["metrics"],
            "metadata": prod["metadata"],
        })

    @app.post("/predict")
    def predict():
        prod = registry.get_production()
        if prod is None:
            return jsonify({"error": "no production model registered"}), 503

        payload = request.get_json(silent=True) or {}
        image = payload.get("image")
        if image is None:
            return jsonify({"error": "missing 'image' field"}), 400

        arr = np.array(image, dtype=np.float32)
        if arr.shape != (IMG_SIZE, IMG_SIZE):
            return jsonify({
                "error": f"expected image shape ({IMG_SIZE}, {IMG_SIZE}), got {arr.shape}"
            }), 400

        model = registry.load_model(prod["version"])
        with torch.no_grad():
            logits = model(torch.from_numpy(arr[None, None, :, :]))
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())

        return jsonify({
            "predicted_class": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "model_version": prod["version"],
        })

    return app
