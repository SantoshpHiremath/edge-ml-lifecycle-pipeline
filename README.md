# edge-ml-lifecycle-pipeline

A tested project covering the full lifecycle of a small computer-vision
model: **train → version/register → promote → export for edge inference →
serve → monitor for drift → decide whether to retrain → roll back**.

My other ML projects each cover training or a single downstream step.
This one treats the lifecycle *around* a model as its own subject.

## What it does

- `src/data.py` / `src/model.py` — a small PyTorch CNN (`TinyEdgeCNN`, two
  conv blocks + one FC layer, sized for a 16×16 grayscale input) trained on
  a synthetic four-class shape dataset (`blank` / `circle` / `square` /
  `triangle`). The model is deliberately small and fast so the *lifecycle
  tooling* is the focus.
- `src/registry.py` — a file-backed model registry: every trained model
  gets an incrementing version, versions carry metrics and a lifecycle
  stage (`staging` / `production` / `archived`), promoting a new version to
  production automatically archives whichever version held it before, and
  the manifest persists to disk (a fresh `ModelRegistry` instance pointed
  at the same directory sees the same state, verified by a test that
  creates a second instance and checks it agrees).
- `src/export.py` — exports the trained PyTorch model to **ONNX**, the
  standard interchange format for running a model on edge hardware via a
  lightweight runtime rather than a full PyTorch install. A numeric parity
  check runs the same input batch through both the original PyTorch model
  and the exported ONNX graph via `onnxruntime`, and confirms the two
  outputs agree to within `1e-4` (max observed difference ≈ `3.8e-6`).
- `src/serve.py` — a small Flask API (`/health`, `/model-info`, `/predict`)
  that always serves whatever the registry currently has marked
  `production`. Promoting or rolling back a version in the registry changes
  what the server returns with no code change, verified by a test that
  promotes a second model mid-test and confirms `/model-info` reflects it
  immediately.
- `src/monitor.py` — drift monitoring on a rolling window of live
  requests, using two independent statistical signals: a two-sample
  Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`) comparing live
  input-image statistics against the training-time baseline, and a
  comparison of the model's own mean prediction confidence against a
  baseline confidence.
- `src/retrain_trigger.py` — turns a drift-check result (plus, optionally,
  a recent labeled-accuracy spot-check) into an explicit, explainable
  retrain-or-not decision with stated reasons.
- `run_pipeline.py` — runs the entire loop end to end: train v1 → register
  → promote → export to ONNX → verify parity → monitor normal traffic (no
  drift) → monitor a simulated distribution shift (drift fires) → train v2
  on more data → register/compare/promote → roll back to v1.

## Scope

- The data is synthetic. The four-class shape dataset is procedurally
  generated with controlled noise and jitter, enough to require real
  training (loss decreases, accuracy is measured on a held-out split). The
  pipeline is built so a real image stream can replace it.
- The model is exported to an edge *format* (ONNX) and validated with
  `onnxruntime` on the host CPU. The resulting `.onnx` file is the same
  artifact a Raspberry Pi or similar device would run via `onnxruntime` or
  a comparable lightweight engine.
- The task is intentionally easy: four visually distinct shapes on a 16×16
  canvas, so validation accuracy reaches 100% within a handful of epochs.
  The focus is the lifecycle machinery around the model.
- The "shifted traffic" and "recent labeled accuracy" values fed into
  `decide_retrain()` in the demo run are simulated, and labeled as such in
  `run_pipeline.py`.

## Results

When `run_pipeline.py` feeds the monitor a deliberately shifted (much
brighter and noisier) input distribution, the KS test fires (`p ≈ 1.68e-37`,
far below the `alpha=0.01` threshold). The model's mean *confidence* on
that same shifted data went **up** (baseline 0.995 → 1.000 on the observed
run), so the confidence-drift signal did not fire alongside the
input-drift signal. Small models are often *more* overconfident on
out-of-distribution input, and this is the concrete reason the project
measures two independent drift signals: input-distribution drift catches
what confidence-based monitoring alone would miss.

Sample output (from a run of `run_pipeline.py`):

```
======================================================================
1. TRAIN — v1
======================================================================
Trained on 192 images, validated on 48.
Loss: 1.3899 -> 0.0176
Validation accuracy: 100.00%

======================================================================
3. EXPORT — ONNX for edge inference
======================================================================
PyTorch vs. ONNX-Runtime parity check: max abs diff = 3.81e-06 (OK)

======================================================================
4. MONITOR — simulated normal live traffic
======================================================================
KS p-value: 0.3969 (drift threshold alpha=0.01) -> drift_detected=False

======================================================================
5. MONITOR — simulated distribution shift on the edge device
======================================================================
KS p-value: 1.681e-37 -> input_drift_detected=True
Mean confidence: 1.000 (baseline 0.995) -> confidence_drift_detected=False
Retrain decision: {'trigger_retrain': True, 'reasons': [
  'input distribution drift detected (KS p-value=1.681e-37 < alpha)',
  'recent labeled-sample accuracy 55.00% below minimum 75.00%']}

======================================================================
7. ROLLBACK — demonstrate reverting a bad promotion
======================================================================
Rolled back to v1. Current production: v1
```

## Tests

- `python -m pytest tests/ -v` — 43/43 tests pass. They cover every module
  above, including the registry's persistence-across-instances guarantee,
  the ONNX export's numeric parity, the Flask API's status codes, and the
  drift monitor firing correctly on both a shifted-distribution case and a
  matching-distribution case.
- The ONNX parity check is a cross-engine comparison, not a shape-only
  check: it runs inference through both PyTorch and `onnxruntime` on the
  same input batch and asserts numeric agreement.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python run_pipeline.py
```

## Notes

`torch.onnx.export()` on torch 2.13 requires the `onnxscript` package for
its default dynamo-based exporter (`pip install onnxscript`), which I
installed rather than falling back to the legacy exporter.

## Possible extensions

- Run the exported `.onnx` model on a physical edge device (Raspberry Pi,
  Jetson) and feed the monitor from its live traffic.
- Replace the synthetic shape data with a real camera or drone image feed.
