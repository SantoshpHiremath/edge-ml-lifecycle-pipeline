# edge-ml-lifecycle-pipeline

A real, tested project covering the full lifecycle of a small
computer-vision model — **train → version/register → promote → export
for edge inference → serve → monitor for drift → decide whether to
retrain → roll back** — built to close a specific gap identified
against Mitsubishi Heavy Industries EMEA's "Werkstudent Software
Development Edge AI" posting: "Den Lifecycle von Machine Learning
Applications beherrschst Du." Every other project in this portfolio
that touches ML (`steel-defect-cv-classifier`,
`agentic-embedded-firmware-assistant`, `mlir-edge-lowering`, and
others) covers training or a single downstream step; none of them
cover the lifecycle *around* a model as its own subject. This one does.

## What this is (read before citing anywhere)

- `src/data.py` / `src/model.py` — a small, real PyTorch CNN
  (`TinyEdgeCNN`, two conv blocks + one FC layer, sized for a 16×16
  grayscale input) trained on a synthetic four-class shape dataset
  (`blank` / `circle` / `square` / `triangle`). Deliberately small and
  fast so the *lifecycle tooling* is what's being demonstrated, not a
  large model.
- `src/registry.py` — a real, file-backed model registry: every
  trained model gets an incrementing version, versions carry real
  metrics and a lifecycle stage (`staging` / `production` /
  `archived`), promoting a new version to production automatically
  archives whichever version held it before, and the manifest persists
  to disk (a fresh `ModelRegistry` instance pointed at the same
  directory sees the same state — verified directly by a test that
  creates a second instance and checks it agrees).
- `src/export.py` — exports the trained PyTorch model to **ONNX**, the
  standard interchange format for running a model on edge hardware via
  a lightweight runtime rather than a full PyTorch install. Verified
  with a real numeric parity check: the same input batch is run through
  both the original PyTorch model and the exported ONNX graph via
  `onnxruntime`, and the two outputs are confirmed to agree to within
  `1e-4` (in practice, max observed difference ≈ `3.8e-6`).
- `src/serve.py` — a small, real Flask API (`/health`, `/model-info`,
  `/predict`) that always serves whatever the registry currently has
  marked `production` — promoting or rolling back a version in the
  registry changes what this server returns with no code change,
  verified by a test that promotes a second model mid-test and confirms
  `/model-info` reflects it immediately.
- `src/monitor.py` — real drift monitoring on a rolling window of live
  requests, using two independent, genuinely statistical signals: a
  two-sample Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`) comparing
  live input-image statistics against the training-time baseline, and
  a comparison of the model's own mean prediction confidence against a
  baseline confidence.
- `src/retrain_trigger.py` — turns a drift-check result (plus,
  optionally, a recent labeled-accuracy spot-check) into an explicit,
  explainable retrain-or-not decision with stated reasons.
- `run_pipeline.py` — runs the entire loop end to end: train v1 →
  register → promote → export to ONNX → verify parity → monitor normal
  traffic (no drift) → monitor a simulated distribution shift (drift
  fires) → train v2 on more data → register/compare/promote → roll
  back to v1. Output below is copied directly from an actual run.
- `tests/` — 43 tests, all passing, covering every module above
  including the registry's persistence-across-instances guarantee, the
  ONNX export's numeric parity, the Flask API's status codes, and the
  drift monitor firing correctly on both a shifted-distribution case
  and a matching-distribution case.

## Honest disclosures — what's real, what's substituted, and why

**The data is synthetic, not a real camera/drone feed.** There's no
real edge-device image stream (surveillance camera, drone, or
otherwise) available in this environment. The four-class shape dataset
is procedurally generated with controlled noise and jitter — enough to
require real training (loss genuinely decreases, accuracy is measured
on a real held-out split), but it is not a claim of having built or
evaluated a production computer-vision model for camera/drone
footage.

**The model is exported to an edge *format*, not deployed on real edge
*hardware*.** `export_to_onnx()` and the parity check against
`onnxruntime` are both real — the resulting `.onnx` file is a genuine,
runnable edge-inference artifact, the same format a Raspberry Pi or
similar device would run via `onnxruntime` or a comparable lightweight
engine. But nothing in this project actually runs on a Raspberry Pi,
Jetson, or any physical edge device — everything executes on this
sandbox's host CPU. This is the same "validated but not executed on
the native target" disclosure pattern already used elsewhere in this
portfolio (Kubernetes manifests schema-validated but not
cluster-applied; the ARM firmware in `agentic-embedded-firmware-assistant`
is the one exception that *is* real hardware emulation, via QEMU).

**The task is easy — accuracy hits 100%, and that's a sign of scope,
not of production quality.** Four visually distinct synthetic shapes
on a 16×16 canvas is not a hard classification problem; the model
reaches near-perfect validation accuracy in a handful of epochs. That's
reported directly rather than presented as evidence of strong modeling
skill — the point of this project is the lifecycle machinery around
the model, not the model's own difficulty.

**A genuinely interesting, non-cherry-picked finding from the drift
test.** When `run_pipeline.py` feeds the monitor a deliberately
shifted (much brighter/noisier) input distribution, the KS test
correctly fires (`p ≈ 1.68e-37`, far below the `alpha=0.01` threshold)
— but the model's mean *confidence* on that same shifted data went
**up**, not down (baseline 0.995 → 1.000 on the observed run), so the
confidence-drift signal did *not* fire alongside the input-drift
signal. This is a real, known phenomenon (small models are often
*more* overconfident on out-of-distribution input, not less) and is
reported here exactly as observed rather than tuned away — it's also
the concrete reason this project measures two independent drift
signals instead of one: input-distribution drift caught what
confidence-based monitoring alone would have missed.

**No live production traffic, no real retraining triggered by real
production data.** The "shifted traffic" and "recent labeled accuracy"
values fed into `decide_retrain()` in the demo run are simulated for
demonstration purposes, disclosed as such directly in `run_pipeline.py`
rather than presented as real operational data.

## Sample output (from an actual run of `run_pipeline.py`)

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

## Verification performed

- `python -m pytest tests/ -v` — 43/43 tests pass, re-verified fresh
  in this environment immediately before writing this README.
- `python run_pipeline.py` — runs end to end; the "Sample output"
  section above is copied directly from this run's actual stdout.
- A real environment gap was hit and fixed during development:
  `torch.onnx.export()` on this environment's torch version (2.13)
  requires the `onnxscript` package for its default dynamo-based
  exporter and failed with `ModuleNotFoundError: No module named
  'onnxscript'` on the first run. Installed it directly
  (`pip install onnxscript`) rather than silently falling back to the
  legacy exporter, then re-ran the full suite to confirm the fix.
- The ONNX parity check is a real cross-engine comparison, not a
  shape-only check: it runs actual inference through both PyTorch and
  `onnxruntime` on the same input batch and asserts numeric agreement.

## Running it yourself

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python run_pipeline.py
```
