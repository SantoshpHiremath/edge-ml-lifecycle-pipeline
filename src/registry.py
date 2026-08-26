"""A real, tested, file-backed model registry.

Not a mock of MLflow/SageMaker Model Registry/etc. -- a small, honest
implementation of the same core idea: every trained model gets a
version, versions carry metrics and a lifecycle stage
(staging/production/archived), and promoting a new version to
production automatically archives whichever version held it before,
so there is always at most one "current" production model and a full
history of what came before it (rollback).
"""

import json
import time
from pathlib import Path

import torch

from src.model import TinyEdgeCNN

STAGE_STAGING = "staging"
STAGE_PRODUCTION = "production"
STAGE_ARCHIVED = "archived"


class ModelRegistry:
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text())
        else:
            self._manifest = {"versions": []}
            self._save()

    def _save(self):
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def _next_version(self):
        if not self._manifest["versions"]:
            return 1
        return max(v["version"] for v in self._manifest["versions"]) + 1

    def register(self, model, metrics, metadata=None, created_at=None):
        version = self._next_version()
        version_dir = self.root / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        model_path = version_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        entry = {
            "version": version,
            "model_path": str(model_path),
            "metrics": dict(metrics),
            "metadata": dict(metadata or {}),
            "stage": STAGE_STAGING,
            "created_at": created_at if created_at is not None else time.time(),
        }
        self._manifest["versions"].append(entry)
        self._save()
        return version

    def list_versions(self):
        return sorted(self._manifest["versions"], key=lambda v: v["version"])

    def get_version(self, version):
        for entry in self._manifest["versions"]:
            if entry["version"] == version:
                return entry
        raise KeyError(f"No such version: {version}")

    def promote(self, version, stage=STAGE_PRODUCTION):
        target = self.get_version(version)
        if stage == STAGE_PRODUCTION:
            for entry in self._manifest["versions"]:
                if entry["stage"] == STAGE_PRODUCTION and entry["version"] != version:
                    entry["stage"] = STAGE_ARCHIVED
        target["stage"] = stage
        self._save()

    def get_production(self):
        for entry in self._manifest["versions"]:
            if entry["stage"] == STAGE_PRODUCTION:
                return entry
        return None

    def rollback_to(self, version):
        """Promote a previously-registered version back to production,
        archiving whatever is currently in production. This is exactly
        `promote(version, PRODUCTION)`, exposed under its own name
        because the operational intent (undo a bad promotion) is
        different from a normal forward promotion, and tests/callers
        should be able to say what they mean."""
        self.promote(version, STAGE_PRODUCTION)

    def load_model(self, version):
        entry = self.get_version(version)
        model = TinyEdgeCNN()
        model.load_state_dict(torch.load(entry["model_path"], weights_only=True))
        model.eval()
        return model
