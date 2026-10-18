"""Version-proof linear heads.

Heads are stored as plain .npz (numpy arrays + a JSON header) instead of
pickled sklearn objects, so the bundle does not depend on the exact
scikit-learn version and loads in milliseconds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearHead:
    name: str
    feature_set: str  # "text" | "meta" | "internal:last:L12" | "internal:mean:L12" | "internal:logits"
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    # Optional projection applied after standardisation (e.g. PCA), shape (d, k)
    proj: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Z = (X - self.mean) / self.scale
        if self.proj is not None:
            Z = Z @ self.proj
        return Z @ self.coef + self.intercept

    def proba(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(self.decision(X), -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: str) -> None:
        header = {"name": self.name, "feature_set": self.feature_set,
                  "intercept": float(self.intercept), "meta": self.meta}
        arrays = {"mean": self.mean, "scale": self.scale, "coef": self.coef,
                  "header": np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)}
        if self.proj is not None:
            arrays["proj"] = self.proj
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "LinearHead":
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(bytes(z["header"]).decode())
            proj = z["proj"].astype(np.float64) if "proj" in z.files else None
            return cls(
                name=header["name"], feature_set=header["feature_set"],
                mean=z["mean"].astype(np.float64), scale=z["scale"].astype(np.float64),
                coef=z["coef"].astype(np.float64), intercept=float(header["intercept"]),
                proj=proj, meta=header.get("meta", {}),
            )
