"""
Drift Estimator
---------------
Monitors embedding shift using L2 or cosine distance.
Supports EWMA smoothing and exposes state for dashboards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

class DriftMode(str, Enum):
    L2 = "l2"
    COSINE = "cosine"

@dataclass
class DriftEstimator:
    threshold: float = 0.2
    mode: DriftMode = DriftMode.L2
    ewma_alpha: float = 0.3  # smoothing for stability

    history: List[np.ndarray] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    ewma: Optional[float] = None

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        if self.mode == DriftMode.L2:
            return float(np.linalg.norm(a - b))
        # cosine distance = 1 - cosine similarity
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(1.0 - float(np.dot(a, b)) / denom)

    def update(self, embedding) -> Tuple[bool, float, float]:
        """
        Args:
            embedding: vector representation of current response
        Returns:
            (drift_alert, distance, ewma)
        """
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim != 1:
            emb = emb.ravel()

        if not self.history:
            self.history.append(emb)
            self.ewma = 0.0
            self.distances.append(0.0)
            return (False, 0.0, 0.0)

        last = self.history[-1]
        dist = self._distance(emb, last)
        self.history.append(emb)
        self.distances.append(dist)

        # EWMA update
        if self.ewma is None:
            self.ewma = dist
        else:
            self.ewma = self.ewma_alpha * dist + (1 - self.ewma_alpha) * self.ewma

        return (bool(self.ewma > self.threshold), float(dist), float(self.ewma))

    def state(self) -> dict:
        return {
            "count": len(self.history),
            "last_distance": self.distances[-1] if self.distances else 0.0,
            "ewma": self.ewma or 0.0,
            "mode": self.mode.value,
            "threshold": self.threshold,
        }
