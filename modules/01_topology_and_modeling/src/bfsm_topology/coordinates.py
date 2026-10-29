"""The common raw-scan ↔ posed registration transform (row-vector convention)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RegistrationTransform:
    center: np.ndarray
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        for name, shape in (("center", (3,)), ("rotation", (3, 3)), ("translation", (3,))):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid registration {name}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Registration scale must be positive")
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("Registration rotation must be orthogonal")
        if np.linalg.det(self.rotation) <= 0:
            raise ValueError("Registration rotation must preserve orientation")

    @staticmethod
    def _vertices(vertices):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("Expected finite N by 3 vertices")
        return vertices

    def to_canonical(self, vertices):
        vertices = self._vertices(vertices)
        return ((vertices - self.center) * self.scale - self.translation) @ np.linalg.inv(self.rotation)

    def to_world(self, vertices):
        vertices = self._vertices(vertices)
        return (vertices @ self.rotation + self.translation) / self.scale + self.center

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, schema_version=np.asarray(1), center=self.center, scale=np.asarray(self.scale),
                 rotation=self.rotation, translation=self.translation, **metadata)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("Unsupported registration coordinate transform")
            return cls(data["center"], float(data["scale"]), data["rotation"], data["translation"])
