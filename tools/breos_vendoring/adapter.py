"""Prototype adapter matching the proposed BREOS BLAST engine contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .catalog import MODEL_SPECS, get_model_class


def _as_1d_float_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def normalize_step_inputs(
    t_secs: Any, soc: Any, temperature_c: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize one BLAST update chunk."""

    t_secs_array = _as_1d_float_array("t_secs", t_secs)
    soc_array = _as_1d_float_array("soc", soc)
    temperature_array = _as_1d_float_array("temperature_c", temperature_c)

    if temperature_array.size == 1 and t_secs_array.size > 1:
        temperature_array = np.full(t_secs_array.shape, temperature_array.item())

    if not (
        t_secs_array.size == soc_array.size == temperature_array.size
    ):
        raise ValueError("t_secs, soc, and temperature_c must have the same length")
    if t_secs_array.size < 2:
        raise ValueError("BLAST update chunks need at least two time points")
    if not np.all(np.diff(t_secs_array) > 0):
        raise ValueError("t_secs must be strictly increasing")
    if np.any((soc_array < -1e-12) | (soc_array > 1 + 1e-12)):
        raise ValueError("soc must be normalized to the range [0, 1]")

    return t_secs_array, soc_array, temperature_array


def build_endpoint_day(
    step_seconds: float,
    soc_samples: Any,
    temperature_c_samples: Any,
    start_soc: float,
    start_temperature_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the full-day endpoint grid BREOS needs for daily BLAST updates.

    BREOS buffers post-step samples. BLAST needs endpoints, so the prior anchor
    is prepended and the time axis spans ``N * step_seconds`` exactly.
    """

    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be a positive finite value")

    soc_post = _as_1d_float_array("soc_samples", soc_samples)
    temperature_post = _as_1d_float_array(
        "temperature_c_samples", temperature_c_samples
    )
    if temperature_post.size == 1 and soc_post.size > 1:
        temperature_post = np.full(soc_post.shape, temperature_post.item())
    if temperature_post.size != soc_post.size:
        raise ValueError("soc_samples and temperature_c_samples must have the same length")
    if soc_post.size == 0:
        raise ValueError("daily endpoint construction needs at least one sample")

    t_secs = np.arange(soc_post.size + 1, dtype=float) * float(step_seconds)
    soc = np.concatenate(([float(start_soc)], soc_post))
    temperature_c = np.concatenate(([float(start_temperature_c)], temperature_post))
    return normalize_step_inputs(t_secs, soc, temperature_c)


class BlastStepEngine:
    """Thin wrapper around a BLAST model instance for incremental daily stepping."""

    _ARRAY_GROUPS = ("states", "outputs", "stressors", "rates")

    def __init__(self, model_key: str, **model_kwargs: Any):
        if model_key not in MODEL_SPECS:
            available = ", ".join(MODEL_SPECS)
            raise KeyError(f"Unknown BLAST model key {model_key!r}. Available: {available}")
        self.model_key = model_key
        self._model_kwargs = dict(model_kwargs)
        self.model = self._new_model()

    def _new_model(self):
        model_cls = get_model_class(self.model_key)
        return model_cls(**self._model_kwargs)

    def step(self, t_secs: Any, soc: Any, temperature_c: Any) -> float:
        """Update the model by one chunk and return current SoH fraction."""

        t_secs_array, soc_array, temperature_array = normalize_step_inputs(
            t_secs, soc, temperature_c
        )
        self.model.update_battery_state(t_secs_array, soc_array, temperature_array)
        return self.soh()

    def soh(self) -> float:
        return float(self.model.outputs["q"][-1])

    def reset(self) -> None:
        self.model = self._new_model()

    def state_snapshot(self, serializable: bool = False) -> dict[str, Any]:
        """Copy the model state required for cross-year threading."""

        snapshot: dict[str, Any] = {
            "model_key": self.model_key,
            "model_kwargs": deepcopy(self._model_kwargs),
        }
        for group_name in self._ARRAY_GROUPS:
            group = getattr(self.model, group_name)
            snapshot[group_name] = {
                key: (value.tolist() if serializable else value.copy())
                for key, value in group.items()
            }
        return snapshot

    @classmethod
    def from_snapshot(cls, model_key: str, snapshot: dict[str, Any]) -> "BlastStepEngine":
        snapshot_key = snapshot.get("model_key")
        if snapshot_key is not None and snapshot_key != model_key:
            raise ValueError(
                f"Snapshot model_key {snapshot_key!r} does not match {model_key!r}"
            )

        engine = cls(model_key, **snapshot.get("model_kwargs", {}))
        for group_name in cls._ARRAY_GROUPS:
            group = snapshot[group_name]
            setattr(
                engine.model,
                group_name,
                {key: np.asarray(value, dtype=float) for key, value in group.items()},
            )
        return engine

