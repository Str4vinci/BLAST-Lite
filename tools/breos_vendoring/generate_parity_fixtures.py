"""Generate multi-condition parity fixtures for the BREOS vendored BLAST source.

Runs every BLAST model over a battery of operating conditions chosen to
activate parameter groups the single-point golden fixture does not: hot/cold
calendar terms, deep-DOD cycling, higher C-rates, and temperature-varying
trapezoidal rate integration. Also dumps each model's literal parameters
(``cap``, ``_params_life``, ``experimental_range``) so BREOS can assert the
ported values are identical, not just behaviorally equivalent at one point.

Run from the repository root with the project environment (numpy<2, original
``np.trapz`` code path):

    uv run python tools/breos_vendoring/generate_parity_fixtures.py <output.json>

The BREOS test suite replays the same profiles through the vendored,
transformed source (numpy>=2, ``np.trapezoid``) and compares at 1e-12.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.breos_vendoring.catalog import MODEL_SPECS, get_model_class  # noqa: E402

DAYS = 60
HOURS = np.arange(25, dtype=float)
T_SECS = HOURS * 3600.0


def _deep_cycle_soc() -> np.ndarray:
    """Triangle 0.95 -> 0.05 -> 0.95 over one day (~0.9 DOD, ~0.9 EFC/day)."""
    half = np.linspace(0.95, 0.05, 13)
    return np.concatenate([half, np.linspace(0.05, 0.95, 13)[1:]])


CONDITIONS: dict[str, dict[str, np.ndarray]] = {
    "hot_storage": {
        "soc": np.full(25, 0.9),
        "temperature_c": np.full(25, 45.0),
    },
    "cold_storage": {
        "soc": np.full(25, 0.3),
        "temperature_c": np.full(25, 5.0),
    },
    "deep_cycle": {
        "soc": _deep_cycle_soc(),
        "temperature_c": np.full(25, 25.0),
    },
    "shallow_fast_cycle": {
        # Four shallow cycles per day: DOD 0.3 at ~0.3C peak.
        "soc": 0.75 + 0.15 * np.sin(HOURS * 4.0 * 2.0 * np.pi / 24.0),
        "temperature_c": np.full(25, 40.0),
    },
    "tvar_deep_cycle": {
        # Deep cycling with intraday temperature swing: exercises the
        # trapezoidal integration of time-varying degradation rates, the
        # exact code path where np.trapz -> np.trapezoid was rewritten.
        "soc": _deep_cycle_soc(),
        "temperature_c": 25.0 + 10.0 * np.sin(HOURS * 2.0 * np.pi / 24.0),
    },
}


def _params_payload(model) -> dict:
    return {
        "cap": float(model.cap),
        "params_life": {k: float(v) for k, v in model._params_life.items()},
        "experimental_range": json.loads(
            json.dumps(model.experimental_range, default=float)
        ),
    }


def main(output_path: str) -> None:
    fixture: dict = {
        "schema": "blast-lite-breos-parity-multicondition-v1",
        "metadata": {
            "source": "local untransformed BLAST-Lite checkout (blast/ at pin d789e00)",
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "days_per_condition": DAYS,
        },
        "conditions": {},
        "parameters": {},
    }

    for key in MODEL_SPECS:
        model = get_model_class(key)()
        fixture["parameters"][key] = _params_payload(model)

    for name, cond in CONDITIONS.items():
        soc = cond["soc"]
        temperature_c = cond["temperature_c"]
        entry: dict = {
            "profile": {
                "time_s": T_SECS.tolist(),
                "soc": soc.tolist(),
                "temperature_c": temperature_c.tolist(),
                "days": DAYS,
            },
            "trajectories": {},
            "final_outputs": {},
        }
        for key in MODEL_SPECS:
            model = get_model_class(key)()
            q, efc = [], []
            for _ in range(DAYS):
                model.update_battery_state(
                    T_SECS.copy(), soc.copy(), temperature_c.copy()
                )
                q.append(float(model.outputs["q"][-1]))
                efc.append(float(model.stressors["efc"][-1]))
            assert abs(model.stressors["t_days"][-1] - DAYS) < 1e-9
            entry["trajectories"][key] = {"q": q, "efc": efc}
            entry["final_outputs"][key] = {
                out_key: float(series[-1]) for out_key, series in model.outputs.items()
            }
        fixture["conditions"][name] = entry

    Path(output_path).write_text(json.dumps(fixture) + "\n")
    n_models = len(MODEL_SPECS)
    print(f"Wrote {output_path}: {len(CONDITIONS)} conditions x {n_models} models x {DAYS} days")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blast_parity_multicondition.json")
