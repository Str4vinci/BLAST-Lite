"""Generate BLAST golden fixtures for BREOS vendoring parity tests."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from .catalog import available_model_keys, get_model_class


DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "breos_vendoring"
    / "blast_golden_soh_100d.json"
)


def _package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def reference_daily_profile() -> tuple[Any, Any, Any]:
    """Return a deterministic 24-hour endpoint profile for golden fixtures."""

    import numpy as np

    t_secs = np.arange(25, dtype=float) * 3600.0
    soc = 0.55 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, 25))
    temperature_c = np.full(t_secs.shape, 25.0)
    return t_secs, soc, temperature_c


def generate_fixture(days: int = 100) -> dict[str, Any]:
    """Generate all-model SoH trajectories from the local BLAST implementation."""

    import numpy as np

    if days <= 0:
        raise ValueError("days must be positive")

    t_secs, soc, temperature_c = reference_daily_profile()
    trajectories: dict[str, dict[str, list[float]]] = {}

    for model_key in available_model_keys():
        model = get_model_class(model_key)()
        soh: list[float] = []
        t_days: list[float] = []
        efc: list[float] = []

        for _ in range(days):
            model.update_battery_state(t_secs.copy(), soc.copy(), temperature_c.copy())
            soh.append(float(model.outputs["q"][-1]))
            t_days.append(float(model.stressors["t_days"][-1]))
            efc.append(float(model.stressors["efc"][-1]))

        trajectories[model_key] = {
            "soh": soh,
            "t_days": t_days,
            "efc": efc,
        }

    return {
        "schema": "blast-lite-breos-golden-soh-v1",
        "source": "local untransformed BLAST-Lite checkout",
        "metadata": {
            "blast_lite_version": _package_version("BLAST-Lite"),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "profile": {
            "days": days,
            "time_s": t_secs.tolist(),
            "soc": soc.tolist(),
            "temperature_c": temperature_c.tolist(),
        },
        "trajectories": trajectories,
    }


def write_fixture(path: Path = DEFAULT_FIXTURE_PATH, days: int = 100) -> Path:
    fixture = generate_fixture(days=days)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate BLAST golden SoH fixtures for BREOS vendoring."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help=f"Output JSON path. Default: {DEFAULT_FIXTURE_PATH}",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=100,
        help="Number of daily update chunks to simulate.",
    )
    args = parser.parse_args()

    path = write_fixture(path=args.output, days=args.days)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
