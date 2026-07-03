"""Catalog and vendoring manifest for BREOS BLAST integration prep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BlastModelSpec:
    """Stable metadata for a BLAST model BREOS can expose by key."""

    key: str
    module: str
    class_name: str
    form: str
    enable_phase: str
    notes: str = ""

    @property
    def source_path(self) -> Path:
        return Path("blast/models") / f"{self.module}.py"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


MODEL_SPECS: dict[str, BlastModelSpec] = {
    "lfp_gr_250ah_prismatic": BlastModelSpec(
        key="lfp_gr_250ah_prismatic",
        module="lfp_gr_250AhPrismatic_2019",
        class_name="Lfp_Gr_250AhPrismatic",
        form="2-bucket power",
        enable_phase="P1",
        notes="Stationary flagship.",
    ),
    "nca_gr_panasonic_3ah": BlastModelSpec(
        key="nca_gr_panasonic_3ah",
        module="nca_gr_Panasonic3Ah_2018",
        class_name="Nca_Gr_Panasonic3Ah_Battery",
        form="2-bucket power",
        enable_phase="P1",
        notes="Proof of concept.",
    ),
    "lmo_gr_nissanleaf_66ah_2nd": BlastModelSpec(
        key="lmo_gr_nissanleaf_66ah_2nd",
        module="lmo_gr_NissanLeaf66Ah_2ndLife_2020",
        class_name="Lmo_Gr_NissanLeaf66Ah_2ndLife_Battery",
        form="2-bucket power",
        enable_phase="P3a",
    ),
    "nmc811_grsi_lgm50_5ah": BlastModelSpec(
        key="nmc811_grsi_lgm50_5ah",
        module="nmc811_grSi_LGM50_5Ah_2021",
        class_name="Nmc811_GrSi_LGM50_5Ah_Battery",
        form="2-bucket power",
        enable_phase="P3a",
    ),
    "nmc811_grsi_lgmj1_4ah": BlastModelSpec(
        key="nmc811_grsi_lgmj1_4ah",
        module="nmc811_grSi_LGMJ1_4Ah_2020",
        class_name="Nmc811_GrSi_LGMJ1_4Ah_Battery",
        form="2-bucket power",
        enable_phase="P3a",
    ),
    "nmc_gr_50ah_b1": BlastModelSpec(
        key="nmc_gr_50ah_b1",
        module="nmc_gr_50Ah_B1_2020",
        class_name="NMC_Gr_50Ah_B1",
        form="2-bucket power",
        enable_phase="P3a",
    ),
    "nmc_gr_50ah_b2": BlastModelSpec(
        key="nmc_gr_50ah_b2",
        module="nmc_gr_50Ah_B2_2020",
        class_name="NMC_Gr_50Ah_B2",
        form="2-bucket power",
        enable_phase="P3a",
    ),
    "nmc_gr_75ah_a": BlastModelSpec(
        key="nmc_gr_75ah_a",
        module="nmc_gr_75Ah_A_2019",
        class_name="NMC_Gr_75Ah_A",
        form="2-bucket power",
        enable_phase="P3a",
    ),
    "nmc111_gr_sanyo_2ah": BlastModelSpec(
        key="nmc111_gr_sanyo_2ah",
        module="nmc111_gr_Sanyo2Ah_2014",
        class_name="Nmc111_Gr_Sanyo2Ah_Battery",
        form="3x power, capacity plus resistance",
        enable_phase="P3a",
    ),
    "nmc_lto_10ah": BlastModelSpec(
        key="nmc_lto_10ah",
        module="nmc_lto_10Ah_2020",
        class_name="Nmc_Lto_10Ah_Battery",
        form="3x power, includes early capacity gain",
        enable_phase="P3a",
    ),
    "lfp_gr_sonymurata_3ah": BlastModelSpec(
        key="lfp_gr_sonymurata_3ah",
        module="lfp_gr_SonyMurata3Ah_2018",
        class_name="Lfp_Gr_SonyMurata3Ah_Battery",
        form="sigmoid plus power_B, multi-mode",
        enable_phase="P3b",
        notes=(
            "Imports scipy.stats for a deterministic skew-normal term in its "
            "rate equations (no random sampling in the model itself)."
        ),
    ),
    "nca_grsi_sonymurata_2p5ah": BlastModelSpec(
        key="nca_grsi_sonymurata_2p5ah",
        module="nca_grsi_SonyMurata2p5Ah_2023",
        class_name="NCA_GrSi_SonyMurata2p5Ah_Battery",
        form="2x sigmoid",
        enable_phase="P3b",
    ),
    "nmc111_gr_kokam_75ah": BlastModelSpec(
        key="nmc111_gr_kokam_75ah",
        module="nmc111_gr_Kokam75Ah_2017",
        class_name="Nmc111_Gr_Kokam75Ah_Battery",
        form="power x4 plus sigmoid, LLI/LAM/R",
        enable_phase="P3b",
    ),
    "nmc622_gr_denso_50ah": BlastModelSpec(
        key="nmc622_gr_denso_50ah",
        module="nmc622_gr_DENSO50Ah_2021",
        class_name="Nmc622_Gr_DENSO50Ah_Battery",
        form="power plus exponential break-in",
        enable_phase="P3b",
    ),
}

VENDORED_SOURCE_FILES: tuple[str, ...] = (
    "LICENSE",
    "NOTICE",
    "blast/models/degradation_model.py",
    "blast/utils/rainflow.py",
    *(str(spec.source_path) for spec in MODEL_SPECS.values()),
)

TRIMMED_HELPERS: dict[str, tuple[str, ...]] = {
    "blast/utils/functions.py": ("rescale_soc",),
}

NUMPY2_REQUIRED_REWRITES: dict[str, str] = {
    "np.trapz": "np.trapezoid",
}

HEAVY_IMPORTS_TO_REMOVE: dict[str, tuple[str, ...]] = {
    "blast/models/degradation_model.py": ("pandas", "matplotlib.pyplot"),
    "blast/utils/functions.py": ("h5pyd", "pandas", "scipy.spatial", "geopy"),
}


def available_model_keys(enable_phase: str | None = None) -> tuple[str, ...]:
    """Return stable model keys, optionally filtered by an enable phase."""

    if enable_phase is None:
        return tuple(MODEL_SPECS)
    return tuple(
        key for key, spec in MODEL_SPECS.items() if spec.enable_phase == enable_phase
    )


def get_model_class(model_key: str) -> type:
    """Import and return the BLAST class for a stable model key."""

    try:
        spec = MODEL_SPECS[model_key]
    except KeyError as exc:
        available = ", ".join(MODEL_SPECS)
        raise KeyError(f"Unknown BLAST model key {model_key!r}. Available: {available}") from exc

    module = import_module(f"blast.models.{spec.module}")
    return getattr(module, spec.class_name)


def validate_manifest(root: Path | str = ".") -> list[str]:
    """Return missing files from the current vendoring manifest."""

    root_path = Path(root)
    required = set(VENDORED_SOURCE_FILES) | set(TRIMMED_HELPERS)
    return sorted(path for path in required if not (root_path / path).exists())


def runtime_catalog() -> dict[str, dict[str, Any]]:
    """Return catalog metadata enriched from live BLAST classes.

    This imports BLAST and therefore requires the package dependencies. Keep
    static tooling on ``MODEL_SPECS`` when dependency-free operation matters.
    """

    catalog: dict[str, dict[str, Any]] = {}
    for key, spec in MODEL_SPECS.items():
        model_cls = get_model_class(key)
        model = model_cls()
        catalog[key] = {
            **spec.as_dict(),
            "label": getattr(model, "_label", ""),
            "capacity_ah": getattr(model, "cap", None),
            "experimental_range": dict(getattr(model, "experimental_range", {})),
            "state_keys": tuple(model.states),
            "output_keys": tuple(model.outputs),
            "stressor_keys": tuple(model.stressors),
            "rate_keys": tuple(model.rates),
        }
    return catalog

