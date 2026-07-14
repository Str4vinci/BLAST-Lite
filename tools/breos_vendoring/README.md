# BREOS Vendoring Prep

This directory contains local preparation helpers for the proposed BREOS BLAST
degradation engine. It is not part of the public BLAST-Lite package.

Use this checkout to validate the integration shape, but vendor final source
from a clean upstream BLAST-Lite commit and record that commit in BREOS
attribution docs.

Current artifacts:

- `catalog.py`: stable BREOS-facing model keys, class mapping, vendored file
  manifest, NumPy 2 rewrites, and heavy imports to trim.
- `adapter.py`: a small prototype of the proposed incremental `BlastEngine`
  contract, including daily endpoint grid construction and cross-year snapshots.
- `generate_golden_fixtures.py`: writes all-model 100-day SoH trajectories from
  this untransformed checkout for later BREOS parity tests.
- `generate_parity_fixtures.py`: writes every model's literal parameters
  (`cap`, `_params_life`, `experimental_range`) plus multi-condition golden
  trajectories (hot/cold storage, deep and shallow-fast cycling,
  intraday-varying temperature) so BREOS can assert parameter equality and
  transform neutrality at 1e-12 across operating points the 100-day fixture
  does not activate.
- `tests/test_breos_vendoring.py`: smoke checks for the manifest, catalog, daily
  endpoint grid, model stepping, snapshot continuity, and golden fixture parity.

Phase 0 BREOS vendoring transforms to apply to clean upstream source:

- Copy the base class, rainflow helper, all 14 model files, `LICENSE`, and
  `NOTICE`.
- Extract only `rescale_soc` from `blast/utils/functions.py`.
- Rewrite `np.trapz` to `np.trapezoid` for NumPy 2.
- Remove unused `pandas` and `matplotlib` imports from the vendored base class.
  Note: `simulate_battery_life`'s `Union[dict, pd.DataFrame]` annotation is
  evaluated at import time, so the annotation and the DataFrame input branch
  must be removed together with the import.
- Do not vendor NSRDB/demo helpers that import `h5pyd`, `geopy`, or
  `scipy.spatial`.

Generate the current golden fixture with:

```bash
uv run python -m tools.breos_vendoring.generate_golden_fixtures
```
