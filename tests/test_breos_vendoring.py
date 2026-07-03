from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


from tools.breos_vendoring.catalog import (
    MODEL_SPECS,
    VENDORED_SOURCE_FILES,
    available_model_keys,
    runtime_catalog,
    validate_manifest,
)
from tools.breos_vendoring.generate_golden_fixtures import (
    DEFAULT_FIXTURE_PATH,
    generate_fixture,
)


DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "pandas", "matplotlib", "scipy")
)


class BreosVendoringCatalogTest(unittest.TestCase):
    def test_manifest_files_exist(self):
        self.assertEqual(validate_manifest(), [])

    def test_manifest_covers_all_model_files(self):
        model_paths = {str(spec.source_path) for spec in MODEL_SPECS.values()}
        self.assertTrue(model_paths.issubset(set(VENDORED_SOURCE_FILES)))

    def test_phase_filter_returns_stable_keys(self):
        self.assertEqual(
            available_model_keys("P1"),
            ("lfp_gr_250ah_prismatic", "nca_gr_panasonic_3ah"),
        )

    def test_golden_fixture_file_exists(self):
        self.assertTrue(Path(DEFAULT_FIXTURE_PATH).exists())

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "BLAST runtime dependencies unavailable")
    def test_runtime_catalog_imports_all_models(self):
        catalog = runtime_catalog()
        self.assertEqual(set(catalog), set(MODEL_SPECS))
        for model_key, metadata in catalog.items():
            with self.subTest(model_key=model_key):
                self.assertIn("q", metadata["output_keys"])
                self.assertIn("t_days", metadata["stressor_keys"])
                self.assertIsNotNone(metadata["capacity_ah"])


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "BLAST runtime dependencies unavailable")
class BreosVendoringAdapterTest(unittest.TestCase):
    def test_build_endpoint_day_spans_full_day(self):
        import numpy as np

        from tools.breos_vendoring.adapter import build_endpoint_day

        soc_samples = np.linspace(0.5, 0.6, 24)
        t_secs, soc, temperature_c = build_endpoint_day(
            step_seconds=3600,
            soc_samples=soc_samples,
            temperature_c_samples=25.0,
            start_soc=0.5,
            start_temperature_c=25.0,
        )

        self.assertEqual(len(t_secs), 25)
        self.assertEqual(t_secs[0], 0)
        self.assertEqual(t_secs[-1], 86400)
        self.assertEqual(len(soc), 25)
        self.assertEqual(len(temperature_c), 25)

    def test_p1_models_step_and_snapshot(self):
        import numpy as np

        from tools.breos_vendoring.adapter import BlastStepEngine

        t_secs = np.arange(25, dtype=float) * 3600
        soc = 0.5 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, 25))
        temperature_c = np.full(25, 25.0)

        for model_key in available_model_keys("P1"):
            with self.subTest(model_key=model_key):
                engine = BlastStepEngine(model_key)
                soh_day_1 = engine.step(t_secs, soc, temperature_c)
                self.assertTrue(np.isfinite(soh_day_1))
                self.assertLessEqual(soh_day_1, 1.0)
                self.assertGreater(soh_day_1, 0.0)

                snapshot = engine.state_snapshot()
                restored = BlastStepEngine.from_snapshot(model_key, snapshot)
                soh_day_2 = restored.step(t_secs, soc, temperature_c)

                self.assertTrue(np.isfinite(soh_day_2))
                self.assertLessEqual(soh_day_2, soh_day_1)
                self.assertAlmostEqual(restored.model.stressors["t_days"][-1], 2.0)

    def test_all_models_smoke_one_day(self):
        import numpy as np

        from tools.breos_vendoring.adapter import BlastStepEngine

        t_secs = np.arange(25, dtype=float) * 3600
        soc = 0.55 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, 25))
        temperature_c = np.full(25, 25.0)

        for model_key in available_model_keys():
            with self.subTest(model_key=model_key):
                engine = BlastStepEngine(model_key)
                soh = engine.step(t_secs, soc, temperature_c)
                self.assertTrue(np.isfinite(soh))
                self.assertGreater(soh, 0.0)
                self.assertTrue(np.all(np.isfinite(engine.model.outputs["q"])))
                self.assertTrue(np.all(np.diff(engine.model.stressors["t_days"]) >= 0))

    def test_all_models_snapshot_restore_continuity(self):
        import numpy as np

        from tools.breos_vendoring.adapter import BlastStepEngine

        t_secs = np.arange(25, dtype=float) * 3600
        soc = 0.55 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, 25))
        temperature_c = np.full(25, 25.0)

        for model_key in available_model_keys():
            with self.subTest(model_key=model_key):
                engine = BlastStepEngine(model_key)
                engine.step(t_secs, soc, temperature_c)

                restored = BlastStepEngine.from_snapshot(
                    model_key, engine.state_snapshot(serializable=True)
                )
                soh_restored = restored.step(t_secs, soc, temperature_c)
                soh_direct = engine.step(t_secs, soc, temperature_c)

                self.assertAlmostEqual(soh_restored, soh_direct, places=12)
                self.assertAlmostEqual(restored.model.stressors["t_days"][-1], 2.0)

    def test_all_models_match_golden_soh_fixture(self):
        import numpy as np

        fixture = json.loads(Path(DEFAULT_FIXTURE_PATH).read_text())
        regenerated = generate_fixture(days=fixture["profile"]["days"])

        self.assertEqual(fixture["schema"], regenerated["schema"])
        self.assertEqual(
            set(fixture["trajectories"]),
            set(regenerated["trajectories"]),
        )

        for model_key in available_model_keys():
            with self.subTest(model_key=model_key):
                expected = fixture["trajectories"][model_key]
                actual = regenerated["trajectories"][model_key]
                np.testing.assert_allclose(
                    actual["soh"], expected["soh"], rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    actual["t_days"], expected["t_days"], rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    actual["efc"], expected["efc"], rtol=0, atol=1e-12
                )


if __name__ == "__main__":
    unittest.main()
