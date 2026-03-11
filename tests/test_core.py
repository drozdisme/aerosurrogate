"""
Unit tests for AeroSurrogate v2.0.
Covers: demo_model, feature engineer, physics_loss.
Compatible with pytest AND unittest.
"""

import math
import sys
import os
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.demo.demo_model import (
    predict_coefficients, generate_cp_distribution,
    _compute_cl, _compute_cd, _compute_cm,
    _prandtl_glauert, _stall_factor,
)
from src.features.engineer import FeatureEngineer


# ─── demo_model ───────────────────────────────────────────────────────────────

class TestDemoModel(unittest.TestCase):
    GEO = dict(thickness_ratio=0.12, camber=0.04, camber_position=0.4,
               leading_edge_radius=0.02, trailing_edge_angle=15.0,
               aspect_ratio=8.0, taper_ratio=0.5, sweep_angle=20.0)
    FLOW = dict(mach=0.3, reynolds=1e6, alpha=5.0, beta=0.0, altitude=0.0)

    def test_returns_all_keys(self):
        r = predict_coefficients(self.GEO, self.FLOW)
        self.assertEqual(set(r), {"Cl", "Cd", "Cm", "K"})

    def test_cl_positive_at_positive_alpha(self):
        r = predict_coefficients(self.GEO, self.FLOW)
        self.assertGreater(r["Cl"], 0)

    def test_cl_negative_at_negative_alpha(self):
        r = predict_coefficients(self.GEO, {**self.FLOW, "alpha": -5.0})
        self.assertLess(r["Cl"], 0)

    def test_cd_always_positive(self):
        for alpha in [-10, 0, 5, 14, 20]:
            with self.subTest(alpha=alpha):
                r = predict_coefficients(self.GEO, {**self.FLOW, "alpha": float(alpha)})
                self.assertGreater(r["Cd"], 0)

    def test_cd_increases_with_high_alpha(self):
        r0 = predict_coefficients(self.GEO, {**self.FLOW, "alpha": 0.0})
        r1 = predict_coefficients(self.GEO, {**self.FLOW, "alpha": 14.0})
        self.assertGreater(r1["Cd"], r0["Cd"])

    def test_cl_increases_with_camber(self):
        r1 = predict_coefficients({**self.GEO, "camber": 0.0},  self.FLOW)
        r2 = predict_coefficients({**self.GEO, "camber": 0.06}, self.FLOW)
        self.assertGreater(r2["Cl"], r1["Cl"])

    def test_k_equals_cl_over_cd(self):
        r = predict_coefficients(self.GEO, self.FLOW)
        expected = r["Cl"] / max(r["Cd"], 1e-6)
        self.assertAlmostEqual(r["K"], expected, delta=0.01)

    def test_compressibility_increases_cl(self):
        r_lo = predict_coefficients(self.GEO, {**self.FLOW, "mach": 0.1})
        r_hi = predict_coefficients(self.GEO, {**self.FLOW, "mach": 0.7})
        self.assertGreater(r_hi["Cl"], r_lo["Cl"])

    def test_prandtl_glauert_at_zero(self):
        self.assertAlmostEqual(_prandtl_glauert(0.0), 1.0, places=6)

    def test_prandtl_glauert_increases(self):
        self.assertLess(_prandtl_glauert(0.3), _prandtl_glauert(0.7))

    def test_stall_factor_pre_stall(self):
        self.assertGreater(_stall_factor(5.0), 0.9)

    def test_stall_factor_post_stall(self):
        self.assertLess(_stall_factor(25.0), 0.2)

    def test_cp_shape(self):
        r = generate_cp_distribution(self.GEO, self.FLOW, n_points=100)
        self.assertEqual(r["n_points"], 100)
        self.assertEqual(len(r["x"]), 100)
        self.assertEqual(len(r["Cp"]), 100)

    def test_cp_x_starts_at_zero(self):
        r = generate_cp_distribution(self.GEO, self.FLOW)
        self.assertAlmostEqual(r["x"][0], 0.0, delta=0.01)

    def test_cp_x_ends_at_one(self):
        r = generate_cp_distribution(self.GEO, self.FLOW)
        self.assertAlmostEqual(r["x"][-1], 1.0, delta=0.01)

    def test_cp_physical_range(self):
        r = generate_cp_distribution(self.GEO, self.FLOW)
        for v in r["Cp"]:
            self.assertTrue(-5.0 < v < 1.5, f"Cp={v} out of range")

    def test_cm_negative_for_positive_camber(self):
        r = predict_coefficients(self.GEO, {**self.FLOW, "alpha": 2.0})
        self.assertLess(r["Cm"], 0)

    def test_empty_geometry_defaults(self):
        r = predict_coefficients({}, {"mach": 0.2, "reynolds": 1e6, "alpha": 3.0})
        self.assertIn("Cl", r)

    def test_zero_alpha_near_zero_cl_symmetric(self):
        r = predict_coefficients({**self.GEO, "camber": 0.0}, {**self.FLOW, "alpha": 0.0})
        self.assertAlmostEqual(r["Cl"], 0.0, delta=0.05)


# ─── feature engineer ─────────────────────────────────────────────────────────

class TestFeatureEngineer(unittest.TestCase):
    def _df(self):
        return pd.DataFrame([{
            "mach": 0.3, "alpha": 5.0, "reynolds": 1e6,
            "thickness_ratio": 0.12, "camber": 0.04,
            "camber_position": 0.4, "aspect_ratio": 8.0,
            "taper_ratio": 0.5, "sweep_angle": 20.0,
        }])

    def test_returns_dataframe(self):
        out = FeatureEngineer().transform(self._df())
        self.assertIsInstance(out, pd.DataFrame)

    def test_mach_alpha_interaction(self):
        df = self._df()
        out = FeatureEngineer().transform(df)
        self.assertIn("mach_alpha_interaction", out.columns)
        self.assertAlmostEqual(out["mach_alpha_interaction"].iloc[0], 1.5, delta=1e-9)

    def test_reynolds_log(self):
        out = FeatureEngineer().transform(self._df())
        self.assertIn("reynolds_log", out.columns)
        self.assertGreater(out["reynolds_log"].iloc[0], 0)

    def test_alpha_squared(self):
        out = FeatureEngineer().transform(self._df())
        self.assertAlmostEqual(out["alpha_squared"].iloc[0], 25.0, delta=1e-9)

    def test_alpha_cubed(self):
        out = FeatureEngineer().transform(self._df())
        self.assertAlmostEqual(out["alpha_cubed"].iloc[0], 125.0, delta=1e-9)

    def test_sweep_mach_interaction(self):
        out = FeatureEngineer().transform(self._df())
        self.assertIn("sweep_mach_interaction", out.columns)

    def test_compressibility_factor_above_one(self):
        out = FeatureEngineer().transform(self._df())
        self.assertIn("compressibility_factor", out.columns)
        self.assertGreater(out["compressibility_factor"].iloc[0], 1.0)

    def test_compressibility_factor_zero_mach(self):
        df = self._df(); df["mach"] = 0.0
        out = FeatureEngineer().transform(df)
        self.assertAlmostEqual(out["compressibility_factor"].iloc[0], 1.0, delta=0.01)

    def test_reduced_frequency(self):
        out = FeatureEngineer().transform(self._df())
        self.assertIn("reduced_frequency", out.columns)

    def test_source_column_dropped(self):
        df = self._df(); df["_source"] = "test"
        out = FeatureEngineer().transform(df)
        self.assertNotIn("_source", out.columns)

    def test_aspect_ratio_taper_interaction(self):
        df = self._df()
        out = FeatureEngineer().transform(df)
        self.assertIn("aspect_ratio_taper_interaction", out.columns)
        self.assertAlmostEqual(out["aspect_ratio_taper_interaction"].iloc[0], 4.0, delta=1e-9)

    def test_get_feature_columns_excludes_targets(self):
        df = self._df()
        out = FeatureEngineer().transform(df)
        out["Cl"] = 0.5; out["Cd"] = 0.02
        feats = FeatureEngineer().get_feature_columns(out, ["Cl", "Cd"])
        self.assertNotIn("Cl", feats)
        self.assertNotIn("Cd", feats)


# ─── physics_loss (conditional on torch) ─────────────────────────────────────

try:
    import torch
    from src.models.physics_loss import PhysicsLoss, SimpleMSELoss

    class TestPhysicsLoss(unittest.TestCase):
        def _t(self, vals):
            return torch.tensor(vals, dtype=torch.float32)

        def test_perfect_prediction_zero_mse(self):
            loss = PhysicsLoss(cd_positivity_weight=0, cl_monotonicity_weight=0, consistency_weight=0)
            p = self._t([0.5, 0.6])
            self.assertAlmostEqual(loss(p, p).item(), 0.0, delta=1e-5)

        def test_cd_penalty_for_negative_cd(self):
            loss = PhysicsLoss(cd_positivity_weight=1.0, cl_monotonicity_weight=0, consistency_weight=0)
            p = self._t([0.5]); t = self._t([0.5])
            v_with = loss(p, t, cd_pred=self._t([-0.01])).item()
            v_without = loss(p, t).item()
            self.assertGreater(v_with, v_without)

        def test_cd_no_penalty_positive_cd(self):
            loss = PhysicsLoss(cd_positivity_weight=1.0, cl_monotonicity_weight=0, consistency_weight=0)
            p = self._t([0.5]); t = self._t([0.5])
            v_with = loss(p, t, cd_pred=self._t([0.02])).item()
            v_without = loss(p, t).item()
            self.assertAlmostEqual(v_with, v_without, delta=1e-5)

        def test_simple_mse_perfect(self):
            loss = SimpleMSELoss()
            p = self._t([1.0, 2.0])
            self.assertAlmostEqual(loss(p, p).item(), 0.0, delta=1e-6)

        def test_simple_mse_error(self):
            loss = SimpleMSELoss()
            p = self._t([0.0]); t = self._t([1.0])
            self.assertAlmostEqual(loss(p, t).item(), 1.0, delta=1e-5)

except ImportError:
    pass   # torch not installed — physics_loss tests skipped


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestDemoModel, TestFeatureEngineer]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    try:
        suite.addTests(loader.loadTestsFromTestCase(TestPhysicsLoss))
    except NameError:
        pass
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    coverage_est = min(100, round(passed / total * 100)) if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    print(f"Estimated coverage proxy: {coverage_est}%")
