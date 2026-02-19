"""Tests for hexellipse — Hidden ellipse parameters, catastrophe line."""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexellipse.hexellipse import EllipseAnalysis


class TestEllipseParameters(unittest.TestCase):
    def setUp(self):
        # Standard ellipse: a=5, b=3
        self.ea = EllipseAnalysis(5.0, 3.0)

    def test_auto_sort_a_ge_b(self):
        ea2 = EllipseAnalysis(3.0, 5.0)
        self.assertEqual(ea2.a, 5.0)
        self.assertEqual(ea2.b, 3.0)

    def test_c(self):
        # c = √(25-9) = √16 = 4
        self.assertAlmostEqual(self.ea.c(), 4.0)

    def test_eccentricity(self):
        # e = c/a = 4/5
        self.assertAlmostEqual(self.ea.eccentricity(), 0.8)

    def test_focal_parameter(self):
        # p₁ = b²/a = 9/5 = 1.8
        self.assertAlmostEqual(self.ea.focal_parameter(), 1.8)

    def test_radial_parameter(self):
        # p₂ = a²/b = 25/3
        self.assertAlmostEqual(self.ea.radial_parameter(), 25.0 / 3.0)

    def test_axial_parameter(self):
        # p = ab/c = 15/4
        self.assertAlmostEqual(self.ea.axial_parameter(), 3.75)

    def test_invalid_a_b(self):
        with self.assertRaises(ValueError):
            EllipseAnalysis(-1.0, 3.0)
        with self.assertRaises(ValueError):
            EllipseAnalysis(5.0, 0.0)


class TestFundamentalIdentity(unittest.TestCase):
    def test_identity_5_3(self):
        ea = EllipseAnalysis(5.0, 3.0)
        ident = ea.verify_identity()
        self.assertTrue(ident["ok"])

    def test_identity_ab_equals_cp(self):
        ea = EllipseAnalysis(5.0, 3.0)
        ident = ea.verify_identity()
        self.assertAlmostEqual(ident["a*b"], 15.0)
        self.assertAlmostEqual(ident["c*p"], 15.0)

    def test_identity_various(self):
        for a, b in [(4.0, 3.0), (10.0, 6.0), (7.0, 2.0)]:
            ea = EllipseAnalysis(a, b)
            ident = ea.verify_identity()
            self.assertTrue(ident["ok"], f"Identity failed for a={a}, b={b}")

    def test_circle_identity(self):
        ea = EllipseAnalysis(5.0, 5.0)
        ident = ea.verify_identity()
        # For circle, a==b → c=0 → ok by special case
        self.assertTrue(ident["ok"])


class TestCircleCase(unittest.TestCase):
    def test_circle_c_is_zero(self):
        ea = EllipseAnalysis(3.0, 3.0)
        self.assertAlmostEqual(ea.c(), 0.0)

    def test_circle_eccentricity_zero(self):
        ea = EllipseAnalysis(3.0, 3.0)
        self.assertAlmostEqual(ea.eccentricity(), 0.0)

    def test_circle_axial_parameter_inf(self):
        ea = EllipseAnalysis(3.0, 3.0)
        self.assertEqual(ea.axial_parameter(), float("inf"))


class TestEquidistant(unittest.TestCase):
    def setUp(self):
        self.ea = EllipseAnalysis(5.0, 3.0)

    def test_equidistant_count(self):
        pts = self.ea.equidistant(1.0, n_points=100)
        self.assertEqual(len(pts), 100)

    def test_equidistant_zero_offset(self):
        # q=0: equidistant = ellipse itself
        pts0 = self.ea.equidistant(0.0, n_points=50)
        # Check first point is near (a, 0)
        x0, y0 = pts0[0]
        self.assertAlmostEqual(x0, self.ea.a, places=5)
        self.assertAlmostEqual(y0, 0.0, places=5)

    def test_equidistant_outward(self):
        # q>0: points farther from center on average
        pts = self.ea.equidistant(2.0, n_points=50)
        dist_avg = sum(math.sqrt(x**2 + y**2) for x, y in pts) / len(pts)
        ellipse_pts = self.ea.equidistant(0.0, n_points=50)
        ellipse_avg = sum(math.sqrt(x**2 + y**2) for x, y in ellipse_pts) / len(ellipse_pts)
        self.assertGreater(dist_avg, ellipse_avg)


class TestCatastropheCurve(unittest.TestCase):
    def setUp(self):
        self.ea = EllipseAnalysis(5.0, 3.0)

    def test_catastrophe_curve_count(self):
        pts = self.ea.catastrophe_curve(100)
        self.assertEqual(len(pts), 100)

    def test_at_catastrophe_q_value(self):
        info = self.ea.at_catastrophe()
        self.assertAlmostEqual(info["q"], self.ea.focal_parameter())

    def test_at_catastrophe_type(self):
        info = self.ea.at_catastrophe()
        self.assertEqual(info["type"], "cusp")

    def test_at_catastrophe_circle_no_cusps(self):
        ea = EllipseAnalysis(3.0, 3.0)
        info = ea.at_catastrophe()
        self.assertEqual(info["type"], "circle_no_catastrophe")

    def test_at_catastrophe_inflection_points(self):
        info = self.ea.at_catastrophe()
        self.assertIn("inflection_points", info)
        # Two special points (t and -t)
        self.assertEqual(len(info["inflection_points"]), 2)


class TestInscribedCircle(unittest.TestCase):
    def setUp(self):
        self.ea = EllipseAnalysis(5.0, 3.0)

    def test_inscribed_circle_radius(self):
        # R = p₂ = a²/b = 25/3
        self.assertAlmostEqual(self.ea.inscribed_circle_radius(), 25.0 / 3.0)

    def test_inscribed_system(self):
        sol = self.ea.inscribed_system_solution()
        self.assertAlmostEqual(sol["R_outer"], 25.0 / 3.0)
        self.assertAlmostEqual(sol["r_inner"], 1.8)
        # R/r = p₂/p₁ = (a²/b)/(b²/a) = a³/b³ = (a/b)³ = (5/3)³ = 125/27
        self.assertAlmostEqual(sol["R/r"], (5.0 / 3.0) ** 3)
        # p₁·p₂ = (b²/a)·(a²/b) = ab = 15
        self.assertAlmostEqual(sol["product"], 15.0)


class TestASCIIPlot(unittest.TestCase):
    def test_plot_returns_string(self):
        ea = EllipseAnalysis(5.0, 3.0)
        plot = ea.plot_ascii(40)
        self.assertIsInstance(plot, str)
        self.assertIn("*", plot)

    def test_summary_string(self):
        ea = EllipseAnalysis(5.0, 3.0)
        s = ea.summary()
        self.assertIn("p₁", s)
        self.assertIn("p₂", s)
        self.assertIn("a·b = c·p", s)


if __name__ == "__main__":
    unittest.main()
