"""Тесты для hexpowerxy — уравнение X^Y = Y^X."""
import unittest
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexpowerxy.hexpowerxy import PowerXY, _PHI, _E

_PI = math.pi


class TestXYFromT(unittest.TestCase):
    def test_t2_gives_2_4(self):
        """t=2: X=2, Y=4."""
        pxy = PowerXY()
        X, Y = pxy.xy_from_t(2.0)
        self.assertAlmostEqual(X, 2.0, places=10)
        self.assertAlmostEqual(Y, 4.0, places=10)

    def test_verify_2_4(self):
        """2^4 = 4^2 = 16."""
        pxy = PowerXY()
        self.assertTrue(pxy.verify(2.0, 4.0))

    def test_invalid_t(self):
        pxy = PowerXY()
        with self.assertRaises(ValueError):
            pxy.xy_from_t(0.5)
        with self.assertRaises(ValueError):
            pxy.xy_from_t(1.0)

    def test_xy_verify_for_curve(self):
        """Все точки кривой удовлетворяют X^Y = Y^X."""
        pxy = PowerXY()
        for t in [1.5, 2.0, 3.0, 4.0, 5.0]:
            X, Y = pxy.xy_from_t(t)
            self.assertTrue(pxy.verify(X, Y), f"Нарушение при t={t}")


class TestFindY(unittest.TestCase):
    def test_find_y_x2(self):
        """X=2 → Y=4."""
        pxy = PowerXY()
        Y = pxy.find_y(2.0)
        self.assertIsNotNone(Y)
        self.assertAlmostEqual(Y, 4.0, places=5)

    def test_exception_e(self):
        """X≈e — исключение, Y=None."""
        pxy = PowerXY()
        Y = pxy.find_y(_E)
        self.assertIsNone(Y)


class TestGoldenRoot(unittest.TestCase):
    def test_golden_root_is_phi(self):
        """Корень золотого уравнения = φ."""
        pxy = PowerXY()
        phi = pxy.find_golden_root()
        self.assertAlmostEqual(phi, _PHI, places=8)

    def test_golden_eq1_at_phi(self):
        """Золотое уравнение 1 обращается в 0 при x=φ."""
        pxy = PowerXY()
        err = pxy.golden_eq1(_PHI)
        self.assertLess(err, 1e-10)

    def test_golden_eq2_at_phi(self):
        """Золотое уравнение 2 обращается в 0 при x=φ."""
        pxy = PowerXY()
        err = pxy.golden_eq2(_PHI)
        self.assertLess(err, 1e-10)


class TestNotablePairs(unittest.TestCase):
    def test_contains_2_4(self):
        pxy = PowerXY()
        pairs = pxy.notable_pairs()
        self.assertIn((2.0, 4.0), pairs)

    def test_all_satisfy_equation(self):
        pxy = PowerXY()
        for x, y in pxy.notable_pairs():
            if x == y:
                continue  # тривиальное решение
            self.assertTrue(pxy.verify(x, y, tol=1e-5),
                            f"Нарушение X^Y=Y^X для ({x}, {y})")


class TestGenerateCurve(unittest.TestCase):
    def test_generates_nonempty(self):
        pxy = PowerXY()
        curve = pxy.generate_curve(steps=20)
        self.assertGreater(len(curve), 0)

    def test_all_points_verified(self):
        pxy = PowerXY()
        curve = pxy.generate_curve(t_min=1.01, t_max=5.0, steps=30)
        for X, Y in curve:
            self.assertTrue(pxy.verify(X, Y), f"Нарушение для ({X}, {Y})")


class TestVerify(unittest.TestCase):
    def setUp(self):
        self.pxy = PowerXY()

    def test_verify_trivial_xy_equal(self):
        """X = Y: X^X = X^X всегда истинно."""
        self.assertTrue(self.pxy.verify(3.0, 3.0))

    def test_verify_invalid_returns_false(self):
        """X=1, Y=2: 1^2 ≠ 2^1 → False."""
        self.assertFalse(self.pxy.verify(1.0, 2.0))

    def test_verify_known_pair(self):
        """2^4 = 4^2 = 16 → True."""
        self.assertTrue(self.pxy.verify(2.0, 4.0))


class TestExceptionConstant(unittest.TestCase):
    def test_is_exception_true(self):
        pxy = PowerXY()
        self.assertTrue(pxy.is_exception(_E))

    def test_is_exception_false(self):
        pxy = PowerXY()
        self.assertFalse(pxy.is_exception(2.0))
        self.assertFalse(pxy.is_exception(4.0))

    def test_exception_constant_is_e(self):
        pxy = PowerXY()
        self.assertAlmostEqual(pxy.exception_constant(), _E, places=12)


class TestPlotCurve(unittest.TestCase):
    def test_plot_curve_returns_string(self):
        pxy = PowerXY()
        s = pxy.plot_curve(steps=20)
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)

    def test_plot_curve_contains_axis_label(self):
        """plot_curve содержит ось Y и символ точки '*'."""
        pxy = PowerXY()
        s = pxy.plot_curve(steps=15)
        self.assertIn("Y", s)
        self.assertIn("*", s)


class TestXYFromTLargeT(unittest.TestCase):
    def test_large_t_gives_x_near_1(self):
        """При t → ∞: X = t/(t-1) → 1, Y = t^(1/(t-1)) → 1."""
        pxy = PowerXY()
        X, Y = pxy.xy_from_t(100.0)
        self.assertGreater(X, 1.0)
        self.assertGreater(Y, 1.0)

    def test_large_t_verified(self):
        """Точка при t=10 удовлетворяет X^Y = Y^X."""
        pxy = PowerXY()
        X, Y = pxy.xy_from_t(10.0)
        self.assertTrue(pxy.verify(X, Y, tol=1e-6))


class TestNotablePairsCount(unittest.TestCase):
    def test_notable_pairs_count_gte_2(self):
        """notable_pairs() содержит хотя бы 2 пары."""
        pxy = PowerXY()
        self.assertGreaterEqual(len(pxy.notable_pairs()), 2)


class TestGenerateCurveDefault(unittest.TestCase):
    def test_default_curve_positive_count(self):
        """generate_curve() с параметрами по умолчанию возвращает точки."""
        pxy = PowerXY()
        curve = pxy.generate_curve()
        self.assertGreater(len(curve), 0)

    def test_default_curve_x_greater_1(self):
        """X-координата каждой точки > 1."""
        pxy = PowerXY()
        for X, Y in pxy.generate_curve(steps=10):
            self.assertGreater(X, 1.0)


class TestFindYExtra(unittest.TestCase):
    def setUp(self):
        self.pxy = PowerXY()

    def test_find_y_non_trivial(self):
        """find_y(1.5) возвращает не None и Y > 1."""
        Y = self.pxy.find_y(1.5)
        self.assertIsNotNone(Y)
        self.assertGreater(Y, 1.0)

    def test_verify_symmetric(self):
        """verify(4, 2) == True (симметричное решение 2^4 = 4^2)."""
        self.assertTrue(self.pxy.verify(4.0, 2.0))


class TestGoldenEqExtra(unittest.TestCase):
    def test_golden_eq1_nonzero_at_2(self):
        """golden_eq1(2.0) ≠ 0 (2 не является φ)."""
        pxy = PowerXY()
        err = pxy.golden_eq1(2.0)
        self.assertGreater(abs(err), 1e-6)

    def test_golden_eq2_nonzero_at_2(self):
        """golden_eq2(2.0) ≠ 0 (2 не является φ)."""
        pxy = PowerXY()
        err = pxy.golden_eq2(2.0)
        self.assertGreater(abs(err), 1e-6)


class TestGenerateCurveSteps(unittest.TestCase):
    def test_exact_steps_count(self):
        """generate_curve(steps=15) возвращает ровно 15 точек."""
        pxy = PowerXY()
        curve = pxy.generate_curve(steps=15)
        self.assertEqual(len(curve), 15)


if __name__ == "__main__":
    unittest.main()
