"""Тесты для hexbuffon — обобщённая формула Бюффона."""
import unittest
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexbuffon.hexbuffon import BuffonParquet, buffon_general, _PHI, _E, _PI


class TestBuffonGeneral(unittest.TestCase):
    def test_square_formula(self):
        """Квадрат a×a: W = 4L/(πa). При a=L=1: W=4/π."""
        W = buffon_general(needle=1.0, perimeter=4.0, area=1.0)
        self.assertAlmostEqual(W, 4 / _PI, places=10)

    def test_invalid_area(self):
        with self.assertRaises(ValueError):
            buffon_general(1.0, 4.0, 0.0)


class TestBuffonParquet(unittest.TestCase):
    def setUp(self):
        self.bp = BuffonParquet()

    def test_rectangular_formula(self):
        """a=1, b=1 → квадрат → W = 4/π."""
        W = self.bp.rectangular(1.0, 1.0, 1.0)
        self.assertAlmostEqual(W, 4 / _PI, places=10)

    def test_rectangular_custom(self):
        """a=2, b=3: W = L·2(a+b)/(π·ab) = (2L/π)·(1/a+1/b)."""
        W = self.bp.rectangular(2.0, 3.0, 1.0)
        expected = (2.0 / _PI) * (1.0 / 2 + 1.0 / 3)
        self.assertAlmostEqual(W, expected, places=10)

    def test_square_equals_rectangular(self):
        """bp.square(a) == bp.rectangular(a, a)."""
        for a in [0.5, 1.0, 2.0, 3.14]:
            self.assertAlmostEqual(self.bp.square(a, 1.0),
                                   self.bp.rectangular(a, a, 1.0))

    def test_hexagonal_formula(self):
        """Шестиугольник r=1: W = L·6r/(π·3√3/2·r²) = 4L/(π√3r)."""
        W = self.bp.hexagonal(1.0, 1.0)
        expected = 4.0 / (_PI * math.sqrt(3))   # = 4/(π√3) ≈ 0.735
        self.assertAlmostEqual(W, expected, places=10)

    def test_golden_rectangle_formula(self):
        """W = φe/π для любого a."""
        for a in [0.5, 1.0, 2.0, 5.0]:
            W = self.bp.golden_rectangle(a)
            expected = _PHI * _E / _PI
            self.assertAlmostEqual(W, expected, places=10)

    def test_golden_verify(self):
        res = self.bp.golden_rectangle_verify()
        self.assertTrue(res["ok"])
        self.assertAlmostEqual(res["exact"], _PHI * _E / _PI, places=12)

    def test_golden_value(self):
        """φe/π ≈ 1.4008..."""
        val = _PHI * _E / _PI
        self.assertGreater(val, 1.40)
        self.assertLess(val, 1.41)

    def test_find_needle_square(self):
        """Обратная задача: L = W·π·F/U = W·π·a²/(4a) = W·π·a/4."""
        L = self.bp.find_needle_length(1.0, tile="square", a=1.0)
        expected = _PI / 4
        self.assertAlmostEqual(L, expected, places=10)

    def test_simulate_basic(self):
        """Симуляция не падает и даёт разумный результат."""
        result = self.bp.simulate("square", needle=0.5, n=10_000, seed=42, a=1.0)
        self.assertIn("estimated_W", result)
        self.assertIn("exact_W", result)
        self.assertIn("error_pct", result)
        # Ошибка < 10% при n=10000
        self.assertLess(result["error_pct"], 10.0)


if __name__ == "__main__":
    unittest.main()
