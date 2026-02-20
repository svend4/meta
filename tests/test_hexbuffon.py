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


    def test_triangular_formula(self):
        """Правильный треугольник side=2: W = needle·3·side / (π·√3/4·side²) = 12/(π·√3·side)."""
        W = self.bp.triangular(2.0, 1.0)
        expected = 12.0 / (_PI * math.sqrt(3) * 2.0)
        self.assertAlmostEqual(W, expected, places=10)

    def test_triangular_positive(self):
        for side in [0.5, 1.0, 2.0, 5.0]:
            W = self.bp.triangular(side, 1.0)
            self.assertGreater(W, 0.0)

    def test_find_needle_rectangle(self):
        """L = W·π·F/U = W·π·ab / (2(a+b))."""
        L = self.bp.find_needle_length(1.0, tile="rectangle", a=2.0, b=3.0)
        expected = 1.0 * _PI * 6.0 / 10.0
        self.assertAlmostEqual(L, expected, places=10)

    def test_find_needle_hexagonal(self):
        """L = W·π·(3√3/2·r²) / (6r) = W·π·√3·r/4."""
        L = self.bp.find_needle_length(1.0, tile="hexagonal", r=1.0)
        expected = _PI * math.sqrt(3) / 4.0
        self.assertAlmostEqual(L, expected, places=10)

    def test_find_needle_invalid_tile(self):
        with self.assertRaises(ValueError):
            self.bp.find_needle_length(1.0, tile="unknown_tile")

    def test_hexagonal_q6_equals_hexagonal(self):
        """hexagonal_q6(r, needle) == hexagonal(r, needle)."""
        for r in [0.5, 1.0, 2.0]:
            self.assertAlmostEqual(
                self.bp.hexagonal_q6(r, 1.0),
                self.bp.hexagonal(r, 1.0),
                places=12
            )

    def test_plot_formula_relation_contains_phi_e_pi(self):
        s = self.bp.plot_formula_relation()
        self.assertIn('φ', s)
        self.assertIn('e', s)
        self.assertIn('π', s)

    def test_simulate_rectangle(self):
        res = self.bp.simulate("rectangle", needle=0.5, n=10_000, seed=7, a=1.0, b=2.0)
        self.assertIn("estimated_W", res)
        self.assertEqual(res["tile"], "rectangle")
        self.assertLess(res["error_pct"], 15.0)

    def test_simulate_golden(self):
        res = self.bp.simulate("golden", needle=0.5, n=10_000, seed=99, a=1.0)
        self.assertIn("exact_W", res)
        self.assertEqual(res["tile"], "golden")

    def test_simulate_invalid_tile(self):
        with self.assertRaises(ValueError):
            self.bp.simulate("unknown", needle=0.5, n=100, seed=1)

    def test_general_formula_method_equals_function(self):
        """general_formula() метода идентичен функции buffon_general()."""
        from projects.hexbuffon.hexbuffon import buffon_general
        W_method = self.bp.general_formula(1.0, 4.0, 1.0)
        W_func = buffon_general(1.0, 4.0, 1.0)
        self.assertAlmostEqual(W_method, W_func, places=12)

    def test_simulate_result_keys(self):
        """simulate возвращает dict с ключами tile, n, estimated_W, exact_W, error_pct."""
        res = self.bp.simulate("square", needle=0.5, n=500, seed=5, a=1.0)
        for key in ("tile", "n", "estimated_W", "exact_W", "error_pct"):
            self.assertIn(key, res)
        self.assertEqual(res["tile"], "square")
        self.assertEqual(res["n"], 500)

    def test_simulate_rectangle_exact_w(self):
        """exact_W в симуляции прямоугольника совпадает с rectangular()."""
        res = self.bp.simulate("rectangle", needle=1.0, n=100, seed=1, a=2.0, b=3.0)
        expected_exact = self.bp.rectangular(2.0, 3.0, 1.0)
        self.assertAlmostEqual(res["exact_W"], expected_exact, places=12)

    def test_square_direct_value(self):
        """square(a=1, needle=1) = 4/π."""
        W = self.bp.square(1.0, 1.0)
        self.assertAlmostEqual(W, 4.0 / _PI, places=10)

    def test_golden_rectangle_verify_dict_keys(self):
        """golden_rectangle_verify() содержит 'formula', 'phi', 'e', 'pi'."""
        res = self.bp.golden_rectangle_verify()
        for key in ("formula", "phi", "e", "pi", "ok", "exact", "computed"):
            self.assertIn(key, res)
        self.assertAlmostEqual(res["phi"], _PHI, places=12)
        self.assertAlmostEqual(res["e"], _E, places=12)

    def test_find_needle_hexagonal_formula(self):
        """find_needle_length(W, tile='hexagonal', r=1) = W·π·√3/4."""
        L = self.bp.find_needle_length(1.0, tile="hexagonal", r=1.0)
        expected = 1.0 * _PI * math.sqrt(3) / 4.0
        self.assertAlmostEqual(L, expected, places=10)

    def test_simulate_golden_exact_w_is_phi_e_pi(self):
        """simulate('golden') → exact_W == φ·e/π."""
        res = self.bp.simulate("golden", needle=0.5, n=100, seed=1, a=1.0)
        self.assertAlmostEqual(res["exact_W"], _PHI * _E / _PI, places=12)

    def test_hexagonal_q6_r2(self):
        """hexagonal_q6(r=2.0) > 0."""
        W = self.bp.hexagonal_q6(r=2.0, needle=1.0)
        self.assertGreater(W, 0.0)


if __name__ == "__main__":
    unittest.main()
