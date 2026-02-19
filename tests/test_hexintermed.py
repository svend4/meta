"""Тесты для hexintermed — промежуточный ряд H Германа."""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexintermed.hexintermed import IntermediateSeries, _h_by_formula, _h_by_search


class TestHFormula(unittest.TestCase):
    """Проверка замкнутой формулы через прямой поиск."""

    KNOWN = [3, 5, 14, 18, 33, 39, 60, 68, 95, 105]

    def test_formula_vs_search(self):
        """Замкнутая формула совпадает с прямым перебором для k=1..10."""
        for k in range(1, 11):
            self.assertEqual(_h_by_formula(k), _h_by_search(k),
                             f"Несоответствие при k={k}")

    def test_known_values(self):
        for k, expected in enumerate(self.KNOWN, start=1):
            self.assertEqual(_h_by_formula(k), expected,
                             f"h({k}) должно быть {expected}")

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            _h_by_formula(0)


class TestIntermediateSeries(unittest.TestCase):
    def setUp(self):
        self.h = IntermediateSeries()

    def test_term(self):
        known = [3, 5, 14, 18, 33, 39, 60, 68, 95, 105]
        for k, expected in enumerate(known, start=1):
            self.assertEqual(self.h.term(k), expected)

    def test_generate_count(self):
        terms = self.h.generate(20)
        self.assertEqual(len(terms), 20)

    def test_generate_first_10(self):
        expected = [3, 5, 14, 18, 33, 39, 60, 68, 95, 105]
        self.assertEqual(self.h.generate(10), expected)

    def test_partial_sum_formula_vs_direct(self):
        """Частичная сумма по формуле == прямая сумма."""
        for n in range(1, 15):
            formula_sum = self.h.partial_sum(n)
            direct_sum = sum(self.h.generate(n))
            self.assertEqual(formula_sum, direct_sum,
                             f"S({n}): формула={formula_sum}, прямая={direct_sum}")

    def test_factorize(self):
        """h(k) = x * y для каждого k."""
        for k in range(1, 15):
            f = self.h.factorize(k)
            self.assertTrue(f["check"], f"h({k}) ≠ x·y: {f}")

    def test_factorize_y_is_2k1(self):
        """y(k) = 2k+1 всегда."""
        for k in range(1, 10):
            f = self.h.factorize(k)
            self.assertEqual(f["y"], 2 * k + 1)

    def test_polygonal_square(self):
        """k=4 → n² (квадратные числа)."""
        for n in range(1, 8):
            self.assertEqual(self.h.polygonal(n, 4), n * n)

    def test_polygonal_triangle(self):
        """k=3 → треугольные числа n(n+1)/2."""
        for n in range(1, 8):
            self.assertEqual(self.h.polygonal(n, 3), n * (n + 1) // 2)

    def test_polygonal_invalid(self):
        with self.assertRaises(ValueError):
            self.h.polygonal(5, 2)

    def test_h_is_multiple_of_2k1(self):
        """h(k) кратно (2k+1)."""
        for k in range(1, 20):
            self.assertEqual(self.h.term(k) % (2 * k + 1), 0)

    def test_h_in_interval(self):
        """h(k) лежит в открытом интервале (k², (k+1)²)."""
        for k in range(1, 20):
            hk = self.h.term(k)
            self.assertGreater(hk, k * k)
            self.assertLess(hk, (k + 1) * (k + 1))


if __name__ == "__main__":
    unittest.main()
