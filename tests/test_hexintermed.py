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


    def test_recurrence_check_structure(self):
        """recurrence_check(5) возвращает список из 5 словарей с нужными ключами."""
        results = self.h.recurrence_check(5)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIn("k", r)
            self.assertIn("h(k)", r)
            self.assertIn("h(k+2)", r)
            self.assertIn("diff", r)

    def test_recurrence_check_diff_is_h_k2_minus_hk(self):
        """diff == h(k+2) - h(k) для каждого k."""
        for r in self.h.recurrence_check(10):
            self.assertEqual(r["diff"], r["h(k+2)"] - r["h(k)"])

    def test_symmetry_check_structure(self):
        """symmetry_check(6, k=4, m=2) возвращает словарь с нужными ключами."""
        res = self.h.symmetry_check(6, k=4, m=2)
        self.assertIn("n", res)
        self.assertIn("m", res)
        self.assertIn("h(n+m)+h(n-m)", res)
        self.assertIn("2·h(n)", res)

    def test_symmetry_check_invalid(self):
        """n <= m → ValueError."""
        with self.assertRaises(ValueError):
            self.h.symmetry_check(3, m=3)
        with self.assertRaises(ValueError):
            self.h.symmetry_check(2, m=5)

    def test_plot_returns_string_with_header(self):
        """plot() возвращает строку, содержащую 'H(k)'."""
        s = self.h.plot(n_max=10)
        self.assertIsInstance(s, str)
        self.assertIn("H(k)", s)

    def test_partial_sum_known_n3(self):
        """S(3) = h(1)+h(2)+h(3) = 3+5+14 = 22."""
        self.assertEqual(self.h.partial_sum(3), 22)

    def test_factorize_product_equals_term(self):
        """x * y == h(k) для k=1..10."""
        for k in range(1, 11):
            f = self.h.factorize(k)
            self.assertEqual(f["x"] * f["y"], self.h.term(k))

    def test_recurrence_check_hk_matches_term(self):
        """h(k) в словаре совпадает с term(k)."""
        for r in self.h.recurrence_check(8):
            self.assertEqual(r["h(k)"], self.h.term(r["k"]))

    def test_symmetry_check_diff_is_int(self):
        """diff — целое число."""
        res = self.h.symmetry_check(6, m=2)
        self.assertIsInstance(res["diff"], int)

    def test_h_by_search_k1_is_3(self):
        """_h_by_search(1) == 3 (первый элемент ряда H)."""
        self.assertEqual(_h_by_search(1), 3)

    def test_partial_sum_zero_returns_zero(self):
        """S(0) = 0 (пустая сумма)."""
        self.assertEqual(self.h.partial_sum(0), 0)

    def test_partial_sum_one_equals_term1(self):
        """S(1) = h(1) = 3."""
        self.assertEqual(self.h.partial_sum(1), self.h.term(1))

    def test_generate_empty(self):
        """generate(0) возвращает пустой список."""
        self.assertEqual(self.h.generate(0), [])

    def test_factorize_k1_values(self):
        """factorize(1): k=1, y=3, x=1, h=3."""
        f = self.h.factorize(1)
        self.assertEqual(f["k"], 1)
        self.assertEqual(f["y"], 3)
        self.assertEqual(f["x"], 1)
        self.assertEqual(f["h"], 3)

    def test_symmetry_check_diff_formula(self):
        """diff == h(n+m)+h(n-m) - 2*h(n)."""
        res = self.h.symmetry_check(6, m=2)
        expected_diff = res["h(n+m)+h(n-m)"] - res["2·h(n)"]
        self.assertEqual(res["diff"], expected_diff)


class TestIntermediateExtra(unittest.TestCase):
    def setUp(self):
        self.h = IntermediateSeries()

    def test_h_by_formula_k5(self):
        """_h_by_formula(5) == 33."""
        self.assertEqual(_h_by_formula(5), 33)

    def test_generate_first_three(self):
        """generate(3) возвращает [3, 5, 14]."""
        self.assertEqual(self.h.generate(3), [3, 5, 14])

    def test_polygonal_triangle_k3(self):
        """polygonal(3, 3) = T(3) = 6 (третье треугольное число)."""
        self.assertEqual(IntermediateSeries.polygonal(3, 3), 6)

    def test_partial_sum_n5(self):
        """partial_sum(5) == 73."""
        self.assertEqual(self.h.partial_sum(5), 73)

    def test_recurrence_check_all_ok(self):
        """recurrence_check(5): все записи имеют ok == True."""
        for entry in self.h.recurrence_check(5):
            self.assertTrue(entry["ok"], f"Рекуррентность нарушена: {entry}")


if __name__ == "__main__":
    unittest.main()
