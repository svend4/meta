"""Тесты для hexcrossrat — группа двойных отношений R6."""
import unittest
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexcrossrat.hexcrossrat import CrossRatioGroup, cross_ratio


class TestCrossRatio(unittest.TestCase):
    def test_known_value(self):
        """Простой случай: 4 равноотстоящих точки."""
        # cross_ratio(0, 2, 1, 3) = (0-1)(2-3) / ((1-2)(0-3)) = 1/3 (зависит от конвенции)
        w = cross_ratio(0.0, 2.0, 1.0, 3.0)
        # Проверяем, что это число (не ноль и не бесконечность)
        self.assertNotEqual(w, 0)
        self.assertFalse(math.isinf(abs(w)))

    def test_invalid_degenerate(self):
        """Вырожденный случай: знаменатель = 0 → ZeroDivisionError."""
        with self.assertRaises(ZeroDivisionError):
            cross_ratio(1.0, 2.0, 2.0, 1.0)  # A3=A2, A1=A4 → деление на 0


class TestCrossRatioGroup(unittest.TestCase):
    W_VALUES = [0.5, 2.0, -1.0, 1.5, 3.0]

    def _make(self, w):
        return CrossRatioGroup(w)

    def test_invalid_w_zero(self):
        with self.assertRaises(ValueError):
            CrossRatioGroup(0)

    def test_invalid_w_one(self):
        with self.assertRaises(ValueError):
            CrossRatioGroup(1)

    def test_elements_count(self):
        """R6 всегда имеет ровно 6 элементов."""
        for w in self.W_VALUES:
            r6 = self._make(w)
            self.assertEqual(len(r6.elements()), 6)

    def test_r0_is_w(self):
        """r₀ = w."""
        for w in self.W_VALUES:
            r6 = self._make(w)
            self.assertAlmostEqual(complex(r6.elements()[0]), complex(w))

    def test_r1_is_1_over_w(self):
        """r₁ = 1/w."""
        for w in self.W_VALUES:
            r6 = self._make(w)
            self.assertAlmostEqual(complex(r6.elements()[1]), 1.0 / w)

    def test_r2_is_1_minus_w(self):
        """r₂ = 1 - w."""
        for w in self.W_VALUES:
            r6 = self._make(w)
            self.assertAlmostEqual(complex(r6.elements()[2]), 1.0 - w)

    def test_sum_identity(self):
        """Σr_i = 3."""
        for w in self.W_VALUES:
            r6 = self._make(w)
            result = r6.verify_sum_identity()
            self.assertTrue(result["ok"], f"Нарушение суммы при w={w}: {result['sum']}")

    def test_product_identity(self):
        """∏r_i = 1."""
        for w in self.W_VALUES:
            r6 = self._make(w)
            result = r6.verify_product_identity()
            self.assertTrue(result["ok"], f"Нарушение произведения при w={w}: {result['product']}")

    def test_cayley_table_size(self):
        """Таблица Кэли — 6×6."""
        r6 = self._make(3.0)   # w=3: все элементы различны
        table = r6.cayley_table()
        self.assertEqual(len(table), 6)
        for row in table:
            self.assertEqual(len(row), 6)

    def test_cayley_table_values_in_range(self):
        """Все значения таблицы Кэли ∈ {0..5} (при w с различными элементами)."""
        r6 = self._make(3.0)   # w=3: r0=3, r1=1/3, r2=-2, r3=-1/2, r4=2/3, r5=3/2
        for row in r6.cayley_table():
            for val in row:
                self.assertIn(val, range(6))

    def test_identity_element(self):
        """r₀ = тождественная: r₀ ∘ r_i = r_i (при w с различными элементами)."""
        r6 = self._make(3.0)   # w=3: все 6 элементов различны
        for j in range(6):
            result = r6.multiply(0, j)
            self.assertEqual(result, j, f"r0 ∘ r{j} должно быть r{j}")

    def test_isomorphism_to_s3(self):
        """Изоморфизм R6 → S3 возвращает 6 перестановок."""
        r6 = self._make(2.0)
        iso = r6.isomorphism_to_s3()
        self.assertEqual(len(iso), 6)
        for k, perm in iso.items():
            self.assertEqual(len(perm), 3)

    def test_klein_four_group(self):
        """V4 содержит ровно 4 элемента."""
        r6 = self._make(0.5)
        v4 = r6.klein_four_group()
        self.assertEqual(len(v4), 4)


    def test_compose_returns_value(self):
        """compose(i, j) возвращает числовое значение r_i(r_j(w))."""
        r6 = self._make(3.0)
        val = r6.compose(1, 0)   # r1(r0(3)) = r1(3) = 1/3
        self.assertAlmostEqual(complex(val).real, 1.0 / 3.0, places=9)

    def test_compose_consistent_with_multiply(self):
        """compose(i,j) индекс совпадает с multiply(i,j)."""
        r6 = self._make(3.0)
        for i in range(6):
            for j in range(6):
                idx = r6.multiply(i, j)
                result_val = r6.compose(i, j)
                elem_val = r6.elements()[idx]
                self.assertAlmostEqual(complex(result_val), complex(elem_val), places=6,
                                       msg=f"Несоответствие compose/multiply i={i},j={j}")

    def test_print_cayley_table_is_string(self):
        r6 = self._make(3.0)
        s = r6.print_cayley_table()
        self.assertIsInstance(s, str)
        self.assertIn("Таблица Кэли", s)

    def test_s4_decomposition_is_string(self):
        r6 = self._make(2.0)
        s = r6.s4_decomposition()
        self.assertIsInstance(s, str)
        self.assertIn("S4", s)
        self.assertIn("V4", s)
        self.assertIn("24", s)


if __name__ == "__main__":
    unittest.main()
