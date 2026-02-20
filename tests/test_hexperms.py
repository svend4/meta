"""Тесты для hexperms — перестановки (алгоритм Германа)."""
import unittest
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexperms.hexperms import PermutationEngine, _factoradic, _factoradic_to_int


class TestFactoradic(unittest.TestCase):
    def test_zero(self):
        """0 в факторадической = [0, 0, 0, ...]."""
        digits = _factoradic(0, 4)
        self.assertEqual(sum(digits), 0)

    def test_known(self):
        """факторадическое(23, 4) соответствует последней перестановке 4! - 1."""
        # 3! - 1 = 5 → [2, 1, 0] (факт. системе длины 3)
        digits = _factoradic(5, 3)
        # Правильный ответ: 5 = 2·2! + 1·1! + 0·0! = [2, 1, 0]
        self.assertEqual(digits, [2, 1, 0])


class TestPermutationEngine(unittest.TestCase):
    def setUp(self):
        self.pe4 = PermutationEngine(4)
        self.pe5 = PermutationEngine(5)

    def test_invalid_n(self):
        with self.assertRaises(ValueError):
            PermutationEngine(0)

    def test_generate_count(self):
        """Число перестановок = n!"""
        count = sum(1 for _ in self.pe4.generate_all())
        self.assertEqual(count, math.factorial(4))

    def test_generate_sorted(self):
        """Первая — минимальная, последняя — максимальная."""
        perms = list(self.pe4.generate_all())
        self.assertEqual(perms[0], [1, 2, 3, 4])
        self.assertEqual(perms[-1], [4, 3, 2, 1])

    def test_generate_lexicographic(self):
        """Порядок строго лексикографический."""
        perms = list(self.pe4.generate_all())
        for i in range(len(perms) - 1):
            self.assertLess(perms[i], perms[i + 1])

    def test_unrank_0(self):
        """unrank(0) = [1,2,...,n]."""
        self.assertEqual(self.pe4.unrank(0), [1, 2, 3, 4])

    def test_unrank_last(self):
        """unrank(n!-1) = [n,...,1]."""
        n = 4
        self.assertEqual(self.pe4.unrank(math.factorial(n) - 1), [4, 3, 2, 1])

    def test_unrank_rank_roundtrip(self):
        """rank(unrank(k)) = k для всех k."""
        for k in range(math.factorial(4)):
            perm = self.pe4.unrank(k)
            self.assertEqual(self.pe4.rank(perm), k)

    def test_rank_known(self):
        """Проверить несколько конкретных значений ранга."""
        self.assertEqual(self.pe4.rank([1, 2, 3, 4]), 0)
        self.assertEqual(self.pe4.rank([4, 3, 2, 1]), math.factorial(4) - 1)

    def test_next_perm(self):
        """next_perm([1,2,3,4]) = [1,2,4,3]."""
        nxt = self.pe4.next_perm([1, 2, 3, 4])
        self.assertEqual(nxt, [1, 2, 4, 3])

    def test_next_perm_last(self):
        """next_perm от последней = None."""
        self.assertIsNone(self.pe4.next_perm([4, 3, 2, 1]))

    def test_prev_perm(self):
        """prev_perm([1,2,4,3]) = [1,2,3,4]."""
        prv = self.pe4.prev_perm([1, 2, 4, 3])
        self.assertEqual(prv, [1, 2, 3, 4])

    def test_prev_perm_first(self):
        """prev_perm от первой = None."""
        self.assertIsNone(self.pe4.prev_perm([1, 2, 3, 4]))

    def test_derangements_count(self):
        """D(n) = n! * Σ(-1)^k/k!"""
        # D(4) = 9
        ders = self.pe4.derangements()
        expected = self.pe4.derangement_count()
        self.assertEqual(len(ders), expected)
        self.assertEqual(len(ders), 9)

    def test_derangements_no_fixed_points(self):
        """Все деранжементы не имеют неподвижных точек."""
        for d in self.pe4.derangements():
            for i, v in enumerate(d):
                self.assertNotEqual(v, i + 1)

    def test_with_fixed_point_count(self):
        """Перестановки с фиксированной точкой 1 = (n-1)!"""
        fixed = self.pe4.with_fixed_point(1)
        self.assertEqual(len(fixed), math.factorial(3))  # 3! = 6

    def test_with_fixed_point_correct(self):
        """Все возвращённые перестановки фиксируют нужную точку."""
        for p in self.pe4.with_fixed_point(2):
            self.assertEqual(p[1], 2)

    def test_n5_2_is_4(self):
        """Для n=5: unrank(0) = [1,2,3,4,5], ранг [2,1,3,4,5] > 0."""
        pe5 = PermutationEngine(5)
        self.assertEqual(pe5.unrank(0), [1, 2, 3, 4, 5])
        self.assertGreater(pe5.rank([2, 1, 3, 4, 5]), 0)

    def test_unrank_invalid(self):
        with self.assertRaises(ValueError):
            self.pe4.unrank(math.factorial(4))

    def test_rank_invalid(self):
        with self.assertRaises(ValueError):
            self.pe4.rank([1, 2, 3, 5])  # не перестановка {1..4}


class TestFactoradicToInt(unittest.TestCase):
    def test_roundtrip_zero(self):
        """_factoradic_to_int(_factoradic(0, n)) == 0."""
        self.assertEqual(_factoradic_to_int(_factoradic(0, 4)), 0)

    def test_known(self):
        """[2,1,0] → 5 = 2·2! + 1·1! + 0·0!"""
        self.assertEqual(_factoradic_to_int([2, 1, 0]), 5)

    def test_roundtrip_arbitrary(self):
        """_factoradic_to_int(_factoradic(k, n)) == k для любого k."""
        for k in range(24):   # 4! = 24
            self.assertEqual(_factoradic_to_int(_factoradic(k, 4)), k)


class TestAutQ6(unittest.TestCase):
    def test_count(self):
        """|Aut(Q6)| = 6! = 720."""
        pe6 = PermutationEngine(6)
        auts = pe6.generate_aut_q6()
        self.assertEqual(len(auts), math.factorial(6))

    def test_wrong_n_raises(self):
        """generate_aut_q6 только для n=6."""
        pe4 = PermutationEngine(4)
        with self.assertRaises(ValueError):
            pe4.generate_aut_q6()

    def test_each_aut_is_permutation(self):
        """Каждый элемент — перестановка {1..6}."""
        pe6 = PermutationEngine(6)
        for p in pe6.generate_aut_q6()[:5]:  # первые 5 достаточно
            self.assertEqual(sorted(p), [1, 2, 3, 4, 5, 6])


class TestBenchmark(unittest.TestCase):
    def test_returns_string(self):
        pe3 = PermutationEngine(3)
        s = pe3.benchmark(3)
        self.assertIsInstance(s, str)

    def test_contains_counts(self):
        """Строка содержит число перестановок."""
        pe3 = PermutationEngine(3)
        s = pe3.benchmark(3)
        self.assertIn('6', s)   # 3! = 6

    def test_default_n(self):
        """benchmark() без аргумента использует self.n."""
        pe3 = PermutationEngine(3)
        s = pe3.benchmark()
        self.assertIsInstance(s, str)


if __name__ == "__main__":
    unittest.main()
