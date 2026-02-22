"""Тесты карты Карно и алгоритма Куайна–МакКласки (karnaugh6)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.karnaugh6.minimize import (
    Implicant, quine_mccluskey, essential_implicants, minimize,
)


class TestImplicant(unittest.TestCase):
    def test_from_minterm(self):
        imp = Implicant.from_minterm(42)  # 101010
        self.assertEqual(imp.bits, '101010')
        self.assertEqual(imp.covered, frozenset({42}))

    def test_combine_valid(self):
        a = Implicant.from_minterm(0)   # 000000
        b = Implicant.from_minterm(1)   # 000001
        merged = a.combine(b)
        self.assertIsNotNone(merged)
        self.assertEqual(merged.bits, '00000-')
        self.assertEqual(merged.covered, frozenset({0, 1}))

    def test_combine_invalid_distance2(self):
        a = Implicant.from_minterm(0)   # 000000
        b = Implicant.from_minterm(3)   # 000011
        self.assertIsNone(a.combine(b))

    def test_combine_dash_mismatch(self):
        a = Implicant('00000-', frozenset({0, 1}))
        b = Implicant('0000-0', frozenset({0, 2}))
        # Различаются в двух позициях (не считая '-')
        self.assertIsNone(a.combine(b))

    def test_covers(self):
        imp = Implicant('00000-', frozenset({0, 1}))
        self.assertTrue(imp.covers(0))
        self.assertTrue(imp.covers(1))
        self.assertFalse(imp.covers(2))

    def test_size(self):
        self.assertEqual(Implicant.from_minterm(0).size(), 1)
        self.assertEqual(Implicant('0000--', frozenset({0,1,2,3})).size(), 4)
        self.assertEqual(Implicant('------', frozenset(range(64))).size(), 64)

    def test_to_expr_single(self):
        imp = Implicant.from_minterm(3)   # 000011: x0=1, x1=1
        expr = imp.to_expr()
        self.assertIn('x0', expr)
        self.assertIn('x1', expr)

    def test_to_expr_tautology(self):
        imp = Implicant('------', frozenset(range(64)))
        self.assertEqual(imp.to_expr(), '1')

    def test_to_expr_negation(self):
        imp = Implicant.from_minterm(0)   # 000000: все ~x
        expr = imp.to_expr()
        self.assertIn('~x0', expr)


class TestQuineMcCluskey(unittest.TestCase):
    def test_single_minterm(self):
        pis = quine_mccluskey([42])
        self.assertTrue(any(p.covers(42) for p in pis))

    def test_all_minterms(self):
        pis = quine_mccluskey(list(range(64)))
        self.assertEqual(len(pis), 1)
        self.assertEqual(pis[0].bits, '------')

    def test_empty(self):
        pis = quine_mccluskey([])
        self.assertEqual(pis, [])

    def test_x0_and_x1(self):
        """f = x0 & x1 → минтермы с битами 0 и 1 = 1."""
        minterms = [m for m in range(64) if (m & 0b11) == 0b11]
        pis = quine_mccluskey(minterms)
        self.assertEqual(len(pis), 1)
        self.assertEqual(pis[0].bits, '----11')

    def test_covers_all_minterms(self):
        """Все простые импликанты вместе покрывают все минтермы."""
        minterms = [0, 1, 2, 3, 4, 5, 6, 7]
        pis = quine_mccluskey(minterms)
        for m in minterms:
            self.assertTrue(any(p.covers(m) for p in pis))

    def test_dont_care(self):
        minterms = [0, 1]
        dont_cares = [2, 3]
        pis = quine_mccluskey(minterms, dont_cares)
        # Имплицанта должна покрыть хотя бы 0 и 1
        self.assertTrue(any(p.covers(0) and p.covers(1) for p in pis))


class TestMinimize(unittest.TestCase):
    def test_constant_zero(self):
        r = minimize([])
        self.assertEqual(r['expression'], '0')

    def test_constant_one(self):
        r = minimize(list(range(64)))
        self.assertEqual(r['expression'], '1')

    def test_single_var(self):
        """f = x0: минтермы с битом 0 = 1."""
        minterms = [m for m in range(64) if m & 1]
        r = minimize(minterms)
        self.assertIn('x0', r['expression'])
        # Один существенный импликант
        self.assertEqual(len(r['essential']), 1)
        self.assertEqual(r['essential'][0].bits, '-----1')

    def test_minimization_correct(self):
        """Проверяем, что минимизированная функция совпадает с исходной."""
        import ast
        minterms = [3, 7, 11, 15, 19, 23]
        r = minimize(minterms)
        # Все минтермы покрыты существенными импликантами
        covered = set()
        for e in r['essential']:
            covered.update(e.covered & set(minterms))
        self.assertEqual(covered, set(minterms))

    def test_two_var_function(self):
        """f = x0 & ~x1 (минтермы: 1, 5, 9, ... — бит0=1, бит1=0)."""
        minterms = [m for m in range(64) if (m & 1) and not (m & 2)]
        r = minimize(minterms)
        self.assertEqual(len(r['essential']), 1)
        # Импликанта: ----01
        self.assertEqual(r['essential'][0].bits, '----01')

    def test_essential_cover_all(self):
        """Существенные импликанты покрывают все минтермы."""
        import random
        random.seed(0)
        minterms = random.sample(range(64), 20)
        r = minimize(minterms)
        covered = set()
        for e in r['essential']:
            covered.update(e.covered & set(minterms))
        self.assertEqual(covered, set(minterms))


class TestEssentialImplicants(unittest.TestCase):
    """Тесты функции essential_implicants (жадный выбор покрытия)."""

    def test_single_minterm_single_pi(self):
        """Один минтерм → одна существенная импликанта."""
        pis = quine_mccluskey([42])
        ess = essential_implicants(pis, [42])
        self.assertEqual(len(ess), 1)
        self.assertTrue(ess[0].covers(42))

    def test_covers_all_minterms(self):
        """Существенные импликанты покрывают все входные минтермы."""
        minterms = [0, 1, 2, 4, 8, 16, 32]
        pis = quine_mccluskey(minterms)
        ess = essential_implicants(pis, minterms)
        covered = set()
        for e in ess:
            covered.update(e.covered & set(minterms))
        self.assertEqual(covered, set(minterms))

    def test_empty_minterms(self):
        """Пустой список минтермов → пустое покрытие."""
        ess = essential_implicants([], [])
        self.assertEqual(ess, [])

    def test_all_minterms_one_implicant(self):
        """Все 64 минтерма → одна тавтологическая импликанта '------'."""
        minterms = list(range(64))
        pis = quine_mccluskey(minterms)
        ess = essential_implicants(pis, minterms)
        self.assertEqual(len(ess), 1)
        self.assertEqual(ess[0].bits, '------')

    def test_no_duplicates(self):
        """Одна импликанта не выбирается дважды."""
        minterms = [0, 1, 2, 3]
        pis = quine_mccluskey(minterms)
        ess = essential_implicants(pis, minterms)
        self.assertEqual(len(ess), len(set(id(e) for e in ess)))

    def test_result_is_subset_of_pis(self):
        """Существенные импликанты — подмножество простых импликант."""
        minterms = [0, 1, 4, 5, 16, 17, 20, 21]
        pis = quine_mccluskey(minterms)
        ess = essential_implicants(pis, minterms)
        for e in ess:
            self.assertIn(e, pis)

    def test_random_function_covered(self):
        """Случайная функция (seed=7) — все минтермы покрыты."""
        import random
        random.seed(7)
        minterms = sorted(random.sample(range(64), 15))
        pis = quine_mccluskey(minterms)
        ess = essential_implicants(pis, minterms)
        covered = set()
        for e in ess:
            covered.update(e.covered & set(minterms))
        self.assertEqual(covered, set(minterms))


if __name__ == '__main__':
    unittest.main(verbosity=2)
