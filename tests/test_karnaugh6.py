"""Тесты карты Карно и алгоритма Куайна–МакКласки (karnaugh6)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import io
import unittest
from contextlib import redirect_stdout
from projects.karnaugh6.minimize import (
    Implicant, quine_mccluskey, essential_implicants, minimize,
    print_truth_table, print_karnaugh_map, print_result,
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


class TestPrintFunctions(unittest.TestCase):
    """Тесты функций вывода: print_truth_table, print_karnaugh_map, print_result."""

    def _capture(self, fn, *args, **kwargs) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*args, **kwargs)
        return buf.getvalue()

    # print_truth_table -------------------------------------------------------

    def test_truth_table_contains_header(self):
        out = self._capture(print_truth_table, [0])
        self.assertIn('x0', out)
        self.assertIn('x5', out)

    def test_truth_table_shows_minterm(self):
        out = self._capture(print_truth_table, [42])
        self.assertIn('42', out)
        self.assertIn('1', out)

    def test_truth_table_shows_dont_care(self):
        out = self._capture(print_truth_table, [0], [1])
        self.assertIn('-', out)

    def test_truth_table_empty_no_crash(self):
        out = self._capture(print_truth_table, [])
        self.assertGreater(len(out), 0)  # header still printed

    def test_truth_table_all_ones_shows_count(self):
        out = self._capture(print_truth_table, list(range(64)))
        # 64 minterm rows + header + separator = many lines
        self.assertGreater(out.count('\n'), 64)

    # print_karnaugh_map -------------------------------------------------------

    def test_karnaugh_map_contains_header(self):
        out = self._capture(print_karnaugh_map, [0])
        self.assertIn('8×8', out)

    def test_karnaugh_map_shows_one_for_minterm(self):
        out = self._capture(print_karnaugh_map, [0])
        self.assertIn('1', out)

    def test_karnaugh_map_shows_dot_for_zero(self):
        out = self._capture(print_karnaugh_map, [0])
        self.assertIn('·', out)

    def test_karnaugh_map_shows_dash_for_dont_care(self):
        out = self._capture(print_karnaugh_map, [], [0])
        self.assertIn('-', out)

    def test_karnaugh_map_shows_star_for_essential(self):
        minterms = [0, 1]
        result = minimize(minterms)
        out = self._capture(print_karnaugh_map, minterms, [], result['essential'])
        self.assertIn('*', out)

    def test_karnaugh_map_8_rows(self):
        """Карта содержит 8 строк данных (коды Грея)."""
        out = self._capture(print_karnaugh_map, [])
        gray_labels = ['000', '001', '011', '010', '110', '111', '101', '100']
        for label in gray_labels:
            self.assertIn(label, out)

    # print_result ------------------------------------------------------------

    def test_print_result_shows_expression(self):
        result = minimize([0, 1, 2, 3])
        out = self._capture(print_result, result)
        self.assertIn(result['expression'], out)

    def test_print_result_shows_prime_implicants(self):
        result = minimize([0, 1])
        out = self._capture(print_result, result)
        self.assertIn('импликант', out.lower())

    def test_print_result_shows_essential(self):
        result = minimize([42])
        out = self._capture(print_result, result)
        self.assertIn('существенн', out.lower())

    def test_print_result_constant_zero(self):
        result = minimize([])
        out = self._capture(print_result, result)
        self.assertIn('0', out)

    def test_print_result_constant_one(self):
        result = minimize(list(range(64)))
        out = self._capture(print_result, result)
        self.assertIn('1', out)


class TestImplicantReprAndEq(unittest.TestCase):
    """Тесты для __repr__ (line 92) и __eq__ NotImplemented (line 96)."""

    def test_repr_returns_string(self):
        imp = Implicant.from_minterm(42)
        r = repr(imp)
        self.assertIn('Implicant', r)
        self.assertIn('101010', r)

    def test_eq_with_non_implicant_returns_not_implemented(self):
        imp = Implicant.from_minterm(0)
        result = imp.__eq__(42)
        self.assertIs(result, NotImplemented)


class TestGreedyCover(unittest.TestCase):
    """Тесты жадного покрытия в essential_implicants (lines 184-191)."""

    def test_cyclic_cover_uses_greedy_selection(self):
        # Cyclic covering: each minterm covered by exactly 2 PIs → no essential PIs
        # Greedy loop must select them
        p1 = Implicant('00000-', frozenset({0, 1}))  # covers 0, 1
        p2 = Implicant('0000-0', frozenset({0, 2}))  # covers 0, 2
        p3 = Implicant('0000-1', frozenset({1, 3}))  # covers 1, 3
        p4 = Implicant('00001-', frozenset({2, 3}))  # covers 2, 3
        selected = essential_implicants([p1, p2, p3, p4], [0, 1, 2, 3])
        # All minterms must be covered
        covered = set()
        for imp in selected:
            covered |= imp.covered
        self.assertGreaterEqual(covered, {0, 1, 2, 3})

    def test_greedy_break_when_no_coverage(self):
        # PI that doesn't cover any of the specified minterms → greedy breaks, returns []
        p_other = Implicant('100000', frozenset({32}))  # covers only 32
        selected = essential_implicants([p_other], [5, 10])
        self.assertEqual(selected, [])


class TestKarnaughCLI(unittest.TestCase):
    """Тесты для main() функции (lines 348-402)."""

    def _run(self, args, expect_exit=None):
        from projects.karnaugh6.minimize import main
        old_argv = sys.argv
        sys.argv = ['minimize.py'] + args
        buf = io.StringIO()
        try:
            if expect_exit is not None:
                with self.assertRaises(SystemExit) as cm:
                    with redirect_stdout(buf):
                        main()
                self.assertEqual(cm.exception.code, expect_exit)
            else:
                with redirect_stdout(buf):
                    main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_main_no_args_shows_example(self):
        out = self._run([])
        self.assertIn('Пример', out)

    def test_main_with_minterms(self):
        out = self._run(['0', '1', '2', '3'])
        self.assertIn('karnaugh6', out.lower())

    def test_main_with_table_flag(self):
        out = self._run(['0', '1', '--table'])
        self.assertIn('x5', out)

    def test_main_with_map_flag(self):
        out = self._run(['0', '1', '--map'])
        self.assertGreater(len(out), 0)

    def test_main_out_of_range_exits_1(self):
        self._run(['99'], expect_exit=1)

    def test_main_overlapping_dc_exits_1(self):
        self._run(['0', '1', '--dc', '1'], expect_exit=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
