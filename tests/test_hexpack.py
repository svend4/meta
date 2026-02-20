"""Тесты hexpack — упаковки замкнутых клеточных полей Германа (K5)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest

from projects.hexpack.hexpack import (
    PackedRing, MagicSquare, Q6_RING,
    period, valid_periods, prime_periods,
    _triangular, _is_power_of_two,
)
from projects.hexpack.pack_glyphs import (
    json_ring, json_antipode, json_fixpoint, json_packable, json_periods, json_magic,
)


class TestTriangularAndPow2(unittest.TestCase):
    def test_triangular(self):
        for n in range(1, 10):
            self.assertEqual(_triangular(n), n * (n - 1) // 2)

    def test_triangular_zero(self):
        self.assertEqual(_triangular(0), 0)

    def test_is_power_of_two(self):
        for k in range(8):
            self.assertTrue(_is_power_of_two(2 ** k))

    def test_is_not_power_of_two(self):
        for n in [3, 5, 6, 7, 9, 12, 65]:
            self.assertFalse(_is_power_of_two(n))

    def test_period_formula(self):
        for n in range(1, 20):
            self.assertEqual(period(n), n * (n - 1) // 2 + 1)


class TestPackedRing(unittest.TestCase):
    """Тесты основного класса PackedRing на примере Q6 (P=64)."""

    def test_q6_ring_packable(self):
        self.assertTrue(Q6_RING.packable)

    def test_ring_length(self):
        self.assertEqual(len(Q6_RING), 64)

    def test_ring_is_permutation(self):
        """ring[0..63] — перестановка {1..64}."""
        self.assertEqual(set(Q6_RING.as_list()), set(range(1, 65)))

    def test_ring_no_zeros(self):
        self.assertNotIn(0, Q6_RING.as_list())

    def test_exceptional_start(self):
        """Следствие 1: исключительный старт m=32."""
        self.assertEqual(Q6_RING.exceptional_start(), 32)

    def test_exceptional_has_no_fixed_points(self):
        exc = Q6_RING.exceptional_start()
        self.assertEqual(Q6_RING.fixed_points(exc), [])

    def test_only_one_start_has_no_fixed_points(self):
        """Ровно один старт (m=32) не имеет фиксированных точек."""
        zero_fp_starts = [m for m in range(64) if not Q6_RING.fixed_points(m)]
        self.assertEqual(len(zero_fp_starts), 1)
        self.assertEqual(zero_fp_starts[0], 32)

    def test_verify_antipodal(self):
        """Следствие 2: ring[k] + ring[k+32] = 65 для всех k."""
        self.assertTrue(Q6_RING.verify_antipodal())

    def test_antipodal_sum_constant(self):
        self.assertEqual(Q6_RING.q6_antipodal_sum_constant(), 65)

    def test_antipodal_pairs_count(self):
        pairs = Q6_RING.antipodal_pairs()
        self.assertEqual(len(pairs), 32)

    def test_antipodal_pairs_sum_65(self):
        for _, _, v1, v2, s in Q6_RING.antipodal_pairs():
            self.assertEqual(s, 65, f"v1={v1}, v2={v2}, sum={s}")

    def test_q6_antipode(self):
        for h in range(64):
            self.assertEqual(Q6_RING.q6_antipode(h), h ^ 63)

    def test_non_power_not_packable(self):
        r = PackedRing(7)
        self.assertFalse(r.packable)

    def test_small_power_packable(self):
        for k in [1, 2, 3, 4]:
            r = PackedRing(2 ** k)
            self.assertTrue(r.packable)

    def test_getitem_wraps(self):
        """__getitem__ работает по модулю P."""
        self.assertEqual(Q6_RING[0], Q6_RING[64])

    def test_as_list_copy(self):
        lst = Q6_RING.as_list()
        self.assertIsInstance(lst, list)
        self.assertEqual(len(lst), 64)


class TestMagicSquare(unittest.TestCase):
    """Тесты магических квадратов из упаковок P=2^(2k)."""

    def test_k1_creates(self):
        ms = MagicSquare(1)
        self.assertIsInstance(ms, MagicSquare)

    def test_k1_side(self):
        self.assertEqual(MagicSquare(1).side, 2)

    def test_k1_P(self):
        self.assertEqual(MagicSquare(1).P, 4)

    def test_k1_magic_constant(self):
        # (P+1) * side // 2 = 5 * 2 // 2 = 5
        self.assertEqual(MagicSquare(1).magic_constant, 5)

    def test_k1_is_magic(self):
        self.assertTrue(MagicSquare(1).is_magic())

    def test_k1_column_sums(self):
        ms = MagicSquare(1)
        for s in ms.column_sums():
            self.assertEqual(s, ms.magic_constant)

    def test_k2_side(self):
        self.assertEqual(MagicSquare(2).side, 4)

    def test_k2_is_magic(self):
        self.assertTrue(MagicSquare(2).is_magic())

    def test_k2_column_sums(self):
        ms = MagicSquare(2)
        for s in ms.column_sums():
            self.assertEqual(s, ms.magic_constant)

    def test_matrix_shape(self):
        ms = MagicSquare(2)
        self.assertEqual(len(ms.matrix), ms.side)
        for row in ms.matrix:
            self.assertEqual(len(row), ms.side)

    def test_k0_raises(self):
        with self.assertRaises(ValueError):
            MagicSquare(0)


class TestValidPeriods(unittest.TestCase):
    def test_returns_list(self):
        vp = valid_periods(64)
        self.assertIsInstance(vp, list)

    def test_all_power_of_two(self):
        for n, P, k in valid_periods(200):
            self.assertTrue(_is_power_of_two(P))
            self.assertEqual(2 ** k, P)

    def test_period_formula_holds(self):
        for n, P, k in valid_periods(200):
            self.assertEqual(period(n), P)

    def test_n1_in_valid(self):
        ns = [n for n, P, k in valid_periods(64)]
        self.assertIn(1, ns)    # period(1)=1=2^0

    def test_prime_periods_is_list(self):
        pp = prime_periods(64)
        self.assertIsInstance(pp, list)

    def test_prime_periods_p_is_prime(self):
        def is_prime(p):
            if p < 2: return False
            if p == 2: return True
            if p % 2 == 0: return False
            return all(p % i != 0 for i in range(3, int(p**0.5)+1, 2))
        for n, P in prime_periods(64):
            self.assertTrue(is_prime(P), f"P={P} is not prime")


class TestJsonRing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json_ring()

    def test_command(self):
        self.assertEqual(self.result['command'], 'ring')

    def test_p_is_64(self):
        self.assertEqual(self.result['P'], 64)

    def test_packable_true(self):
        self.assertTrue(self.result['packable'])

    def test_exceptional_start_32(self):
        self.assertEqual(self.result['exceptional_start'], 32)

    def test_antipodal_sum_65(self):
        self.assertEqual(self.result['antipodal_sum'], 65)

    def test_verify_true(self):
        self.assertTrue(self.result['verify'])

    def test_ring_length_64(self):
        self.assertEqual(len(self.result['ring']), 64)

    def test_ring_is_permutation(self):
        self.assertEqual(set(self.result['ring']), set(range(1, 65)))


class TestJsonAntipode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json_antipode()

    def test_command(self):
        self.assertEqual(self.result['command'], 'antipode')

    def test_verify_true(self):
        self.assertTrue(self.result['verify'])

    def test_antipodal_sum_65(self):
        self.assertEqual(self.result['antipodal_sum'], 65)

    def test_pairs_count_32(self):
        self.assertEqual(len(self.result['pairs']), 32)

    def test_all_sums_equal_65(self):
        for pair in self.result['pairs']:
            self.assertEqual(pair['sum'], 65, f"pair={pair}")

    def test_pair_fields(self):
        for pair in self.result['pairs']:
            for key in ('h', 'h_xor', 'v1', 'v2', 'sum'):
                self.assertIn(key, pair)


class TestJsonFixpoint(unittest.TestCase):
    def test_exceptional_start_is_exceptional(self):
        r = json_fixpoint(start=32)
        self.assertTrue(r['is_exceptional'])
        self.assertEqual(r['count'], 0)
        self.assertEqual(r['fixed_points'], [])

    def test_normal_start_one_fixed_point(self):
        r = json_fixpoint(start=5)   # start=5 → именно 1 фиксированная точка
        self.assertFalse(r['is_exceptional'])
        self.assertEqual(r['count'], 1)
        self.assertEqual(len(r['fixed_points']), 1)

    def test_command(self):
        self.assertEqual(json_fixpoint()['command'], 'fixpoint')

    def test_exceptional_start_field(self):
        r = json_fixpoint(start=5)
        self.assertEqual(r['exceptional_start'], 32)


class TestJsonPackable(unittest.TestCase):
    def test_power_of_two_packable(self):
        r = json_packable(64)
        self.assertTrue(r['packable'])
        self.assertEqual(r['k'], 6)

    def test_non_power_not_packable(self):
        r = json_packable(65)
        self.assertFalse(r['packable'])
        self.assertIn('nearest_lower', r)
        self.assertIn('nearest_upper', r)
        self.assertEqual(r['nearest_lower'], 64)
        self.assertEqual(r['nearest_upper'], 128)

    def test_command(self):
        self.assertEqual(json_packable(64)['command'], 'packable')

    def test_p4_k2(self):
        r = json_packable(4)
        self.assertTrue(r['packable'])
        self.assertEqual(r['k'], 2)


class TestJsonPeriods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json_periods(64)

    def test_command(self):
        self.assertEqual(self.result['command'], 'periods')

    def test_max_n(self):
        self.assertEqual(self.result['max_n'], 64)

    def test_entry_count(self):
        self.assertEqual(len(self.result['entries']), 64)

    def test_entry_fields(self):
        for e in self.result['entries']:
            for key in ('n', 'P', 'is_power_of_2', 'is_prime'):
                self.assertIn(key, e)

    def test_n1_entry(self):
        e = self.result['entries'][0]
        self.assertEqual(e['n'], 1)
        self.assertEqual(e['P'], 1)
        self.assertTrue(e['is_power_of_2'])

    def test_power_of_2_entries_have_k(self):
        for e in self.result['entries']:
            if e['is_power_of_2']:
                self.assertIsNotNone(e['k'])
                self.assertEqual(2 ** e['k'], e['P'])


class TestJsonMagic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r1 = json_magic(1)
        cls.r2 = json_magic(2)

    def test_command(self):
        self.assertEqual(self.r1['command'], 'magic')

    def test_k1_side(self):
        self.assertEqual(self.r1['side'], 2)

    def test_k1_P(self):
        self.assertEqual(self.r1['P'], 4)

    def test_k1_magic_constant(self):
        self.assertEqual(self.r1['magic_constant'], 5)

    def test_k1_is_magic(self):
        self.assertTrue(self.r1['is_magic'])

    def test_k1_column_sums(self):
        for s in self.r1['column_sums']:
            self.assertEqual(s, self.r1['magic_constant'])

    def test_k1_square_shape(self):
        sq = self.r1['square']
        self.assertEqual(len(sq), 2)
        for row in sq:
            self.assertEqual(len(row), 2)

    def test_k2_side(self):
        self.assertEqual(self.r2['side'], 4)

    def test_k2_is_magic(self):
        self.assertTrue(self.r2['is_magic'])

    def test_k2_square_shape(self):
        sq = self.r2['square']
        self.assertEqual(len(sq), 4)
        for row in sq:
            self.assertEqual(len(row), 4)

    def test_k2_column_sums(self):
        for s in self.r2['column_sums']:
            self.assertEqual(s, self.r2['magic_constant'])

    def test_square_is_permutation(self):
        """Все числа 1..P встречаются ровно по одному разу."""
        sq = self.r2['square']
        P = self.r2['P']
        flat = [v for row in sq for v in row]
        self.assertEqual(set(flat), set(range(1, P + 1)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
