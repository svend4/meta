"""Тесты для hexlat — булева решётка B₆ на Q6."""
import sys
import os
import io
import unittest
from contextlib import redirect_stdout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import comb
from projects.hexlat.hexlat import (
    leq, meet, join, complement, rank, rank_elements, whitney_numbers,
    is_chain, is_antichain,
    maximal_chains, largest_antichain, dilworth_theorem,
    mobius, zeta_function, mobius_inversion_check, euler_characteristic,
    interval, principal_filter, principal_ideal, upset_closure, downset_closure,
    covers, hasse_edges, atom_decomposition,
    rank_generating_function, poincare_polynomial, characteristic_polynomial,
    zeta_polynomial,
    lattice_diameter, comparable_pairs_count, incomparable_pairs_count,
    sublattice_boolean,
    mirsky_decomposition, chain_partition_count,
    _cmd_info, _cmd_interval, _cmd_mobius, _cmd_chains, _cmd_antichain,
)


class TestLatticeOrder(unittest.TestCase):
    def test_leq_reflexive(self):
        for x in range(64): self.assertTrue(leq(x, x))
    def test_leq_antisymmetric(self):
        for x in range(64):
            for y in range(64):
                if leq(x, y) and leq(y, x): self.assertEqual(x, y)
    def test_leq_transitive_sample(self):
        self.assertTrue(leq(0, 1) and leq(1, 3) and leq(0, 3))
    def test_leq_zero_minimum(self):
        for x in range(64): self.assertTrue(leq(0, x))
    def test_leq_maximum(self):
        for x in range(64): self.assertTrue(leq(x, 63))
    def test_meet_definition(self):
        for x in range(64):
            for y in range(0, 64, 7): self.assertEqual(meet(x, y), x & y)
    def test_join_definition(self):
        for x in range(64):
            for y in range(0, 64, 7): self.assertEqual(join(x, y), x | y)
    def test_complement_involutive(self):
        for x in range(64): self.assertEqual(complement(complement(x)), x)
    def test_complement_meet_zero(self):
        for x in range(64): self.assertEqual(meet(x, complement(x)), 0)
    def test_complement_join_one(self):
        for x in range(64): self.assertEqual(join(x, complement(x)), 63)
    def test_rank_zero(self):
        self.assertEqual(rank(0), 0)
    def test_rank_max(self):
        self.assertEqual(rank(63), 6)
    def test_rank_atoms(self):
        for bit in range(6): self.assertEqual(rank(1 << bit), 1)
    def test_rank_elements_count(self):
        for k in range(7): self.assertEqual(len(rank_elements(k)), comb(6, k))
    def test_rank_elements_disjoint(self):
        all_elems = []
        for k in range(7): all_elems.extend(rank_elements(k))
        self.assertEqual(sorted(all_elems), list(range(64)))


class TestWhitneyNumbers(unittest.TestCase):
    def test_whitney_count(self):
        self.assertEqual(len(whitney_numbers()), 7)
    def test_whitney_values(self):
        wn = whitney_numbers()
        for k in range(7): self.assertEqual(wn[k], comb(6, k))
    def test_whitney_sum(self):
        self.assertEqual(sum(whitney_numbers()), 64)
    def test_whitney_symmetric(self):
        wn = whitney_numbers()
        for k in range(7): self.assertEqual(wn[k], wn[6 - k])
    def test_whitney_max(self):
        self.assertEqual(max(whitney_numbers()), comb(6, 3))


class TestChainsAntichains(unittest.TestCase):
    def test_is_chain_empty(self):
        self.assertTrue(is_chain([]))
    def test_is_chain_single(self):
        self.assertTrue(is_chain([5]))
    def test_is_chain_valid(self):
        self.assertTrue(is_chain([0, 1, 3, 7, 15, 31, 63]))
    def test_is_chain_invalid(self):
        self.assertFalse(is_chain([1, 2]))
    def test_is_antichain_empty(self):
        self.assertTrue(is_antichain([]))
    def test_is_antichain_single(self):
        self.assertTrue(is_antichain([42]))
    def test_is_antichain_atoms(self):
        self.assertTrue(is_antichain([1 << b for b in range(6)]))
    def test_is_antichain_invalid(self):
        self.assertFalse(is_antichain([0, 1]))
    def test_maximal_chains_count(self):
        self.assertEqual(len(maximal_chains()), 720)
    def test_maximal_chains_start_end(self):
        for chain in maximal_chains():
            self.assertEqual(chain[0], 0); self.assertEqual(chain[-1], 63)
    def test_maximal_chains_length(self):
        for chain in maximal_chains(): self.assertEqual(len(chain), 7)
    def test_maximal_chains_are_chains(self):
        for chain in maximal_chains()[:10]: self.assertTrue(is_chain(chain))
    def test_largest_antichain_size(self):
        self.assertEqual(len(largest_antichain()), comb(6, 3))
    def test_largest_antichain_is_antichain(self):
        self.assertTrue(is_antichain(largest_antichain()))
    def test_dilworth_width(self):
        self.assertEqual(dilworth_theorem()['width'], 20)


class TestMobius(unittest.TestCase):
    def test_mobius_diagonal(self):
        for x in range(64): self.assertEqual(mobius(x, x), 1)
    def test_mobius_atoms(self):
        for bit in range(6): self.assertEqual(mobius(0, 1 << bit), -1)
    def test_mobius_max(self):
        self.assertEqual(mobius(0, 63), 1)
    def test_mobius_incomparable(self):
        self.assertEqual(mobius(1, 2), 0)
    def test_mobius_formula(self):
        for x in range(0, 64, 5):
            for y in range(0, 64, 5):
                if leq(x, y):
                    expected = (-1) ** bin(y ^ x).count('1')
                    self.assertEqual(mobius(x, y), expected)
    def test_mobius_inversion(self):
        self.assertTrue(mobius_inversion_check())
    def test_euler_characteristic(self):
        self.assertEqual(euler_characteristic(), 0)
    def test_zeta_function(self):
        for x in range(10):
            for y in range(10):
                if leq(x, y): self.assertEqual(zeta_function(x, y), 1)
                else: self.assertEqual(zeta_function(x, y), 0)


class TestIntervals(unittest.TestCase):
    def test_interval_same(self):
        self.assertEqual(interval(5, 5), [5])
    def test_interval_full(self):
        self.assertEqual(sorted(interval(0, 63)), list(range(64)))
    def test_interval_incomparable(self):
        self.assertEqual(interval(1, 2), [])
    def test_interval_size(self):
        x, y = 5, 63
        if leq(x, y):
            self.assertEqual(len(interval(x, y)), 2 ** rank(y ^ x))
    def test_principal_filter_zero(self):
        self.assertEqual(sorted(principal_filter(0)), list(range(64)))
    def test_principal_filter_max(self):
        self.assertEqual(principal_filter(63), [63])
    def test_principal_ideal_zero(self):
        self.assertEqual(principal_ideal(0), [0])
    def test_principal_ideal_max(self):
        self.assertEqual(sorted(principal_ideal(63)), list(range(64)))
    def test_upset_closure(self):
        self.assertEqual(sorted(upset_closure([0])), list(range(64)))
    def test_downset_closure(self):
        self.assertEqual(sorted(downset_closure([63])), list(range(64)))
    def test_downset_atoms_is_atoms_and_zero(self):
        atoms = [1, 2, 4]
        ds = downset_closure(atoms)
        self.assertIn(0, ds)
        for a in atoms: self.assertIn(a, ds)


class TestHasse(unittest.TestCase):
    def test_hasse_edges_count(self):
        self.assertEqual(len(hasse_edges()), 192)
    def test_covers_basic(self):
        self.assertTrue(covers(0, 1)); self.assertTrue(covers(0, 2))
        self.assertFalse(covers(0, 3))
    def test_covers_max(self):
        for x in range(64): self.assertFalse(covers(63, x))
    def test_atom_decomposition_zero(self):
        self.assertEqual(atom_decomposition(0), [])
    def test_atom_decomposition_max(self):
        self.assertEqual(sorted(atom_decomposition(63)), [1, 2, 4, 8, 16, 32])
    def test_atom_decomposition_join(self):
        for x in range(64):
            result = 0
            for a in atom_decomposition(x): result |= a
            self.assertEqual(result, x)


class TestPolynomials(unittest.TestCase):
    def test_rank_generating_function(self):
        self.assertEqual(rank_generating_function(), [comb(6, k) for k in range(7)])
    def test_poincare_polynomial(self):
        self.assertEqual(poincare_polynomial(), [comb(6, k) for k in range(7)])
    def test_characteristic_polynomial_values(self):
        cp = characteristic_polynomial()
        self.assertEqual(len(cp), 7)
        self.assertEqual(cp[0], 1); self.assertEqual(cp[1], -6); self.assertEqual(cp[6], 1)
    def test_characteristic_polynomial_sum(self):
        cp = characteristic_polynomial()
        self.assertEqual(sum(cp[i] * (1 ** (6 - i)) for i in range(7)), 0)
    def test_zeta_polynomial_n1(self):
        self.assertEqual(zeta_polynomial(1), 64)
    def test_zeta_polynomial_n2(self):
        self.assertEqual(zeta_polynomial(2), comparable_pairs_count())
    def test_zeta_polynomial_positive(self):
        for n in range(1, 5): self.assertGreater(zeta_polynomial(n), 0)

    def test_zeta_polynomial_n0(self):
        """Z(0) = 0 для n ≤ 0."""
        assert zeta_polynomial(0) == 0
        assert zeta_polynomial(-1) == 0


class TestLatticeStructure(unittest.TestCase):
    def test_lattice_diameter(self):
        self.assertEqual(lattice_diameter(), 6)
    def test_comparable_pairs_count(self):
        manual = sum(1 for x in range(64) for y in range(64) if leq(x, y))
        self.assertEqual(comparable_pairs_count(), manual)
    def test_incomparable_pairs_count(self):
        manual = sum(1 for x in range(64) for y in range(x+1, 64)
                     if not leq(x, y) and not leq(y, x))
        self.assertEqual(incomparable_pairs_count(), manual)
    def test_sublattice_boolean_2(self):
        self.assertEqual(len(sublattice_boolean(2)), comb(6, 2))
    def test_sublattice_boolean_3(self):
        self.assertEqual(len(sublattice_boolean(3)), comb(6, 3))
    def test_sublattice_boolean_elements_count(self):
        for k in range(1, 5):
            for mask, elements in sublattice_boolean(k):
                self.assertEqual(len(elements), 1 << k)
    def test_mirsky_decomposition(self):
        md = mirsky_decomposition()
        self.assertEqual(len(md), 7)
        for k in range(7): self.assertEqual(sorted(md[k]), sorted(rank_elements(k)))
    def test_chain_partition_count(self):
        self.assertEqual(chain_partition_count(), comb(6, 3))
    def test_total_elements(self):
        self.assertEqual(sum(len(rank_elements(k)) for k in range(7)), 64)


# ============================================================
# CLI-функции
# ============================================================

class TestCLI:
    def _capture(self, fn, args=None):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(args or [])
        return buf.getvalue()

    def test_cmd_info_produces_output(self):
        out = self._capture(_cmd_info)
        assert 'Уитни' in out or 'B' in out

    def test_cmd_interval_valid(self):
        out = self._capture(_cmd_interval, ['0', '7'])
        assert 'нтервал' in out or '0' in out

    def test_cmd_interval_no_args(self):
        out = self._capture(_cmd_interval, [])
        assert 'Использование' in out

    def test_cmd_mobius_valid(self):
        out = self._capture(_cmd_mobius, ['0', '7'])
        assert 'μ' in out or '0' in out

    def test_cmd_mobius_no_args(self):
        out = self._capture(_cmd_mobius, [])
        assert 'Использование' in out

    def test_cmd_chains_produces_output(self):
        out = self._capture(_cmd_chains)
        assert '720' in out

    def test_cmd_antichain_produces_output(self):
        out = self._capture(_cmd_antichain)
        assert '20' in out


# ============================================================
# CLI main()
# ============================================================

class TestMainCLI:
    def _run(self, args):
        import sys
        from projects.hexlat.hexlat import main
        old_argv = sys.argv
        sys.argv = ['hexlat.py'] + args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_no_args_shows_usage(self):
        out = self._run([])
        assert 'Использование' in out

    def test_cmd_info(self):
        out = self._run(['info'])
        assert len(out) > 0

    def test_cmd_interval(self):
        out = self._run(['interval', '0', '7'])
        assert len(out) > 0

    def test_cmd_mobius(self):
        out = self._run(['mobius', '0', '63'])
        assert len(out) > 0

    def test_cmd_chains(self):
        out = self._run(['chains'])
        assert '720' in out

    def test_cmd_antichain(self):
        out = self._run(['antichain'])
        assert '20' in out

    def test_unknown_cmd(self):
        out = self._run(['xyz'])
        assert 'Неизвестная' in out


if __name__ == '__main__':
    unittest.main()
