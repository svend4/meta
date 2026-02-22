"""Тесты для hexlat — булева решётка B₆ на Q6."""
import sys
import os
import io
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


# ============================================================
# Частичный порядок и базовые операции
# ============================================================

class TestLatticeOrder:
    def test_leq_reflexive(self):
        for x in range(64):
            assert leq(x, x)

    def test_leq_antisymmetric(self):
        for x in range(64):
            for y in range(64):
                if leq(x, y) and leq(y, x):
                    assert x == y

    def test_leq_transitive_sample(self):
        # 0 ≤ 1 ≤ 3 ≤ 7
        assert leq(0, 1) and leq(1, 3) and leq(0, 3)

    def test_leq_zero_minimum(self):
        for x in range(64):
            assert leq(0, x)

    def test_leq_maximum(self):
        for x in range(64):
            assert leq(x, 63)

    def test_meet_definition(self):
        for x in range(64):
            for y in range(0, 64, 7):
                assert meet(x, y) == (x & y)

    def test_join_definition(self):
        for x in range(64):
            for y in range(0, 64, 7):
                assert join(x, y) == (x | y)

    def test_complement_involutive(self):
        for x in range(64):
            assert complement(complement(x)) == x

    def test_complement_meet_zero(self):
        for x in range(64):
            assert meet(x, complement(x)) == 0

    def test_complement_join_one(self):
        for x in range(64):
            assert join(x, complement(x)) == 63

    def test_rank_zero(self):
        assert rank(0) == 0

    def test_rank_max(self):
        assert rank(63) == 6

    def test_rank_atoms(self):
        for bit in range(6):
            assert rank(1 << bit) == 1

    def test_rank_elements_count(self):
        for k in range(7):
            assert len(rank_elements(k)) == comb(6, k)

    def test_rank_elements_disjoint(self):
        all_elems = []
        for k in range(7):
            all_elems.extend(rank_elements(k))
        assert sorted(all_elems) == list(range(64))


# ============================================================
# Числа Уитни
# ============================================================

class TestWhitneyNumbers:
    def test_whitney_count(self):
        wn = whitney_numbers()
        assert len(wn) == 7

    def test_whitney_values(self):
        wn = whitney_numbers()
        for k in range(7):
            assert wn[k] == comb(6, k)

    def test_whitney_sum(self):
        assert sum(whitney_numbers()) == 64

    def test_whitney_symmetric(self):
        wn = whitney_numbers()
        for k in range(7):
            assert wn[k] == wn[6 - k]

    def test_whitney_max(self):
        wn = whitney_numbers()
        assert max(wn) == comb(6, 3)


# ============================================================
# Цепи и антицепи
# ============================================================

class TestChainsAntichains:
    def test_is_chain_empty(self):
        assert is_chain([])

    def test_is_chain_single(self):
        assert is_chain([5])

    def test_is_chain_valid(self):
        assert is_chain([0, 1, 3, 7, 15, 31, 63])

    def test_is_chain_invalid(self):
        assert not is_chain([1, 2])  # 1 и 2 несравнимы

    def test_is_antichain_empty(self):
        assert is_antichain([])

    def test_is_antichain_single(self):
        assert is_antichain([42])

    def test_is_antichain_atoms(self):
        atoms = [1 << b for b in range(6)]
        assert is_antichain(atoms)

    def test_is_antichain_invalid(self):
        assert not is_antichain([0, 1])  # 0 ≤ 1

    def test_maximal_chains_count(self):
        chains = maximal_chains()
        assert len(chains) == 720  # 6! = 720

    def test_maximal_chains_start_end(self):
        for chain in maximal_chains():
            assert chain[0] == 0
            assert chain[-1] == 63

    def test_maximal_chains_length(self):
        for chain in maximal_chains():
            assert len(chain) == 7

    def test_maximal_chains_are_chains(self):
        chains = maximal_chains()
        for chain in chains[:10]:  # проверить первые 10
            assert is_chain(chain)

    def test_largest_antichain_size(self):
        ac = largest_antichain()
        assert len(ac) == comb(6, 3)  # = 20

    def test_largest_antichain_is_antichain(self):
        assert is_antichain(largest_antichain())

    def test_dilworth_width(self):
        d = dilworth_theorem()
        assert d['width'] == 20


# ============================================================
# Функция Мёбиуса
# ============================================================

class TestMobius:
    def test_mobius_diagonal(self):
        for x in range(64):
            assert mobius(x, x) == 1

    def test_mobius_atoms(self):
        # μ(0, atom) = -1
        for bit in range(6):
            assert mobius(0, 1 << bit) == -1

    def test_mobius_max(self):
        # μ(0, 63) = (-1)^6 = 1
        assert mobius(0, 63) == 1

    def test_mobius_incomparable(self):
        # 1 и 2 несравнимы: μ = 0
        assert mobius(1, 2) == 0

    def test_mobius_formula(self):
        # μ(x, y) = (-1)^{popcount(y^x)} для x ≤ y
        for x in range(0, 64, 5):
            for y in range(0, 64, 5):
                if leq(x, y):
                    expected = (-1) ** bin(y ^ x).count('1')
                    assert mobius(x, y) == expected

    def test_mobius_inversion(self):
        assert mobius_inversion_check()

    def test_euler_characteristic(self):
        assert euler_characteristic() == 0

    def test_zeta_function(self):
        for x in range(10):
            for y in range(10):
                if leq(x, y):
                    assert zeta_function(x, y) == 1
                else:
                    assert zeta_function(x, y) == 0


# ============================================================
# Интервалы и фильтры
# ============================================================

class TestIntervals:
    def test_interval_same(self):
        assert interval(5, 5) == [5]

    def test_interval_full(self):
        iv = interval(0, 63)
        assert sorted(iv) == list(range(64))

    def test_interval_incomparable(self):
        assert interval(1, 2) == []

    def test_interval_size(self):
        # [x, y]: размер = 2^{rank(y^x)}
        x, y = 5, 63
        if leq(x, y):
            iv = interval(x, y)
            expected_size = 2 ** rank(y ^ x)
            assert len(iv) == expected_size

    def test_principal_filter_zero(self):
        pf = principal_filter(0)
        assert sorted(pf) == list(range(64))

    def test_principal_filter_max(self):
        assert principal_filter(63) == [63]

    def test_principal_ideal_zero(self):
        assert principal_ideal(0) == [0]

    def test_principal_ideal_max(self):
        pi = principal_ideal(63)
        assert sorted(pi) == list(range(64))

    def test_upset_closure(self):
        # up({0}) = все элементы
        assert sorted(upset_closure([0])) == list(range(64))

    def test_downset_closure(self):
        # down({63}) = все элементы
        assert sorted(downset_closure([63])) == list(range(64))

    def test_downset_atoms_is_atoms_and_zero(self):
        atoms = [1, 2, 4]
        ds = downset_closure(atoms)
        # Должны содержать 0 и все атомы
        assert 0 in ds
        for a in atoms:
            assert a in ds


# ============================================================
# Диаграмма Хассе
# ============================================================

class TestHasse:
    def test_hasse_edges_count(self):
        edges = hasse_edges()
        # 64 вершины × 6 битов / каждое ребро считается от меньшего к большему
        assert len(edges) == 192  # = 64*6/2? Нет: ненаправленных 192, но у каждой из 64 вершин ≤ 6 покрытий вверх
        # Точно: для каждой вершины x, число покрытий вверх = 6 - rank(x)
        # Итого: Σ_k C(6,k) * (6-k) = 6 * Σ_k C(6,k) - Σ_k k*C(6,k)
        # = 6*64 - 6*32 = 384 - 192 = 192
        assert len(edges) == 192

    def test_covers_basic(self):
        assert covers(0, 1)
        assert covers(0, 2)
        assert not covers(0, 3)  # 0 < 1 < 3

    def test_covers_max(self):
        # 63 ничего не покрывает (он максимум)
        for x in range(64):
            assert not covers(63, x)

    def test_atom_decomposition_zero(self):
        assert atom_decomposition(0) == []

    def test_atom_decomposition_max(self):
        atoms = atom_decomposition(63)
        assert sorted(atoms) == [1, 2, 4, 8, 16, 32]

    def test_atom_decomposition_join(self):
        # join атомов = исходный элемент
        for x in range(64):
            atoms = atom_decomposition(x)
            result = 0
            for a in atoms:
                result |= a
            assert result == x


# ============================================================
# Многочлены
# ============================================================

class TestPolynomials:
    def test_rank_generating_function(self):
        rgf = rank_generating_function()
        assert rgf == [comb(6, k) for k in range(7)]

    def test_poincare_polynomial(self):
        pp = poincare_polynomial()
        assert pp == [comb(6, k) for k in range(7)]

    def test_characteristic_polynomial_values(self):
        cp = characteristic_polynomial()
        assert len(cp) == 7
        # (t-1)^6: коэффициент при t^6 = 1
        assert cp[0] == 1
        # при t^5 = -6
        assert cp[1] == -6
        # при t^0 = 1 ((-1)^6 = 1)
        assert cp[6] == 1

    def test_characteristic_polynomial_sum(self):
        # χ(1) = (1-1)^6 = 0
        cp = characteristic_polynomial()
        val_at_1 = sum(cp[i] * (1 ** (6 - i)) for i in range(7))
        assert val_at_1 == 0

    def test_zeta_polynomial_n1(self):
        # Z(1) = 64 (число элементов)
        assert zeta_polynomial(1) == 64

    def test_zeta_polynomial_n2(self):
        # Z(2) = число пар x ≤ y
        assert zeta_polynomial(2) == comparable_pairs_count()

    def test_zeta_polynomial_positive(self):
        for n in range(1, 5):
            assert zeta_polynomial(n) > 0

    def test_zeta_polynomial_n0(self):
        """Z(0) = 0 для n ≤ 0."""
        assert zeta_polynomial(0) == 0
        assert zeta_polynomial(-1) == 0


# ============================================================
# Глобальная структура
# ============================================================

class TestLatticeStructure:
    def test_lattice_diameter(self):
        assert lattice_diameter() == 6

    def test_comparable_pairs_count(self):
        # Должно быть равно ручному подсчёту
        manual = sum(1 for x in range(64) for y in range(64) if leq(x, y))
        assert comparable_pairs_count() == manual

    def test_incomparable_pairs_count(self):
        # Ручной подсчёт несравнимых пар {x,y} с x≠y
        manual = sum(1 for x in range(64) for y in range(x+1, 64)
                     if not leq(x, y) and not leq(y, x))
        assert incomparable_pairs_count() == manual

    def test_sublattice_boolean_2(self):
        # B₂ в B₆: C(6,2) = 15 подрешёток
        subs = sublattice_boolean(2)
        assert len(subs) == comb(6, 2)

    def test_sublattice_boolean_3(self):
        subs = sublattice_boolean(3)
        assert len(subs) == comb(6, 3)

    def test_sublattice_boolean_elements_count(self):
        # B_k подрешётка содержит 2^k элементов
        for k in range(1, 5):
            subs = sublattice_boolean(k)
            for mask, elements in subs:
                assert len(elements) == (1 << k)

    def test_mirsky_decomposition(self):
        md = mirsky_decomposition()
        assert len(md) == 7
        for k in range(7):
            assert sorted(md[k]) == sorted(rank_elements(k))

    def test_chain_partition_count(self):
        assert chain_partition_count() == comb(6, 3)

    def test_total_elements(self):
        all_count = sum(len(rank_elements(k)) for k in range(7))
        assert all_count == 64


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
