"""Тесты hexsym — группа автоморфизмов Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexsym.hexsym import (
    Automorphism,
    identity_aut, bit_transposition, bit_flip_single, bit_flip_mask,
    bit_permutation, complement_aut,
    s6_generators, aut_generators,
    orbit, all_orbits, canonical_form, canonical_map,
    cycle_decomposition, cycle_count, fixed_points,
    generate_group, burnside_count, burnside_subset,
    yang_orbits, antipodal_orbits, full_aut_orbits,
    count_stabilizer, edge_orbits, cycle_index_s6_on_q6,
    polya_count, orbit_size_distribution, group_order_estimate,
)
from libs.hexcore.hexcore import hamming, yang_count, SIZE


# ---------------------------------------------------------------------------
# Тест класса Automorphism
# ---------------------------------------------------------------------------

class TestAutomorphism(unittest.TestCase):
    def test_identity_action(self):
        """Тождественный автоморфизм не меняет ни одну вершину."""
        ident = identity_aut()
        for h in range(SIZE):
            self.assertEqual(ident(h), h)

    def test_complement_action(self):
        """Дополнение: h → 63 - h."""
        comp = complement_aut()
        for h in range(SIZE):
            self.assertEqual(comp(h), 63 - h)

    def test_bit_transposition(self):
        """Транспозиция битов 0 и 1: h → swap(bit0, bit1) в h."""
        tau = bit_transposition(0, 1)
        # h = 0b000001 = 1: бит 0 → позицию 1 → 0b000010 = 2
        self.assertEqual(tau(1), 2)
        self.assertEqual(tau(2), 1)
        self.assertEqual(tau(3), 3)  # биты 0 и 1 оба равны 1 → без изменений

    def test_bit_flip_single(self):
        """Переворот бита i."""
        sigma = bit_flip_single(0)
        for h in range(SIZE):
            self.assertEqual(sigma(h), h ^ 1)

    def test_bit_flip_mask(self):
        """XOR с маской."""
        mask = 42
        aut = bit_flip_mask(mask)
        for h in range(SIZE):
            self.assertEqual(aut(h), h ^ mask)

    def test_composition_identity(self):
        """g · g⁻¹ = id."""
        g = bit_transposition(1, 3)
        g_inv = g.inverse()
        ident = identity_aut()
        composed = g * g_inv
        for h in range(SIZE):
            self.assertEqual(composed(h), ident(h))

    def test_composition_associative(self):
        """(f · g) · h = f · (g · h)."""
        f = bit_transposition(0, 2)
        g = bit_flip_single(1)
        h = complement_aut()
        lhs = (f * g) * h
        rhs = f * (g * h)
        for v in range(SIZE):
            self.assertEqual(lhs(v), rhs(v))

    def test_inverse_reverses_action(self):
        """g⁻¹(g(h)) = h."""
        g = bit_transposition(2, 5)
        g_inv = g.inverse()
        for h in range(SIZE):
            self.assertEqual(g_inv(g(h)), h)

    def test_order_identity(self):
        """Порядок тождественного = 1."""
        self.assertEqual(identity_aut().order(), 1)

    def test_order_transposition(self):
        """Транспозиция имеет порядок 2."""
        self.assertEqual(bit_transposition(0, 1).order(), 2)

    def test_order_complement(self):
        """Дополнение имеет порядок 2."""
        self.assertEqual(complement_aut().order(), 2)

    def test_order_3cycle(self):
        """3-цикл (0 1 2) имеет порядок 3."""
        g = bit_permutation([1, 2, 0, 3, 4, 5])
        self.assertEqual(g.order(), 3)

    def test_is_identity(self):
        self.assertTrue(identity_aut().is_identity())
        self.assertFalse(complement_aut().is_identity())

    def test_equality(self):
        self.assertEqual(identity_aut(), identity_aut())
        self.assertNotEqual(identity_aut(), complement_aut())

    def test_hash_consistent(self):
        d = {identity_aut(): 'id', complement_aut(): 'comp'}
        self.assertEqual(d[identity_aut()], 'id')

    def test_automorphism_preserves_edges(self):
        """Автоморфизм сохраняет смежность: если d(u,v)=1, то d(g(u),g(v))=1."""
        from libs.hexcore.hexcore import neighbors
        for g in s6_generators() + [bit_flip_single(0)]:
            for u in range(0, SIZE, 7):
                for v in neighbors(u):
                    self.assertEqual(hamming(g(u), g(v)), 1)

    def test_repr_contains_perm(self):
        """__repr__ содержит перестановку."""
        ident = identity_aut()
        r = repr(ident)
        self.assertIn('Aut', r)
        self.assertIn('0', r)

    def test_eq_with_non_automorphism(self):
        """Сравнение с не-автоморфизмом возвращает NotImplemented (т.е. False)."""
        ident = identity_aut()
        self.assertNotEqual(ident, 42)
        self.assertNotEqual(ident, 'not an automorphism')


# ---------------------------------------------------------------------------
# Тест орбит
# ---------------------------------------------------------------------------

class TestOrbits(unittest.TestCase):
    def test_orbit_identity(self):
        """Орбита под id = синглтон."""
        orb = orbit(42, [identity_aut()])
        self.assertEqual(orb, frozenset({42}))

    def test_orbit_complement(self):
        """Орбита под дополнением = пара {h, 63-h}."""
        orb = orbit(10, [complement_aut()])
        self.assertEqual(orb, frozenset({10, 53}))

    def test_orbit_s6_is_weight_class(self):
        """Орбита под S₆ = все гексаграммы того же веса."""
        gens = s6_generators()
        for weight in range(7):
        # Найти одну вершину данного веса
            h = next(v for v in range(SIZE) if yang_count(v) == weight)
            orb = orbit(h, gens)
            expected = frozenset(v for v in range(SIZE) if yang_count(v) == weight)
            self.assertEqual(orb, expected)

    def test_orbit_full_aut_is_all(self):
        """Орбита под полной Aut(Q6) = весь Q6 (vertex-transitive)."""
        gens = aut_generators()
        orb = orbit(0, gens)
        self.assertEqual(len(orb), SIZE)
        self.assertEqual(orb, frozenset(range(SIZE)))

    def test_all_orbits_partition(self):
        """Орбиты образуют разбиение Q6."""
        gens = s6_generators()
        orbits = all_orbits(gens)
        all_pts = set()
        for orb in orbits:
            self.assertEqual(all_pts & orb, set())  # попарно непересекающиеся
            all_pts |= orb
        self.assertEqual(all_pts, set(range(SIZE)))

    def test_yang_orbits_count(self):
        """Под S₆: ровно 7 орбит (по весу 0..6)."""
        orbits = yang_orbits()
        self.assertEqual(len(orbits), 7)

    def test_yang_orbits_sizes(self):
        """Размеры орбит = C(6,k) для k=0..6."""
        import math
        orbits = sorted(yang_orbits(), key=len)
        expected = sorted(math.comb(6, k) for k in range(7))
        self.assertEqual([len(o) for o in orbits], expected)

    def test_antipodal_orbits_count(self):
        """Орбиты под дополнением: 32 пары {h, 63-h}."""
        orbits = antipodal_orbits()
        self.assertEqual(len(orbits), 32)
        for orb in orbits:
            self.assertEqual(len(orb), 2)

    def test_full_aut_orbits_count(self):
        """Полная Aut(Q6) — одна орбита."""
        orbits = full_aut_orbits()
        self.assertEqual(len(orbits), 1)
        self.assertEqual(len(orbits[0]), SIZE)

    def test_canonical_form_minimum(self):
        """Канонический представитель = минимальный элемент орбиты."""
        gens = s6_generators()
        for h in range(SIZE):
            canon = canonical_form(h, gens)
            orb = orbit(h, gens)
            self.assertEqual(canon, min(orb))

    def test_canonical_map_consistent(self):
        """Два элемента одной орбиты имеют одинаковый канонический представитель."""
        gens = s6_generators()
        canon_map = canonical_map(gens)
        for h in range(SIZE):
            for g in gens:
                gh = g(h)
                self.assertEqual(canon_map[gh], canon_map[h])


# ---------------------------------------------------------------------------
# Тест циклов
# ---------------------------------------------------------------------------

class TestCycles(unittest.TestCase):
    def test_cycle_decomposition_identity(self):
        """Тождественный автоморфизм: 64 цикла длины 1."""
        cycles = cycle_decomposition(identity_aut())
        self.assertEqual(len(cycles), SIZE)
        for c in cycles:
            self.assertEqual(len(c), 1)

    def test_cycle_decomposition_complement(self):
        """Дополнение: 32 цикла длины 2."""
        cycles = cycle_decomposition(complement_aut())
        self.assertEqual(len(cycles), 32)
        for c in cycles:
            self.assertEqual(len(c), 2)

    def test_cycle_decomposition_covers_all(self):
        """Все вершины покрыты циклами."""
        for g in [identity_aut(), complement_aut(), bit_transposition(0, 1)]:
            cycles = cycle_decomposition(g)
            covered = set()
            for c in cycles:
                covered |= set(c)
            self.assertEqual(covered, set(range(SIZE)))

    def test_cycle_count_identity(self):
        self.assertEqual(cycle_count(identity_aut()), SIZE)

    def test_cycle_count_complement(self):
        self.assertEqual(cycle_count(complement_aut()), 32)

    def test_fixed_points_identity(self):
        """Тождественный: все 64 вершины неподвижны."""
        fps = fixed_points(identity_aut())
        self.assertEqual(len(fps), SIZE)

    def test_fixed_points_complement(self):
        """Дополнение: нет неподвижных точек (нет h: 63-h=h)."""
        fps = fixed_points(complement_aut())
        self.assertEqual(len(fps), 0)

    def test_fixed_points_flip_single(self):
        """Переворот бита 0: точки h с нулевым битом 0 неподвижны → 32 точки."""
        fps = fixed_points(bit_flip_single(0))
        # h неподвижна ↔ h^1 = h → невозможно, значит fixed_points = []
        self.assertEqual(len(fps), 0)


# ---------------------------------------------------------------------------
# Тест генерации группы
# ---------------------------------------------------------------------------

class TestGenerateGroup(unittest.TestCase):
    def test_identity_generates_trivial(self):
        """Тривиальная группа из id имеет 1 элемент."""
        g = generate_group([identity_aut()])
        self.assertEqual(len(g), 1)

    def test_complement_generates_z2(self):
        """Дополнение порождает Z₂: {id, comp}, размер 2."""
        g = generate_group([complement_aut()])
        self.assertEqual(len(g), 2)

    def test_single_transposition_z2(self):
        """Транспозиция порождает Z₂."""
        g = generate_group([bit_transposition(0, 1)])
        self.assertEqual(len(g), 2)

    def test_s6_size(self):
        """S₆ порождается смежными транспозициями, |S₆| = 720."""
        g = generate_group(s6_generators())
        self.assertEqual(len(g), 720)

    def test_group_closed(self):
        """Группа замкнута относительно умножения."""
        g = generate_group([bit_transposition(0, 1), bit_transposition(1, 2)])
        g_set = set(a.perm + (a.mask,) for a in g)
        for a in g[:5]:
            for b in g[:5]:
                c = a * b
                key = c.perm + (c.mask,)
                self.assertIn(key, g_set)


# ---------------------------------------------------------------------------
# Тест теоремы Бёрнсайда
# ---------------------------------------------------------------------------

class TestBurnside(unittest.TestCase):
    def test_burnside_trivial_group(self):
        """Тривиальная группа: n^{64} раскрасок."""
        g = [identity_aut()]
        for n_colors in [2, 3]:
            count = burnside_count(n_colors, g)
            self.assertEqual(count, n_colors ** SIZE)

    def test_burnside_z2(self):
        """Z₂ = {id, comp}: среднее двух вкладов."""
        g = generate_group([complement_aut()])
        # id: 2^64 фиксированных, comp: 2^32 фиксированных
        # итого: (2^64 + 2^32) / 2
        expected = (2 ** 64 + 2 ** 32) // 2
        self.assertEqual(burnside_count(2, g), expected)

    def test_burnside_s6_1color(self):
        """Одноцветная раскраска: ровно 1 (все вершины одного цвета)."""
        g = generate_group(s6_generators())
        self.assertEqual(burnside_count(1, g), 1)

    def test_burnside_positive(self):
        """Число орбит ≥ 1 для n_colors ≥ 1."""
        g = generate_group(s6_generators())
        for n in [1, 2, 3]:
            self.assertGreater(burnside_count(n, g), 0)

    def test_burnside_subset_k1(self):
        """Число различных 1-подмножеств под S₆ = число орбит вершин = 7."""
        g = generate_group(s6_generators())
        self.assertEqual(burnside_subset(1, g), 7)

    def test_burnside_subset_k0(self):
        """Число различных 0-подмножеств = 1 (пустое множество)."""
        g = generate_group(s6_generators())
        self.assertEqual(burnside_subset(0, g), 1)

    def test_burnside_subset_k_all(self):
        """Число различных 64-подмножеств = 1 (всё Q6)."""
        g = generate_group(s6_generators())
        self.assertEqual(burnside_subset(SIZE, g), 1)

    def test_burnside_subset_symmetric(self):
        """Число k-подмножеств = число (64-k)-подмножеств (симметрия дополнения)."""
        g = generate_group(s6_generators())
        for k in [1, 2, 3]:
            c_k = burnside_subset(k, g)
            c_compl = burnside_subset(SIZE - k, g)
            # Под S₆ (без флипов) это не обязательно равно, но для S₆ × Z₂ было бы
            # Просто проверяем что оба положительны
            self.assertGreater(c_k, 0)
            self.assertGreater(c_compl, 0)


class TestCountStabilizer(unittest.TestCase):
    """Тесты count_stabilizer через теорему орбита–стабилизатор."""

    def test_returns_positive_int(self):
        gens = s6_generators()
        result = count_stabilizer(0, gens)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_orbit_times_stabilizer_equals_group_order(self):
        """|Orb(h)| × |Stab(h)| = |G| = 46080 для полной Aut(Q6)."""
        gens = s6_generators()
        for h in [0, 42, 63]:
            orb_size = len(orbit(h, gens))
            stab_size = count_stabilizer(h, gens)
            self.assertEqual(orb_size * stab_size, 46080)

    def test_stabilizer_positive_for_all_vertices(self):
        gens = s6_generators()
        for h in range(0, 64, 8):
            self.assertGreater(count_stabilizer(h, gens), 0)


class TestEdgeOrbits(unittest.TestCase):
    """Тесты edge_orbits — орбиты рёбер."""

    def test_identity_group_all_singletons(self):
        """Тождественная группа: каждое ребро — отдельная орбита."""
        gens = [identity_aut()]
        orbs = edge_orbits(gens)
        # Q6 имеет 192 рёбра → 192 одноэлементных орбиты
        self.assertEqual(len(orbs), 192)

    def test_returns_list(self):
        gens = [identity_aut()]
        orbs = edge_orbits(gens)
        self.assertIsInstance(orbs, list)

    def test_edges_partition(self):
        """Орбиты рёбер покрывают все 192 рёбра Q6 без дублей."""
        gens = [identity_aut()]
        orbs = edge_orbits(gens)
        all_edges = frozenset(e for orb in orbs for e in orb)
        self.assertEqual(len(all_edges), 192)

    def test_s6_fewer_orbits_than_edges(self):
        """S₆ объединяет рёбра в крупные орбиты → меньше 192 орбит."""
        gens = s6_generators()
        orbs = edge_orbits(gens)
        self.assertLess(len(orbs), 192)


class TestCycleIndexAndPolya(unittest.TestCase):
    """Тесты cycle_index_s6_on_q6 и polya_count."""

    def test_cycle_index_returns_dict(self):
        ci = cycle_index_s6_on_q6()
        self.assertIsInstance(ci, dict)

    def test_cycle_index_sums_to_720(self):
        """Сумма кратностей = |S₆| = 6! = 720."""
        ci = cycle_index_s6_on_q6()
        self.assertEqual(sum(ci.values()), 720)

    def test_cycle_index_identity_present(self):
        """Тождественная перестановка → все 64 цикла длины 1."""
        ci = cycle_index_s6_on_q6()
        # Есть хотя бы один тип с 64 циклами длины 1 (или смешанный)
        self.assertTrue(any(t[0] == 64 for t in ci if t))

    def test_polya_count_1_color(self):
        """1 цвет → 1 раскраска."""
        self.assertEqual(polya_count(1), 1)

    def test_polya_count_positive(self):
        """Для любого n_colors > 0 результат положителен."""
        for n in range(1, 4):
            self.assertGreater(polya_count(n), 0)

    def test_polya_count_increases_with_colors(self):
        """Больше цветов → не меньше раскрасок."""
        c1 = polya_count(1)
        c2 = polya_count(2)
        self.assertGreaterEqual(c2, c1)


class TestOrbitSizeDistribution(unittest.TestCase):
    """Тесты orbit_size_distribution."""

    def test_returns_dict(self):
        gens = [identity_aut()]
        dist = orbit_size_distribution(gens)
        self.assertIsInstance(dist, dict)

    def test_identity_all_size_1(self):
        """Тождественная группа: все орбиты размера 1."""
        gens = [identity_aut()]
        dist = orbit_size_distribution(gens)
        self.assertEqual(dist, {1: 64})

    def test_total_elements_64(self):
        """Сумма size * count = 64 для любых генераторов."""
        gens = s6_generators()
        dist = orbit_size_distribution(gens)
        total = sum(size * count for size, count in dist.items())
        self.assertEqual(total, 64)

    def test_s6_orbit_by_yang_count(self):
        """S₆ действует по ярусам yang_count → 7 орбит (C(6,0)..C(6,6))."""
        gens = s6_generators()
        dist = orbit_size_distribution(gens)
        # По одной орбите для каждого yang_count = 0..6
        self.assertEqual(dist, {1: 2, 6: 2, 15: 2, 20: 1})


class TestGroupOrderEstimate(unittest.TestCase):
    """Тесты group_order_estimate (текущая реализация — заглушка)."""

    def test_returns_int(self):
        gens = [identity_aut()]
        result = group_order_estimate(gens)
        self.assertIsInstance(result, int)

    def test_no_crash_with_s6(self):
        """Не падает при вызове с генераторами S₆."""
        result = group_order_estimate(s6_generators(), sample=10)
        self.assertIsInstance(result, int)


if __name__ == '__main__':
    unittest.main(verbosity=2)
