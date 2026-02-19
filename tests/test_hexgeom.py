"""Тесты для hexgeom — метрическая геометрия на Q6."""
import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from projects.hexgeom.hexgeom import (
    hamming, hamming_ball, hamming_sphere, ball_size, sphere_size,
    ball_intersection, ball_union_size,
    voronoi_cells, voronoi_sizes, delaunay_graph,
    metric_interval, gate, median, steiner_point, sum_of_distances, find_1_median,
    antipodal_pair, all_antipodal_pairs, equidistant_set,
    packing_number, covering_number_greedy, is_perfect_code,
    distance_distribution, dual_distance_distribution,
    minimum_distance, diameter_of_set,
    vertex_boundary, edge_boundary, isoperimetric_ratio,
    johnson_bound, singleton_bound, plotkin_bound,
)
from math import comb


# ── расстояние Хэмминга ──────────────────────────────────────────────────────

class TestHamming(unittest.TestCase):

    def test_hamming_zero(self):
        """d(h, h) = 0."""
        for h in [0, 1, 31, 63]:
            self.assertEqual(hamming(h, h), 0)

    def test_hamming_antipodal(self):
        """d(h, h⊕63) = 6 (все биты различны)."""
        for h in [0, 1, 42, 63]:
            self.assertEqual(hamming(h, h ^ 63), 6)

    def test_hamming_symmetric(self):
        """d(a, b) = d(b, a)."""
        self.assertEqual(hamming(7, 42), hamming(42, 7))

    def test_hamming_triangle_inequality(self):
        """d(a, c) ≤ d(a, b) + d(b, c)."""
        for a, b, c in [(0, 7, 63), (1, 2, 4), (15, 30, 60)]:
            self.assertLessEqual(hamming(a, c), hamming(a, b) + hamming(b, c))

    def test_hamming_single_bit(self):
        """Соседи на расстоянии 1."""
        for i in range(6):
            self.assertEqual(hamming(0, 1 << i), 1)


# ── шары и сферы Хэмминга ────────────────────────────────────────────────────

class TestBallSphere(unittest.TestCase):

    def test_ball_size_formula(self):
        """Размер шара = Σ C(6,k)."""
        for r in range(7):
            b = hamming_ball(0, r)
            self.assertEqual(len(b), ball_size(r))

    def test_sphere_size_formula(self):
        """Размер сферы = C(6, r)."""
        for r in range(7):
            s = hamming_sphere(0, r)
            self.assertEqual(len(s), sphere_size(r))
            self.assertEqual(len(s), comb(6, r))

    def test_ball_r0_is_singleton(self):
        """B(c, 0) = {c}."""
        for c in [0, 7, 42]:
            b = hamming_ball(c, 0)
            self.assertEqual(b, frozenset({c}))

    def test_ball_r6_is_all(self):
        """B(c, 6) = Q6."""
        for c in [0, 63]:
            b = hamming_ball(c, 6)
            self.assertEqual(b, frozenset(range(64)))

    def test_sphere_r0_is_center(self):
        """S(c, 0) = {c}."""
        self.assertEqual(hamming_sphere(5, 0), frozenset({5}))

    def test_sphere_r1_is_6_neighbors(self):
        """S(c, 1) состоит из 6 соседей."""
        self.assertEqual(len(hamming_sphere(0, 1)), 6)

    def test_ball_partitioned_by_spheres(self):
        """B(c, r) = ∪ S(c, k) для k=0..r."""
        for c in [0, 15]:
            for r in range(4):
                union = frozenset().union(*[hamming_sphere(c, k) for k in range(r + 1)])
                self.assertEqual(union, hamming_ball(c, r))

    def test_ball_center_invariant(self):
        """Размер шара не зависит от центра."""
        for r in range(4):
            sizes = [len(hamming_ball(c, r)) for c in [0, 7, 42, 63]]
            self.assertEqual(len(set(sizes)), 1)

    def test_ball_intersection_triangle(self):
        """B(0,2) ∩ B(7,2): вершины на расстоянии ≤2 от обоих."""
        inter = ball_intersection(0, 2, 7, 2)
        for h in inter:
            self.assertLessEqual(hamming(h, 0), 2)
            self.assertLessEqual(hamming(h, 7), 2)

    def test_ball_union_size(self):
        """Размер объединения шаров ≤ сумме."""
        centers = [0, 7, 42]
        union = ball_union_size(centers, 1)
        total = sum(len(hamming_ball(c, 1)) for c in centers)
        self.assertLessEqual(union, total)


# ── диаграмма Вороного ───────────────────────────────────────────────────────

class TestVoronoi(unittest.TestCase):

    def test_voronoi_partition(self):
        """Ячейки Вороного образуют разбиение Q6."""
        centers = [0, 21, 42, 63]
        cells = voronoi_cells(centers)
        all_verts = set()
        for verts in cells.values():
            self.assertTrue(all_verts.isdisjoint(verts))
            all_verts |= set(verts)
        self.assertEqual(all_verts, set(range(64)))

    def test_voronoi_center_in_own_cell(self):
        """Каждый центр лежит в своей ячейке."""
        centers = [0, 21, 42, 63]
        cells = voronoi_cells(centers)
        for c in centers:
            self.assertIn(c, cells[c])

    def test_voronoi_sizes_sum(self):
        """Сумма размеров ячеек = 64."""
        centers = [0, 7, 56, 63]
        sizes = voronoi_sizes(centers)
        self.assertEqual(sum(sizes.values()), 64)

    def test_voronoi_single_center(self):
        """Один центр: ячейка = Q6."""
        cells = voronoi_cells([0])
        self.assertEqual(cells[0], frozenset(range(64)))

    def test_delaunay_graph_nonempty(self):
        """Граф Делоне непустой: d(h,0)=d(h,63)=3 для yang_count=3."""
        # 0=000000, 63=111111: любая вершина с yang_count=3 равноудалена
        centers = [0, 63]
        edges = delaunay_graph(centers)
        self.assertGreater(len(edges), 0)

    def test_delaunay_edge_format(self):
        """Рёбра Делоне — упорядоченные пары (c1 < c2)."""
        centers = [0, 7, 56, 63]
        for c1, c2 in delaunay_graph(centers):
            self.assertLess(c1, c2)
            self.assertIn(c1, centers)
            self.assertIn(c2, centers)


# ── геодезические интервалы и медиана ────────────────────────────────────────

class TestIntervalMedian(unittest.TestCase):

    def test_interval_endpoints_included(self):
        """u, v ∈ I(u, v)."""
        for u, v in [(0, 63), (7, 42), (0, 0)]:
            I = metric_interval(u, v)
            self.assertIn(u, I)
            self.assertIn(v, I)

    def test_interval_size_equals_ball_product(self):
        """|I(u, v)| = C(d, k) последовательно = 2^d для d-куба."""
        # I(0, 63) = весь Q6 (d=6, все вершины лежат на кратчайшем пути)
        I = metric_interval(0, 63)
        self.assertEqual(I, frozenset(range(64)))

    def test_interval_single_point(self):
        """I(h, h) = {h}."""
        for h in [0, 42]:
            self.assertEqual(metric_interval(h, h), frozenset({h}))

    def test_interval_adjacent_vertices(self):
        """I(h, n) для соседей h,n на расстоянии 1: только {h, n}."""
        I = metric_interval(0, 1)
        self.assertEqual(I, frozenset({0, 1}))

    def test_gate_forward_step(self):
        """Все вершины gate(u, v) ближе к v на 1."""
        u, v = 0, 63
        for g in gate(u, v):
            self.assertEqual(hamming(g, v), hamming(u, v) - 1)

    def test_median_three_points(self):
        """Медиана трёх точек — побитовое большинство."""
        # median(0, 1, 3) = побитово: bit0: 0+1+1=2>1→1; bit1: 0+0+1=1=1→0
        # 0=000000, 1=000001, 3=000011
        m = median([0, 1, 3])
        self.assertEqual(m, 1)  # bit0=1,1,1→1; bit1=0,0,1→0: m=001=1

    def test_median_minimizes_distance(self):
        """Медиана минимизирует сумму расстояний."""
        points = [0, 7, 14, 21]
        m = median(points)
        sm = sum_of_distances(m, points)
        for h in range(64):
            self.assertGreaterEqual(sum_of_distances(h, points), sm)

    def test_steiner_equals_median(self):
        """steiner_point = median для (Z₂)ⁿ."""
        pts = [3, 5, 12, 20]
        self.assertEqual(steiner_point(pts), median(pts))

    def test_find_1_median_correct(self):
        """find_1_median возвращает вершину с минимальной суммой расстояний."""
        points = [0, 1, 3, 7, 15]
        h_opt, sm_opt = find_1_median(points)
        for h in range(64):
            self.assertGreaterEqual(sum_of_distances(h, points), sm_opt)


# ── антиподальные пары и равноудалённые множества ────────────────────────────

class TestAntipodal(unittest.TestCase):

    def test_antipodal_pair_distance(self):
        """d(h, h⊕63) = 6."""
        for h in range(10):
            a, b = antipodal_pair(h)
            self.assertEqual(hamming(a, b), 6)

    def test_antipodal_pair_xor(self):
        """b = a ⊕ 63."""
        a, b = antipodal_pair(7)
        self.assertEqual(a ^ b, 63)

    def test_all_antipodal_pairs_count(self):
        """Ровно 32 антиподальные пары."""
        pairs = all_antipodal_pairs()
        self.assertEqual(len(pairs), 32)

    def test_all_antipodal_covers_q6(self):
        """Объединение всех пар = Q6."""
        pairs = all_antipodal_pairs()
        all_verts = set()
        for a, b in pairs:
            all_verts.add(a)
            all_verts.add(b)
        self.assertEqual(all_verts, set(range(64)))

    def test_equidistant_set_distance_correct(self):
        """Все пары в equidistant_set имеют нужное расстояние."""
        for d in [2, 4]:
            S = equidistant_set(d, seed=1)
            for a in S:
                for b in S:
                    if a != b:
                        self.assertEqual(hamming(a, b), d)

    def test_equidistant_set_distance_2_nonempty(self):
        """Равноудалённое множество с d=2 непусто."""
        S = equidistant_set(2)
        self.assertGreater(len(S), 0)


# ── пакинг и покрытие ────────────────────────────────────────────────────────

class TestPackingCovering(unittest.TestCase):

    def test_packing_disjoint_balls(self):
        """Шары пакинга не пересекаются."""
        for r in range(3):
            centers = packing_number(r)
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    self.assertGreater(hamming(centers[i], centers[j]), 2 * r)

    def test_covering_covers_all(self):
        """Покрытие покрывает все 64 вершины."""
        for r in range(4):
            centers = covering_number_greedy(r)
            covered = set()
            for c in centers:
                covered |= hamming_ball(c, r)
            self.assertEqual(covered, set(range(64)))

    def test_perfect_code_r1(self):
        """Единственный совершенный 1-код на Q6: нет (объём шара 7, 64/7 не целое)."""
        # 64 / ball_size(1) = 64/7 — не целое, поэтому совершенного кода нет
        self.assertFalse(64 % ball_size(1) == 0)

    def test_perfect_code_trivial(self):
        """is_perfect_code работает: {весь Q6} с r=0."""
        self.assertTrue(is_perfect_code(list(range(64)), 0))

    def test_packing_r0_is_all(self):
        """Пакинг с r=0: все 64 вершины (шары — точки)."""
        centers = packing_number(0)
        self.assertEqual(len(centers), 64)


# ── распределение расстояний ─────────────────────────────────────────────────

class TestDistanceDistribution(unittest.TestCase):

    def test_dist_dist_sums_to_m(self):
        """Σ A[d] = |C| (нормировка: Σ A[d] = |C|/|C| × |C| = |C|... нет, Σ = |C|)."""
        code = [0, 7, 42, 63]
        A = distance_distribution(code)
        # Σ A[d] = |C| / |C| × Σ_d count[d] = Σ_d count[d] / |C|
        # При нормировке /|C|: A[0]=1 + ... = |C|? Нет:
        # A[d] = count[d] / |C|, A[0] = |C|/|C| = 1
        # Σ_d A[d] = (Σ_d count[d]) / |C| = |C|² / |C| = |C|
        self.assertAlmostEqual(sum(A), len(code), places=10)

    def test_dist_dist_a0_equals_1(self):
        """A[0] = 1 (каждое кодовое слово на расстоянии 0 от себя)."""
        code = [0, 21, 42, 63]
        A = distance_distribution(code)
        self.assertAlmostEqual(A[0], 1.0, places=10)

    def test_minimum_distance_correct(self):
        """minimum_distance корректен."""
        # d(0, 7) = popcount(7) = 3
        code = [0, 7, 56]
        self.assertEqual(minimum_distance(code), 3)

    def test_diameter_of_set_correct(self):
        """diameter_of_set: max попарное расстояние."""
        S = [0, 63]
        self.assertEqual(diameter_of_set(S), 6)

    def test_diameter_of_singleton(self):
        """diameter({h}) = 0."""
        self.assertEqual(diameter_of_set([42]), 0)

    def test_dual_distance_distribution_length(self):
        """dual_distance_distribution имеет длину 7."""
        code = [0, 21, 42, 63]
        B = dual_distance_distribution(code)
        self.assertEqual(len(B), 7)


# ── изоперметрия ─────────────────────────────────────────────────────────────

class TestIsoperimetric(unittest.TestCase):

    def test_boundary_empty(self):
        """Граница пустого множества пуста."""
        self.assertEqual(vertex_boundary(set()), frozenset())

    def test_boundary_all_q6(self):
        """Граница всего Q6 пуста."""
        self.assertEqual(vertex_boundary(set(range(64))), frozenset())

    def test_boundary_single(self):
        """Граница {0} = 6 соседей."""
        b = vertex_boundary({0})
        self.assertEqual(len(b), 6)

    def test_edge_boundary_bisection(self):
        """Для бисекции (32 вершины с bit5=0 vs bit5=1): 32 пересекающих ребра."""
        S = frozenset(h for h in range(64) if not (h >> 5 & 1))
        eb = edge_boundary(S)
        self.assertEqual(len(eb), 32)

    def test_isoperimetric_ratio_nonnegative(self):
        """Изоперметрическое отношение ≥ 0."""
        S = {0, 1, 3}
        self.assertGreaterEqual(isoperimetric_ratio(S), 0.0)


# ── границы кодов ─────────────────────────────────────────────────────────────

class TestBounds(unittest.TestCase):

    def test_singleton_bound_d1(self):
        """Singleton(6, 1) = 2^6 = 64."""
        self.assertEqual(singleton_bound(1), 64)

    def test_singleton_bound_d6(self):
        """Singleton(6, 6) = 2^1 = 2 (только антиподальная пара)."""
        self.assertEqual(singleton_bound(6), 2)

    def test_plotkin_bound_d4(self):
        """Plotkin(6, 4): 2×4/(2×4-6) = 8/2 = 4."""
        self.assertEqual(plotkin_bound(4), 4)

    def test_plotkin_bound_small_d(self):
        """Plotkin не применима при d ≤ 3 (n/2 = 3)."""
        self.assertIsNone(plotkin_bound(3))

    def test_johnson_bound_d1_is_64(self):
        """Johnson(6,1) = 64."""
        self.assertEqual(johnson_bound(1), 64)

    def test_bounds_positive(self):
        """Все границы положительны и ≤ 64."""
        for d in range(1, 7):
            sb = singleton_bound(d)
            jb = johnson_bound(d)
            self.assertGreater(sb, 0)
            self.assertGreater(jb, 0)
            self.assertLessEqual(sb, 64)
            self.assertLessEqual(jb, 64)

    def test_bounds_decrease_with_d(self):
        """Singleton bound строго убывает с d."""
        for d in range(1, 6):
            self.assertGreater(singleton_bound(d), singleton_bound(d + 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
