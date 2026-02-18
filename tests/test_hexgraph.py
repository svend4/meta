"""Тесты hexgraph — граф-теоретический анализ Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexgraph.hexgraph import (
    Subgraph, q6_full, q6_layer, q6_ball, induced_subgraph,
    largest_eigenvalue, spectrum_approx,
    bit_permutation_automorphism, xor_automorphism, count_automorphisms,
    distance_graph, analyze,
)
from libs.hexcore.hexcore import neighbors, yang_count, hamming, SIZE


class TestSubgraphBasic(unittest.TestCase):
    def test_q6_full_order(self):
        g = q6_full()
        self.assertEqual(g.order(), 64)

    def test_q6_full_size(self):
        g = q6_full()
        self.assertEqual(g.size(), 192)

    def test_q6_full_repr(self):
        g = q6_full()
        r = repr(g)
        self.assertIn('64', r)
        self.assertIn('192', r)

    def test_induced_subgraph_edges(self):
        """Рёбра = только Q6-рёбра между вершинами подмножества."""
        g = induced_subgraph({0, 1, 2, 4})
        for u, v in g.edges:
            self.assertIn(u, g.vertices)
            self.assertIn(v, g.vertices)
            self.assertEqual(hamming(u, v), 1)

    def test_contains(self):
        g = induced_subgraph({5, 10, 15})
        self.assertIn(5, g)
        self.assertNotIn(42, g)

    def test_degree_q6(self):
        """Каждая вершина Q6 имеет степень 6."""
        g = q6_full()
        for h in range(SIZE):
            self.assertEqual(g.degree(h), 6)

    def test_degree_sequence_q6(self):
        g = q6_full()
        ds = g.degree_sequence()
        self.assertEqual(ds, [6] * 64)

    def test_adjacency_matrix_symmetry(self):
        g = induced_subgraph({0, 1, 2, 3})
        mat = g.adjacency_matrix()
        n = len(mat)
        for i in range(n):
            for j in range(n):
                self.assertEqual(mat[i][j], mat[j][i])

    def test_adjacency_matrix_diagonal_zero(self):
        g = induced_subgraph({0, 1, 2, 3})
        mat = g.adjacency_matrix()
        for i in range(len(mat)):
            self.assertEqual(mat[i][i], 0)


class TestConnectivity(unittest.TestCase):
    def test_q6_connected(self):
        self.assertTrue(q6_full().is_connected())

    def test_single_vertex_connected(self):
        self.assertTrue(induced_subgraph({42}).is_connected())

    def test_disconnected(self):
        # 0 и 42 не соединены напрямую, но через других вершин
        # Зато {0} и {42} как отдельный подграф (без промежуточных) — несвязны
        g = induced_subgraph({0, 42})
        self.assertFalse(g.is_connected())

    def test_connected_components_q6(self):
        comps = q6_full().connected_components()
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0], frozenset(range(SIZE)))

    def test_connected_components_two(self):
        g = induced_subgraph({0, 1, 42, 43})
        comps = g.connected_components()
        # 0-1 связаны (hamming=1), 42-43 связаны; 0-42 и 1-42 не Q6-соседи
        # 42=101010, 43=101011 → hamming(42,43)=1 ✓
        # 0-42: hamming=3 → не рёбра
        # 0-1: hamming=1 → ребро
        for comp in comps:
            self.assertIn(len(comp), (1, 2, 3, 4))
        total = sum(len(c) for c in comps)
        self.assertEqual(total, 4)

    def test_diameter_q6(self):
        self.assertEqual(q6_full().diameter(), 6)

    def test_diameter_disconnected(self):
        g = induced_subgraph({0, 42})
        self.assertEqual(g.diameter(), -1)

    def test_girth_q6(self):
        """Гиперкуб Q6 имеет обхват 4 (наименьший цикл — квадрат)."""
        g = q6_full()
        self.assertEqual(g.girth(), 4)

    def test_girth_path_no_cycle(self):
        """Путь 0→1→3 не имеет циклов."""
        g = induced_subgraph({0, 1, 3})
        self.assertEqual(g.girth(), float('inf'))


class TestColoring(unittest.TestCase):
    def test_q6_bipartite(self):
        self.assertTrue(q6_full().is_bipartite())

    def test_two_coloring_valid(self):
        g = q6_full()
        col = g.two_coloring()
        self.assertIsNotNone(col)
        # Проверить, что соседи имеют разные цвета
        for u, v in g.edges:
            self.assertNotEqual(col[u], col[v])

    def test_two_coloring_yang_parity(self):
        """Цвет = четность ян-счёта."""
        g = q6_full()
        col = g.two_coloring()
        for h, c in col.items():
            self.assertEqual(c, yang_count(h) % 2)

    def test_chromatic_number_q6(self):
        g = q6_full()
        self.assertEqual(g.chromatic_number_upper(), 2)


class TestIndependenceAndCover(unittest.TestCase):
    def test_independence_number_q6(self):
        """Максимальное независимое множество Q6 = 32 (одна доля двудольного графа)."""
        g = q6_full()
        self.assertEqual(g.independence_number(), 32)

    def test_max_independent_set_is_independent(self):
        g = q6_full()
        mis = g.max_independent_set()
        for v in mis:
            for nb in neighbors(v):
                if nb in g.vertices:
                    self.assertNotIn(nb, mis)

    def test_vertex_cover_number_q6(self):
        """По теореме Кёнига: vertex_cover + independence = n."""
        g = q6_full()
        self.assertEqual(g.vertex_cover_number() + g.independence_number(), g.order())

    def test_vertex_cover_is_cover(self):
        """Каждое ребро должно быть покрыто."""
        g = induced_subgraph(set(range(16)))
        vc = g.min_vertex_cover()
        for u, v in g.edges:
            self.assertTrue(u in vc or v in vc)

    def test_domination_q6(self):
        """Число доминирования Q6 ≤ 64/7 ≈ 16 (жадный алгоритм)."""
        g = q6_full()
        dom = g.domination_number()
        self.assertLessEqual(dom, 16)
        self.assertGreater(dom, 0)


class TestClique(unittest.TestCase):
    def test_clique_number_q6(self):
        """Максимальная клика в гиперкубе — ребро (размер 2)."""
        g = q6_full()
        self.assertEqual(g.clique_number(), 2)

    def test_clique_small(self):
        """В треугольнике (если бы существовал) клика = 3.
        В Q6 треугольников нет (двудольный), поэтому проверяем клику=2."""
        g = induced_subgraph({0, 1, 2})
        # 0-1 смежны (hamming=1), 0-2 смежны (hamming=1), 1-2 не смежны (hamming=2)
        self.assertEqual(g.clique_number(), 2)


class TestHamiltonian(unittest.TestCase):
    def test_hamiltonian_path_small(self):
        """Путь Грея на первых 4 вершинах: 0→1→3→2."""
        g = induced_subgraph({0, 1, 2, 3})
        path = g.find_hamiltonian_path(start=0)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 4)
        self.assertEqual(set(path), {0, 1, 2, 3})
        for i in range(len(path) - 1):
            self.assertEqual(hamming(path[i], path[i + 1]), 1)

    def test_hamiltonian_cycle_small(self):
        """Цикл 0→1→3→2→0."""
        g = induced_subgraph({0, 1, 2, 3})
        cycle = g.find_hamiltonian_cycle()
        self.assertIsNotNone(cycle)
        self.assertEqual(cycle[0], cycle[-1])
        self.assertEqual(len(set(cycle)), 4)

    def test_no_hamiltonian_path_isolated(self):
        """Изолированная вершина — нет гамильтонова пути через все 3."""
        g = induced_subgraph({0, 42, 63})   # 0-42 и 0-63 не смежны (hamming>1)
        path = g.find_hamiltonian_path(start=0)
        # Ни одна пара из {0,42,63} не является Q6-соседями
        self.assertIsNone(path)


class TestLayers(unittest.TestCase):
    def test_layer_sizes(self):
        """Слой yang=k имеет C(6,k) вершин."""
        import math
        for k in range(7):
            g = q6_layer(k)
            self.assertEqual(g.order(), math.comb(6, k))

    def test_layer_no_edges(self):
        """В слое нет рёбер (соседи меняют yang на ±1)."""
        for k in range(7):
            g = q6_layer(k)
            self.assertEqual(g.size(), 0)


class TestBall(unittest.TestCase):
    def test_ball_r0(self):
        g = q6_ball(0, 0)
        self.assertEqual(g.order(), 1)
        self.assertIn(0, g)

    def test_ball_r1(self):
        g = q6_ball(0, 1)
        self.assertEqual(g.order(), 7)   # 1 + 6

    def test_ball_r6_full(self):
        g = q6_ball(0, 6)
        self.assertEqual(g.order(), SIZE)

    def test_ball_connected(self):
        for r in range(1, 5):
            g = q6_ball(0, r)
            self.assertTrue(g.is_connected())


class TestSpectral(unittest.TestCase):
    def test_largest_eigenvalue_q6(self):
        """λ_max Q6 = 6 (степень регулярного графа)."""
        g = q6_full()
        lam = largest_eigenvalue(g)
        self.assertAlmostEqual(lam, 6.0, places=2)

    def test_largest_eigenvalue_edge(self):
        """λ_max для единственного ребра = 1."""
        g = induced_subgraph({0, 1})
        lam = largest_eigenvalue(g)
        self.assertAlmostEqual(lam, 1.0, places=2)

    def test_spectrum_length(self):
        g = induced_subgraph({0, 1, 2, 3})
        sp = spectrum_approx(g, k=4)
        self.assertEqual(len(sp), 4)


class TestAutomorphisms(unittest.TestCase):
    def test_count(self):
        self.assertEqual(count_automorphisms(), 46080)

    def test_xor_automorphism_preserves_edges(self):
        """XOR-сдвиг — автоморфизм: соседи переходят в соседей."""
        f = xor_automorphism(42)
        for h in range(SIZE):
            for nb in neighbors(h):
                self.assertIn(f(nb), neighbors(f(h)))

    def test_bit_permutation_automorphism(self):
        """Перестановка битов — автоморфизм."""
        perm = [1, 0, 2, 3, 4, 5]   # поменять биты 0 и 1
        f = bit_permutation_automorphism(perm)
        for h in range(SIZE):
            for nb in neighbors(h):
                self.assertIn(f(nb), neighbors(f(h)))


class TestIsomorphism(unittest.TestCase):
    def test_self_isomorphic(self):
        g = induced_subgraph({0, 1, 3})
        self.assertTrue(g.is_isomorphic_to(g))

    def test_path3_isomorphic_to_path3(self):
        """Два пути длины 2 изоморфны."""
        g1 = induced_subgraph({0, 1, 3})    # 0-1-3
        g2 = induced_subgraph({0, 2, 6})    # 0-2-6 (биты 1 и 2)
        # 0-1: hamming=1 ✓, 1-3: hamming=1 ✓; 0-2: hamming=1 ✓, 2-6: hamming=1 ✓
        self.assertTrue(g1.is_isomorphic_to(g2))

    def test_different_size_not_isomorphic(self):
        g1 = induced_subgraph({0, 1})
        g2 = induced_subgraph({0, 1, 3})
        self.assertFalse(g1.is_isomorphic_to(g2))

    def test_different_degree_sequence(self):
        g1 = induced_subgraph({0, 1, 3})   # путь: степени [1,2,1]
        g2 = induced_subgraph({0, 1, 2, 4})  # 4 вершины, разные степени
        self.assertFalse(g1.is_isomorphic_to(g2))


class TestDistanceGraph(unittest.TestCase):
    def test_distance_1_edges(self):
        """D_1(Q6) = Q6: те же рёбра."""
        g = distance_graph(1)
        self.assertEqual(g.order(), SIZE)
        self.assertEqual(g.size(), 192)

    def test_distance_6_edges(self):
        """D_6(Q6): каждая вершина соединена с антиподом. 32 рёбра."""
        g = distance_graph(6)
        self.assertEqual(g.size(), 32)

    def test_distance_graph_sizes(self):
        """Число рёбер D_k = 64 * C(6,k) / 2."""
        import math
        for k in range(1, 7):
            g = distance_graph(k)
            expected = 64 * math.comb(6, k) // 2
            self.assertEqual(g.size(), expected)


class TestAnalyze(unittest.TestCase):
    def test_analyze_q6_keys(self):
        g = q6_full()
        info = analyze(g)
        for key in ('order', 'size', 'connected', 'bipartite',
                    'independence_number', 'clique_number', 'diameter', 'girth'):
            self.assertIn(key, info)

    def test_analyze_q6_values(self):
        g = q6_full()
        info = analyze(g)
        self.assertEqual(info['order'], 64)
        self.assertEqual(info['size'], 192)
        self.assertTrue(info['connected'])
        self.assertTrue(info['bipartite'])
        self.assertEqual(info['diameter'], 6)
        self.assertEqual(info['girth'], 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
