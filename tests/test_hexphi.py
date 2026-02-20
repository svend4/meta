"""Тесты hexphi — числа Фибоначчи и φ в Q6 (SC-7, K7×K5)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from math import sqrt

from projects.hexphi.phi_glyphs import (
    json_fibonacci_q6,
    _fibonacci_cube_vertices,
    _fibonacci_cube_edges,
    _fibonacci_up_to,
    _PHI,
)
from projects.hexpack.pack_glyphs import json_ring
from libs.hexcore.hexcore import yang_count


class TestFibonacciHelpers(unittest.TestCase):
    def test_phi_value(self):
        self.assertAlmostEqual(_PHI, (1 + sqrt(5)) / 2, places=10)

    def test_fibonacci_up_to_64(self):
        fibs = _fibonacci_up_to(64)
        self.assertEqual(fibs, [1, 2, 3, 5, 8, 13, 21, 34, 55])

    def test_fibonacci_up_to_sorted(self):
        fibs = _fibonacci_up_to(1000)
        self.assertEqual(fibs, sorted(fibs))

    def test_fibonacci_up_to_all_in_range(self):
        limit = 100
        for f in _fibonacci_up_to(limit):
            self.assertLessEqual(f, limit)


class TestFibonacciCubeVertices(unittest.TestCase):
    """Тесты вершин Γ₆ — гексаграмм без смежных янов."""

    @classmethod
    def setUpClass(cls):
        cls.vertices = _fibonacci_cube_vertices()

    def test_count_is_21(self):
        """|V(Γ₆)| = 21 = F(8)."""
        self.assertEqual(len(self.vertices), 21)

    def test_vertices_in_q6(self):
        for v in self.vertices:
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, 64)

    def test_no_adjacent_yangs(self):
        """Ни одна вершина не имеет двух смежных янов (1-битов)."""
        for h in self.vertices:
            for i in range(5):
                self.assertFalse(
                    (h >> i) & 1 and (h >> (i + 1)) & 1,
                    f"h={h:06b} has adjacent yangs at bits {i},{i+1}"
                )

    def test_zero_hexagram_included(self):
        """Гексаграмма 0 (все инь) принадлежит Γ₆."""
        self.assertIn(0, self.vertices)

    def test_yang_distribution(self):
        """Ян-распределение: [1, 6, 10, 4]."""
        from collections import Counter
        dist = Counter(yang_count(h) for h in self.vertices)
        self.assertEqual(dist[0], 1)
        self.assertEqual(dist[1], 6)
        self.assertEqual(dist[2], 10)
        self.assertEqual(dist[3], 4)
        self.assertEqual(dist.get(4, 0), 0)

    def test_yang_ratio_phi(self):
        """yang=2 / yang=1 = 10/6 = 5/3 = F(5)/F(4) ≈ φ."""
        from collections import Counter
        dist = Counter(yang_count(h) for h in self.vertices)
        ratio = dist[2] / dist[1]
        self.assertAlmostEqual(ratio, 5 / 3, places=6)

    def test_no_duplicates(self):
        self.assertEqual(len(self.vertices), len(set(self.vertices)))

    def test_sorted(self):
        self.assertEqual(self.vertices, sorted(self.vertices))


class TestFibonacciCubeEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vertices = _fibonacci_cube_vertices()
        cls.edges = _fibonacci_cube_edges(cls.vertices)
        cls.vset = set(cls.vertices)

    def test_edge_count_38(self):
        """|E(Γ₆)| = 38."""
        self.assertEqual(len(self.edges), 38)

    def test_edges_in_gamma6(self):
        for u, v in self.edges:
            self.assertIn(u, self.vset)
            self.assertIn(v, self.vset)

    def test_edges_hamming_1(self):
        """Все рёбра — пары гексаграмм на расстоянии 1."""
        from libs.hexcore.hexcore import hamming
        for u, v in self.edges:
            self.assertEqual(hamming(u, v), 1)

    def test_edges_ordered(self):
        for u, v in self.edges:
            self.assertLess(u, v)

    def test_no_self_loops(self):
        for u, v in self.edges:
            self.assertNotEqual(u, v)


class TestJsonFibonacciQ6(unittest.TestCase):
    """Тесты json_fibonacci_q6() без ring_data (только K7)."""

    @classmethod
    def setUpClass(cls):
        cls.result = json_fibonacci_q6()

    def test_command(self):
        self.assertEqual(self.result['command'], 'fibonacci')

    def test_phi_approx(self):
        self.assertAlmostEqual(self.result['phi'], 1.618034, places=5)

    def test_gamma6_n_vertices(self):
        self.assertEqual(self.result['gamma6_structure']['n_vertices'], 21)

    def test_gamma6_n_edges(self):
        self.assertEqual(self.result['gamma6_structure']['n_edges'], 38)

    def test_gamma6_diameter(self):
        self.assertEqual(self.result['gamma6_structure']['diameter'], 6)

    def test_gamma6_is_induced(self):
        self.assertTrue(self.result['gamma6_structure']['is_induced_subgraph_of_q6'])

    def test_gamma6_yang_distribution(self):
        self.assertEqual(self.result['gamma6_structure']['yang_distribution'], [1, 6, 10, 4])

    def test_phi_ratios_nonempty(self):
        self.assertGreater(len(self.result['phi_ratios_in_yang']), 0)

    def test_phi_ratios_have_fib_match(self):
        """Хотя бы одно соотношение ян-слоёв совпадает с Фибоначчи."""
        ratios = self.result['phi_ratios_in_yang']
        has_fib = any(r['is_fib_ratio'] for r in ratios)
        self.assertTrue(has_fib)

    def test_yang_2_over_1_is_fib(self):
        """yang=2/yang=1 = 10/6 = F(5)/F(4) — Фибоначчи-соотношение."""
        ratios = {r['yang_ratio']: r for r in self.result['phi_ratios_in_yang']}
        r21 = ratios.get('yang=2 / yang=1')
        self.assertIsNotNone(r21)
        self.assertTrue(r21['is_fib_ratio'])
        self.assertEqual(r21['fib_match'], 'F(n+1)/F(n)=5/3')

    def test_phi_facts_binet(self):
        pf = self.result['phi_facts']
        self.assertEqual(pf['binet_f8_int'], 21)
        self.assertAlmostEqual(pf['binet_f8'], 21.009519, places=4)
        self.assertTrue(pf['binet_matches_gamma6'])

    def test_phi_facts_fibonacci_list(self):
        fibs = self.result['phi_facts']['fibonacci_up_to_64']
        self.assertEqual(fibs, [1, 2, 3, 5, 8, 13, 21, 34, 55])

    def test_phi_facts_f8_amino_acids(self):
        self.assertTrue(self.result['phi_facts']['f8_equals_n_amino_acids'])

    def test_amino_acid_coincidence(self):
        aa = self.result['amino_acid_coincidence']
        self.assertEqual(aa['amino_acids_in_genetic_code'], 21)
        self.assertEqual(aa['fibonacci_cube_vertices'], 21)
        self.assertTrue(aa['match'])
        self.assertTrue(aa['both_use_21'])

    def test_ring_analysis_none_without_data(self):
        """Без ring_data анализ кольца отсутствует."""
        self.assertIsNone(self.result['ring_analysis'])

    def test_sc7_finding_nonempty(self):
        self.assertGreater(len(self.result['sc7_finding']), 20)

    def test_sc7_finding_mentions_f8(self):
        self.assertIn('F(8)', self.result['sc7_finding'])

    def test_top_level_keys(self):
        for key in ('command', 'phi', 'gamma6_structure', 'phi_ratios_in_yang',
                    'phi_facts', 'amino_acid_coincidence', 'ring_analysis', 'sc7_finding'):
            self.assertIn(key, self.result)


class TestJsonFibonacciWithRing(unittest.TestCase):
    """SC-7 интеграция: K5 (ring) → K7 (Fibonacci cube)."""

    @classmethod
    def setUpClass(cls):
        ring_data = json_ring()
        cls.result = json_fibonacci_q6(ring_data=ring_data)

    def test_ring_analysis_present(self):
        self.assertIsNotNone(self.result['ring_analysis'])

    def test_ring_analysis_n_vertices(self):
        ra = self.result['ring_analysis']
        self.assertEqual(ra['n_vertices'], 21)

    def test_ring_analysis_mean_in_range(self):
        """Среднее ring-значений на Γ₆ ∈ (1, 64)."""
        ra = self.result['ring_analysis']
        self.assertGreater(ra['mean_ring_at_gamma6'], 1)
        self.assertLess(ra['mean_ring_at_gamma6'], 64)

    def test_global_mean_is_32_5(self):
        self.assertAlmostEqual(self.result['ring_analysis']['mean_ring_global'], 32.5, places=1)

    def test_ring_values_length_21(self):
        ra = self.result['ring_analysis']
        self.assertEqual(len(ra['ring_values_at_gamma6']), 21)

    def test_ring_values_in_1_64(self):
        for v in self.result['ring_analysis']['ring_values_at_gamma6']:
            self.assertGreaterEqual(v, 1)
            self.assertLessEqual(v, 64)

    def test_n_fibonacci_ring_values_positive(self):
        ra = self.result['ring_analysis']
        self.assertGreater(ra['n_fibonacci_ring_values'], 0)

    def test_ring_bias_negative(self):
        """Γ₆-гексаграммы смещены к меньшим значениям ring (< 32.5)."""
        ra = self.result['ring_analysis']
        self.assertLess(ra['mean_ring_at_gamma6'], ra['mean_ring_global'])

    def test_k5_k7_finding_nonempty(self):
        ra = self.result['ring_analysis']
        self.assertIsInstance(ra['k5_k7_finding'], str)
        self.assertGreater(len(ra['k5_k7_finding']), 20)

    def test_sc7_finding_includes_delta(self):
        """С ring_data finding включает Δ смещения."""
        finding = self.result['sc7_finding']
        self.assertIn('Δ', finding)


if __name__ == '__main__':
    unittest.main(verbosity=2)
