"""
Интеграционные тесты: импорт всех 24 проектов и инварианты hexcore.

Цель: убедиться, что ни один модуль не ломается при импорте и что
базовая математика Q6 стабильна.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import importlib


# ── список всех 24 проектов и ожидаемые символы ───────────────────────────
PROJECTS = [
    ('projects.hexnav.hexnav',        ['fmt_hex']),
    ('projects.hexca.hexca',          ['CA1D', 'CA2D']),
    ('projects.hexpath.game',         ['new_game', 'best_move', 'Player']),
    ('projects.hexforth.interpreter', ['HexForth', 'HexForthError']),
    ('projects.karnaugh6.minimize',   ['minimize', 'quine_mccluskey']),
    ('projects.hexspec.verifier',     ['Spec', 'verify']),
    ('projects.hexgraph.hexgraph',    ['Subgraph', 'analyze']),
    ('projects.hexvis.hexvis',        ['render_grid', 'render']),
    ('projects.hexcode.hexcode',      ['BinaryCode', 'even_weight_code']),
    ('projects.hexlearn.hexlearn',    ['KNN', 'KMedoids']),
    ('projects.hexopt.hexopt',        ['SetOptimizer', 'genetic_algorithm']),
    ('projects.hexring.hexring',      ['BoolFunc', 'ReedMullerCode']),
    ('projects.hexsym.hexsym',        ['Automorphism', 'all_orbits']),
    ('projects.hexnet.hexnet',        ['broadcast_schedule', 'adaptive_route']),
    ('projects.hexcrypt.hexcrypt',    ['SBox', 'FeistelCipher']),
    ('projects.hexstat.hexstat',      ['Q6Distribution', 'RandomWalk']),
    ('projects.hexgeom.hexgeom',      ['ball_size', 'all_antipodal_pairs']),
    ('projects.hexdim.hexdim',        ['all_tesseracts', 'dimension_info']),
    ('projects.hexalg.hexalg',        ['cayley_graph', 'all_characters']),
    ('projects.hexphys.hexphys',      ['IsingChain', 'MetropolisQ6']),
    ('projects.hexgf.hexgf',          ['all_elements', 'all_cyclotomic_cosets']),
    ('projects.hexmat.hexmat',        ['gl6_order', 'is_invertible']),
    ('projects.hexbio.hexbio',        ['codon_to_int', 'amino_acid_distance']),
    ('projects.hexlat.hexlat',        ['covers', 'complement']),
]


class TestImports(unittest.TestCase):
    """Все 24 модуля должны импортироваться и экспортировать ключевые символы."""

    def _check(self, module_path: str, symbols: list[str]):
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            self.fail(f"Не удалось импортировать {module_path}: {e}")
        for sym in symbols:
            self.assertTrue(
                hasattr(mod, sym),
                f"{module_path} не экспортирует '{sym}'",
            )


# Динамически создаём по одному test-методу на проект
for _path, _syms in PROJECTS:
    def _make_test(p, s):
        def _test(self):
            self._check(p, s)
        _test.__name__ = f'test_{p.split(".")[-2]}_imports'
        return _test

    setattr(
        TestImports,
        f'test_{_path.split(".")[-2]}_imports',
        _make_test(_path, _syms),
    )


# ── hexcore: математические инварианты ────────────────────────────────────
from libs.hexcore.hexcore import (
    neighbors, hamming, flip, shortest_path, gray_code, antipode, describe,
)


class TestHexcoreInvariants(unittest.TestCase):
    """Математические свойства Q6 должны выполняться всегда."""

    def test_neighbors_count(self):
        """Каждая вершина Q6 имеет ровно 6 соседей."""
        for v in range(64):
            self.assertEqual(len(neighbors(v)), 6, f"вершина {v}")

    def test_neighbors_hamming1(self):
        """Каждый сосед отличается ровно на 1 бит."""
        for v in range(64):
            for nb in neighbors(v):
                self.assertEqual(hamming(v, nb), 1)

    def test_neighbors_symmetric(self):
        """Если b — сосед a, то a — сосед b."""
        for v in range(64):
            for nb in neighbors(v):
                self.assertIn(v, neighbors(nb))

    def test_hamming_metric(self):
        """d(a,a)=0, d(a,b)=d(b,a), неравенство треугольника."""
        import random
        random.seed(1)
        samples = random.sample(range(64), 12)
        for a in samples:
            self.assertEqual(hamming(a, a), 0)
        for a in samples:
            for b in samples:
                self.assertEqual(hamming(a, b), hamming(b, a))
                for c in samples:
                    self.assertLessEqual(
                        hamming(a, c), hamming(a, b) + hamming(b, c)
                    )

    def test_hamming_range(self):
        """Расстояние Хэмминга от 0 до 63 равно 6 (максимум на Q6)."""
        self.assertEqual(hamming(0, 63), 6)
        self.assertEqual(hamming(0, 0), 0)

    def test_flip_involution(self):
        """Двойной flip возвращает исходную вершину."""
        for v in range(64):
            for bit in range(6):
                self.assertEqual(flip(flip(v, bit), bit), v)

    def test_flip_changes_one_bit(self):
        """flip меняет ровно один бит (расстояние Хэмминга = 1)."""
        for v in [0, 7, 42, 63]:
            for bit in range(6):
                self.assertEqual(hamming(v, flip(v, bit)), 1)

    def test_shortest_path_length(self):
        """Длина кратчайшего пути = расстояние Хэмминга."""
        for a, b in [(0, 63), (0, 0), (42, 21), (1, 62)]:
            path = shortest_path(a, b)
            self.assertEqual(len(path) - 1, hamming(a, b))
            self.assertEqual(path[0], a)
            self.assertEqual(path[-1], b)

    def test_antipode(self):
        """Антипод на расстоянии 6, двойной антипод = исходная вершина."""
        for v in [0, 1, 42, 63]:
            self.assertEqual(hamming(v, antipode(v)), 6)
            self.assertEqual(antipode(antipode(v)), v)
        self.assertEqual(antipode(0), 63)
        self.assertEqual(antipode(63), 0)

    def test_gray_code_hamiltonian(self):
        """Код Грея: 64 вершины, каждая ровно раз, соседние на 1 бит."""
        path = gray_code()
        self.assertEqual(len(path), 64)
        self.assertEqual(len(set(path)), 64)
        for i in range(63):
            self.assertEqual(hamming(path[i], path[i + 1]), 1)

    def test_diameter(self):
        """Диаметр Q6 = 6."""
        max_d = max(hamming(a, b) for a in range(64) for b in range(64))
        self.assertEqual(max_d, 6)

    def test_vertex_count(self):
        """Q6 имеет ровно 64 вершины."""
        all_v = set(range(64))
        for v in range(64):
            all_v.update(neighbors(v))
        self.assertEqual(len(all_v), 64)

    def test_edge_count(self):
        """Q6 имеет 64 × 6 / 2 = 192 рёбра."""
        edges = set()
        for v in range(64):
            for nb in neighbors(v):
                edges.add((min(v, nb), max(v, nb)))
        self.assertEqual(len(edges), 192)

    def test_describe_keys(self):
        """describe() возвращает ожидаемые ключи."""
        d = describe(0)
        for key in ('index', 'bits', 'yang', 'antipode', 'neighbors'):
            self.assertIn(key, d)

    def test_describe_values(self):
        """describe(0): 0 ян-линий, антипод = 63, 6 соседей."""
        d = describe(0)
        self.assertEqual(d['yang'], 0)
        self.assertEqual(d['antipode'], 63)
        self.assertEqual(len(d['neighbors']), 6)

    def test_describe_63(self):
        """describe(63): 6 ян-линий, антипод = 0."""
        d = describe(63)
        self.assertEqual(d['yang'], 6)
        self.assertEqual(d['antipode'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
