"""Тесты библиотеки hexcore (граф Q6)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from libs.hexcore.hexcore import (
    neighbors, hamming, flip, shortest_path, all_paths,
    gray_code, antipode, describe, render, to_bits, yang_count,
    upper_trigram, lower_trigram, SIZE,
)


class TestHamming(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(hamming(0, 0), 0)
        self.assertEqual(hamming(63, 63), 0)

    def test_one_bit(self):
        self.assertEqual(hamming(0, 1), 1)
        self.assertEqual(hamming(0, 32), 1)

    def test_antipodes(self):
        self.assertEqual(hamming(0, 63), 6)
        self.assertEqual(hamming(42, 21), 6)

    def test_symmetry(self):
        self.assertEqual(hamming(7, 42), hamming(42, 7))


class TestNeighbors(unittest.TestCase):
    def test_count(self):
        for h in range(SIZE):
            self.assertEqual(len(neighbors(h)), 6)

    def test_all_adjacent(self):
        for h in range(SIZE):
            for nb in neighbors(h):
                self.assertEqual(hamming(h, nb), 1)

    def test_symmetric(self):
        for h in range(SIZE):
            for nb in neighbors(h):
                self.assertIn(h, neighbors(nb))

    def test_zero(self):
        self.assertEqual(set(neighbors(0)), {1, 2, 4, 8, 16, 32})

    def test_full(self):
        self.assertEqual(set(neighbors(63)), {62, 61, 59, 55, 47, 31})


class TestFlip(unittest.TestCase):
    def test_flip_bit0(self):
        self.assertEqual(flip(0, 0), 1)
        self.assertEqual(flip(1, 0), 0)

    def test_flip_involution(self):
        for h in range(SIZE):
            for b in range(6):
                self.assertEqual(flip(flip(h, b), b), h)

    def test_flip_result_in_neighbors(self):
        for h in range(SIZE):
            for b in range(6):
                self.assertIn(flip(h, b), neighbors(h))


class TestShortestPath(unittest.TestCase):
    def test_trivial(self):
        self.assertEqual(shortest_path(0, 0), [0])

    def test_length(self):
        # Длина пути = расстояние Хэмминга
        for a in range(0, 64, 7):
            for b in range(0, 64, 7):
                path = shortest_path(a, b)
                self.assertEqual(len(path) - 1, hamming(a, b))

    def test_path_validity(self):
        path = shortest_path(0, 63)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 63)
        for i in range(len(path) - 1):
            self.assertEqual(hamming(path[i], path[i + 1]), 1)

    def test_antipode_path_length(self):
        path = shortest_path(0, 63)
        self.assertEqual(len(path), 7)


class TestAntipode(unittest.TestCase):
    def test_antipode_zero(self):
        self.assertEqual(antipode(0), 63)

    def test_antipode_involution(self):
        for h in range(SIZE):
            self.assertEqual(antipode(antipode(h)), h)

    def test_antipode_distance(self):
        for h in range(SIZE):
            self.assertEqual(hamming(h, antipode(h)), 6)


class TestGrayCode(unittest.TestCase):
    def test_length(self):
        path = gray_code()
        self.assertEqual(len(path), SIZE)

    def test_all_visited(self):
        path = gray_code()
        self.assertEqual(set(path), set(range(SIZE)))

    def test_each_step_one_bit(self):
        path = gray_code()
        for i in range(len(path) - 1):
            self.assertEqual(hamming(path[i], path[i + 1]), 1)


class TestYangCount(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(yang_count(0), 0)

    def test_full(self):
        self.assertEqual(yang_count(63), 6)

    def test_specific(self):
        self.assertEqual(yang_count(42), 3)   # 101010
        self.assertEqual(yang_count(21), 3)   # 010101

    def test_complement(self):
        for h in range(SIZE):
            self.assertEqual(yang_count(h) + yang_count(63 ^ h), 6)


class TestToBits(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(to_bits(0), '000000')

    def test_full(self):
        self.assertEqual(to_bits(63), '111111')

    def test_length(self):
        for h in range(SIZE):
            self.assertEqual(len(to_bits(h)), 6)

    def test_roundtrip(self):
        for h in range(SIZE):
            self.assertEqual(int(to_bits(h), 2), h)


class TestDescribe(unittest.TestCase):
    def test_keys(self):
        d = describe(42)
        for key in ('yang', 'yin', 'upper_tri', 'lower_tri', 'antipode', 'neighbors', 'bits'):
            self.assertIn(key, d)

    def test_yang_yin_sum(self):
        for h in range(SIZE):
            d = describe(h)
            self.assertEqual(d['yang'] + d['yin'], 6)

    def test_antipode_field(self):
        for h in range(SIZE):
            self.assertEqual(describe(h)['antipode'], antipode(h))


class TestRender(unittest.TestCase):
    def test_lines(self):
        r = render(0)
        lines = r.split('\n')
        self.assertEqual(len(lines), 6)

    def test_content(self):
        r = render(0)
        # Все черты инь → разрывные линии
        self.assertNotIn('━━━━━━', r)

    def test_full_yang(self):
        r = render(63)
        lines = r.split('\n')
        for line in lines:
            self.assertIn('━━━━━━', line)


class TestGraphProperties(unittest.TestCase):
    def test_size(self):
        self.assertEqual(SIZE, 64)

    def test_edge_count(self):
        """Q6 должен иметь 192 рёбра (64 × 6 / 2)."""
        edges = set()
        for h in range(SIZE):
            for nb in neighbors(h):
                edges.add((min(h, nb), max(h, nb)))
        self.assertEqual(len(edges), 192)

    def test_bipartite(self):
        """Q6 двудольный: чётные/нечётные числа ян-линий."""
        for h in range(SIZE):
            parity = yang_count(h) % 2
            for nb in neighbors(h):
                self.assertNotEqual(yang_count(nb) % 2, parity)

    def test_connectivity(self):
        """Q6 связен: из любой вершины достижимы все остальные."""
        from collections import deque
        visited = {0}
        queue = deque([0])
        while queue:
            cur = queue.popleft()
            for nb in neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        self.assertEqual(visited, set(range(SIZE)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
