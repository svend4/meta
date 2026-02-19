"""Тесты библиотеки hexcore (граф Q6)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from libs.hexcore.hexcore import (
    neighbors, hamming, flip, shortest_path, all_paths,
    gray_code, antipode, describe, render, to_bits, yang_count,
    upper_trigram, lower_trigram, SIZE,
    orbit, orbit_length, all_orbits, subcubes, distance_spectrum,
    sphere, ball, apply_permutation,
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


class TestOrbit(unittest.TestCase):
    def test_orbit_antipode_length2(self):
        """antipode применённый дважды возвращает исходное — орбита длины 2."""
        from libs.hexcore.hexcore import antipode
        orb = orbit(0, antipode)
        self.assertEqual(len(orb), 2)
        self.assertEqual(orb[0], 0)
        self.assertEqual(orb[1], 63)

    def test_orbit_flip0_length2(self):
        """FLIP бита 0 — орбита длины 2 для любого h."""
        f = lambda h: h ^ 1
        for h in range(0, SIZE, 7):
            orb = orbit(h, f)
            self.assertEqual(len(orb), 2)

    def test_orbit_length_matches_orbit(self):
        f = lambda h: h ^ 1
        for h in [0, 7, 42, 63]:
            self.assertEqual(orbit_length(h, f), len(orbit(h, f)))

    def test_all_orbits_partition(self):
        """all_orbits разбивает все 64 гексаграммы без пересечений."""
        from libs.hexcore.hexcore import antipode
        orbits = all_orbits(antipode)
        all_nodes = set()
        for orb in orbits:
            s = set(orb)
            self.assertEqual(s & all_nodes, set(), "пересечение орбит")
            all_nodes |= s
        self.assertEqual(all_nodes, set(range(SIZE)))

    def test_all_orbits_antipode_count(self):
        """antipode делит 64 на пары → 32 орбиты."""
        from libs.hexcore.hexcore import antipode
        orbits = all_orbits(antipode)
        self.assertEqual(len(orbits), 32)
        for orb in orbits:
            self.assertEqual(len(orb), 2)


class TestSubcubes(unittest.TestCase):
    def test_subcubes_0(self):
        """0-мерные подкубы — отдельные вершины, должно быть 64."""
        sc = subcubes(0)
        self.assertEqual(len(sc), 64)
        for s in sc:
            self.assertEqual(len(s), 1)

    def test_subcubes_1_count(self):
        """1-мерные подкубы = рёбра Q6 = 192."""
        sc = subcubes(1)
        self.assertEqual(len(sc), 192)

    def test_subcubes_1_all_edges(self):
        """Каждое ребро Q6 — 1-мерный подкуб."""
        sc = subcubes(1)
        edges_in_subcubes = set()
        for s in sc:
            lst = sorted(s)
            edges_in_subcubes.add((lst[0], lst[1]))
        expected = set()
        for h in range(SIZE):
            for nb in neighbors(h):
                expected.add((min(h, nb), max(h, nb)))
        self.assertEqual(edges_in_subcubes, expected)

    def test_subcubes_2_count(self):
        """2-мерных подкубов в Q6: C(6,2)*2^(6-2) = 15*16 = 240."""
        sc = subcubes(2)
        self.assertEqual(len(sc), 240)

    def test_subcubes_6_count(self):
        """6-мерный подкуб — весь граф Q6, только 1."""
        sc = subcubes(6)
        self.assertEqual(len(sc), 1)
        self.assertEqual(sc[0], frozenset(range(64)))

    def test_subcubes_disjoint_vertices(self):
        """Каждый k-подкуб содержит ровно 2^k вершин."""
        for k in range(4):
            for s in subcubes(k):
                self.assertEqual(len(s), 2 ** k)


class TestDistanceSpectrum(unittest.TestCase):
    def test_spectrum_values(self):
        """Спектр расстояний = биномиальные коэффициенты C(6,0..6)."""
        expected = [1, 6, 15, 20, 15, 6, 1]
        for h in range(SIZE):
            self.assertEqual(distance_spectrum(h), expected)

    def test_spectrum_sum(self):
        """Сумма спектра = 64."""
        self.assertEqual(sum(distance_spectrum(0)), 64)

    def test_spectrum_length(self):
        self.assertEqual(len(distance_spectrum(42)), 7)


class TestSphereAndBall(unittest.TestCase):
    def test_sphere_radius0(self):
        """Сфера радиуса 0 = {h}."""
        for h in [0, 42, 63]:
            self.assertEqual(sphere(h, 0), [h])

    def test_sphere_radius1(self):
        """Сфера радиуса 1 = соседи."""
        for h in range(0, SIZE, 7):
            self.assertEqual(set(sphere(h, 1)), set(neighbors(h)))

    def test_sphere_sizes(self):
        """Размер сферы = биномиальный коэффициент C(6, r)."""
        import math
        for r in range(7):
            self.assertEqual(len(sphere(0, r)), math.comb(6, r))

    def test_ball_contains_sphere(self):
        for h in [0, 21, 63]:
            for r in range(4):
                b = set(ball(h, r))
                s = set(sphere(h, r))
                self.assertTrue(s.issubset(b))

    def test_ball_size_cumulative(self):
        """Размер шара = сумма C(6,0..r)."""
        import math
        for r in range(7):
            expected = sum(math.comb(6, i) for i in range(r + 1))
            self.assertEqual(len(ball(0, r)), expected)

    def test_ball_radius6(self):
        """Шар радиуса 6 = весь Q6."""
        self.assertEqual(len(ball(0, 6)), 64)


class TestApplyPermutation(unittest.TestCase):
    def test_identity_permutation(self):
        """Тождественная перестановка не меняет значение."""
        perm = list(range(6))
        for h in range(SIZE):
            self.assertEqual(apply_permutation(h, perm), h)

    def test_reverse_permutation(self):
        """Перестановка [5,4,3,2,1,0] обращает биты."""
        perm = [5, 4, 3, 2, 1, 0]
        # h=1 (бит 0 = 1) → бит 5 = 1 → результат = 32
        self.assertEqual(apply_permutation(1, perm), 32)
        self.assertEqual(apply_permutation(32, perm), 1)

    def test_cycle_permutation(self):
        """Циклическая перестановка — применение 6 раз возвращает исходное."""
        perm = [1, 2, 3, 4, 5, 0]  # сдвиг влево на 1
        for h in range(SIZE):
            result = h
            for _ in range(6):
                result = apply_permutation(result, perm)
            self.assertEqual(result, h)

    def test_preserves_yang_count(self):
        """Перестановка не меняет число ян-черт."""
        import random
        rng = random.Random(42)
        for _ in range(20):
            perm = list(range(6))
            rng.shuffle(perm)
            h = rng.randrange(SIZE)
            self.assertEqual(yang_count(apply_permutation(h, perm)), yang_count(h))


if __name__ == '__main__':
    unittest.main(verbosity=2)
