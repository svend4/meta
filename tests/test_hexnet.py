"""Тесты hexnet — Q6 как коммуникационная сеть."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexnet.hexnet import (
    ecube_route, all_shortest_paths, count_shortest_paths,
    fault_tolerant_route, adaptive_route,
    broadcast_tree, broadcast_schedule, gossip, broadcast_steps,
    network_diameter, bisection_width, node_connectivity, edge_connectivity,
    average_path_length,
    bond_percolation, site_percolation, k_fault_diameter,
    ecube_traffic_load, traffic_statistics, hot_spot_edges,
    find_hamiltonian_path, gray_code_cycle,
    is_hamiltonian_path, is_hamiltonian_cycle,
    ecube_path_length_distribution,
)
from libs.hexcore.hexcore import hamming, neighbors, SIZE


# ---------------------------------------------------------------------------
# E-cube маршрутизация
# ---------------------------------------------------------------------------

class TestEcubeRoute(unittest.TestCase):
    def test_route_to_self(self):
        path = ecube_route(42, 42)
        self.assertEqual(path, [42])

    def test_route_length(self):
        """Длина маршрута = расстояние Хэмминга."""
        for src, dst in [(0, 63), (0, 1), (7, 56), (21, 42)]:
            path = ecube_route(src, dst)
            self.assertEqual(len(path) - 1, hamming(src, dst))

    def test_route_valid_steps(self):
        """Каждый шаг — ребро Q6."""
        for src, dst in [(0, 63), (1, 62), (3, 21)]:
            path = ecube_route(src, dst)
            for i in range(len(path) - 1):
                self.assertEqual(hamming(path[i], path[i + 1]), 1)

    def test_route_ends_at_dst(self):
        for src in [0, 7, 42]:
            for dst in [1, 63, 21]:
                path = ecube_route(src, dst)
                self.assertEqual(path[-1], dst)

    def test_route_starts_at_src(self):
        path = ecube_route(10, 50)
        self.assertEqual(path[0], 10)

    def test_ecube_dimension_order(self):
        """E-cube исправляет биты по порядку 0, 1, 2, ..."""
        # src=0 (000000), dst=7 (000111): должен исправить биты 0,1,2 по порядку
        path = ecube_route(0, 7)
        self.assertEqual(path, [0, 1, 3, 7])


# ---------------------------------------------------------------------------
# Все кратчайшие пути
# ---------------------------------------------------------------------------

class TestAllShortestPaths(unittest.TestCase):
    def test_same_node(self):
        paths = all_shortest_paths(0, 0)
        self.assertEqual(paths, [[0]])

    def test_adjacent_nodes(self):
        """Для соседей: один путь длиной 1."""
        paths = all_shortest_paths(0, 1)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], [0, 1])

    def test_count_matches_factorial(self):
        """Число кратчайших путей = d! (факториал расстояния)."""
        import math
        for src, dst in [(0, 7), (0, 63), (1, 14)]:
            d = hamming(src, dst)
            paths = all_shortest_paths(src, dst)
            self.assertEqual(len(paths), math.factorial(d))

    def test_count_shortest_paths(self):
        import math
        for src, dst in [(0, 63), (3, 21), (7, 56)]:
            d = hamming(src, dst)
            self.assertEqual(count_shortest_paths(src, dst), math.factorial(d))

    def test_paths_are_minimal(self):
        """Все пути имеют длину = hamming(src, dst)."""
        d = hamming(0, 63)
        for path in all_shortest_paths(0, 63):
            self.assertEqual(len(path) - 1, d)

    def test_paths_valid_steps(self):
        """Каждый шаг — ребро Q6."""
        for path in all_shortest_paths(0, 7):
            for i in range(len(path) - 1):
                self.assertEqual(hamming(path[i], path[i + 1]), 1)


# ---------------------------------------------------------------------------
# Отказоустойчивая маршрутизация
# ---------------------------------------------------------------------------

class TestFaultTolerantRoute(unittest.TestCase):
    def test_no_faults(self):
        """Без отказов: путь существует."""
        path = fault_tolerant_route(0, 63)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 63)

    def test_path_valid(self):
        """Путь корректен."""
        path = fault_tolerant_route(0, 42)
        for i in range(len(path) - 1):
            self.assertEqual(hamming(path[i], path[i + 1]), 1)

    def test_avoids_failed_nodes(self):
        """Путь не проходит через отказавшие узлы."""
        bad = {1, 2, 4}
        path = fault_tolerant_route(0, 63, failed_nodes=bad)
        if path:
            for v in path:
                self.assertNotIn(v, bad)

    def test_src_failed_returns_none(self):
        self.assertIsNone(fault_tolerant_route(5, 10, failed_nodes={5}))

    def test_dst_failed_returns_none(self):
        self.assertIsNone(fault_tolerant_route(0, 63, failed_nodes={63}))

    def test_5_failures_still_reachable(self):
        """Q6 6-связен: при 5 отказах всё ещё существует путь 0→63."""
        bad = {1, 2, 4, 8, 16}  # 5 соседей вершины 0, но не 32 и 63
        path = fault_tolerant_route(0, 63, failed_nodes=bad)
        self.assertIsNotNone(path)

    def test_isolated_node(self):
        """Если все соседи dst удалены, dst недостижима."""
        # Все соседи 0: 1, 2, 4, 8, 16, 32 → удалить все 6
        bad = set(neighbors(63))
        path = fault_tolerant_route(0, 63, failed_nodes=bad)
        self.assertIsNone(path)


# ---------------------------------------------------------------------------
# Широковещание
# ---------------------------------------------------------------------------

class TestBroadcast(unittest.TestCase):
    def test_broadcast_tree_covers_all(self):
        """BFS-дерево покрывает все 64 узла."""
        tree = broadcast_tree(0)
        self.assertEqual(len(tree), SIZE)
        self.assertEqual(set(tree.keys()), set(range(SIZE)))

    def test_broadcast_tree_root_has_no_parent(self):
        tree = broadcast_tree(42)
        self.assertIsNone(tree[42])

    def test_broadcast_tree_parent_is_neighbor(self):
        """Каждый узел соединён с родителем ребром Q6."""
        tree = broadcast_tree(0)
        for v, parent in tree.items():
            if parent is not None:
                self.assertEqual(hamming(v, parent), 1)

    def test_broadcast_schedule_steps(self):
        """Расписание завершается за ≤ 6 шагов."""
        sched = broadcast_schedule(0)
        self.assertLessEqual(len(sched), 6)

    def test_broadcast_schedule_covers_all(self):
        """После всех шагов все узлы получили сообщение."""
        sched = broadcast_schedule(0)
        received = {0}
        for step in sched:
            for sender, receiver in step:
                self.assertIn(sender, received)
                received.add(receiver)
        self.assertEqual(received, set(range(SIZE)))

    def test_broadcast_steps_equals_schedule(self):
        steps = broadcast_steps(0)
        sched = broadcast_schedule(0)
        self.assertEqual(steps, len(sched))

    def test_gossip_estimate(self):
        """Gossip: 2n-2 = 10 раундов для n=6."""
        self.assertEqual(gossip(), 10)


# ---------------------------------------------------------------------------
# Характеристики сети
# ---------------------------------------------------------------------------

class TestNetworkProperties(unittest.TestCase):
    def test_diameter_no_faults(self):
        """Без отказов: диаметр Q6 = 6."""
        self.assertEqual(network_diameter(), 6)

    def test_bisection_width(self):
        """Ширина бисекции Q6 = 32 = 2^5."""
        self.assertEqual(bisection_width(), 32)

    def test_node_connectivity(self):
        """Узловая связность Q6 = 6."""
        self.assertEqual(node_connectivity(), 6)

    def test_edge_connectivity(self):
        """Рёберная связность Q6 = 6."""
        self.assertEqual(edge_connectivity(), 6)

    def test_average_path_length(self):
        """Среднее расстояние = Σ k×C(6,k)/63 = 192/63 ≈ 3.0476."""
        import math
        expected = sum(k * math.comb(6, k) for k in range(1, 7)) * 64 / (63 * 64)
        apl = average_path_length()
        self.assertAlmostEqual(apl, expected, places=4)

    def test_path_length_distribution(self):
        """Распределение длин = {d: 64 × C(6,d)} для d=1..6."""
        import math
        dist = ecube_path_length_distribution()
        for d in range(1, 7):
            self.assertEqual(dist[d], SIZE * math.comb(6, d))


# ---------------------------------------------------------------------------
# Трафик
# ---------------------------------------------------------------------------

class TestTraffic(unittest.TestCase):
    def test_traffic_load_all_edges(self):
        """Нагрузка задана для всех 192 рёбер."""
        load = ecube_traffic_load()
        total_edges = sum(len(neighbors(h)) for h in range(SIZE)) // 2
        self.assertEqual(len(load), total_edges)

    def test_traffic_balanced(self):
        """E-cube создаёт идеально равномерный трафик (нагрузка = 64 на каждое ребро)."""
        load = ecube_traffic_load()
        for edge, val in load.items():
            self.assertEqual(val, 64)

    def test_traffic_statistics(self):
        stats = traffic_statistics()
        self.assertEqual(stats['avg_load'], 64.0)
        self.assertEqual(stats['max_load'], 64)
        self.assertAlmostEqual(stats['load_imbalance'], 1.0)

    def test_hot_spot_edges_count(self):
        hot = hot_spot_edges(5)
        self.assertEqual(len(hot), 5)


# ---------------------------------------------------------------------------
# Гамильтоновы пути и циклы
# ---------------------------------------------------------------------------

class TestHamiltonian(unittest.TestCase):
    def test_hamiltonian_path_length(self):
        """Гамильтонов путь содержит все 64 вершины."""
        path = find_hamiltonian_path()
        self.assertEqual(len(path), SIZE)

    def test_hamiltonian_path_no_repeats(self):
        path = find_hamiltonian_path()
        self.assertEqual(len(set(path)), SIZE)

    def test_is_hamiltonian_path(self):
        path = find_hamiltonian_path()
        self.assertTrue(is_hamiltonian_path(path))

    def test_gray_code_is_hamiltonian(self):
        path = gray_code_cycle()
        self.assertTrue(is_hamiltonian_path(path))

    def test_is_hamiltonian_cycle_check(self):
        """Код Грея образует Гамильтонов цикл Q6."""
        path = gray_code_cycle()
        self.assertTrue(is_hamiltonian_cycle(path))

    def test_invalid_path_rejected(self):
        """Список с повтором — не Гамильтонов путь."""
        self.assertFalse(is_hamiltonian_path([0, 1, 0, 2]))

    def test_invalid_cycle_rejected(self):
        """Путь с несмежными концами — не Гамильтонов цикл."""
        bad = list(range(SIZE))  # не Гамильтонов цикл (соседи могут не совпадать)
        # Проверяем, что случайный список не является циклом
        self.assertFalse(is_hamiltonian_cycle(bad))


# ---------------------------------------------------------------------------
# Перколяция
# ---------------------------------------------------------------------------

class TestPercolation(unittest.TestCase):
    def test_bond_percolation_zero_fail(self):
        """При p_fail=0: все рёбра целы → P(связь) = 1."""
        prob = bond_percolation(0, 63, p_fail=0.0, n_trials=100, seed=0)
        self.assertAlmostEqual(prob, 1.0)

    def test_bond_percolation_all_fail(self):
        """При p_fail=1: все рёбра удалены → P(связь) = 0."""
        prob = bond_percolation(0, 63, p_fail=1.0, n_trials=100, seed=0)
        self.assertAlmostEqual(prob, 0.0)

    def test_bond_percolation_range(self):
        """P(связь) ∈ [0, 1]."""
        prob = bond_percolation(0, 42, p_fail=0.3, n_trials=200, seed=7)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_bond_percolation_monotone(self):
        """P(связь) убывает с ростом p_fail."""
        probs = [bond_percolation(0, 63, p, n_trials=300, seed=0)
                 for p in [0.0, 0.3, 0.6, 0.9]]
        for i in range(len(probs) - 1):
            self.assertGreaterEqual(probs[i], probs[i + 1] - 0.1)  # нестрогий монотон

    def test_site_percolation_range(self):
        prob = site_percolation(p_fail=0.5, n_trials=100, seed=0)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_site_percolation_zero_fail(self):
        """При p_fail=0: вся сеть цела → все узлы в одной компоненте."""
        prob = site_percolation(p_fail=0.0, n_trials=10, seed=0)
        self.assertAlmostEqual(prob, 1.0)

    def test_k_fault_diameter_positive(self):
        d = k_fault_diameter(k=3, n_trials=10, seed=0)
        self.assertGreaterEqual(d, 0)


class TestAdaptiveRoute(unittest.TestCase):
    """Тесты adaptive_route — альтернативная маршрутизация с учётом отказов."""

    def test_route_to_self(self):
        path = adaptive_route(0, 0)
        self.assertIsNotNone(path)
        self.assertEqual(path, [0])

    def test_route_length(self):
        """Путь имеет длину = hamming + 1."""
        from libs.hexcore.hexcore import hamming
        path = adaptive_route(0, 63)
        self.assertIsNotNone(path)
        self.assertEqual(len(path) - 1, hamming(0, 63))

    def test_route_valid_steps(self):
        """Каждый переход — сосед (Хэмминг = 1)."""
        from libs.hexcore.hexcore import hamming
        path = adaptive_route(0, 42)
        self.assertIsNotNone(path)
        for a, b in zip(path, path[1:]):
            self.assertEqual(hamming(a, b), 1)

    def test_route_ends_at_dst(self):
        path = adaptive_route(7, 56)
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], 56)

    def test_no_faults_same_as_ecube(self):
        """Без отказов adaptive_route и ecube_route дают одинаковую длину."""
        from projects.hexnet.hexnet import ecube_route
        from libs.hexcore.hexcore import hamming
        for src, dst in [(0, 63), (3, 60), (15, 48)]:
            p = adaptive_route(src, dst)
            self.assertIsNotNone(p)
            self.assertEqual(len(p) - 1, hamming(src, dst))

    def test_avoids_failed_nodes(self):
        """Путь не проходит через отказавшие узлы (если путь существует)."""
        failed = {1, 2, 4, 8}
        path = adaptive_route(0, 63, failed_nodes=failed)
        if path is not None:
            for v in path[1:-1]:  # конечные точки не проверяем
                self.assertNotIn(v, failed)

    def test_src_failed_returns_none(self):
        path = adaptive_route(0, 63, failed_nodes={0})
        self.assertIsNone(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
