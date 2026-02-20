"""Тесты для hexcubenets — развёртки куба."""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexcubenets.hexcubenets import CubeNets, Net, _is_spanning_tree, CUBE_EDGES


class TestSpanningTree(unittest.TestCase):
    def test_valid_tree(self):
        """5 рёбер, образующих дерево."""
        # Путь T-F-B-L-R + K соединим K-B
        tree_edges = frozenset([
            ("T", "F"), ("F", "B"), ("B", "L"), ("L", "K"), ("K", "R")
        ])
        # Приведём к сортированным парам
        edge_tuples = {tuple(sorted(e)) for e in tree_edges}
        self.assertTrue(_is_spanning_tree(edge_tuples))

    def test_cycle_not_tree(self):
        """Цикл → не дерево."""
        # T-F-B-L-T — цикл
        cyclic = {
            tuple(sorted(("T", "F"))),
            tuple(sorted(("F", "B"))),
            tuple(sorted(("B", "L"))),
            tuple(sorted(("L", "T"))),
            tuple(sorted(("T", "K"))),
        }
        self.assertFalse(_is_spanning_tree(cyclic))

    def test_wrong_size(self):
        """Не 5 рёбер → False."""
        self.assertFalse(_is_spanning_tree(frozenset()))


class TestCubeNets(unittest.TestCase):
    def setUp(self):
        self.cn = CubeNets()

    def test_enumerate_returns_11(self):
        """Должно быть ровно 11 развёрток."""
        nets = self.cn.enumerate_all()
        self.assertEqual(len(nets), 11)

    def test_get_net_valid_index(self):
        net = self.cn.get_net(0)
        self.assertIsInstance(net, Net)

    def test_get_net_invalid_index(self):
        with self.assertRaises(ValueError):
            self.cn.get_net(11)
        with self.assertRaises(ValueError):
            self.cn.get_net(-1)

    def test_net_has_6_faces(self):
        """Каждая развёртка содержит ровно 6 граней."""
        for net in self.cn.enumerate_all():
            self.assertEqual(len(net.coords), 6)

    def test_net_symmetry_valid(self):
        """Симметрия каждой развёртки — одна из допустимых строк."""
        valid = {"mirror", "central", "none"}
        for net in self.cn.enumerate_all():
            self.assertIn(net.symmetry(), valid)

    def test_classify_total(self):
        """Симметричных = 6, асимметричных = 5."""
        cls = self.cn.classify()
        non_none = sum(v for k, v in cls.items() if k != "none")
        none_count = cls.get("none", 0)
        self.assertEqual(non_none, 6)
        self.assertEqual(none_count, 5)

    def test_prove_count(self):
        """Перебор: 792 всего, 384 валидных."""
        res = self.cn.prove_count()
        self.assertEqual(res["total_subsets"], 792)
        self.assertEqual(res["valid"], 384)
        self.assertEqual(res["invalid"], 408)
        self.assertTrue(res["check_total"])
        self.assertTrue(res["check_valid"])

    def test_tetrahedron_nets(self):
        self.assertEqual(self.cn.tetrahedron_nets(), 2)

    def test_octahedron_nets(self):
        self.assertEqual(self.cn.octahedron_nets(), 11)

    def test_hypercube_3(self):
        self.assertEqual(self.cn.hypercube_nets(3), 11)

    def test_hypercube_4(self):
        self.assertEqual(self.cn.hypercube_nets(4), 261)

    def test_net_ascii_contains_faces(self):
        """ASCII-представление содержит все 6 граней."""
        from projects.hexcubenets.hexcubenets import FACES
        for net in self.cn.enumerate_all():
            ascii_str = net.to_ascii()
            for face in FACES:
                self.assertIn(face, ascii_str, f"Грань {face} отсутствует в развёртке {net.index}")


    def test_is_valid_net_valid(self):
        """Разрезаем 7 рёбер так, что остаются 5 образующих дерево."""
        # Оставляем: T-F, T-K, T-L, T-R, B-F (дерево: T-звезда + B через F)
        # Разрезаем остальные 7: B-K, B-L, B-R, F-L, F-R, K-L, K-R
        cut = ['B-K', 'B-L', 'B-R', 'F-L', 'F-R', 'K-L', 'K-R']
        self.assertTrue(self.cn.is_valid_net(cut))

    def test_is_valid_net_invalid(self):
        """Разрезаем 7 рёбер так, что остаток образует цикл."""
        # Оставляем: B-R, F-L, F-R, K-L, K-R (содержит цикл через K и R)
        # Разрезаем: T-F, T-K, T-L, T-R, B-F, B-K, B-L
        cut = ['T-F', 'T-K', 'T-L', 'T-R', 'B-F', 'B-K', 'B-L']
        self.assertFalse(self.cn.is_valid_net(cut))

    def test_to_colored_ascii_contains_faces(self):
        """to_colored_ascii() содержит метки всех 6 граней."""
        from projects.hexcubenets.hexcubenets import FACES
        net = self.cn.get_net(0)
        s = net.to_colored_ascii()
        self.assertIsInstance(s, str)
        for face in FACES:
            self.assertIn(face, s)

    def test_hypercube_unknown_returns_string(self):
        """hypercube_nets(5) возвращает строку (неизвестный случай)."""
        result = self.cn.hypercube_nets(5)
        self.assertIsInstance(result, str)

    def test_net_index_attribute(self):
        """Net.index совпадает с позицией в enumerate_all()."""
        for i, net in enumerate(self.cn.enumerate_all()):
            self.assertEqual(net.index, i)

    def test_get_net_index_matches(self):
        """get_net(k).index == k."""
        for k in range(11):
            net = self.cn.get_net(k)
            self.assertEqual(net.index, k)

    def test_to_ascii_contains_header(self):
        """to_ascii() содержит слово 'Развёртка'."""
        net = self.cn.get_net(0)
        s = net.to_ascii()
        self.assertIn("Развёртка", s)

    def test_classify_keys_present(self):
        """classify() содержит ключи 'mirror', 'central', 'none'."""
        cls = self.cn.classify()
        self.assertIn("mirror", cls)
        self.assertIn("central", cls)
        self.assertIn("none", cls)

    def test_to_colored_ascii_with_custom_colors(self):
        """to_colored_ascii с явными цветами не падает."""
        net = self.cn.get_net(0)
        colors = {f: f.lower() for f in ["T", "B", "F", "K", "L", "R"]}
        s = net.to_colored_ascii(colors=colors)
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)


if __name__ == "__main__":
    unittest.main()
