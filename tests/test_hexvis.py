"""Тесты hexvis — визуализация Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexvis.hexvis import (
    render_grid, render_path, render_hexagram, render_transition,
    to_dot, to_svg, _GRAY3,
)
from libs.hexcore.hexcore import neighbors, hamming, yang_count, SIZE


class TestRenderGrid(unittest.TestCase):
    def test_contains_all_hexagrams(self):
        """Решётка содержит все 64 гексаграммы."""
        grid = render_grid(color=False)
        for h in range(SIZE):
            self.assertIn(str(h), grid)

    def test_grid_row_count(self):
        """Сетка 8×8: заголовок столбцов + separator + 8 строк = 10."""
        grid = render_grid(color=False)
        lines = grid.split('\n')
        self.assertEqual(len(lines), 10)   # header + sep + 8 rows

    def test_grid_row_count_with_title(self):
        """Сетка с заголовком: title + header + sep + 8 строк = 11."""
        grid = render_grid(color=False, title='Q6')
        lines = grid.split('\n')
        self.assertEqual(len(lines), 11)   # title + header + sep + 8 rows

    def test_highlight_marked(self):
        """Выделенные вершины отмечены символом *."""
        grid = render_grid(highlights={0, 42}, color=False)
        self.assertIn('*00', grid)
        self.assertIn('*42', grid)

    def test_gray3_coverage(self):
        """_GRAY3 содержит все числа 0..7."""
        self.assertEqual(set(_GRAY3), set(range(8)))

    def test_adjacent_cells_differ_by_one_bit(self):
        """
        Соседние клетки по строке (один шаг по Gray-коду столбцов)
        отличаются ровно на 1 бит в битах 0-2.
        """
        for row_idx in range(7):
            for col_idx in range(7):
                row_g = _GRAY3[row_idx]
                col_g1 = _GRAY3[col_idx]
                col_g2 = _GRAY3[col_idx + 1]
                h1 = (row_g << 3) | col_g1
                h2 = (row_g << 3) | col_g2
                self.assertEqual(hamming(h1, h2), 1,
                                 f"Нарушена смежность в строке: {h1} vs {h2}")

    def test_title_in_output(self):
        grid = render_grid(color=False, title='Тест')
        self.assertIn('Тест', grid)

    def test_empty_highlights(self):
        """Без выделений нет символа *."""
        grid = render_grid(color=False)
        self.assertNotIn('*', grid)

    def test_color_mode_produces_output(self):
        """color=True не ломает вывод."""
        grid = render_grid(color=True)
        self.assertGreater(len(grid), 0)

    def test_color_highlight_produces_output(self):
        """color=True + highlights работает без ошибок."""
        grid = render_grid(color=True, highlights={0, 42})
        self.assertGreater(len(grid), 0)


class TestRenderPath(unittest.TestCase):
    def test_empty_path(self):
        result = render_path([], color=False)
        self.assertIn('пустой', result)

    def test_single_node(self):
        result = render_path([42], color=False)
        self.assertIn('42', result)
        self.assertIn('0', result)   # шаг 0

    def test_path_contains_nodes(self):
        path = [0, 1, 3, 7]
        result = render_path(path, color=False)
        for h in path:
            self.assertIn(str(h), result)

    def test_path_step_count(self):
        path = [0, 1, 3, 7]
        result = render_path(path, color=False)
        self.assertIn('3', result)   # 3 шага

    def test_path_with_bits(self):
        result = render_path([0, 1], color=False, show_bits=True)
        self.assertIn('000000', result)   # bits of 0
        self.assertIn('000001', result)   # bits of 1

    def test_hamming_shown(self):
        result = render_path([0, 63], color=False)
        # [0→1→...→63] — показывает hamming distance
        # Но path=[0,63] сам по себе — 2 узла, расстояние 6
        self.assertIn('6', result)

    def test_color_path_no_crash(self):
        """render_path с color=True работает без ошибок."""
        result = render_path([0, 1, 3, 7], color=True)
        self.assertGreater(len(result), 0)


class TestRenderHexagram(unittest.TestCase):
    def test_contains_hexagram_number(self):
        result = render_hexagram(42, color=False)
        self.assertIn('42', result)

    def test_contains_bits(self):
        result = render_hexagram(0, color=False)
        self.assertIn('000000', result)

    def test_contains_yang(self):
        result = render_hexagram(63, color=False)
        self.assertIn('6', result)   # ян=6

    def test_multiline(self):
        result = render_hexagram(0, color=False)
        lines = result.split('\n')
        self.assertGreater(len(lines), 2)   # заголовок + 6 черт


class TestRenderTransition(unittest.TestCase):
    def test_valid_transition(self):
        result = render_transition(0, 1, color=False)
        self.assertIn('0', result)
        self.assertIn('1', result)
        self.assertIn('бит 0', result)

    def test_invalid_transition_raises(self):
        with self.assertRaises(ValueError):
            render_transition(0, 42, color=False)   # hamming=3

    def test_direction_up(self):
        """0→1: бит 0 переходит 0→1 (↑)."""
        result = render_transition(0, 1, color=False)
        self.assertIn('↑', result)

    def test_direction_down(self):
        """1→0: бит 0 переходит 1→0 (↓)."""
        result = render_transition(1, 0, color=False)
        self.assertIn('↓', result)

    def test_color_transition_no_crash(self):
        """render_transition с color=True работает без ошибок."""
        result = render_transition(0, 1, color=True)
        self.assertGreater(len(result), 0)

    def test_color_transition_down_no_crash(self):
        """render_transition color=True, направление ↓."""
        result = render_transition(1, 0, color=True)
        self.assertGreater(len(result), 0)


class TestToDot(unittest.TestCase):
    def test_dot_header(self):
        dot = to_dot({0, 1, 3}, title='test')
        self.assertIn('graph "test"', dot)

    def test_dot_contains_vertices(self):
        dot = to_dot({0, 1, 2})
        self.assertIn(' 0 [', dot)
        self.assertIn(' 1 [', dot)
        self.assertIn(' 2 [', dot)

    def test_dot_contains_edge(self):
        """0 и 1 — соседи, ребро должно быть в DOT."""
        dot = to_dot({0, 1})
        self.assertIn('0 -- 1', dot)

    def test_dot_path_edge_bold(self):
        """Рёбра пути должны быть bold."""
        dot = to_dot({0, 1, 3}, path=[0, 1, 3])
        self.assertIn('bold', dot)

    def test_dot_directed(self):
        dot = to_dot({0, 1}, directed=True)
        self.assertIn('digraph', dot)
        self.assertIn('->', dot)

    def test_dot_valid_structure(self):
        """DOT начинается с graph/digraph и заканчивается }."""
        dot = to_dot({0, 1, 2})
        self.assertTrue(dot.strip().startswith('graph'))
        self.assertTrue(dot.strip().endswith('}'))


class TestToSvg(unittest.TestCase):
    def test_svg_header(self):
        svg = to_svg({0, 1, 2})
        self.assertIn('<svg', svg)
        self.assertIn('</svg>', svg)

    def test_svg_contains_circles(self):
        svg = to_svg({0, 1})
        self.assertIn('<circle', svg)

    def test_svg_vertex_count(self):
        """По одному <circle> на каждую вершину."""
        svg = to_svg({0, 1, 3})
        self.assertEqual(svg.count('<circle'), 3)

    def test_svg_contains_edge(self):
        """0 и 1 — соседи, должна быть линия."""
        svg = to_svg({0, 1})
        self.assertIn('<line', svg)

    def test_svg_title(self):
        svg = to_svg({0, 1}, title='MyTest')
        self.assertIn('MyTest', svg)

    def test_svg_layout_circle(self):
        """Оба layout генерируют валидный SVG."""
        for layout in ('grid', 'circle'):
            svg = to_svg({0, 1, 3, 7}, layout=layout)
            self.assertIn('<svg', svg)

    def test_svg_path_highlighted(self):
        """Рёбра пути отрисовываются ярче."""
        svg_with_path = to_svg({0, 1, 3}, path=[0, 1, 3])
        svg_no_path = to_svg({0, 1, 3})
        # SVG с путём содержит белые линии (#ffffff)
        self.assertIn('#ffffff', svg_with_path)

    def test_svg_empty_vertices(self):
        """Пустой набор вершин — валидный SVG."""
        svg = to_svg(set())
        self.assertIn('<svg', svg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
