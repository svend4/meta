"""Тесты hexvis — визуализация Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexvis.hexvis import (
    render_grid, render_path, render_hexagram, render_transition,
    to_dot, to_svg, _GRAY3,
    render_glyph, render_hasse_glyphs,
    render_glyph_grid, to_svg_hasse_glyphs,
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

    def test_svg_with_highlights(self):
        """to_svg с highlights рисует выделенную вершину."""
        svg = to_svg({0, 1, 3}, highlights={0})
        self.assertIn('<svg', svg)
        # Выделенная вершина имеет белый stroke
        self.assertIn('#ffffff', svg)

    def test_svg_edge_with_missing_vertex(self):
        """Рёбра с вершинами вне vertices пропускаются (continue)."""
        # Edge (0, 42) but 42 not in vertices → should be skipped gracefully
        svg = to_svg({0, 1}, edges={(0, 1), (0, 42)})
        self.assertIn('<svg', svg)

    def test_svg_path_with_vertex_outside_set(self):
        """Вершина пути вне vertices пропускается (continue)."""
        # Path includes vertex 42 which is not in vertices {0, 1}
        svg = to_svg({0, 1}, path=[0, 1, 42])
        self.assertIn('<svg', svg)


class TestRenderGlyph(unittest.TestCase):
    def test_returns_three_rows(self):
        """render_glyph always returns exactly 3 strings."""
        for h in range(64):
            self.assertEqual(len(render_glyph(h)), 3)

    def test_rows_width_3(self):
        """Every row of every glyph is exactly 3 characters wide."""
        for h in range(64):
            for row in render_glyph(h):
                self.assertEqual(len(row), 3, f"h={h}, row={row!r}")

    def test_all_64_unique(self):
        """All 64 glyphs are visually distinct (different string tuples)."""
        glyphs = [tuple(render_glyph(h)) for h in range(64)]
        self.assertEqual(len(set(glyphs)), 64)

    def test_glyph_zero_all_spaces(self):
        """h=0 (no bits set) → all three rows are spaces."""
        rows = render_glyph(0)
        for row in rows:
            self.assertEqual(row, '   ')

    def test_glyph_63_full(self):
        """h=63 (all bits set) → top bar, X centre, bottom bar."""
        rows = render_glyph(63)
        self.assertEqual(rows[0], ' _ ')
        self.assertEqual(rows[1], '|X|')
        self.assertEqual(rows[2], ' _ ')

    def test_glyph_bit0_top_bar(self):
        """bit 0 = top bar: h=1 → row0 contains '_'."""
        rows = render_glyph(1)
        self.assertIn('_', rows[0])

    def test_glyph_bit1_bottom_bar(self):
        """bit 1 = bottom bar: h=2 → row2 contains '_'."""
        rows = render_glyph(2)
        self.assertIn('_', rows[2])

    def test_glyph_bit2_left_bar(self):
        """bit 2 = left bar: h=4 → row1 starts with '|'."""
        rows = render_glyph(4)
        self.assertEqual(rows[1][0], '|')

    def test_glyph_bit3_right_bar(self):
        """bit 3 = right bar: h=8 → row1 ends with '|'."""
        rows = render_glyph(8)
        self.assertEqual(rows[1][2], '|')

    def test_glyph_bit4_backslash(self):
        """bit 4 only → centre of row1 is '\\'."""
        rows = render_glyph(16)
        self.assertEqual(rows[1][1], '\\')

    def test_glyph_bit5_slash(self):
        """bit 5 only → centre of row1 is '/'."""
        rows = render_glyph(32)
        self.assertEqual(rows[1][1], '/')

    def test_glyph_both_diag(self):
        """bits 4 and 5 → centre of row1 is 'X'."""
        rows = render_glyph(48)   # 0b110000
        self.assertEqual(rows[1][1], 'X')

    def test_no_segments_no_bar_or_pipe(self):
        """h=0 → no '_', '|', '\\', '/', 'X' characters."""
        rows = render_glyph(0)
        combined = ''.join(rows)
        for ch in '_|/\\X':
            self.assertNotIn(ch, combined)


class TestRenderHasseGlyphs(unittest.TestCase):
    def test_nonempty(self):
        out = render_hasse_glyphs(color=False)
        self.assertGreater(len(out), 0)

    def test_no_crash_color(self):
        """Does not raise with color=True."""
        render_hasse_glyphs(color=True)

    def test_seven_rank_bands(self):
        """Output has content for all 7 ranks (0–6)."""
        out = render_hasse_glyphs(color=False)
        # Use 'if l' (not 'if l.strip()') so rank-0's all-space rows are counted.
        lines = [l for l in out.split('\n') if l]
        # 7 ranks × 3 glyph rows = 21 lines (rank-0 rows are spaces, still non-empty)
        self.assertGreaterEqual(len(lines), 21)

    def test_show_numbers_contains_63(self):
        """show_numbers=True → '63' appears in the output."""
        out = render_hasse_glyphs(color=False, show_numbers=True)
        self.assertIn('63', out)

    def test_show_numbers_contains_0(self):
        out = render_hasse_glyphs(color=False, show_numbers=True)
        self.assertIn(' 0', out)

    def test_highlight_no_crash(self):
        """Highlighting a subset doesn't raise."""
        render_hasse_glyphs(color=True, highlights={0, 42, 63})

    def test_rank3_is_widest_band(self):
        """The rank-3 glyph rows are the widest lines in the output."""
        out = render_hasse_glyphs(color=False)
        line_widths = [len(l) for l in out.split('\n') if l.strip()]
        max_w = max(line_widths)
        # rank-3 band: 20 glyphs × 3 chars + 19 separators = 79 chars
        self.assertEqual(max_w, 79)

    def test_rank0_single_glyph_shorter_than_rank3(self):
        """Rank-0 row (1 glyph, centred) is shorter than rank-3 row (20 glyphs)."""
        out = render_hasse_glyphs(color=False)
        lines = [l for l in out.split('\n') if l]  # non-empty strings
        max_w = max(len(l) for l in lines)   # rank-3 row = 79
        min_w = min(len(l) for l in lines)   # rank-0/6 row < 79
        self.assertEqual(max_w, 79)
        self.assertLess(min_w, 79)

    def test_glyph_segments_visible_in_rank1(self):
        """Rank-1 elements (single-bit) each show exactly one active segment."""
        out = render_hasse_glyphs(color=False)
        # rank-1 elements: 1,2,4,8,16,32 — each has one bar or diagonal
        # The combined output must contain at least one '_' and one '|'
        self.assertIn('_', out)
        self.assertIn('|', out)


class TestRenderGlyphGrid(unittest.TestCase):
    def test_nonempty(self):
        out = render_glyph_grid(color=False)
        self.assertGreater(len(out), 0)

    def test_contains_separator(self):
        """Grid contains a column-header separator line."""
        out = render_glyph_grid(color=False)
        self.assertIn('─', out)

    def test_8x8_glyph_rows(self):
        """8 Gray-code rows × 3 glyph lines each = 24 glyph content lines."""
        out = render_glyph_grid(color=False)
        # Count lines that start with the row prefix pattern '  xxx │ '
        glyph_lines = [l for l in out.split('\n') if '│ ' in l]
        self.assertEqual(len(glyph_lines), 8 * 3)

    def test_all_segments_visible(self):
        """Grid output contains '_', '|', '\\', '/' and 'X' characters."""
        out = render_glyph_grid(color=False)
        for ch in '_|/\\X':
            self.assertIn(ch, out, f"Missing character {ch!r} in glyph grid")

    def test_gray_code_labels_present(self):
        """All 8 Gray-code 3-bit labels appear in the header."""
        out = render_glyph_grid(color=False)
        for g in _GRAY3:
            self.assertIn(format(g, '03b'), out)

    def test_title_displayed(self):
        out = render_glyph_grid(color=False, title='Test title')
        self.assertIn('Test title', out)

    def test_highlight_no_crash(self):
        render_glyph_grid(color=True, highlights={0, 42, 63})

    def test_neighbor_cells_differ_by_one_bit(self):
        """Adjacent cells in a row are Q6 neighbours (hamming distance = 1)."""
        for row_idx in range(7):
            for col_idx in range(7):
                h1 = (_GRAY3[row_idx] << 3) | _GRAY3[col_idx]
                h2 = (_GRAY3[row_idx] << 3) | _GRAY3[col_idx + 1]
                self.assertEqual(hamming(h1, h2), 1)

    def test_glyph_63_appears(self):
        """The full glyph (h=63) row1='|X|' appears in the grid."""
        out = render_glyph_grid(color=False)
        self.assertIn('|X|', out)


class TestToSvgHasseGlyphs(unittest.TestCase):
    def test_returns_svg(self):
        svg = to_svg_hasse_glyphs()
        self.assertIn('<svg', svg)
        self.assertIn('</svg>', svg)

    def test_contains_64_rects(self):
        """One <rect> per glyph node (plus the background rect = 65 total)."""
        svg = to_svg_hasse_glyphs()
        # background rect + 64 node rects = 65
        self.assertEqual(svg.count('<rect'), 65)

    def test_contains_paths(self):
        """Non-empty elements produce <path> segments."""
        svg = to_svg_hasse_glyphs()
        self.assertIn('<path', svg)

    def test_title_in_svg(self):
        svg = to_svg_hasse_glyphs(title='MyHasse')
        self.assertIn('MyHasse', svg)

    def test_edges_present_by_default(self):
        """Cover-relation edges (lines) are drawn by default."""
        svg = to_svg_hasse_glyphs(show_edges=True)
        self.assertIn('<line', svg)

    def test_no_edges_option(self):
        """show_edges=False removes all <line> elements."""
        svg = to_svg_hasse_glyphs(show_edges=False)
        self.assertNotIn('<line', svg)

    def test_highlight_no_crash(self):
        to_svg_hasse_glyphs(highlights={0, 42, 63})

    def test_no_color_no_crash(self):
        to_svg_hasse_glyphs(color=False)

    def test_custom_cell_size(self):
        """Different cell sizes produce different SVG dimensions."""
        svg_small = to_svg_hasse_glyphs(cell=12)
        svg_large = to_svg_hasse_glyphs(cell=24)
        # Extract width attribute
        import re
        w_small = int(re.search(r'width="(\d+)"', svg_small).group(1))
        w_large = int(re.search(r'width="(\d+)"', svg_large).group(1))
        self.assertLess(w_small, w_large)

    def test_element_0_has_no_path(self):
        """h=0 has no segments, so its node produces no <path>."""
        # All segments come from bits 1-6; h=0 has no bits set → no path.
        # The total path count equals the number of non-zero elements (63).
        svg = to_svg_hasse_glyphs(color=False, show_edges=False)
        self.assertEqual(svg.count('<path'), 63)


if __name__ == '__main__':
    unittest.main(verbosity=2)
