"""Tests for hexglyph — Solan font / Stargate glyph system."""

import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from projects.hexglyph.hexglyph import (
    font_data,
    font_path,
    font_info,
    glyph_bitmap,
    render_bitmap,
    detect_segments,
    char_to_h,
    h_to_char,
    encode,
    decode,
    viewer_path,
    RUSSIAN_PHONETIC,
    PHONETIC_H_TO_RU,
    _CHARSET_64,
    _Q6_ORDER,
)


class TestFontData(unittest.TestCase):

    def test_font_data_loads(self):
        data = font_data(4)
        self.assertIsInstance(data, dict)

    def test_font_name_solan4(self):
        self.assertEqual(font_data(4)["name"], "myFront4Solan4")

    def test_font_name_solan3(self):
        self.assertEqual(font_data(3)["name"], "myFront4Solan3")

    def test_copyright_svend(self):
        self.assertEqual(font_data(4)["copy"], "svend")

    def test_glyph_count_71(self):
        data = font_data(4)
        keys = [k for k in data if k.isdigit()]
        self.assertEqual(len(keys), 71)

    def test_font_path_exists(self):
        self.assertTrue(font_path(4).exists())

    def test_font_path_is_ttf(self):
        self.assertEqual(font_path(4).suffix, ".ttf")


class TestGlyphBitmap(unittest.TestCase):

    def test_bitmap_returns_8_rows(self):
        bm = glyph_bitmap('A')
        self.assertEqual(len(bm), 8)

    def test_bitmap_values_are_uint8(self):
        bm = glyph_bitmap('Z')
        for v in bm:
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 255)

    def test_render_bitmap_returns_8_strings(self):
        lines = render_bitmap('A')
        self.assertEqual(len(lines), 8)

    def test_render_bitmap_width_8(self):
        lines = render_bitmap('0')
        for line in lines:
            self.assertEqual(len(line), 8)

    def test_render_bitmap_only_block_chars(self):
        lines = render_bitmap('B')
        for line in lines:
            self.assertTrue(all(c in '█·' for c in line))

    def test_digit_0_is_symmetric(self):
        bm = glyph_bitmap('0')
        # rows 0,1 == rows 6,7 (vertical symmetry)
        self.assertEqual(bm[0], bm[6])
        self.assertEqual(bm[1], bm[7])

    def test_unknown_char_raises(self):
        with self.assertRaises(KeyError):
            glyph_bitmap('~')  # code 126, not in font


class TestQ6Mapping(unittest.TestCase):

    def test_charset_64_length(self):
        self.assertEqual(len(_CHARSET_64), 64)

    def test_charset_64_unique(self):
        self.assertEqual(len(set(_CHARSET_64)), 64)

    def test_q6_order_length(self):
        self.assertEqual(len(_Q6_ORDER), 64)

    def test_q6_order_is_permutation(self):
        self.assertEqual(sorted(_Q6_ORDER), list(range(64)))

    def test_q6_order_sorted_by_hamming(self):
        weights = [bin(h).count('1') for h in _Q6_ORDER]
        self.assertEqual(weights, sorted(weights))

    def test_char_to_h_zero(self):
        # '0' is first char → maps to h=0 (lightest vertex)
        self.assertEqual(char_to_h('0'), 0)

    def test_char_to_h_range(self):
        for ch in _CHARSET_64:
            h = char_to_h(ch)
            self.assertIsNotNone(h)
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(h, 63)

    def test_h_to_char_range(self):
        for h in range(64):
            ch = h_to_char(h)
            self.assertIsNotNone(ch)
            self.assertEqual(len(ch), 1)

    def test_roundtrip_char_h_char(self):
        for ch in _CHARSET_64:
            self.assertEqual(h_to_char(char_to_h(ch)), ch)

    def test_roundtrip_h_char_h(self):
        for h in range(64):
            self.assertEqual(char_to_h(h_to_char(h)), h)

    def test_unknown_char_returns_none(self):
        self.assertIsNone(char_to_h('~'))

    def test_unknown_h_returns_none(self):
        self.assertIsNone(h_to_char(64))
        self.assertIsNone(h_to_char(-1))


class TestEncodeDecode(unittest.TestCase):

    def test_encode_returns_list(self):
        self.assertIsInstance(encode("ABC"), list)

    def test_encode_values_in_range(self):
        for h in encode("Hello"):
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(h, 63)

    def test_decode_returns_str(self):
        self.assertIsInstance(decode([0, 1, 2]), str)

    def test_encode_decode_roundtrip(self):
        text = "Hello"
        self.assertEqual(decode(encode(text)), text)

    def test_encode_skip_unknown(self):
        # '~' not in charset; should be silently skipped
        result = encode("A~B")
        self.assertEqual(len(result), 2)

    def test_encode_empty(self):
        self.assertEqual(encode(""), [])

    def test_decode_empty(self):
        self.assertEqual(decode([]), "")

    def test_encode_full_charset(self):
        result = encode(_CHARSET_64)
        self.assertEqual(len(result), 64)
        self.assertEqual(sorted(result), list(range(64)))


class TestFontInfo(unittest.TestCase):

    def test_font_info_keys(self):
        info = font_info(4)
        for key in ["name", "copyright", "glyph_count", "q6_chars", "ttf_path"]:
            self.assertIn(key, info)

    def test_font_info_q6_chars_64(self):
        self.assertEqual(font_info(4)["q6_chars"], 64)

    def test_font_info_glyph_count_71(self):
        self.assertEqual(font_info(4)["glyph_count"], 71)


class TestDetectSegments(unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        result = detect_segments('A')
        for key in ('T', 'B', 'L', 'R', 'D1', 'D2', 'h', 'bits'):
            self.assertIn(key, result)

    def test_h_is_int_in_range(self):
        for ch in 'AHELTBaez0':
            seg = detect_segments(ch)
            self.assertIsInstance(seg['h'], int)
            self.assertGreaterEqual(seg['h'], 0)
            self.assertLessEqual(seg['h'], 63)

    def test_bits_length_6(self):
        self.assertEqual(len(detect_segments('A')['bits']), 6)

    def test_bits_consistent_with_h(self):
        seg = detect_segments('E')
        self.assertEqual(int(seg['bits'], 2), seg['h'])

    # Confirmed pixel-decoded mappings
    def test_H_is_LR_only(self):
        # 'H' visually = two vertical bars = L+R = h=12
        seg = detect_segments('H')
        self.assertFalse(seg['T'])
        self.assertFalse(seg['B'])
        self.assertTrue(seg['L'])
        self.assertTrue(seg['R'])
        self.assertEqual(seg['h'], 12)

    def test_A_is_frame(self):
        # 'A' = full square frame = T+B+L+R = h=15
        seg = detect_segments('A')
        self.assertTrue(seg['T'])
        self.assertTrue(seg['B'])
        self.assertTrue(seg['L'])
        self.assertTrue(seg['R'])
        self.assertEqual(seg['h'], 15)

    def test_0_is_diagonals(self):
        # '0' = X shape = D1+D2 = h=48
        seg = detect_segments('0')
        self.assertTrue(seg['D1'])
        self.assertTrue(seg['D2'])
        self.assertFalse(seg['T'])
        self.assertFalse(seg['B'])
        self.assertEqual(seg['h'], 48)

    def test_E_is_full(self):
        # 'E' = ⊠ = all 6 segments = h=63
        self.assertEqual(detect_segments('E')['h'], 63)

    def test_h_formula_correct(self):
        # h = T*1 + B*2 + L*4 + R*8 + D1*16 + D2*32
        seg = detect_segments('H')  # L+R only
        expected = seg['L'] * 4 + seg['R'] * 8
        self.assertEqual(seg['h'], expected)


class TestRussianPhonetic(unittest.TestCase):

    def test_russian_phonetic_has_core_letters(self):
        for letter in ('А', 'Р', 'Т'):
            self.assertIn(letter, RUSSIAN_PHONETIC)

    def test_high_confidence_mappings(self):
        # А=63, Р=15, Т=48 confirmed from image analysis
        self.assertEqual(RUSSIAN_PHONETIC['А']['h'], 63)
        self.assertEqual(RUSSIAN_PHONETIC['Р']['h'], 15)
        self.assertEqual(RUSSIAN_PHONETIC['Т']['h'], 48)

    def test_a_maps_to_full_symbol(self):
        # А = ⊠ full = h=63
        self.assertEqual(RUSSIAN_PHONETIC['А']['h'], 63)

    def test_phonetic_h_values_in_range(self):
        for letter, info in RUSSIAN_PHONETIC.items():
            self.assertGreaterEqual(info['h'], 0)
            self.assertLessEqual(info['h'], 63)

    def test_phonetic_h_to_ru_covers_high_confidence(self):
        # All high-confidence mappings appear in PHONETIC_H_TO_RU
        for letter, info in RUSSIAN_PHONETIC.items():
            if info['confidence'] == 'high':
                self.assertIn(info['h'], PHONETIC_H_TO_RU)

    def test_viewer_path_exists(self):
        self.assertTrue(viewer_path().exists())

    def test_viewer_is_html(self):
        self.assertEqual(viewer_path().suffix, '.html')

    def test_viewer_contains_font(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Solan4', content)
        self.assertIn('font-face', content)

    def test_viewer_triangle_has_64_cells(self):
        import re
        content = viewer_path().read_text(encoding='utf-8')
        cells = re.findall(r'class="tri-cell', content)
        self.assertEqual(len(cells), 64)

    def test_viewer_triangle_correct_row_counts(self):
        import re, math
        content = viewer_path().read_text(encoding='utf-8')
        for k in range(7):
            expected = math.comb(6, k)
            cells_in_row = re.findall(rf'class="tri-cell w{k} ', content)
            self.assertEqual(len(cells_in_row), expected,
                             f'rank {k}: expected {expected} cells')

    def test_viewer_triangle_has_pixel_and_seq(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('class="tri-cell w6 pixel"', content)  # h=63 confirmed
        self.assertIn('class="tri-cell w0 seq"',   content)  # h=0  sequential

    def test_viewer_triangle_apex_is_E(self):
        # h=63 apex must show 'E' (pixel-confirmed all-segments glyph)
        import re
        content = viewer_path().read_text(encoding='utf-8')
        apex = re.search(r'class="tri-cell w6 pixel"[^>]*>(.)</div>', content)
        self.assertIsNotNone(apex)
        self.assertEqual(apex.group(1), 'E')


class TestSolanTriangle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_triangle import (
            _build_solan_map,
            _build_full_map,
            _compact4,
            _solan_cell,
            _hexvis_cell,
            _rank_elements,
            render_triangle,
            detection_stats,
            _SOLAN_MAP,
            _FULL_MAP,
        )
        cls._build_solan_map = staticmethod(_build_solan_map)
        cls._build_full_map  = staticmethod(_build_full_map)
        cls._compact4 = staticmethod(_compact4)
        cls._solan_cell = staticmethod(_solan_cell)
        cls._hexvis_cell = staticmethod(_hexvis_cell)
        cls._rank_elements = staticmethod(_rank_elements)
        cls.render_triangle = staticmethod(render_triangle)
        cls.detection_stats = staticmethod(detection_stats)
        cls._SOLAN_MAP = _SOLAN_MAP
        cls._FULL_MAP  = _FULL_MAP

    # --- _build_solan_map ---

    def test_solan_map_is_dict(self):
        m = self._build_solan_map()
        self.assertIsInstance(m, dict)

    def test_solan_map_keys_in_range(self):
        for h in self._SOLAN_MAP:
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(h, 63)

    def test_solan_map_values_are_single_chars(self):
        for ch in self._SOLAN_MAP.values():
            self.assertEqual(len(ch), 1)

    def test_solan_map_has_at_least_20_entries(self):
        self.assertGreaterEqual(len(self._SOLAN_MAP), 20)

    def test_solan_map_includes_confirmed_vertices(self):
        # h=12 → 'H', h=15 → 'A', h=48 → '0', h=63 → 'E'
        self.assertIn(12, self._SOLAN_MAP)
        self.assertIn(15, self._SOLAN_MAP)
        self.assertIn(48, self._SOLAN_MAP)
        self.assertIn(63, self._SOLAN_MAP)

    # --- _build_full_map / _FULL_MAP ---

    def test_full_map_is_dict(self):
        m = self._build_full_map()
        self.assertIsInstance(m, dict)

    def test_full_map_covers_all_64(self):
        self.assertEqual(len(self._FULL_MAP), 64)

    def test_full_map_keys_are_all_vertices(self):
        self.assertEqual(sorted(self._FULL_MAP.keys()), list(range(64)))

    def test_full_map_values_are_single_chars(self):
        for ch in self._FULL_MAP.values():
            self.assertEqual(len(ch), 1)

    def test_full_map_values_unique(self):
        # Each vertex gets its own character
        self.assertEqual(len(set(self._FULL_MAP.values())), 64)

    def test_full_map_contains_solan_map(self):
        # Every pixel-confirmed mapping appears in the full map
        for h in self._SOLAN_MAP:
            self.assertIn(h, self._FULL_MAP)

    def test_full_map_h0_is_zero_char(self):
        # h=0 (no segments) → char '0' (first char in sequential order)
        self.assertEqual(self._FULL_MAP[0], '0')

    def test_full_map_matches_h_to_char(self):
        # _FULL_MAP must agree with h_to_char for every vertex
        for h in range(64):
            self.assertEqual(self._FULL_MAP[h], h_to_char(h))

    # --- detection_stats (extended) ---

    def test_detection_stats_assigned_64(self):
        self.assertEqual(self.detection_stats()['assigned'], 64)

    def test_detection_stats_by_rank_has_assigned(self):
        st = self.detection_stats()
        for k, info in st['by_rank'].items():
            self.assertIn('assigned', info)
            self.assertIn('chars_seq', info)

    def test_detection_stats_by_rank_confirmed_plus_sequential_equals_total(self):
        st = self.detection_stats()
        for k, info in st['by_rank'].items():
            self.assertEqual(info['detected'] + info['assigned'], info['total'])

    # --- _compact4 ---

    def test_compact4_returns_4_rows(self):
        rows = glyph_bitmap('A')
        result = self._compact4(rows)
        self.assertEqual(len(result), 4)

    def test_compact4_each_row_length_4(self):
        rows = glyph_bitmap('E')
        for r in self._compact4(rows):
            self.assertEqual(len(r), 4)

    def test_compact4_only_bits(self):
        rows = glyph_bitmap('H')
        for r in self._compact4(rows):
            self.assertTrue(all(c in '01' for c in r))

    # --- _solan_cell ---

    def test_solan_cell_returns_4_lines(self):
        lines = self._solan_cell('A', '')
        self.assertEqual(len(lines), 4)

    def test_solan_cell_contains_block_chars(self):
        import re
        lines = self._solan_cell('E', '')
        for line in lines:
            stripped = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
            self.assertTrue(all(c in '█·' for c in stripped))

    # --- _hexvis_cell ---

    def test_hexvis_cell_returns_4_lines(self):
        lines = self._hexvis_cell(12, '')
        self.assertEqual(len(lines), 4)

    def test_hexvis_cell_lines_not_empty(self):
        lines = self._hexvis_cell(0, '')
        self.assertEqual(len(lines), 4)

    # --- _rank_elements ---

    def test_rank_elements_7_ranks(self):
        ranks = self._rank_elements()
        self.assertEqual(len(ranks), 7)

    def test_rank_elements_total_64(self):
        ranks = self._rank_elements()
        self.assertEqual(sum(len(r) for r in ranks), 64)

    def test_rank_elements_binomial_counts(self):
        import math
        ranks = self._rank_elements()
        for k, elems in enumerate(ranks):
            self.assertEqual(len(elems), math.comb(6, k))

    def test_rank_elements_correct_hamming_weight(self):
        ranks = self._rank_elements()
        for k, elems in enumerate(ranks):
            for h in elems:
                self.assertEqual(bin(h).count('1'), k)

    # --- render_triangle ---

    def test_render_triangle_returns_str(self):
        self.assertIsInstance(self.render_triangle(), str)

    def test_render_triangle_hexvis_mode(self):
        out = self.render_triangle(mode='hexvis', color=False)
        self.assertIn('hexvis', out)

    def test_render_triangle_side_mode(self):
        out = self.render_triangle(mode='side', color=False)
        self.assertIn('│', out)

    def test_render_triangle_solan_mode_has_block_chars(self):
        out = self.render_triangle(mode='solan', color=False)
        self.assertIn('█', out)

    def test_render_triangle_has_all_weight_labels(self):
        out = self.render_triangle(mode='solan', color=False)
        for k in range(7):
            self.assertIn(f'w={k}', out)

    def test_render_triangle_nocolor_no_ansi(self):
        import re
        out = self.render_triangle(mode='solan', color=False)
        # Only _RST codes may be present — test no color codes
        ansi_color = re.findall(r'\x1b\[3[0-9]', out)
        self.assertEqual(ansi_color, [])

    def test_render_triangle_color_has_ansi(self):
        out = self.render_triangle(mode='solan', color=True)
        self.assertIn('\033[', out)

    # --- detection_stats ---

    def test_detection_stats_keys(self):
        st = self.detection_stats()
        for key in ('total', 'detected', 'missing', 'detected_list', 'by_rank'):
            self.assertIn(key, st)

    def test_detection_stats_total_64(self):
        self.assertEqual(self.detection_stats()['total'], 64)

    def test_detection_stats_sum_correct(self):
        st = self.detection_stats()
        self.assertEqual(st['detected'] + st['missing'], 64)

    def test_detection_stats_at_least_20_detected(self):
        self.assertGreaterEqual(self.detection_stats()['detected'], 20)

    def test_detection_stats_by_rank_7_levels(self):
        self.assertEqual(len(self.detection_stats()['by_rank']), 7)

    def test_detection_stats_rank6_detected(self):
        # h=63 must always be detected (E = all segments)
        st = self.detection_stats()
        self.assertGreaterEqual(st['by_rank'][6]['detected'], 1)


class TestSolanPhonetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_phonetic import (
            phonetic_h, transliterate, encode_phonetic,
            render_phonetic_table, _solan_char, _glyph4,
        )
        cls.phonetic_h      = staticmethod(phonetic_h)
        cls.transliterate   = staticmethod(transliterate)
        cls.encode_phonetic = staticmethod(encode_phonetic)
        cls.render_table    = staticmethod(render_phonetic_table)
        cls._solan_char     = staticmethod(_solan_char)
        cls._glyph4         = staticmethod(_glyph4)

    # --- phonetic_h ---

    def test_phonetic_h_A_is_63(self):
        self.assertEqual(self.phonetic_h('А'), 63)

    def test_phonetic_h_R_is_15(self):
        self.assertEqual(self.phonetic_h('Р'), 15)

    def test_phonetic_h_T_is_48(self):
        self.assertEqual(self.phonetic_h('Т'), 48)

    def test_phonetic_h_unknown_returns_none(self):
        self.assertIsNone(self.phonetic_h('Я'))
        self.assertIsNone(self.phonetic_h('Z'))

    def test_phonetic_h_all_known_in_range(self):
        all16 = ('А', 'Р', 'Т', 'Л', 'Н', 'О', 'Б', 'М', 'В', 'Д',
                 'Г', 'У', 'И', 'З', 'Ж', 'Ч')
        for h in [self.phonetic_h(ru) for ru in all16]:
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(h, 63)

    def test_phonetic_h_G_is_49(self):
        self.assertEqual(self.phonetic_h('Г'), 49)

    def test_phonetic_h_U_is_51(self):
        self.assertEqual(self.phonetic_h('У'), 51)

    def test_phonetic_h_I_is_44(self):
        self.assertEqual(self.phonetic_h('И'), 44)

    def test_phonetic_h_Z_is_19(self):
        self.assertEqual(self.phonetic_h('З'), 19)

    def test_phonetic_h_ZH_is_28(self):
        self.assertEqual(self.phonetic_h('Ж'), 28)

    def test_phonetic_h_CH_is_35(self):
        self.assertEqual(self.phonetic_h('Ч'), 35)

    # --- transliterate ---

    def test_transliterate_A_gives_E(self):
        self.assertEqual(self.transliterate('А'), 'E')

    def test_transliterate_R_gives_A(self):
        self.assertEqual(self.transliterate('Р'), 'A')

    def test_transliterate_T_gives_L(self):
        self.assertEqual(self.transliterate('Т'), 'L')

    def test_transliterate_art_length(self):
        self.assertEqual(len(self.transliterate('АРТ')), 3)

    def test_transliterate_unknown_kept(self):
        self.assertIn('Я', self.transliterate('Я'))

    def test_transliterate_case_insensitive(self):
        self.assertEqual(self.transliterate('а'), self.transliterate('А'))

    def test_transliterate_empty(self):
        self.assertEqual(self.transliterate(''), '')

    # --- encode_phonetic ---

    def test_encode_returns_list_of_triples(self):
        result = self.encode_phonetic('АРТ')
        self.assertIsInstance(result, list)
        for item in result:
            self.assertEqual(len(item), 3)

    def test_encode_known_letter_has_h(self):
        orig, h, sc = self.encode_phonetic('А')[0]
        self.assertEqual(orig, 'А')
        self.assertEqual(h, 63)

    def test_encode_unknown_letter_has_none_h(self):
        _, h, _ = self.encode_phonetic('Я')[0]
        self.assertIsNone(h)

    def test_encode_length_matches_input(self):
        text = 'АРТМИР'
        self.assertEqual(len(self.encode_phonetic(text)), len(text))

    # --- render_phonetic_table ---

    def test_render_table_returns_str(self):
        self.assertIsInstance(self.render_table(), str)

    def test_render_table_contains_known_letters(self):
        table = self.render_table(color=False)
        for ru in ('А', 'Р', 'Т', 'Л', 'Н', 'О', 'Б', 'М', 'В', 'Д',
                   'Г', 'У', 'И', 'З', 'Ж', 'Ч'):
            self.assertIn(ru, table)

    def test_render_table_has_h_values(self):
        table = self.render_table(color=False)
        for h_str in ('h=63', 'h=15', 'h=48'):
            self.assertIn(h_str, table)

    def test_render_table_no_color_no_ansi(self):
        import re
        self.assertEqual(re.findall(r'\x1b\[', self.render_table(color=False)), [])

    def test_render_table_color_has_ansi(self):
        self.assertIn('\033[', self.render_table(color=True))

    # --- _glyph4 ---

    def test_glyph4_returns_4_lines(self):
        self.assertEqual(len(self._glyph4(63)), 4)

    def test_glyph4_only_block_chars(self):
        for h in (63, 15, 48):
            for line in self._glyph4(h):
                self.assertTrue(all(c in '█·' for c in line))

    # --- viewer phonetic section ---

    def test_viewer_has_phon_grid(self):
        self.assertIn('phon-grid', viewer_path().read_text(encoding='utf-8'))

    def test_viewer_phonetic_16_cells(self):
        import re
        cells = re.findall(r'class="phon-cell"',
                           viewer_path().read_text(encoding='utf-8'))
        self.assertEqual(len(cells), 16)

    def test_viewer_phonetic_transliterate_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ru-inp', content)
        self.assertIn('ru-out', content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
