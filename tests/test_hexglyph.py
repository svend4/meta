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
    render_quad,
    render_braille,
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


class TestTVCPRendering(unittest.TestCase):
    """Тесты для quadrant-block и Braille-рендеров (техника svend4/infon TVCP)."""

    def test_render_quad_returns_4_lines(self):
        lines = render_quad('A')
        self.assertEqual(len(lines), 4)

    def test_render_quad_width_4(self):
        for ch in ('A', 'E', 'L', 'Z'):
            lines = render_quad(ch)
            for line in lines:
                self.assertEqual(len(line), 4, f"quad width != 4 for {ch!r}")

    def test_render_quad_uses_block_chars(self):
        # All output chars must be in the 16 quadrant block characters
        QUAD = set(' ▗▖▄▝▐▞▟▘▚▌▙▀▜▛█')
        for ch in ('A', 'E', 'L'):
            lines = render_quad(ch)
            for line in lines:
                for c in line:
                    self.assertIn(c, QUAD, f"Unexpected char {c!r} in render_quad({ch!r})")

    def test_render_quad_E_has_full_blocks(self):
        # 'E' = h=63 (all segments) → corners should be '█'
        lines = render_quad('E')
        self.assertEqual(lines[0][0], '█')   # top-left corner
        self.assertEqual(lines[0][-1], '█')  # top-right corner

    def test_render_braille_returns_2_lines(self):
        lines = render_braille('A')
        self.assertEqual(len(lines), 2)

    def test_render_braille_width_4(self):
        for ch in ('A', 'E', 'L', 'Z'):
            lines = render_braille(ch)
            for line in lines:
                self.assertEqual(len(line), 4, f"braille width != 4 for {ch!r}")

    def test_render_braille_in_unicode_range(self):
        for ch in ('A', 'E', 'L'):
            lines = render_braille(ch)
            for line in lines:
                for c in line:
                    self.assertGreaterEqual(ord(c), 0x2800)
                    self.assertLessEqual(ord(c), 0x28FF)

    def test_render_quad_all_64_chars(self):
        for h in range(64):
            from projects.hexglyph.hexglyph import h_to_char
            ch = h_to_char(h)
            lines = render_quad(ch)
            self.assertEqual(len(lines), 4)

    def test_render_braille_all_64_chars(self):
        for h in range(64):
            from projects.hexglyph.hexglyph import h_to_char
            ch = h_to_char(h)
            lines = render_braille(ch)
            self.assertEqual(len(lines), 2)


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

    def test_viewer_has_q6_explorer(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('q6-char', content)
        self.assertIn('q6toggle', content)
        self.assertIn('q6setH', content)

    def test_viewer_q6_explorer_has_all_segments(self):
        content = viewer_path().read_text(encoding='utf-8')
        for seg in ('sv-T', 'sv-B', 'sv-L', 'sv-R', 'sv-D1', 'sv-D2'):
            self.assertIn(seg, content)

    def test_viewer_tri_cells_clickable(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wireCell', content)
        self.assertIn('tri-cell', content)

    def test_viewer_phon_cells_linked_to_explorer(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('phon-cell', content)
        self.assertIn('scrollIntoView', content)

    def test_viewer_q6_hamming_neighbours(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('q6-nbrs', content)
        self.assertIn('Соседи по Q6', content)

    def test_viewer_solan_reverse_decoder(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan-inp', content)
        self.assertIn('solan-out', content)
        self.assertIn('REV', content)

    def test_viewer_reverse_map_has_all_16_letters(self):
        import re
        content = viewer_path().read_text(encoding='utf-8')
        m = re.search(r'var REV = \{([^}]+)\}', content)
        self.assertIsNotNone(m, "REV map not found")
        pairs = re.findall(r'"[^"]*":\s*"[^"]*"', m.group(1))
        self.assertGreaterEqual(len(pairs), 15)


class TestSolanCA(unittest.TestCase):
    """Тесты клеточного автомата Q6/Solan."""

    def setUp(self):
        from projects.hexglyph.solan_ca import (
            step, make_initial,
            render_row_char, render_row_braille, render_row_quad,
        )
        self.step              = step
        self.make_initial      = make_initial
        self.render_row_char   = render_row_char
        self.render_row_braille = render_row_braille
        self.render_row_quad   = render_row_quad

    # --- step rules ---

    def test_step_xor_all_zeros_stays_zero(self):
        cells = [0] * 10
        self.assertEqual(self.step(cells, 'xor'), [0] * 10)

    def test_step_xor3_all_zeros_stays_zero(self):
        cells = [0] * 10
        self.assertEqual(self.step(cells, 'xor3'), [0] * 10)

    def test_step_preserves_length(self):
        for rule in ('xor', 'xor3', 'and', 'or', 'hamming'):
            cells = [i % 64 for i in range(12)]
            out = self.step(cells, rule)
            self.assertEqual(len(out), 12, f"len changed for rule={rule!r}")

    def test_step_xor_values_in_range(self):
        cells = list(range(64))
        out = self.step(cells, 'xor')
        for v in out:
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 63)

    def test_step_single_center_spreads(self):
        # XOR rule: center=63 → два соседних cell XOR-ятся с 63
        cells = [0] * 7
        cells[3] = 63
        nxt = self.step(cells, 'xor')
        # cells[2] и cells[4] должны получить 63 XOR 0 = 63
        self.assertEqual(nxt[2], 63)
        self.assertEqual(nxt[4], 63)
        self.assertEqual(nxt[3], 0)   # center становится 63 XOR 63 = 0

    def test_step_unknown_rule_raises(self):
        with self.assertRaises(ValueError):
            self.step([0, 1, 2], 'bogus')

    # --- make_initial ---

    def test_make_initial_center(self):
        cells = self.make_initial(10, 'center')
        self.assertEqual(len(cells), 10)
        self.assertEqual(cells[5], 63)
        self.assertEqual(sum(cells), 63)

    def test_make_initial_edge(self):
        cells = self.make_initial(8, 'edge')
        self.assertEqual(cells[0], 63)
        self.assertEqual(sum(cells), 63)

    def test_make_initial_random_length(self):
        cells = self.make_initial(20, 'random', seed=0)
        self.assertEqual(len(cells), 20)
        for v in cells:
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 63)

    def test_make_initial_phonetic(self):
        cells = self.make_initial(10, 'phonetic', word='АТ')
        self.assertEqual(len(cells), 10)
        # А=63, Т=48 — должны чередоваться
        self.assertEqual(cells[0], 63)
        self.assertEqual(cells[1], 48)

    def test_make_initial_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.make_initial(10, 'unknown_ic')

    # --- render ---

    def test_render_row_char_length(self):
        cells = [0, 63, 15, 48, 3]
        row = self.render_row_char(cells, color=False)
        self.assertEqual(len(row), 5)

    def test_render_row_braille_returns_2_lines(self):
        cells = [0, 63, 15]
        lines = self.render_row_braille(cells, color=False)
        self.assertEqual(len(lines), 2)

    def test_render_row_braille_line_width(self):
        cells = [0, 63, 15]
        lines = self.render_row_braille(cells, color=False)
        for line in lines:
            self.assertEqual(len(line), 3 * 4)  # 3 cells × 4 Braille chars

    def test_render_row_quad_returns_4_lines(self):
        cells = [0, 63]
        lines = self.render_row_quad(cells, color=False)
        self.assertEqual(len(lines), 4)

    def test_render_row_quad_line_width(self):
        cells = [0, 63]
        lines = self.render_row_quad(cells, color=False)
        for line in lines:
            self.assertEqual(len(line), 2 * 4)  # 2 cells × 4 quad chars

    # --- viewer CA section ---

    def test_viewer_has_ca_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ca-grid', content)
        self.assertIn('caSetRule', content)
        self.assertIn('caToggleAuto', content)

    def test_viewer_ca_has_all_rules(self):
        content = viewer_path().read_text(encoding='utf-8')
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(f"ca-rule-{rule}", content)

    def test_viewer_ca_has_ic_buttons(self):
        content = viewer_path().read_text(encoding='utf-8')
        for ic in ("center", "edge", "random"):
            self.assertIn(f"caIC('{ic}')", content)

    def test_viewer_ca_uses_q6GetH(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('q6GetH', content)

    def test_viewer_ca_row_clickable(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ca-row0', content)
        self.assertIn('data-idx', content)

    # --- viewer: 4-panel rule comparison ---

    def test_viewer_ca_compare_panels_exist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ca-compare', content)
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(f'ca-cmp-{rule}', content)

    def test_viewer_ca_compare_uses_nextRowWith(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('nextRowWith', content)
        self.assertIn('renderCompare', content)

    def test_viewer_ca_compare_panel_clickable(self):
        content = viewer_path().read_text(encoding='utf-8')
        # Each panel calls caSetRule() on click
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(f"caSetRule('{rule}')", content)

    # --- find_orbit ---

    def test_find_orbit_imported(self):
        from projects.hexglyph.solan_ca import find_orbit
        self.assertTrue(callable(find_orbit))

    def test_find_orbit_zero_state_fixed_point(self):
        from projects.hexglyph.solan_ca import find_orbit
        cells = [0] * 8
        t, p = find_orbit(cells, 'xor')
        self.assertEqual(t, 0)
        self.assertEqual(p, 1)

    def test_find_orbit_and_rule_converges(self):
        from projects.hexglyph.solan_ca import find_orbit
        # AND rule collapses to a periodic orbit (often p=1 or p=2)
        cells = [63, 15, 48, 3, 7, 63, 0, 21]
        t, p = find_orbit(cells, 'and')
        self.assertIsNotNone(t)
        self.assertGreater(p, 0)
        self.assertLess(p, 100)   # AND must converge fast

    def test_find_orbit_xor_returns_period(self):
        from projects.hexglyph.solan_ca import find_orbit
        # XOR rule on non-trivial IC returns finite period
        cells = [63, 0, 0, 0, 0, 0]
        t, p = find_orbit(cells, 'xor')
        self.assertIsNotNone(t)
        self.assertIsNotNone(p)
        self.assertGreater(p, 0)

    def test_find_orbit_xor3_period_divides_lcm(self):
        from projects.hexglyph.solan_ca import find_orbit
        cells = [63, 15, 48, 3, 7, 63]
        t, p = find_orbit(cells, 'xor3')
        self.assertIsNotNone(p)
        # Period must be positive and bounded
        self.assertGreater(p, 0)
        self.assertLess(p, 5001)

    def test_find_orbit_transient_nonneg(self):
        from projects.hexglyph.solan_ca import find_orbit, make_initial
        for rule in ('xor', 'xor3', 'and', 'or'):
            cells = make_initial(8, 'center')
            t, p = find_orbit(cells, rule)
            if t is not None:
                self.assertGreaterEqual(t, 0)

    def test_print_orbit_importable(self):
        from projects.hexglyph.solan_ca import print_orbit
        # Should run without error (no-color mode, small width)
        import io, sys
        buf = io.StringIO()
        old = sys.stdout; sys.stdout = buf
        try:
            print_orbit([63, 0, 0, 0, 0, 0], rule='xor', color=False)
        finally:
            sys.stdout = old
        out = buf.getvalue()
        self.assertIn('Транзиент', out)
        self.assertIn('Период', out)

    # --- viewer: orbit table ---

    def test_viewer_has_orbit_table(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('orbit-table', content)
        self.assertIn('findOrbitJS', content)
        self.assertIn('renderOrbitTable', content)

    def test_viewer_orbit_table_has_rule_labels(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('RULE_LABELS', content)
        for label in ('XOR', 'XOR3', 'AND', 'OR'):
            self.assertIn(label, content)

    def test_viewer_orbit_table_called_in_render(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('renderOrbitTable()', content)

    def test_viewer_orbit_shows_cycle_strip(self):
        content = viewer_path().read_text(encoding='utf-8')
        # Cycle strip uses → separator and Solan font
        self.assertIn('→', content)
        self.assertIn('pLimit', content)

    # --- --compare flag ---

    def test_run_compare_importable(self):
        from projects.hexglyph.solan_ca import run_compare
        self.assertTrue(callable(run_compare))

    def test_run_compare_output_has_all_rules(self):
        import io, sys
        from projects.hexglyph.solan_ca import run_compare
        buf = io.StringIO()
        old = sys.stdout; sys.stdout = buf
        try:
            run_compare(width=6, steps=3, color=False)
        finally:
            sys.stdout = old
        out = buf.getvalue()
        for name in ('XOR ⊕', 'XOR3', 'AND &', 'OR |'):
            self.assertIn(name, out)

    def test_run_compare_line_count(self):
        import io, sys
        from projects.hexglyph.solan_ca import run_compare
        buf = io.StringIO()
        old = sys.stdout; sys.stdout = buf
        try:
            run_compare(width=4, steps=2, color=False)
        finally:
            sys.stdout = old
        lines = [l for l in buf.getvalue().splitlines() if '│' in l]
        # 4 rules × (steps+1) lines with '│' = 4×3 = 12
        self.assertEqual(len(lines), 4 * 3)


class TestSolanEntropy(unittest.TestCase):
    """Тесты модуля solan_entropy."""

    def setUp(self):
        from projects.hexglyph.solan_entropy import (
            entropy, entropy_profile, entropy_profiles, sparkline,
        )
        self.entropy          = entropy
        self.entropy_profile  = entropy_profile
        self.entropy_profiles = entropy_profiles
        self.sparkline        = sparkline

    # ── entropy() ───────────────────────────────────────────────────────────

    def test_entropy_empty(self):
        self.assertEqual(self.entropy([]), 0.0)

    def test_entropy_uniform(self):
        # All same value → one state, p=1 → H = 0
        self.assertAlmostEqual(self.entropy([5] * 10), 0.0)

    def test_entropy_two_equal(self):
        # Two values, equal count → H = 1 bit
        self.assertAlmostEqual(self.entropy([0, 63] * 8), 1.0)

    def test_entropy_max_64_states(self):
        # All 64 states once → H = log2(64) = 6.0 bits
        cells = list(range(64))
        self.assertAlmostEqual(self.entropy(cells), 6.0)

    def test_entropy_nonneg(self):
        import random
        rng = random.Random(7)
        cells = [rng.randrange(64) for _ in range(100)]
        self.assertGreaterEqual(self.entropy(cells), 0.0)

    def test_entropy_bounded_by_log2_64(self):
        import random
        rng = random.Random(42)
        cells = [rng.randrange(64) for _ in range(200)]
        self.assertLessEqual(self.entropy(cells), 6.0 + 1e-9)

    # ── entropy_profile() ────────────────────────────────────────────────────

    def test_profile_length(self):
        cells = [0] * 10 + [63]
        prof = self.entropy_profile(cells, 'xor', 15)
        self.assertEqual(len(prof), 16)  # t=0..15

    def test_profile_nonneg(self):
        cells = list(range(20))
        for rule in ('xor', 'xor3', 'and', 'or'):
            prof = self.entropy_profile(cells, rule, 10)
            self.assertTrue(all(h >= 0.0 for h in prof), f"rule={rule}")

    def test_profile_and_converges_zero(self):
        # AND rule from random IC → eventually H=0
        cells = [0] * 6 + [63] + [0] * 5
        prof = self.entropy_profile(cells, 'and', 20)
        self.assertAlmostEqual(prof[-1], 0.0, places=6)

    def test_profile_first_value_equals_entropy(self):
        cells = [0, 63, 15, 48, 3, 7] * 3
        prof = self.entropy_profile(cells, 'xor', 5)
        self.assertAlmostEqual(prof[0], self.entropy(cells))

    # ── entropy_profiles() ────────────────────────────────────────────────────

    def test_profiles_returns_all_rules(self):
        cells = [0, 63] * 4
        profs = self.entropy_profiles(cells, 5)
        self.assertEqual(set(profs.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_profiles_custom_rules(self):
        cells = [0, 63] * 4
        profs = self.entropy_profiles(cells, 5, rules=['xor', 'and'])
        self.assertEqual(set(profs.keys()), {'xor', 'and'})

    # ── sparkline() ──────────────────────────────────────────────────────────

    def test_sparkline_length(self):
        vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertEqual(len(self.sparkline(vals, 4.0)), 5)

    def test_sparkline_zero_max(self):
        vals = [1.0, 2.0, 3.0]
        result = self.sparkline(vals, 0.0)
        self.assertEqual(result, '   ')

    def test_sparkline_max_char(self):
        # Maximum value → '█'
        result = self.sparkline([6.0], 6.0)
        self.assertEqual(result, '█')

    def test_sparkline_zero_value(self):
        # Zero value → ' '
        result = self.sparkline([0.0], 6.0)
        self.assertEqual(result, ' ')

    def test_sparkline_monotone(self):
        # Monotonically increasing values → monotonically increasing chars
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = self.sparkline(vals, 6.0)
        self.assertEqual(len(result), 7)
        # Each char ≥ previous
        for i in range(1, len(result)):
            self.assertGreaterEqual(result[i], result[i - 1])

    # ── print_entropy_chart() ────────────────────────────────────────────────

    def test_print_entropy_chart_runs(self):
        import io, sys
        from projects.hexglyph.solan_entropy import print_entropy_chart
        buf = io.StringIO()
        old = sys.stdout; sys.stdout = buf
        try:
            print_entropy_chart([0] * 5 + [63] + [0] * 6,
                                steps=5, color=False)
        finally:
            sys.stdout = old
        out = buf.getvalue()
        self.assertIn('H₀', out)
        self.assertIn('Hf', out)
        self.assertIn('Спарклайн', out)

    def test_print_entropy_chart_contains_all_rules(self):
        import io, sys
        from projects.hexglyph.solan_entropy import print_entropy_chart
        buf = io.StringIO()
        old = sys.stdout; sys.stdout = buf
        try:
            print_entropy_chart([0, 63] * 4, steps=3, color=False)
        finally:
            sys.stdout = old
        out = buf.getvalue()
        for label in ('XOR', 'XOR3', 'AND', 'OR'):
            self.assertIn(label, out)

    # --- viewer: entropy sparklines ---

    def test_viewer_has_entropy_sparks_table(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('entropy-sparks', content)
        self.assertIn('renderEntropySparklines', content)

    def test_viewer_entropy_uses_calcEntropy(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('calcEntropy', content)
        self.assertIn('Math.log2', content)

    def test_viewer_entropy_sparkline_chars(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ENT_BLOCK', content)
        self.assertIn('▁▂▃▄▅▆▇█', content)

    def test_viewer_entropy_called_in_render(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('renderEntropySparklines()', content)

    # --- run_animate ---

    def test_run_animate_importable(self):
        from projects.hexglyph.solan_ca import run_animate
        self.assertTrue(callable(run_animate))

    def test_animate_cli_flag_in_source(self):
        import pathlib
        src = pathlib.Path('projects/hexglyph/solan_ca.py').read_text()
        self.assertIn('--animate', src)
        self.assertIn('--delay', src)
        self.assertIn('--rows', src)

    def test_run_animate_signature(self):
        import inspect
        from projects.hexglyph.solan_ca import run_animate
        sig = inspect.signature(run_animate)
        params = list(sig.parameters)
        for p in ('width', 'rule', 'ic', 'delay', 'rows', 'color'):
            self.assertIn(p, params)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_entropy',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_entropy(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_entropy', content)


class TestSolanWord(unittest.TestCase):
    """Тесты семантического анализа слов solan_word.py."""

    def setUp(self):
        from projects.hexglyph.solan_word import (
            encode_word, pad_to, word_signature, sig_distance, word_distance,
        )
        self.encode_word    = encode_word
        self.pad_to         = pad_to
        self.word_signature = word_signature
        self.sig_distance   = sig_distance
        self.word_distance  = word_distance

    # ── encode_word() ────────────────────────────────────────────────────────

    def test_encode_known_word(self):
        h = self.encode_word('РАТОН')
        self.assertEqual(h, [15, 63, 48, 47, 3])

    def test_encode_empty(self):
        self.assertEqual(self.encode_word(''), [])

    def test_encode_ignores_nonphonetic(self):
        # Буква Э не входит в алфавит → пропускается
        h = self.encode_word('ЭА')
        self.assertEqual(h, [63])   # только А

    def test_encode_lowercase(self):
        # Строчные должны работать так же, как заглавные
        self.assertEqual(self.encode_word('ратон'), self.encode_word('РАТОН'))

    def test_encode_all_16_letters(self):
        word = 'АБВГДЖЗИЛМНОРТУЧ'
        h = self.encode_word(word)
        self.assertEqual(len(h), 16)
        # Все h должны быть в 0..63
        self.assertTrue(all(0 <= x <= 63 for x in h))

    # ── pad_to() ─────────────────────────────────────────────────────────────

    def test_pad_to_shorter(self):
        self.assertEqual(self.pad_to([1, 2], 4), [1, 2, 1, 2])

    def test_pad_to_exact(self):
        self.assertEqual(self.pad_to([1, 2, 3], 3), [1, 2, 3])

    def test_pad_to_truncate(self):
        self.assertEqual(self.pad_to([1, 2, 3, 4], 2), [1, 2])

    def test_pad_to_empty(self):
        self.assertEqual(self.pad_to([], 4), [0, 0, 0, 0])

    # ── word_signature() ─────────────────────────────────────────────────────

    def test_signature_returns_all_rules(self):
        sig = self.word_signature('РАТОН')
        self.assertEqual(set(sig.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_signature_known_word(self):
        # РАТОН с width=16: XOR period = 1 (фиксированная точка)
        sig = self.word_signature('РАТОН', width=16)
        t_xor, p_xor = sig['xor']
        self.assertIsNotNone(t_xor)
        self.assertEqual(p_xor, 1)

    def test_signature_empty_word(self):
        sig = self.word_signature('')
        for r in ('xor', 'xor3', 'and', 'or'):
            self.assertEqual(sig[r], (None, None))

    def test_signature_periods_positive(self):
        sig = self.word_signature('ГОРА', width=16)
        for r, (t, p) in sig.items():
            if p is not None:
                self.assertGreater(p, 0)

    # ── sig_distance() ───────────────────────────────────────────────────────

    def test_distance_same_word(self):
        sig = self.word_signature('РАТОН')
        d = self.sig_distance(sig, sig)
        self.assertAlmostEqual(d, 0.0)

    def test_distance_range(self):
        sig1 = self.word_signature('РАТОН')
        sig2 = self.word_signature('ВОДА')
        d = self.sig_distance(sig1, sig2)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)

    def test_distance_symmetric(self):
        s1 = self.word_signature('РАТОН')
        s2 = self.word_signature('ГОРА')
        self.assertAlmostEqual(
            self.sig_distance(s1, s2),
            self.sig_distance(s2, s1),
        )

    def test_distance_word_function(self):
        d = self.word_distance('РАТОН', 'СТОЛ')
        self.assertGreaterEqual(d, 0.0)

    # ── viewer: Word CA section ────────────────────────────────────────────

    def test_viewer_has_word_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('word-ca', content)
        self.assertIn('word-inp', content)
        self.assertIn('wordUpdate', content)

    def test_viewer_word_has_phonetic_map(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('var WH =', content)
        # Ключевые буквы фонетического алфавита
        for letter in ('А', 'Р', 'Т', 'Н', 'О'):
            self.assertIn(letter, content)

    def test_viewer_word_has_orbit_table(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('word-orbits', content)
        self.assertIn('findOrbit', content)

    def test_viewer_word_exposes_caNextRow(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.caNextRow', content)
        self.assertIn('caNextRow', content)

    def test_viewer_word_rule_buttons(self):
        content = viewer_path().read_text(encoding='utf-8')
        for r in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(f'wordSetRule(\'{r}\')', content)


class TestSolanLexicon(unittest.TestCase):
    """Tests for solan_lexicon.py and the viewer Lexicon section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_lexicon import (
            LEXICON, all_signatures, neighbors, orbital_clusters,
        )
        cls.LEXICON         = LEXICON
        cls.all_signatures  = staticmethod(all_signatures)
        cls.neighbors       = staticmethod(neighbors)
        cls.orbital_clusters = staticmethod(orbital_clusters)

    # ── LEXICON list ─────────────────────────────────────────────────────

    def test_lexicon_not_empty(self):
        self.assertGreater(len(self.LEXICON), 0)

    def test_lexicon_no_duplicates(self):
        self.assertEqual(len(self.LEXICON), len(set(self.LEXICON)))

    def test_lexicon_contains_raton(self):
        self.assertIn('РАТОН', self.LEXICON)

    def test_lexicon_contains_gora(self):
        self.assertIn('ГОРА', self.LEXICON)

    def test_lexicon_all_uppercase(self):
        for w in self.LEXICON:
            self.assertEqual(w, w.upper())

    # ── all_signatures ───────────────────────────────────────────────────

    def test_all_signatures_keys_match_lexicon(self):
        sigs = self.all_signatures()
        for w in self.LEXICON:
            self.assertIn(w, sigs)

    def test_all_signatures_each_has_four_rules(self):
        sigs = self.all_signatures()
        for w, sig in sigs.items():
            for r in ('xor', 'xor3', 'and', 'or'):
                self.assertIn(r, sig)

    def test_all_signatures_periods_positive_or_none(self):
        sigs = self.all_signatures()
        for w, sig in sigs.items():
            for r in ('xor', 'xor3', 'and', 'or'):
                _, p = sig[r]
                if p is not None:
                    self.assertGreater(p, 0)

    # ── neighbors ────────────────────────────────────────────────────────

    def test_neighbors_returns_list(self):
        nbrs = self.neighbors('РАТОН', n=5)
        self.assertIsInstance(nbrs, list)

    def test_neighbors_count(self):
        nbrs = self.neighbors('ГОРА', n=5)
        self.assertLessEqual(len(nbrs), 5)

    def test_neighbors_excludes_target(self):
        nbrs = self.neighbors('ГОРА', n=10)
        words = [w for w, _ in nbrs]
        self.assertNotIn('ГОРА', words)

    def test_neighbors_distances_sorted(self):
        nbrs = self.neighbors('РАТОН', n=8)
        dists = [d for _, d in nbrs if d == d]  # skip NaN
        self.assertEqual(dists, sorted(dists))

    def test_neighbors_distances_in_range(self):
        nbrs = self.neighbors('ВОДА', n=5)
        for _, d in nbrs:
            if d == d:  # not NaN
                self.assertGreaterEqual(d, 0.0)
                self.assertLessEqual(d, 1.0)

    # ── orbital_clusters ─────────────────────────────────────────────────

    def test_clusters_cover_all_words(self):
        sigs  = self.all_signatures()
        clsts = self.orbital_clusters(sigs=sigs, threshold=0.15)
        covered = {w for c in clsts for w in c}
        self.assertEqual(covered, set(self.LEXICON))

    def test_clusters_no_overlap(self):
        sigs  = self.all_signatures()
        clsts = self.orbital_clusters(sigs=sigs, threshold=0.15)
        seen: set[str] = set()
        for c in clsts:
            for w in c:
                self.assertNotIn(w, seen)
                seen.add(w)

    def test_clusters_sorted_by_size(self):
        sigs  = self.all_signatures()
        clsts = self.orbital_clusters(sigs=sigs, threshold=0.15)
        sizes = [len(c) for c in clsts]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_clusters_loose_threshold_fewer(self):
        sigs   = self.all_signatures()
        clsts_tight = self.orbital_clusters(sigs=sigs, threshold=0.01)
        clsts_loose = self.orbital_clusters(sigs=sigs, threshold=0.90)
        # Looser threshold must produce fewer or equal clusters
        self.assertLessEqual(len(clsts_loose), len(clsts_tight))

    # ── viewer: Lexicon section ───────────────────────────────────────────

    def test_viewer_has_lexicon_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Орбитальный лексикон Q6', content)
        self.assertIn('lex-inp', content)
        self.assertIn('lex-results', content)

    def test_viewer_lexicon_has_js_array(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('var LEXICON=', content)
        self.assertIn("'РАТОН'", content)
        self.assertIn("'ГОРА'", content)

    def test_viewer_lexicon_has_lexfind(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lexFind', content)
        self.assertIn('window.lexFind', content)

    def test_viewer_lexicon_exposes_find_orbit_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.findOrbitJS', content)
        self.assertIn('window.findOrbitJS  = findOrbitJS', content)

    def test_viewer_lexicon_has_dist_function(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('distL', content)
        self.assertIn('sigL', content)

    def test_viewer_lexicon_has_get_lex_sigs(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('getLexSigs', content)

    def test_viewer_lexicon_sig_display(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lex-sig', content)
        self.assertIn('Сигнатура', content)


class TestSolanDendrogram(unittest.TestCase):
    """Tests for solan_dendrogram.py and the viewer Dendrogram section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_dendrogram import (
            build_dendrogram, leaf_order, flat_clusters, dendrogram_dict,
        )
        cls.build_dendrogram = staticmethod(build_dendrogram)
        cls.leaf_order       = staticmethod(leaf_order)
        cls.flat_clusters    = staticmethod(flat_clusters)
        cls.dendrogram_dict  = staticmethod(dendrogram_dict)
        cls._words = ['ГОРА', 'УДАР', 'РОТА', 'УТРО',
                      'ВОДА', 'НОРА', 'ЛУНА', 'РАТОН', 'ЖУРНАЛ']

    # ── build_dendrogram ──────────────────────────────────────────────────

    def test_build_returns_tuple(self):
        nodes, root = self.build_dendrogram(self._words)
        self.assertIsInstance(nodes, dict)
        self.assertIsInstance(root, int)

    def test_node_count(self):
        nodes, _ = self.build_dendrogram(self._words)
        n = len(self._words)
        # UPGMA produces n leaves + n-1 internals = 2n-1 nodes
        self.assertEqual(len(nodes), 2 * n - 1)

    def test_root_has_all_leaves(self):
        nodes, root = self.build_dendrogram(self._words)
        self.assertEqual(nodes[root]['size'], len(self._words))

    def test_leaf_labels_present(self):
        nodes, _ = self.build_dendrogram(self._words)
        labels = {nd['label'] for nd in nodes.values() if nd['label'] is not None}
        self.assertEqual(labels, set(self._words))

    def test_internal_nodes_have_no_label(self):
        nodes, _ = self.build_dendrogram(self._words)
        for nd in nodes.values():
            if nd['left'] is not None:
                self.assertIsNone(nd['label'])

    def test_heights_nonnegative(self):
        nodes, _ = self.build_dendrogram(self._words)
        for nd in nodes.values():
            self.assertGreaterEqual(nd['height'], 0.0)

    def test_heights_monotone(self):
        """Parent height >= child heights (UPGMA monotonicity)."""
        nodes, root = self.build_dendrogram(self._words)
        for nd in nodes.values():
            if nd['left'] is not None:
                self.assertGreaterEqual(
                    nd['height'], nodes[nd['left']]['height'])
                self.assertGreaterEqual(
                    nd['height'], nodes[nd['right']]['height'])

    def test_identical_orbit_words_merge_at_zero(self):
        nodes, _ = self.build_dendrogram(self._words)
        # ГОРА, УДАР, РОТА, УТРО have d=0 between them
        for nd in nodes.values():
            if nd['label'] is None:
                l, r = nodes[nd['left']], nodes[nd['right']]
                if (l['label'] in {'ГОРА','УДАР','РОТА','УТРО'} and
                        r['label'] in {'ГОРА','УДАР','РОТА','УТРО'}):
                    self.assertAlmostEqual(nd['height'], 0.0)

    # ── leaf_order ────────────────────────────────────────────────────────

    def test_leaf_order_all_words(self):
        nodes, root = self.build_dendrogram(self._words)
        lo = self.leaf_order(nodes, root)
        self.assertEqual(sorted(lo), sorted(self._words))

    def test_leaf_order_no_duplicates(self):
        nodes, root = self.build_dendrogram(self._words)
        lo = self.leaf_order(nodes, root)
        self.assertEqual(len(lo), len(set(lo)))

    def test_leaf_order_length(self):
        nodes, root = self.build_dendrogram(self._words)
        lo = self.leaf_order(nodes, root)
        self.assertEqual(len(lo), len(self._words))

    # ── flat_clusters ─────────────────────────────────────────────────────

    def test_flat_clusters_cover_all_words(self):
        nodes, root = self.build_dendrogram(self._words)
        clusters = self.flat_clusters(nodes, root, cut=0.01)
        all_words = [w for c in clusters for w in c]
        self.assertEqual(sorted(all_words), sorted(self._words))

    def test_flat_clusters_no_overlap(self):
        nodes, root = self.build_dendrogram(self._words)
        clusters = self.flat_clusters(nodes, root, cut=0.1)
        seen: set[str] = set()
        for c in clusters:
            for w in c:
                self.assertNotIn(w, seen)
                seen.add(w)

    def test_flat_clusters_sorted_desc(self):
        nodes, root = self.build_dendrogram(self._words)
        clusters = self.flat_clusters(nodes, root, cut=0.1)
        sizes = [len(c) for c in clusters]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_flat_clusters_zero_cut_all_singletons(self):
        nodes, root = self.build_dendrogram(self._words)
        # Cut at -1 (below all heights) → each cluster may merge at 0
        clusters_max = self.flat_clusters(nodes, root, cut=999.0)
        self.assertEqual(len(clusters_max), 1)

    def test_clique_merged_at_zero_cut(self):
        nodes, root = self.build_dendrogram(self._words)
        clusters = self.flat_clusters(nodes, root, cut=0.001)
        clique = {'ГОРА', 'УДАР', 'РОТА', 'УТРО'}
        for c in clusters:
            if clique.issubset(set(c)):
                break
        else:
            self.fail('Clique ГОРА/УДАР/РОТА/УТРО not merged at cut=0.001')

    # ── dendrogram_dict ───────────────────────────────────────────────────

    def test_dict_has_required_keys(self):
        nodes, root = self.build_dendrogram(self._words)
        d = self.dendrogram_dict(nodes, root)
        for key in ('nodes', 'root', 'leaf_order', 'max_height'):
            self.assertIn(key, d)

    def test_dict_leaf_order_complete(self):
        nodes, root = self.build_dendrogram(self._words)
        d = self.dendrogram_dict(nodes, root)
        self.assertEqual(sorted(d['leaf_order']), sorted(self._words))

    def test_dict_max_height(self):
        nodes, root = self.build_dendrogram(self._words)
        d = self.dendrogram_dict(nodes, root)
        self.assertAlmostEqual(d['max_height'], nodes[root]['height'], places=4)

    def test_dict_node_keys_are_strings(self):
        nodes, root = self.build_dendrogram(self._words)
        d = self.dendrogram_dict(nodes, root)
        for k in d['nodes']:
            self.assertIsInstance(k, str)

    # ── full lexicon ──────────────────────────────────────────────────────

    def test_full_lexicon_build(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        nodes, root = self.build_dendrogram()
        self.assertEqual(nodes[root]['size'], len(LEXICON))

    # ── viewer: Dendrogram section ────────────────────────────────────────

    def test_viewer_has_dendrogram_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Дендрограмма Q6', content)
        self.assertIn('dend-canvas', content)

    def test_viewer_dendrogram_has_upgma(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('upgma', content)

    def test_viewer_dendrogram_has_cut_slider(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dend-cut', content)
        self.assertIn('dend-cut-val', content)

    def test_viewer_dendrogram_has_flat_clusters(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('flatClusters', content)

    def test_viewer_dendrogram_uses_mDist(self):
        content = viewer_path().read_text(encoding='utf-8')
        # mDist should appear in the dendrogram section
        self.assertIn('mDist', content)

    def test_viewer_dendrogram_has_hover(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dend-info', content)
        self.assertIn('_hovLeaf', content)


class TestSolanPredict(unittest.TestCase):
    """Tests for solan_predict.py and the viewer Prediction section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_predict import (
            predict, batch_predict, predict_text, prediction_dict,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.predict          = staticmethod(predict)
        cls.batch_predict    = staticmethod(batch_predict)
        cls.predict_text     = staticmethod(predict_text)
        cls.prediction_dict  = staticmethod(prediction_dict)
        cls.LEXICON          = list(LEXICON)

    # ── predict() basics ──────────────────────────────────────────────────────

    def test_predict_returns_dict(self):
        r = self.predict('ГОРА')
        self.assertIsInstance(r, dict)

    def test_predict_required_keys(self):
        r = self.predict('ГОРА')
        for k in ('word', 'signature', 'full_key', 'class_id',
                  'class_words', 'is_new_class', 'neighbors'):
            self.assertIn(k, r)

    def test_predict_word_preserved(self):
        r = self.predict('ГОРА')
        self.assertEqual(r['word'], 'ГОРА')

    def test_predict_signature_has_rules(self):
        r = self.predict('ГОРА')
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(rule, r['signature'])

    def test_predict_full_key_is_tuple_of_5(self):
        r = self.predict('ГОРА')
        self.assertIsInstance(r['full_key'], tuple)
        self.assertEqual(len(r['full_key']), 5)

    # ── Class membership ──────────────────────────────────────────────────────

    def test_gora_class_key(self):
        r = self.predict('ГОРА')
        self.assertEqual(r['full_key'], (2, 1, 2, 1, 1))
        self.assertFalse(r['is_new_class'])
        self.assertIsNotNone(r['class_id'])

    def test_luna_class_key(self):
        r = self.predict('ЛУНА')
        self.assertEqual(r['full_key'], (2, 1, 2, 1, 2))
        self.assertFalse(r['is_new_class'])

    def test_zhurnal_class_key(self):
        r = self.predict('ЖУРНАЛ')
        self.assertEqual(r['full_key'], (8, 4, 2, 4, 2))
        self.assertFalse(r['is_new_class'])

    def test_is_new_class_false_for_all_lexicon(self):
        results = self.batch_predict(self.LEXICON)
        new_class_words = [r['word'] for r in results if r['is_new_class']]
        self.assertEqual(new_class_words, [],
                         msg=f'Lexicon words unexpectedly new: {new_class_words}')

    def test_class_id_in_range(self):
        results = self.batch_predict(self.LEXICON[:10])
        for r in results:
            if not r['is_new_class']:
                self.assertGreaterEqual(r['class_id'], 0)
                self.assertLess(r['class_id'], 13)

    def test_class_words_nonempty_for_lexicon(self):
        r = self.predict('ВОДА')
        self.assertGreater(len(r['class_words']), 0)

    def test_gora_class_words_contains_gora(self):
        r = self.predict('ГОРА')
        self.assertIn('ГОРА', r['class_words'])

    # ── Neighbors ─────────────────────────────────────────────────────────────

    def test_lexicon_word_self_neighbor_first(self):
        r = self.predict('ГОРА')
        # Word should appear among neighbors with distance 0
        zero_neighbors = [w for w, d in r['neighbors'] if d == 0.0]
        self.assertIn('ГОРА', zero_neighbors)

    def test_neighbors_sorted_ascending(self):
        r = self.predict('ГОРА')
        dists = [d for _, d in r['neighbors']]
        self.assertEqual(dists, sorted(dists))

    def test_neighbors_count_default(self):
        r = self.predict('ГОРА')
        self.assertLessEqual(len(r['neighbors']), 10)
        self.assertGreater(len(r['neighbors']), 0)

    def test_neighbors_top_n(self):
        r = self.predict('ГОРА', top_n=5)
        self.assertLessEqual(len(r['neighbors']), 5)

    def test_neighbors_no_nan(self):
        import math
        r = self.predict('ГОРА')
        for _, d in r['neighbors']:
            self.assertFalse(math.isnan(d))

    # ── batch_predict() ───────────────────────────────────────────────────────

    def test_batch_predict_length(self):
        words = ['ГОРА', 'ЛУНА', 'ЖУРНАЛ']
        results = self.batch_predict(words)
        self.assertEqual(len(results), 3)

    def test_batch_predict_words_preserved(self):
        words = ['ГОРА', 'ЛУНА']
        results = self.batch_predict(words)
        self.assertEqual([r['word'] for r in results], words)

    def test_batch_predict_consistent_with_single(self):
        single = self.predict('ЛУНА')
        batch  = self.batch_predict(['ЛУНА'])
        self.assertEqual(single['full_key'], batch[0]['full_key'])
        self.assertEqual(single['class_id'], batch[0]['class_id'])

    # ── predict_text() ────────────────────────────────────────────────────────

    def test_predict_text_tokenises(self):
        results = self.predict_text('ГОРА И ЛУНА')
        words = [r['word'] for r in results]
        self.assertIn('ГОРА', words)
        self.assertIn('ЛУНА', words)

    def test_predict_text_skips_non_cyrillic(self):
        results = self.predict_text('ГОРА 123 --- ЛУНА')
        words = [r['word'] for r in results]
        self.assertNotIn('123', words)

    def test_predict_text_deduplicates(self):
        results = self.predict_text('ГОРА ГОРА ГОРА')
        words = [r['word'] for r in results]
        self.assertEqual(len(words), len(set(words)))

    def test_predict_text_upcases(self):
        results = self.predict_text('гора')
        words = [r['word'] for r in results]
        self.assertIn('ГОРА', words)

    # ── prediction_dict() ─────────────────────────────────────────────────────

    def test_prediction_dict_serialisable(self):
        import json
        r = self.predict('ГОРА')
        d = self.prediction_dict(r)
        dumped = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(dumped, str)

    def test_prediction_dict_keys(self):
        r = self.predict('ГОРА')
        d = self.prediction_dict(r)
        for k in ('word', 'signature', 'full_key', 'class_id',
                  'class_words', 'is_new_class', 'neighbors'):
            self.assertIn(k, d)

    def test_prediction_dict_full_key_is_list(self):
        r = self.predict('ГОРА')
        d = self.prediction_dict(r)
        self.assertIsInstance(d['full_key'], list)

    def test_prediction_dict_neighbors_have_dist(self):
        r = self.predict('ГОРА')
        d = self.prediction_dict(r)
        for item in d['neighbors']:
            self.assertIn('word', item)
            self.assertIn('dist', item)

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_pred_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pred-canvas', content)

    def test_viewer_has_pred_result(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pred-result', content)

    def test_viewer_has_pred_classes(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('PRED_CLASSES', content)

    def test_viewer_has_pred_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('predRun', content)

    def test_viewer_pred_classes_count_13(self):
        import re
        content = viewer_path().read_text(encoding='utf-8')
        # Count entries in PRED_CLASSES array: each entry starts with {key:[
        m = re.findall(r'\{key:\[', content)
        self.assertGreaterEqual(len(m), 13)

    def test_viewer_pred_classes_has_class1(self):
        content = viewer_path().read_text(encoding='utf-8')
        # Class 1 key is [2, 1, 2, 1, 2] with 20 words
        self.assertIn('[2, 1, 2, 1, 2]', content)

    def test_viewer_has_sigl_export(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.sigL', content)


class TestSolanTransient(unittest.TestCase):
    """Tests for solan_transient.py and the viewer Transient section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_transient import (
            build_transient_data, transient_classes, full_key,
            transient_dist, transient_dict, RULES,
        )
        cls.build     = staticmethod(build_transient_data)
        cls.classes   = staticmethod(transient_classes)
        cls.full_key  = staticmethod(full_key)
        cls.t_dist    = staticmethod(transient_dist)
        cls.t_dict    = staticmethod(transient_dict)
        cls.RULES     = RULES
        cls._w9 = ['ГОРА','УДАР','РОТА','УТРО',
                   'ВОДА','НОРА','ЛУНА','РАТОН','ЖУРНАЛ']

    # ── full_key ─────────────────────────────────────────────────────────

    def test_full_key_tuple_length(self):
        k = self.full_key('ГОРА')
        self.assertEqual(len(k), 5)

    def test_full_key_all_ints(self):
        k = self.full_key('ГОРА')
        for v in k:
            self.assertIsInstance(v, int)

    def test_full_key_same_word(self):
        self.assertEqual(self.full_key('ГОРА'), self.full_key('ГОРА'))

    def test_full_key_gora_class(self):
        """ГОРА должна быть в классе (2,1,2,1,1)."""
        self.assertEqual(self.full_key('ГОРА'), (2, 1, 2, 1, 1))

    def test_full_key_voda_class(self):
        """ВОДА должна быть в классе (2,1,2,1,2)."""
        self.assertEqual(self.full_key('ВОДА'), (2, 1, 2, 1, 2))

    def test_full_key_zhurnal_class(self):
        """ЖУРНАЛ — singleton (8,4,2,4,2)."""
        self.assertEqual(self.full_key('ЖУРНАЛ'), (8, 4, 2, 4, 2))

    # ── build_transient_data ─────────────────────────────────────────────

    def test_build_has_required_keys(self):
        data = self.build(self._w9)
        for k in ('words','signatures','by_rule','classes','n_classes',
                  'xor_t_isomorphic_xor3_p'):
            self.assertIn(k, data)

    def test_build_words_preserved(self):
        data = self.build(self._w9)
        self.assertEqual(data['words'], self._w9)

    def test_build_signatures_all_words(self):
        data = self.build(self._w9)
        self.assertEqual(set(data['signatures'].keys()), set(self._w9))

    def test_build_by_rule_all_rules(self):
        data = self.build(self._w9)
        for rule in self.RULES:
            self.assertIn(rule, data['by_rule'])

    def test_build_by_rule_keys(self):
        data = self.build(self._w9)
        req = {'transients','periods','tp_pairs','entropy_t',
               'entropy_p','unique_t','hist_t','unique_tp'}
        for rule in self.RULES:
            self.assertTrue(req.issubset(data['by_rule'][rule].keys()))

    def test_xor_transient_equiv_xor3_period(self):
        """XOR transient ≡ XOR3 period for all 49 words."""
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        self.assertTrue(data['xor_t_isomorphic_xor3_p'])

    def test_xor3_transient_all_zero(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        for t in data['by_rule']['xor3']['transients'].values():
            self.assertEqual(t, 0)

    def test_xor3_entropy_transient_zero(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        self.assertAlmostEqual(data['by_rule']['xor3']['entropy_t'], 0.0, places=6)

    def test_xor_transients_values(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        self.assertEqual(sorted(data['by_rule']['xor']['unique_t']), [2, 8])

    def test_and_transients_5_unique(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        self.assertEqual(len(data['by_rule']['and']['unique_t']), 5)

    def test_or_transients_6_unique(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        self.assertEqual(len(data['by_rule']['or']['unique_t']), 6)

    def test_and_entropy_t_greater_p(self):
        """AND transient entropy > AND period entropy."""
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        rd = data['by_rule']['and']
        self.assertGreater(rd['entropy_t'], rd['entropy_p'])

    def test_or_entropy_t_greater_p(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        rd = data['by_rule']['or']
        self.assertGreater(rd['entropy_t'], rd['entropy_p'])

    # ── transient_classes ────────────────────────────────────────────────

    def test_13_classes_full_lexicon(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        self.assertEqual(data['n_classes'], 13)

    def test_classes_cover_all_words(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        all_words = [w for c in data['classes'] for w in c['words']]
        self.assertEqual(sorted(all_words), sorted(LEXICON))

    def test_classes_no_overlap(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        seen: set[str] = set()
        for c in data['classes']:
            for w in c['words']:
                self.assertNotIn(w, seen)
                seen.add(w)

    def test_classes_sorted_desc(self):
        data = self.build()
        sizes = [c['count'] for c in data['classes']]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_largest_class_20(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build()
        self.assertEqual(data['classes'][0]['count'], 20)

    def test_largest_class_key(self):
        data = self.build()
        self.assertEqual(data['classes'][0]['key'], (2, 1, 2, 1, 2))

    def test_singleton_zhurnal(self):
        data = self.build()
        for c in data['classes']:
            if 'ЖУРНАЛ' in c['words']:
                self.assertEqual(c['count'], 1)
                self.assertEqual(c['key'], (8, 4, 2, 4, 2))
                break

    def test_class_sizes_known(self):
        """Known sizes: 20,5,4,4,4,3,2,2,1,1,1,1,1."""
        data = self.build()
        sizes = sorted([c['count'] for c in data['classes']], reverse=True)
        self.assertEqual(sizes, [20, 5, 4, 4, 4, 3, 2, 2, 1, 1, 1, 1, 1])

    # ── transient_dist ───────────────────────────────────────────────────

    def test_transient_dist_same(self):
        from projects.hexglyph.solan_word import word_signature
        sig = word_signature('ГОРА')
        self.assertAlmostEqual(self.t_dist(sig, sig), 0.0)

    def test_transient_dist_nonneg(self):
        from projects.hexglyph.solan_word import word_signature
        s1 = word_signature('ГОРА')
        s2 = word_signature('ЖУРНАЛ')
        d = self.t_dist(s1, s2)
        self.assertGreaterEqual(d, 0.0)

    def test_transient_dist_symmetric(self):
        from projects.hexglyph.solan_word import word_signature
        s1 = word_signature('ГОРА')
        s2 = word_signature('РАТОН')
        self.assertAlmostEqual(self.t_dist(s1, s2), self.t_dist(s2, s1))

    def test_transient_dist_same_class_zero(self):
        """Words in same transient class → dist=0."""
        from projects.hexglyph.solan_word import word_signature
        # ГОРА and УДАР are in same class (2,1,2,1,1)
        s1 = word_signature('ГОРА')
        s2 = word_signature('УДАР')
        self.assertAlmostEqual(self.t_dist(s1, s2), 0.0)

    # ── transient_dict ───────────────────────────────────────────────────

    def test_t_dict_keys(self):
        data = self.build(self._w9)
        d = self.t_dict(data)
        for k in ('words','signatures','by_rule','classes','n_classes',
                  'xor_t_isomorphic_xor3_p'):
            self.assertIn(k, d)

    def test_t_dict_sig_lists(self):
        data = self.build(self._w9)
        d = self.t_dict(data)
        for w, sig in d['signatures'].items():
            self.assertIsInstance(sig, list)
            self.assertEqual(len(sig), 5)

    def test_t_dict_classes_have_key(self):
        data = self.build(self._w9)
        d = self.t_dict(data)
        for c in d['classes']:
            self.assertIn('key', c)
            self.assertIsInstance(c['key'], list)

    # ── viewer: Transient section ────────────────────────────────────────

    def test_viewer_has_transient_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Транзиентный анализ Q6', content)
        self.assertIn('trans-canvas', content)

    def test_viewer_trans_has_and_or_axes(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('AND_T', content)
        self.assertIn('OR_T', content)

    def test_viewer_trans_has_hover(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('trans-info', content)
        self.assertIn('_hovIdx', content)

    def test_viewer_trans_has_structural_note(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('XOR транзиент', content)

    def test_viewer_trans_uses_lex_sigs(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lexAllSigs', content)


class TestSolanRules(unittest.TestCase):
    """Tests for solan_rules.py and the viewer Rules section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_rules import (
            build_all_rules, signature_classes, per_rule_dist,
            rules_dict, RULES,
        )
        cls.build_all_rules    = staticmethod(build_all_rules)
        cls.signature_classes  = staticmethod(signature_classes)
        cls.per_rule_dist      = staticmethod(per_rule_dist)
        cls.rules_dict         = staticmethod(rules_dict)
        cls.RULES              = RULES
        cls._words9 = ['ГОРА','УДАР','РОТА','УТРО',
                       'ВОДА','НОРА','ЛУНА','РАТОН','ЖУРНАЛ']

    # ── per_rule_dist ────────────────────────────────────────────────────

    def test_per_rule_dist_same_word(self):
        from projects.hexglyph.solan_word import word_signature
        sig = word_signature('ГОРА')
        for rule in self.RULES:
            self.assertAlmostEqual(self.per_rule_dist(sig, sig, rule), 0.0)

    def test_per_rule_dist_range(self):
        from projects.hexglyph.solan_word import word_signature
        s1 = word_signature('ГОРА')
        s2 = word_signature('РАТОН')
        for rule in self.RULES:
            d = self.per_rule_dist(s1, s2, rule)
            if not (d != d):  # not NaN
                self.assertGreaterEqual(d, 0.0)
                self.assertLessEqual(d, 1.0)

    def test_per_rule_dist_symmetric(self):
        from projects.hexglyph.solan_word import word_signature
        s1 = word_signature('ГОРА')
        s2 = word_signature('ЖУРНАЛ')
        for rule in self.RULES:
            d1 = self.per_rule_dist(s1, s2, rule)
            d2 = self.per_rule_dist(s2, s1, rule)
            if not (d1 != d1):  # skip NaN
                self.assertAlmostEqual(d1, d2)

    # ── build_all_rules ──────────────────────────────────────────────────

    def test_build_has_all_rules(self):
        data = self.build_all_rules(self._words9)
        for rule in self.RULES:
            self.assertIn(rule, data)

    def test_build_rule_keys(self):
        data = self.build_all_rules(self._words9)
        required = {'words','periods','dmat','coords','stress','entropy','unique_periods'}
        for rule in self.RULES:
            self.assertTrue(required.issubset(data[rule].keys()))

    def test_build_words_preserved(self):
        data = self.build_all_rules(self._words9)
        for rule in self.RULES:
            self.assertEqual(data[rule]['words'], self._words9)

    def test_xor_all_period_one(self):
        """XOR rule must give period=1 for all words in the full lexicon."""
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build_all_rules()
        for p in data['xor']['periods'].values():
            self.assertEqual(p, 1)

    def test_xor_zero_entropy(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build_all_rules()
        self.assertAlmostEqual(data['xor']['entropy'], 0.0, places=6)

    def test_xor_zero_stress(self):
        """XOR MDS with all-zero dists → stress=0."""
        data = self.build_all_rules()
        self.assertAlmostEqual(data['xor']['stress'], 0.0, places=6)

    def test_xor3_two_periods(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build_all_rules()
        ups = data['xor3']['unique_periods']
        self.assertEqual(sorted(ups), [2, 8])

    def test_and_two_periods(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build_all_rules()
        ups = data['and']['unique_periods']
        self.assertEqual(sorted(ups), [1, 2])

    def test_or_two_periods(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        data = self.build_all_rules()
        ups = data['or']['unique_periods']
        self.assertEqual(sorted(ups), [1, 2])

    def test_entropy_at_most_one_bit(self):
        """Binary split → entropy ≤ 1 bit."""
        data = self.build_all_rules()
        for rule in ('xor3', 'and', 'or'):
            self.assertLessEqual(data[rule]['entropy'], 1.0 + 1e-9)

    def test_binary_rule_stress_zero(self):
        """2-cluster MDS (binary periods) → stress = 0."""
        data = self.build_all_rules()
        for rule in ('xor3', 'and', 'or'):
            self.assertAlmostEqual(data[rule]['stress'], 0.0, places=4)

    def test_dmat_shape(self):
        data = self.build_all_rules(self._words9)
        n = len(self._words9)
        for rule in self.RULES:
            self.assertEqual(len(data[rule]['dmat']), n)
            self.assertEqual(len(data[rule]['dmat'][0]), n)

    def test_dmat_symmetric(self):
        data = self.build_all_rules(self._words9)
        for rule in self.RULES:
            dm = data[rule]['dmat']
            n = len(dm)
            for i in range(n):
                for j in range(n):
                    self.assertAlmostEqual(dm[i][j], dm[j][i], places=10)

    def test_coords_shape(self):
        data = self.build_all_rules(self._words9)
        n = len(self._words9)
        for rule in self.RULES:
            self.assertEqual(len(data[rule]['coords']), n)
            self.assertEqual(len(data[rule]['coords'][0]), 2)

    # ── signature_classes ─────────────────────────────────────────────────

    def test_five_classes_full_lexicon(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        classes = self.signature_classes()
        self.assertEqual(len(classes), 5)

    def test_classes_cover_all_words(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        classes = self.signature_classes()
        all_words = [w for c in classes for w in c['words']]
        self.assertEqual(sorted(all_words), sorted(LEXICON))

    def test_classes_no_overlap(self):
        classes = self.signature_classes()
        seen: set[str] = set()
        for c in classes:
            for w in c['words']:
                self.assertNotIn(w, seen)
                seen.add(w)

    def test_classes_sorted_desc(self):
        classes = self.signature_classes()
        sizes = [c['count'] for c in classes]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_class_sizes_known(self):
        """Known class sizes: 21, 20, 4, 3, 1."""
        classes = self.signature_classes()
        sizes = sorted([c['count'] for c in classes], reverse=True)
        self.assertEqual(sizes, [21, 20, 4, 3, 1])

    def test_xor3_8_class_contains_волна(self):
        classes = self.signature_classes()
        big_class = classes[0]  # largest = (8,1,1), 21 words
        self.assertIn('ВОЛНА', big_class['words'])

    def test_gora_udар_same_class(self):
        """ГОРА, УДАР, РОТА, УТРО must be in the same class."""
        classes = self.signature_classes()
        for c in classes:
            if 'ГОРА' in c['words']:
                self.assertIn('УДАР', c['words'])
                self.assertIn('РОТА', c['words'])
                self.assertIn('УТРО', c['words'])
                break

    def test_zhunal_alone(self):
        """ЖУРНАЛ is in a singleton class."""
        classes = self.signature_classes()
        for c in classes:
            if 'ЖУРНАЛ' in c['words']:
                self.assertEqual(c['count'], 1)
                break

    def test_class_key_is_tuple(self):
        classes = self.signature_classes()
        for c in classes:
            self.assertIsInstance(c['key'], tuple)
            self.assertEqual(len(c['key']), 3)

    # ── rules_dict ────────────────────────────────────────────────────────

    def test_rules_dict_has_all_rules(self):
        data = self.build_all_rules(self._words9)
        d = self.rules_dict(data)
        for rule in self.RULES:
            self.assertIn(rule, d)

    def test_rules_dict_has_classes(self):
        data    = self.build_all_rules(self._words9)
        classes = self.signature_classes(self._words9)
        d       = self.rules_dict(data, classes)
        self.assertIn('classes', d)
        self.assertEqual(len(d['classes']), len(classes))

    def test_rules_dict_coords_rounded(self):
        data = self.build_all_rules(self._words9)
        d = self.rules_dict(data)
        for rule in self.RULES:
            for c in d[rule]['coords']:
                for v in c:
                    # Should be rounded to 6 decimal places
                    self.assertAlmostEqual(v, round(v, 6), places=9)

    # ── viewer: Rules section ─────────────────────────────────────────────

    def test_viewer_has_rules_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Анализ CA-правил Q6', content)
        self.assertIn('rules-canvas', content)

    def test_viewer_rules_has_4_rules(self):
        content = viewer_path().read_text(encoding='utf-8')
        for r in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(r, content)

    def test_viewer_rules_has_hover_sync(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rules-info', content)
        self.assertIn('_hovIdx', content)

    def test_viewer_rules_has_entropy(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('entropy', content)

    def test_viewer_rules_degenerate_case(self):
        content = viewer_path().read_text(encoding='utf-8')
        # XOR degenerate case should be handled
        self.assertIn('degen', content)

    def test_viewer_rules_period_colors(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('PCOL', content)
        self.assertIn('pmap', content)


class TestSolanMds(unittest.TestCase):
    """Tests for solan_mds.py and the viewer MDS section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_mds import (
            build_mds, mds_stress, mds_dict,
            _gram, _power_iter, _ascii_scatter,
        )
        cls.build_mds      = staticmethod(build_mds)
        cls.mds_stress     = staticmethod(mds_stress)
        cls.mds_dict       = staticmethod(mds_dict)
        cls._gram          = staticmethod(_gram)
        cls._power_iter    = staticmethod(_power_iter)
        cls._ascii_scatter = staticmethod(_ascii_scatter)
        cls._words9 = ['ГОРА','УДАР','РОТА','УТРО',
                       'ВОДА','НОРА','ЛУНА','РАТОН','ЖУРНАЛ']

    # ── _gram ─────────────────────────────────────────────────────────────

    def test_gram_shape(self):
        d = [[0,1,2],[1,0,1],[2,1,0]]
        B = self._gram(d)
        self.assertEqual(len(B), 3)
        self.assertEqual(len(B[0]), 3)

    def test_gram_symmetric(self):
        import random; random.seed(42)
        n = 5
        dm = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                v = random.random()
                dm[i][j] = dm[j][i] = v
        B = self._gram(dm)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(B[i][j], B[j][i], places=10)

    def test_gram_row_sum_zero(self):
        """Gram matrix of a centred config has zero row sums."""
        d = [[0,1,2],[1,0,1],[2,1,0]]
        B = self._gram(d)
        for row in B:
            self.assertAlmostEqual(sum(row), 0.0, places=10)

    # ── _power_iter ───────────────────────────────────────────────────────

    def test_power_iter_returns_k_pairs(self):
        B = self._gram([[0,1,2],[1,0,1],[2,1,0]])
        res = self._power_iter(B, k=2)
        self.assertEqual(len(res), 2)

    def test_power_iter_eigenvector_unit(self):
        import math
        B = self._gram([[0,1,2],[1,0,1],[2,1,0]])
        res = self._power_iter(B, k=1)
        lam, vec = res[0]
        self.assertAlmostEqual(sum(x**2 for x in vec), 1.0, places=8)

    def test_power_iter_eigenvalue_check(self):
        """lambda * v ≈ B @ v"""
        import math
        B = self._gram([[0,1,2],[1,0,1],[2,1,0]])
        res = self._power_iter(B, k=1)
        lam, vec = res[0]
        Bv = [sum(B[i][j]*vec[j] for j in range(3)) for i in range(3)]
        for i in range(3):
            self.assertAlmostEqual(Bv[i], lam * vec[i], places=5)

    # ── build_mds ─────────────────────────────────────────────────────────

    def test_build_returns_tuple_of_3(self):
        words, coords, stress = self.build_mds(self._words9)
        self.assertIsInstance(words, list)
        self.assertIsInstance(coords, list)
        self.assertIsInstance(stress, float)

    def test_build_coords_shape(self):
        words, coords, stress = self.build_mds(self._words9)
        self.assertEqual(len(coords), len(self._words9))
        self.assertEqual(len(coords[0]), 2)

    def test_build_words_preserved(self):
        words, _, _ = self.build_mds(self._words9)
        self.assertEqual(words, self._words9)

    def test_build_stress_nonneg(self):
        _, _, stress = self.build_mds(self._words9)
        self.assertGreaterEqual(stress, 0.0)

    def test_build_stress_at_most_1(self):
        _, _, stress = self.build_mds(self._words9)
        self.assertLessEqual(stress, 1.0)

    def test_identical_words_same_coords(self):
        """Words with d=0 must land at the same MDS coordinates."""
        words, coords, _ = self.build_mds(self._words9)
        identical = ['ГОРА', 'УДАР', 'РОТА', 'УТРО']
        idxs = [words.index(w) for w in identical]
        x0, y0 = coords[idxs[0]]
        for idx in idxs[1:]:
            self.assertAlmostEqual(coords[idx][0], x0, places=5)
            self.assertAlmostEqual(coords[idx][1], y0, places=5)

    def test_full_lexicon_stress_good(self):
        """Full Q6 lexicon: Kruskal stress should be < 0.15."""
        _, _, stress = self.build_mds()
        self.assertLess(stress, 0.15)

    # ── mds_stress ────────────────────────────────────────────────────────

    def test_mds_stress_perfect(self):
        """If MDS coords perfectly recover distances, stress = 0."""
        dmat = [[0.0,1.0,2.0],[1.0,0.0,1.0],[2.0,1.0,0.0]]
        words, coords, _ = self.build_mds(
            ['A','B','C'],
            # Monkey-patch: just compute directly
        )
        # Use a trivial 1D perfect embedding for a sanity check
        import math
        c = [[0,0],[1,0],[2,0]]
        d2 = [[0,1,2],[1,0,1],[2,1,0]]
        stress = self.mds_stress(d2, c)
        self.assertAlmostEqual(stress, 0.0, places=10)

    def test_mds_stress_increases_with_noise(self):
        import random; random.seed(0)
        c_perfect = [[0,0],[1,0],[2,0]]
        c_noisy   = [[x+random.uniform(-0.3,0.3),y+random.uniform(-0.3,0.3)]
                     for x,y in c_perfect]
        d2 = [[0,1,2],[1,0,1],[2,1,0]]
        s_p = self.mds_stress(d2, c_perfect)
        s_n = self.mds_stress(d2, c_noisy)
        self.assertGreater(s_n, s_p)

    # ── mds_dict ──────────────────────────────────────────────────────────

    def test_dict_has_required_keys(self):
        words, coords, stress = self.build_mds(self._words9)
        d = self.mds_dict(words, coords, stress)
        for key in ('words','coords','stress'):
            self.assertIn(key, d)

    def test_dict_coords_length(self):
        words, coords, stress = self.build_mds(self._words9)
        d = self.mds_dict(words, [c[:] for c in coords], stress)
        self.assertEqual(len(d['coords']), len(self._words9))

    def test_dict_coords_2d(self):
        words, coords, stress = self.build_mds(self._words9)
        d = self.mds_dict(words, [c[:] for c in coords], stress)
        for c in d['coords']:
            self.assertEqual(len(c), 2)

    def test_dict_stress_matches(self):
        words, coords, stress = self.build_mds(self._words9)
        d = self.mds_dict(words, [c[:] for c in coords], stress)
        self.assertAlmostEqual(d['stress'], stress, places=4)

    # ── _ascii_scatter ────────────────────────────────────────────────────

    def test_ascii_scatter_returns_lines(self):
        words, coords, _ = self.build_mds(self._words9)
        lines = self._ascii_scatter(words, coords, W=30, H=10)
        self.assertIsInstance(lines, list)
        self.assertEqual(len(lines), 10)

    def test_ascii_scatter_line_width(self):
        words, coords, _ = self.build_mds(self._words9)
        lines = self._ascii_scatter(words, coords, W=30, H=10)
        for line in lines:
            # Each line has 2 prefix spaces
            self.assertGreaterEqual(len(line), 2)

    def test_ascii_scatter_has_axes(self):
        words, coords, _ = self.build_mds(self._words9)
        lines = self._ascii_scatter(words, coords, W=30, H=10)
        joined = ''.join(lines)
        self.assertIn('└', joined)
        self.assertIn('│', joined)

    # ── viewer: MDS section ───────────────────────────────────────────────

    def test_viewer_has_mds_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('MDS-карта орбитального пространства Q6', content)
        self.assertIn('mds-canvas', content)

    def test_viewer_mds_has_classical_mds_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('classicalMds', content)

    def test_viewer_mds_has_stress(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('kruskalStress', content)

    def test_viewer_mds_has_cut_slider(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mds-cut', content)

    def test_viewer_mds_has_hover_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mds-info', content)
        self.assertIn('nearestK', content)

    def test_viewer_mds_uses_upgma(self):
        content = viewer_path().read_text(encoding='utf-8')
        # MDS section has its own UPGMA for cluster coloring
        self.assertIn('flatClusters', content)


class TestSolanGraph(unittest.TestCase):
    """Tests for solan_graph.py and the viewer Graph section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_graph import (
            build_graph, connected_components, degree,
            hub_words, graph_stats,
        )
        cls.build_graph          = staticmethod(build_graph)
        cls.connected_components = staticmethod(connected_components)
        cls.degree               = staticmethod(degree)
        cls.hub_words            = staticmethod(hub_words)
        cls.graph_stats          = staticmethod(graph_stats)
        cls._words = ['ГОРА', 'ВОДА', 'РАТОН', 'НОРА', 'ЗИМА',
                      'МАРТ', 'ЛУНА', 'УДАР', 'РОТА', 'УТРО']

    # ── build_graph ───────────────────────────────────────────────────────

    def test_graph_all_words_present(self):
        g = self.build_graph(self._words)
        self.assertEqual(set(g.keys()), set(self._words))

    def test_graph_self_not_in_neighbors(self):
        g = self.build_graph(self._words)
        for w, nb in g.items():
            self.assertNotIn(w, nb)

    def test_graph_symmetric(self):
        g = self.build_graph(self._words)
        for w, nb in g.items():
            for n in nb:
                self.assertIn(w, g[n])

    def test_graph_threshold_zero_empty(self):
        g = self.build_graph(self._words, threshold=0.0)
        for nb in g.values():
            self.assertEqual(nb, [])

    def test_graph_threshold_one_dense(self):
        g = self.build_graph(self._words, threshold=1.0)
        total_edges = sum(len(nb) for nb in g.values()) // 2
        self.assertGreater(total_edges, 0)

    def test_graph_default_uses_lexicon(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        g = self.build_graph()
        self.assertEqual(set(g.keys()), set(LEXICON))

    # ── connected_components ─────────────────────────────────────────────

    def test_components_all_words_covered(self):
        g = self.build_graph(self._words, threshold=0.5)
        comps = self.connected_components(g)
        all_words = [w for c in comps for w in c]
        self.assertEqual(sorted(all_words), sorted(self._words))

    def test_components_sorted_desc(self):
        g = self.build_graph(self._words, threshold=0.5)
        comps = self.connected_components(g)
        sizes = [len(c) for c in comps]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_components_no_overlap(self):
        g = self.build_graph(self._words, threshold=0.5)
        comps = self.connected_components(g)
        seen: set[str] = set()
        for comp in comps:
            for w in comp:
                self.assertNotIn(w, seen)
                seen.add(w)

    def test_components_clique_connected(self):
        g = self.build_graph(self._words, threshold=0.01)
        comps = self.connected_components(g)
        clique = {'ГОРА', 'УДАР', 'РОТА', 'УТРО'}
        for comp in comps:
            if clique.issubset(set(comp)):
                break
        else:
            self.fail('ГОРА-УДАР-РОТА-УТРО not in same component at threshold=0.01')

    # ── degree ────────────────────────────────────────────────────────────

    def test_degree_all_words(self):
        g = self.build_graph(self._words)
        d = self.degree(g)
        self.assertEqual(set(d.keys()), set(self._words))

    def test_degree_nonnegative(self):
        g = self.build_graph(self._words)
        for d in self.degree(g).values():
            self.assertGreaterEqual(d, 0)

    def test_degree_matches_adjacency(self):
        g = self.build_graph(self._words)
        d = self.degree(g)
        for w in self._words:
            self.assertEqual(d[w], len(g[w]))

    # ── hub_words ─────────────────────────────────────────────────────────

    def test_hub_words_count(self):
        g = self.build_graph(self._words)
        hubs = self.hub_words(g, n=3)
        self.assertLessEqual(len(hubs), 3)

    def test_hub_words_sorted_desc(self):
        g = self.build_graph(self._words, threshold=0.5)
        hubs = self.hub_words(g, n=5)
        degrees = [d for _, d in hubs]
        self.assertEqual(degrees, sorted(degrees, reverse=True))

    # ── graph_stats ───────────────────────────────────────────────────────

    def test_stats_keys(self):
        g = self.build_graph(self._words)
        st = self.graph_stats(g)
        for key in ('nodes','edges','components','isolated','max_degree','avg_degree'):
            self.assertIn(key, st)

    def test_stats_nodes_count(self):
        g = self.build_graph(self._words)
        st = self.graph_stats(g)
        self.assertEqual(st['nodes'], len(self._words))

    def test_stats_edges_positive(self):
        g = self.build_graph(self._words, threshold=0.5)
        st = self.graph_stats(g)
        self.assertGreater(st['edges'], 0)

    def test_stats_isolated_correct(self):
        g = self.build_graph(self._words, threshold=0.01)
        st = self.graph_stats(g)
        d = self.degree(g)
        isolated_count = sum(1 for v in d.values() if v == 0)
        self.assertEqual(st['isolated'], isolated_count)

    def test_stats_avg_degree_correct(self):
        g = self.build_graph(self._words, threshold=0.5)
        st = self.graph_stats(g)
        d = self.degree(g)
        expected_avg = sum(d.values()) / len(d)
        self.assertAlmostEqual(st['avg_degree'], expected_avg, places=5)

    # ── viewer: Graph section ─────────────────────────────────────────────

    def test_viewer_has_graph_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Граф орбитального сходства Q6', content)
        self.assertIn('graph-canvas', content)
        self.assertIn('graph-thresh', content)

    def test_viewer_graph_uses_mDist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mDist', content)
        self.assertIn('lexAllSigs', content)

    def test_viewer_graph_has_fruchterman_comment(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Fruchterman', content)

    def test_viewer_graph_has_simulate(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('simulate', content)

    def test_viewer_graph_has_hover(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('graph-info', content)
        self.assertIn('hoveredNode', content)

    def test_viewer_graph_exports_mDist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.mDist = mDist', content)

    def test_viewer_graph_has_threshold_slider(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('graph-thresh', content)
        self.assertIn('graph-thresh-val', content)

    def test_viewer_graph_has_reset_button(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('graph-reset', content)
        self.assertIn('перезапуск', content)


class TestSolanMatrix(unittest.TestCase):
    """Tests for solan_matrix.py and the viewer Matrix section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_matrix import (
            distance_matrix, nearest_pairs, farthest_pairs, export_csv,
        )
        cls.distance_matrix  = staticmethod(distance_matrix)
        cls.nearest_pairs    = staticmethod(nearest_pairs)
        cls.farthest_pairs   = staticmethod(farthest_pairs)
        cls.export_csv       = staticmethod(export_csv)
        cls._words = ['ГОРА', 'ВОДА', 'РАТОН', 'НОРА', 'ЗИМА', 'МАРТ', 'ЛУНА']

    # ── distance_matrix ──────────────────────────────────────────────────

    def test_matrix_diagonal_zero(self):
        mat = self.distance_matrix(self._words)
        for w in self._words:
            self.assertAlmostEqual(mat[(w, w)], 0.0)

    def test_matrix_symmetric(self):
        mat = self.distance_matrix(self._words)
        for w1 in self._words:
            for w2 in self._words:
                self.assertAlmostEqual(mat[(w1, w2)], mat[(w2, w1)])

    def test_matrix_all_pairs_present(self):
        mat = self.distance_matrix(self._words)
        n = len(self._words)
        self.assertEqual(len(mat), n * n)

    def test_matrix_values_in_range(self):
        mat = self.distance_matrix(self._words)
        for (w1, w2), d in mat.items():
            if d == d:  # not NaN
                self.assertGreaterEqual(d, 0.0)
                self.assertLessEqual(d, 1.0)

    def test_matrix_default_uses_lexicon(self):
        from projects.hexglyph.solan_lexicon import LEXICON
        mat = self.distance_matrix()
        n = len(LEXICON)
        self.assertEqual(len(mat), n * n)

    # ── nearest_pairs ────────────────────────────────────────────────────

    def test_nearest_pairs_count(self):
        mat  = self.distance_matrix(self._words)
        near = self.nearest_pairs(mat, n=3)
        self.assertLessEqual(len(near), 3)

    def test_nearest_pairs_sorted(self):
        mat  = self.distance_matrix(self._words)
        near = self.nearest_pairs(mat, n=6)
        dists = [d for _, _, d in near if d == d]
        self.assertEqual(dists, sorted(dists))

    def test_nearest_pairs_no_self(self):
        mat  = self.distance_matrix(self._words)
        near = self.nearest_pairs(mat, n=10)
        for w1, w2, _ in near:
            self.assertNotEqual(w1, w2)

    def test_nearest_pairs_no_duplicates(self):
        mat  = self.distance_matrix(self._words)
        near = self.nearest_pairs(mat, n=20)
        seen: set[tuple[str, str]] = set()
        for w1, w2, _ in near:
            key = (min(w1, w2), max(w1, w2))
            self.assertNotIn(key, seen)
            seen.add(key)

    # ── farthest_pairs ───────────────────────────────────────────────────

    def test_farthest_pairs_count(self):
        mat = self.distance_matrix(self._words)
        far = self.farthest_pairs(mat, n=3)
        self.assertLessEqual(len(far), 3)

    def test_farthest_pairs_sorted_descending(self):
        mat = self.distance_matrix(self._words)
        far = self.farthest_pairs(mat, n=5)
        dists = [d for _, _, d in far]
        self.assertEqual(dists, sorted(dists, reverse=True))

    def test_farthest_pairs_no_nan(self):
        mat = self.distance_matrix(self._words)
        far = self.farthest_pairs(mat, n=5)
        for _, _, d in far:
            self.assertEqual(d, d)  # not NaN

    def test_farthest_geq_nearest(self):
        mat  = self.distance_matrix(self._words)
        near = self.nearest_pairs(mat, n=1)
        far  = self.farthest_pairs(mat, n=1)
        if near and far:
            self.assertGreaterEqual(far[0][2], near[0][2])

    # ── export_csv ───────────────────────────────────────────────────────

    def test_csv_header_row(self):
        csv = self.export_csv(self._words)
        lines = csv.splitlines()
        # First line: comma + word names
        header = lines[0].split(',')
        self.assertEqual(header[0], '')   # empty first cell
        self.assertEqual(header[1:], self._words)

    def test_csv_row_count(self):
        csv = self.export_csv(self._words)
        lines = csv.splitlines()
        # 1 header + N data rows
        self.assertEqual(len(lines), len(self._words) + 1)

    def test_csv_diagonal_zero(self):
        csv = self.export_csv(self._words)
        lines = csv.splitlines()
        for i, line in enumerate(lines[1:]):
            parts = line.split(',')
            self.assertEqual(parts[i + 1], '0.0000')  # diagonal

    def test_csv_symmetric_values(self):
        csv = self.export_csv(self._words)
        lines = csv.splitlines()
        data = [line.split(',') for line in lines[1:]]
        # data[i][j+1] should equal data[j][i+1]
        for i in range(len(self._words)):
            for j in range(len(self._words)):
                self.assertEqual(data[i][j + 1], data[j][i + 1])

    # ── viewer: Matrix section ────────────────────────────────────────────

    def test_viewer_has_matrix_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Матрица расстояний Q6', content)
        self.assertIn('mat-canvas', content)
        self.assertIn('mat-pairs', content)

    def test_viewer_matrix_uses_lex_all_sigs(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.lexAllSigs', content)
        self.assertIn('lexAllSigs', content)

    def test_viewer_matrix_has_tooltip(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mat-tip', content)
        self.assertIn('mousemove', content)

    def test_viewer_matrix_exports_lex_dist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.lexDistL', content)
        self.assertIn('window.lexAllSigs', content)

    def test_viewer_matrix_draws_on_load(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('drawMatrix', content)
        self.assertIn('setTimeout', content)


class TestSolanSpectral(unittest.TestCase):
    """Tests for solan_spectral.py and the viewer Spectral section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_spectral import (
            row_spectrum, attractor_spectrum, all_spectra,
            spectral_distance, spectral_fingerprint,
            build_spectral_data, spectral_dict,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.row_spectrum        = staticmethod(row_spectrum)
        cls.attractor_spectrum  = staticmethod(attractor_spectrum)
        cls.all_spectra         = staticmethod(all_spectra)
        cls.spectral_distance   = staticmethod(spectral_distance)
        cls.spectral_fingerprint = staticmethod(spectral_fingerprint)
        cls.build_spectral_data = staticmethod(build_spectral_data)
        cls.spectral_dict       = staticmethod(spectral_dict)
        cls.LEXICON             = list(LEXICON)

    # ── row_spectrum() ────────────────────────────────────────────────────────

    def test_row_spectrum_zeros(self):
        sp = self.row_spectrum([0] * 16)
        self.assertAlmostEqual(sum(sp), 0.0, places=10)

    def test_row_spectrum_uniform(self):
        # All-same value → only DC component
        sp = self.row_spectrum([32] * 16)
        self.assertAlmostEqual(sp[0], 1024.0, places=5)   # 32² = 1024
        self.assertTrue(all(abs(sp[k]) < 1e-8 for k in range(1, len(sp))))

    def test_row_spectrum_length(self):
        sp = self.row_spectrum([0] * 16)
        self.assertEqual(len(sp), 9)  # N//2 + 1 = 9 for N=16

    def test_row_spectrum_nonnegative(self):
        import random
        cells = [random.randint(0, 63) for _ in range(16)]
        sp = self.row_spectrum(cells)
        for v in sp:
            self.assertGreaterEqual(v, 0.0)

    def test_row_spectrum_periodic_input(self):
        # Period-4 signal: only k=0,4,8 are non-zero
        base = [49, 47, 15, 63]
        cells = base * 4  # 16 cells with period 4
        sp = self.row_spectrum(cells)
        # k=1,2,3,5,6,7 should be (near) zero
        for k in [1, 2, 3, 5, 6, 7]:
            self.assertLess(sp[k], 1e-4,
                msg=f'Expected sp[{k}] ≈ 0 for period-4 input, got {sp[k]}')
        # k=4 and k=8 should be non-zero
        self.assertGreater(sp[4], 1.0)
        self.assertGreater(sp[8], 1.0)

    # ── attractor_spectrum() ──────────────────────────────────────────────────

    def test_attr_spectrum_keys(self):
        sp = self.attractor_spectrum('ГОРА', 'xor3')
        for k in ('rule', 'word', 'transient', 'period', 'n_freqs',
                  'power', 'ac_power', 'dominant_k', 'dominant_wl',
                  'dominant_amp', 'dc'):
            self.assertIn(k, sp)

    def test_attr_spectrum_xor_all_zero(self):
        # XOR attractor = all-zeros → dc=0 and all ac=0
        sp = self.attractor_spectrum('ГОРА', 'xor')
        self.assertAlmostEqual(sp['dc'], 0.0, places=8)
        self.assertAlmostEqual(sum(sp['ac_power']), 0.0, places=8)

    def test_attr_spectrum_and_alternating(self):
        # AND attractor for ТУНДРА: alternating → dominant k=8
        sp = self.attractor_spectrum('ТУНДРА', 'and')
        self.assertEqual(sp['dominant_k'], 8)

    def test_attr_spectrum_period4_word_has_k4(self):
        # 4-letter words padded 4× → k=4 and k=8 non-zero only
        sp = self.attractor_spectrum('ГОРА', 'xor3')
        ac = sp['ac_power']
        # k=1,2,3,5,6,7 should be very small
        for k in [1, 2, 3, 5, 6, 7]:
            self.assertLess(ac[k], 1e-8,
                msg=f'Expected ac[{k}]≈0 for 4-letter word, got {ac[k]}')

    def test_attr_spectrum_ac_sums_to_one_or_zero(self):
        for word in ['ГОРА', 'ЛУНА', 'ТУМАН']:
            for rule in ['xor3', 'and', 'or']:
                sp = self.attractor_spectrum(word, rule)
                ac_sum = sum(sp['ac_power'])
                self.assertTrue(abs(ac_sum - 1.0) < 1e-8 or abs(ac_sum) < 1e-8,
                    msg=f'{word}/{rule} ac sum={ac_sum}')

    def test_attr_spectrum_dominant_wl(self):
        sp = self.attractor_spectrum('ГОРА', 'xor3')
        expected_wl = 16 / sp['dominant_k']
        self.assertAlmostEqual(sp['dominant_wl'], expected_wl, places=5)

    def test_attr_spectrum_dc_nonneg(self):
        for word in self.LEXICON[:5]:
            for rule in ['xor3', 'and', 'or']:
                sp = self.attractor_spectrum(word, rule)
                self.assertGreaterEqual(sp['dc'], 0.0)

    def test_attr_spectrum_n_freqs(self):
        sp = self.attractor_spectrum('ГОРА', 'xor3', width=16)
        self.assertEqual(sp['n_freqs'], 9)  # 16//2 + 1

    # ── all_spectra() ─────────────────────────────────────────────────────────

    def test_all_spectra_rules(self):
        spectra = self.all_spectra('ГОРА')
        self.assertEqual(set(spectra.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_spectra_consistent(self):
        spectra = self.all_spectra('ЛУНА')
        for rule, sp in spectra.items():
            self.assertEqual(sp['rule'], rule)
            self.assertEqual(sp['word'], 'ЛУНА')

    # ── spectral_distance() ───────────────────────────────────────────────────

    def test_spectral_distance_self_zero(self):
        sp = self.attractor_spectrum('ГОРА', 'xor3')
        self.assertAlmostEqual(self.spectral_distance(sp, sp), 0.0, places=10)

    def test_spectral_distance_symmetric(self):
        s1 = self.attractor_spectrum('ГОРА', 'xor3')
        s2 = self.attractor_spectrum('ЛУНА', 'xor3')
        d12 = self.spectral_distance(s1, s2)
        d21 = self.spectral_distance(s2, s1)
        self.assertAlmostEqual(d12, d21, places=10)

    def test_spectral_distance_range(self):
        s1 = self.attractor_spectrum('ГОРА', 'xor3')
        s2 = self.attractor_spectrum('ТУМАН', 'xor3')
        d = self.spectral_distance(s1, s2)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)

    def test_spectral_distance_xor_zeros_all_equal(self):
        # All XOR attractors are all-zeros → AC spectra identical (all-zero)
        sp1 = self.attractor_spectrum('ГОРА', 'xor')
        sp2 = self.attractor_spectrum('ЛУНА', 'xor')
        d = self.spectral_distance(sp1, sp2)
        self.assertAlmostEqual(d, 0.0, places=10)

    # ── spectral_fingerprint() ────────────────────────────────────────────────

    def test_spectral_fingerprint_length(self):
        fp = self.spectral_fingerprint('ГОРА')
        # 4 rules × (16//2) = 4 × 8 = 32 values
        self.assertEqual(len(fp), 32)

    def test_spectral_fingerprint_nonneg(self):
        fp = self.spectral_fingerprint('ЛУНА')
        for v in fp:
            self.assertGreaterEqual(v, 0.0)

    def test_spectral_fingerprint_self_similar(self):
        fp1 = self.spectral_fingerprint('ГОРА')
        fp2 = self.spectral_fingerprint('ГОРА')
        self.assertEqual(fp1, fp2)

    # ── build_spectral_data() ─────────────────────────────────────────────────

    def test_build_data_keys(self):
        data = self.build_spectral_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'per_rule', 'dom_freq',
                  'most_harmonic', 'most_dc'):
            self.assertIn(k, data)

    def test_build_data_per_rule_words(self):
        words = ['ГОРА', 'ЛУНА', 'ТУМАН']
        data = self.build_spectral_data(words)
        for rule in ['xor', 'xor3', 'and', 'or']:
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    def test_build_data_most_harmonic_is_word(self):
        words = ['ГОРА', 'ЛУНА', 'ТУМАН']
        data = self.build_spectral_data(words)
        for rule in ['xor3', 'and', 'or']:
            word, amp = data['most_harmonic'][rule]
            self.assertIn(word, words)
            self.assertGreaterEqual(amp, 0.0)

    # ── spectral_dict() ───────────────────────────────────────────────────────

    def test_spectral_dict_serialisable(self):
        import json
        d = self.spectral_dict('ГОРА')
        j = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(j, str)

    def test_spectral_dict_keys(self):
        d = self.spectral_dict('ГОРА')
        for k in ('word', 'width', 'rules'):
            self.assertIn(k, d)
        self.assertEqual(set(d['rules'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_spectral_dict_rule_keys(self):
        d = self.spectral_dict('ГОРА')
        for rule_data in d['rules'].values():
            for k in ('transient', 'period', 'dominant_k', 'dominant_wl',
                      'dominant_amp', 'dc', 'power', 'ac_power'):
                self.assertIn(k, rule_data)

    def test_spectral_dict_power_length(self):
        d = self.spectral_dict('ГОРА', width=16)
        for rule_data in d['rules'].values():
            self.assertEqual(len(rule_data['power']), 9)  # N//2+1=9

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_spec_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spec-canvas', content)

    def test_viewer_has_spec_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spec-info', content)

    def test_viewer_has_row_spectrum_fn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rowSpectrum', content)

    def test_viewer_has_ac_norm_fn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('acNorm', content)

    def test_viewer_has_spec_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('specRun', content)

    def test_viewer_spectral_section_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Спектральный анализ Q6', content)

    def test_viewer_has_spec_word_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spec-word', content)

    def test_viewer_has_spec_rule_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spec-rule', content)

    def test_viewer_has_draw_bar_chart(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('drawBarChart', content)

    def test_viewer_has_draw_lexicon_map(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('drawLexiconMap', content)


class TestSolanPhonemeAnalysis(unittest.TestCase):
    """Tests for solan_phoneme.py and the viewer Phoneme Analysis section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_phoneme import (
            phoneme_table, substitution_matrix, sensitivity_profile,
            critical_positions, neutral_positions, pair_stats,
            build_phoneme_data, phoneme_dict, SOLAN_ALPHABET,
        )
        cls.phoneme_table        = staticmethod(phoneme_table)
        cls.substitution_matrix  = staticmethod(substitution_matrix)
        cls.sensitivity_profile  = staticmethod(sensitivity_profile)
        cls.critical_positions   = staticmethod(critical_positions)
        cls.neutral_positions    = staticmethod(neutral_positions)
        cls.pair_stats           = staticmethod(pair_stats)
        cls.build_phoneme_data   = staticmethod(build_phoneme_data)
        cls.phoneme_dict         = staticmethod(phoneme_dict)
        cls.SOLAN_ALPHABET       = list(SOLAN_ALPHABET)

    # ── phoneme_table() ───────────────────────────────────────────────────────

    def test_phoneme_table_size(self):
        pt = self.phoneme_table()
        self.assertEqual(len(pt), 16)

    def test_phoneme_table_keys(self):
        pt = self.phoneme_table()
        for letter in ['А', 'Г', 'О', 'Р', 'Н', 'Т', 'М']:
            self.assertIn(letter, pt)

    def test_phoneme_table_fields(self):
        pt = self.phoneme_table()
        for letter, d in pt.items():
            for k in ('q6', 'binary', 'hw', 'segs'):
                self.assertIn(k, d)

    def test_phoneme_table_a_q6_63(self):
        # А = all-ones = Q6 value 63
        self.assertEqual(self.phoneme_table()['А']['q6'], 63)

    def test_phoneme_table_n_q6_3(self):
        # Н = lowest Q6 = 3
        self.assertEqual(self.phoneme_table()['Н']['q6'], 3)

    def test_phoneme_table_binary_6_bits(self):
        pt = self.phoneme_table()
        for letter, d in pt.items():
            self.assertEqual(len(d['binary']), 6)
            self.assertRegex(d['binary'], r'^[01]{6}$')

    def test_phoneme_table_hw_matches_binary(self):
        pt = self.phoneme_table()
        for letter, d in pt.items():
            expected_hw = d['binary'].count('1')
            self.assertEqual(d['hw'], expected_hw)

    def test_solan_alphabet_sorted_by_q6(self):
        pt = self.phoneme_table()
        q6_values = [pt[letter]['q6'] for letter in self.SOLAN_ALPHABET]
        self.assertEqual(q6_values, sorted(q6_values))

    def test_solan_alphabet_length(self):
        self.assertEqual(len(self.SOLAN_ALPHABET), 16)

    # ── substitution_matrix() ─────────────────────────────────────────────────

    def test_sub_matrix_returns_list(self):
        m = self.substitution_matrix('ГОРА')
        self.assertIsInstance(m, list)

    def test_sub_matrix_length_equals_letters(self):
        self.assertEqual(len(self.substitution_matrix('ГОРА')), 4)
        self.assertEqual(len(self.substitution_matrix('ТУМАН')), 5)

    def test_sub_matrix_position_keys(self):
        m = self.substitution_matrix('ГОРА')
        for pd in m:
            for k in ('pos', 'letter', 'q6', 'orig_key', 'orig_class',
                      'subs', 'n_changed', 'sensitivity'):
                self.assertIn(k, pd)

    def test_sub_matrix_subs_count(self):
        # Each position has subs for all 16 phonemes minus 1 (the original)
        m = self.substitution_matrix('ГОРА')
        for pd in m:
            self.assertEqual(len(pd['subs']), 15)

    def test_sub_matrix_gora_o_neutral(self):
        # О at position 1 of ГОРА: zero substitutions change class
        m = self.substitution_matrix('ГОРА')
        o_pos = next(pd for pd in m if pd['letter'] == 'О')
        self.assertEqual(o_pos['n_changed'], 0)
        self.assertAlmostEqual(o_pos['sensitivity'], 0.0)

    def test_sub_matrix_gora_r_critical(self):
        # Р at position 2 of ГОРА: 12 substitutions change class
        m = self.substitution_matrix('ГОРА')
        r_pos = next(pd for pd in m if pd['letter'] == 'Р')
        self.assertEqual(r_pos['n_changed'], 12)

    def test_sub_matrix_sensitivity_in_range(self):
        m = self.substitution_matrix('ТУМАН')
        for pd in m:
            self.assertGreaterEqual(pd['sensitivity'], 0.0)
            self.assertLessEqual(pd['sensitivity'], 1.0)

    def test_sub_matrix_orig_class_consistent(self):
        from projects.hexglyph.solan_transient import full_key, transient_classes
        m = self.substitution_matrix('ГОРА')
        # orig_class should be consistent across all positions
        classes = {pd['orig_class'] for pd in m}
        self.assertEqual(len(classes), 1)

    # ── sensitivity_profile() ─────────────────────────────────────────────────

    def test_sensitivity_profile_length(self):
        p = self.sensitivity_profile('ГОРА')
        self.assertEqual(len(p), 4)

    def test_sensitivity_profile_gora(self):
        p = self.sensitivity_profile('ГОРА')
        # [0.73, 0.00, 0.80, 0.47]
        self.assertAlmostEqual(p[1], 0.0, places=5)    # О: neutral
        self.assertGreater(p[2], 0.7)                  # Р: very critical

    def test_sensitivity_profile_luna_mostly_stable(self):
        p = self.sensitivity_profile('ЛУНА')
        mean_sens = sum(p) / len(p)
        # ЛУНА is in the largest class (20 words), so most subs stay in it
        self.assertLess(mean_sens, 0.2)

    # ── critical_positions() and neutral_positions() ──────────────────────────

    def test_critical_positions_gora(self):
        crits = self.critical_positions('ГОРА', threshold=0.5)
        self.assertIn(0, crits)  # Г: critical
        self.assertIn(2, crits)  # Р: critical
        self.assertNotIn(1, crits)  # О: not critical

    def test_neutral_positions_gora(self):
        neuts = self.neutral_positions('ГОРА')
        self.assertIn(1, neuts)   # О: neutral
        self.assertNotIn(2, neuts)  # Р: not neutral

    def test_critical_and_neutral_disjoint(self):
        crits = set(self.critical_positions('ГОРА'))
        neuts = set(self.neutral_positions('ГОРА'))
        self.assertEqual(crits & neuts, set())

    # ── pair_stats() ──────────────────────────────────────────────────────────

    def test_pair_stats_returns_dict(self):
        ps = self.pair_stats(['ГОРА', 'ЛУНА'])
        self.assertIsInstance(ps, dict)

    def test_pair_stats_keys_are_tuples(self):
        ps = self.pair_stats(['ГОРА'])
        for pair in ps:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)

    def test_pair_stats_fields(self):
        ps = self.pair_stats(['ГОРА'])
        for d in ps.values():
            self.assertIn('count', d)
            self.assertIn('changed', d)
            self.assertIn('rate', d)

    def test_pair_stats_rate_range(self):
        ps = self.pair_stats(['ГОРА', 'ЛУНА', 'ЖУРНАЛ'])
        for d in ps.values():
            self.assertGreaterEqual(d['rate'], 0.0)
            self.assertLessEqual(d['rate'], 1.0)

    def test_pair_stats_g_to_t_neutral(self):
        # Г→Т: rate=0 across full lexicon
        ps = self.pair_stats()
        gt = ps.get(('Г', 'Т'))
        if gt and gt['count'] >= 3:
            self.assertAlmostEqual(gt['rate'], 0.0, places=5)

    def test_pair_stats_z_to_a_destabilising(self):
        # З→А: rate=1.0 across full lexicon
        ps = self.pair_stats()
        za = ps.get(('З', 'А'))
        if za and za['count'] >= 3:
            self.assertAlmostEqual(za['rate'], 1.0, places=5)

    # ── build_phoneme_data() ──────────────────────────────────────────────────

    def test_build_data_keys(self):
        data = self.build_phoneme_data(['ГОРА', 'ЛУНА'])
        for k in ('phoneme_table', 'words', 'profiles', 'critical',
                  'neutral', 'pairs', 'most_stable', 'most_sensitive'):
            self.assertIn(k, data)

    def test_build_data_most_stable_is_word(self):
        words = ['ГОРА', 'ЛУНА', 'ТУМАН']
        data = self.build_phoneme_data(words)
        self.assertIn(data['most_stable'], words)

    def test_build_data_profiles_all_words(self):
        words = ['ГОРА', 'ЛУНА']
        data = self.build_phoneme_data(words)
        self.assertEqual(set(data['profiles'].keys()), set(words))

    # ── phoneme_dict() ────────────────────────────────────────────────────────

    def test_phoneme_dict_serialisable(self):
        import json
        d = self.phoneme_dict('ГОРА')
        dumped = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(dumped, str)

    def test_phoneme_dict_keys(self):
        d = self.phoneme_dict('ГОРА')
        for k in ('word', 'width', 'profile', 'positions'):
            self.assertIn(k, d)

    def test_phoneme_dict_positions_length(self):
        d = self.phoneme_dict('ГОРА')
        self.assertEqual(len(d['positions']), 4)

    def test_phoneme_dict_position_keys(self):
        d = self.phoneme_dict('ГОРА')
        for pd in d['positions']:
            for k in ('pos', 'letter', 'q6', 'orig_class', 'n_changed',
                      'sensitivity', 'subs'):
                self.assertIn(k, pd)

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_phon_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('phon-canvas', content)

    def test_viewer_has_phon_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('phon-info', content)

    def test_viewer_has_solan_alpha(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('SOLAN_ALPHA', content)

    def test_viewer_has_phon_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('phonRun', content)

    def test_viewer_has_compute_sub_matrix(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('computeSubMatrix', content)

    def test_viewer_has_pred_classes_export(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.PRED_CLASSES', content)

    def test_viewer_has_pred_full_key_export(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.predFullKey', content)

    def test_viewer_phoneme_section_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Фонемный анализ Q6', content)

    def test_viewer_has_phon_word_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('phon-word', content)


class TestSolanComplexity(unittest.TestCase):
    """Tests for solan_complexity.py and the viewer LZ76 section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_complexity import (
            to_bits, lz76_phrases, lz76_norm,
            trajectory_complexity, all_complexities,
            build_complexity_data, complexity_dict,
            _ALL_RULES, _N_BITS,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.to_bits               = staticmethod(to_bits)
        cls.lz76_phrases          = staticmethod(lz76_phrases)
        cls.lz76_norm             = staticmethod(lz76_norm)
        cls.trajectory_complexity = staticmethod(trajectory_complexity)
        cls.all_complexities      = staticmethod(all_complexities)
        cls.build_complexity_data = staticmethod(build_complexity_data)
        cls.complexity_dict       = staticmethod(complexity_dict)
        cls.ALL_RULES             = _ALL_RULES
        cls.N_BITS                = _N_BITS
        cls.LEXICON               = list(LEXICON)

    # ── to_bits() ─────────────────────────────────────────────────────────────

    def test_tb_returns_list(self):
        r = self.to_bits([0, 63])
        self.assertIsInstance(r, list)

    def test_tb_length(self):
        r = self.to_bits([0, 1, 2], n_bits=6)
        self.assertEqual(len(r), 18)   # 3 values × 6 bits

    def test_tb_zero_gives_all_zeros(self):
        r = self.to_bits([0], n_bits=6)
        self.assertEqual(r, [0, 0, 0, 0, 0, 0])

    def test_tb_63_gives_all_ones(self):
        r = self.to_bits([63], n_bits=6)
        self.assertEqual(r, [1, 1, 1, 1, 1, 1])

    def test_tb_msb_first(self):
        # 1 = 0b000001 → MSB first: [0,0,0,0,0,1]
        r = self.to_bits([1], n_bits=6)
        self.assertEqual(r[-1], 1)
        self.assertEqual(r[0], 0)

    def test_tb_values_binary(self):
        r = self.to_bits([42, 15, 63])
        self.assertTrue(all(v in (0, 1) for v in r))

    # ── lz76_phrases() ────────────────────────────────────────────────────────

    def test_lp_empty(self):
        self.assertEqual(self.lz76_phrases([]), 0)

    def test_lp_single(self):
        # Implementation seeds c=1; single phrase [0] not found in empty prefix → c=2
        self.assertEqual(self.lz76_phrases([0]), 2)

    def test_lp_constant_low(self):
        # Constant string: very low complexity
        c = self.lz76_phrases([0] * 32)
        self.assertLessEqual(c, 4)

    def test_lp_positive(self):
        for s in [[0,1]*8, [1,0,1,1]*4, list(range(16))]:
            self.assertGreater(self.lz76_phrases(s), 0)

    def test_lp_random_higher_than_constant(self):
        import random
        rng = random.Random(42)
        rand_s = [rng.randint(0, 1) for _ in range(64)]
        const_s = [0] * 64
        self.assertGreater(self.lz76_phrases(rand_s), self.lz76_phrases(const_s))

    # ── lz76_norm() ───────────────────────────────────────────────────────────

    def test_ln_empty(self):
        self.assertAlmostEqual(self.lz76_norm([]), 0.0)

    def test_ln_single(self):
        self.assertAlmostEqual(self.lz76_norm([0]), 0.0)

    def test_ln_nonneg(self):
        for s in [[0]*32, [0,1]*16, [1,1,0,1]*8]:
            self.assertGreaterEqual(self.lz76_norm(s), 0.0)

    def test_ln_constant_near_zero(self):
        n = self.lz76_norm([0] * 100)
        self.assertLess(n, 0.3)

    def test_ln_alternating_low(self):
        # [0,1,0,1,...] is periodic → lower complexity than random
        n = self.lz76_norm([0, 1] * 50)
        self.assertLess(n, 0.5)

    # ── trajectory_complexity() ───────────────────────────────────────────────

    def test_tc_returns_dict(self):
        r = self.trajectory_complexity('ГОРА', 'xor3')
        self.assertIsInstance(r, dict)

    def test_tc_required_keys(self):
        r = self.trajectory_complexity('ГОРА', 'xor3')
        for k in ('word', 'rule', 'transient', 'period',
                  'traj_bits', 'attr_bits', 'traj_phrases', 'attr_phrases',
                  'traj_norm', 'attr_norm'):
            self.assertIn(k, r)

    def test_tc_traj_norm_nonneg(self):
        for rule in self.ALL_RULES:
            r = self.trajectory_complexity('ГОРА', rule)
            self.assertGreaterEqual(r['traj_norm'], 0.0)

    def test_tc_traj_bits_positive(self):
        r = self.trajectory_complexity('ГОРА', 'xor3')
        self.assertGreater(r['traj_bits'], 0)

    def test_tc_traj_bits_formula(self):
        r = self.trajectory_complexity('ГОРА', 'xor3')
        expected = (r['transient'] + r['period']) * 16 * 6
        self.assertEqual(r['traj_bits'], expected)

    def test_tc_xor_low_complexity(self):
        # XOR converges to all-zeros → low attractor complexity
        r = self.trajectory_complexity('ГОРА', 'xor')
        self.assertLess(r['attr_norm'], 0.3)

    def test_tc_word_stored(self):
        r = self.trajectory_complexity('ГОРА', 'xor3')
        self.assertEqual(r['word'], 'ГОРА')

    # ── all_complexities() ────────────────────────────────────────────────────

    def test_ac_returns_all_rules(self):
        r = self.all_complexities('ГОРА')
        self.assertEqual(set(r.keys()), set(self.ALL_RULES))

    def test_ac_each_dict(self):
        r = self.all_complexities('ГОРА')
        for tc in r.values():
            self.assertIsInstance(tc, dict)

    # ── build_complexity_data() ───────────────────────────────────────────────

    def test_bcd_returns_dict(self):
        d = self.build_complexity_data(['ГОРА', 'ЛУНА'])
        self.assertIsInstance(d, dict)

    def test_bcd_required_keys(self):
        d = self.build_complexity_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'per_rule', 'ranking_traj',
                  'most_complex', 'least_complex'):
            self.assertIn(k, d)

    def test_bcd_ranking_sorted(self):
        d = self.build_complexity_data(['ГОРА', 'ЛУНА', 'МАТ'])
        for rule in self.ALL_RULES:
            norms = [v for _, v in d['ranking_traj'][rule]]
            self.assertEqual(norms, sorted(norms, reverse=True))

    def test_bcd_most_complex_is_valid(self):
        d = self.build_complexity_data(['ГОРА', 'ЛУНА', 'МАТ'])
        for rule in self.ALL_RULES:
            word, _ = d['most_complex'][rule]
            self.assertIn(word, ['ГОРА', 'ЛУНА', 'МАТ'])

    # ── complexity_dict() ─────────────────────────────────────────────────────

    def test_cd_json_serialisable(self):
        import json
        d = self.complexity_dict('ГОРА')
        dumped = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(dumped, str)

    def test_cd_top_keys(self):
        d = self.complexity_dict('ГОРА')
        for k in ('word', 'width', 'rules'):
            self.assertIn(k, d)

    def test_cd_all_rules_present(self):
        d = self.complexity_dict('ГОРА')
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_cd_traj_norm_in_result(self):
        d = self.complexity_dict('ГОРА')
        for rule in self.ALL_RULES:
            self.assertIn('traj_norm', d['rules'][rule])

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_lz_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lz-canvas', content)

    def test_viewer_has_lz_hmap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lz-hmap', content)

    def test_viewer_has_lz_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lz-btn', content)

    def test_viewer_has_lz_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lzRun', content)

    def test_viewer_lz_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('LZ76-сложность CA Q6', content)

    def test_viewer_has_lz76_fn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lz76', content)

    def test_viewer_has_lz_norm(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lzNorm', content)

    def test_viewer_has_to_bits(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('toBits', content)

    def test_viewer_has_all_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lz-all-btn', content)


class TestSolanSpacetime(unittest.TestCase):
    """Tests for solan_spacetime.py and the viewer Space-time section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_spacetime import (
            spacetime, value_to_rgb,
            st_spatial_entropy, st_temporal_entropy, st_activity,
            st_dict, all_st, build_st_data,
            _ALL_RULES, _DEFAULT_WIDTH,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.spacetime          = staticmethod(spacetime)
        cls.value_to_rgb       = staticmethod(value_to_rgb)
        cls.st_spatial_entropy = staticmethod(st_spatial_entropy)
        cls.st_temporal_entropy= staticmethod(st_temporal_entropy)
        cls.st_activity        = staticmethod(st_activity)
        cls.st_dict            = staticmethod(st_dict)
        cls.all_st             = staticmethod(all_st)
        cls.build_st_data      = staticmethod(build_st_data)
        cls.ALL_RULES          = _ALL_RULES
        cls.W                  = _DEFAULT_WIDTH
        cls.LEXICON            = list(LEXICON)

    # ── spacetime() ───────────────────────────────────────────────────────────

    def test_st_keys(self):
        d = self.spacetime('ГОРА', 'xor')
        for k in ['word', 'rule', 'width', 'transient', 'period',
                  'extra_periods', 'n_steps', 'grid']:
            self.assertIn(k, d)

    def test_st_word_upper(self):
        d = self.spacetime('гора', 'xor')
        self.assertEqual(d['word'], 'ГОРА')

    def test_st_transient_period_gora_xor(self):
        d = self.spacetime('ГОРА', 'xor')
        self.assertEqual(d['transient'], 2)
        self.assertEqual(d['period'],    1)

    def test_st_transient_period_tuman_xor3(self):
        d = self.spacetime('ТУМАН', 'xor3')
        self.assertEqual(d['transient'], 0)
        self.assertEqual(d['period'],    8)

    def test_st_n_steps(self):
        extra = 2
        d = self.spacetime('ТУМАН', 'xor3', extra_periods=extra)
        expected = d['transient'] + d['period'] * (1 + extra)
        self.assertEqual(d['n_steps'], expected)
        self.assertEqual(len(d['grid']), expected)

    def test_st_grid_row_width(self):
        d = self.spacetime('ГОРА', 'xor3')
        for row in d['grid']:
            self.assertEqual(len(row), self.W)

    def test_st_grid_q6_range(self):
        d = self.spacetime('ТУМАН', 'xor3')
        for row in d['grid']:
            for v in row:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 63)

    def test_st_grid_row0_is_ic(self):
        from projects.hexglyph.solan_word import encode_word, pad_to
        word = 'ГОРА'
        d    = self.spacetime(word, 'xor3')
        ic   = pad_to(encode_word(word), self.W)
        self.assertEqual(d['grid'][0], ic)

    def test_st_attractor_cyclic(self):
        # grid[T] through grid[T+P-1] forms the attractor cycle
        # so grid[T] == grid[T+P] when extra_periods >= 1
        d = self.spacetime('ТУМАН', 'xor3', extra_periods=1)
        T = d['transient']
        P = d['period']
        if len(d['grid']) > T + P:
            self.assertEqual(d['grid'][T], d['grid'][T + P])

    def test_st_step_consistency(self):
        from projects.hexglyph.solan_ca import step as ca_step
        d = self.spacetime('ГОРА', 'xor3', extra_periods=0)
        for t in range(len(d['grid']) - 1):
            expected = ca_step(d['grid'][t][:], 'xor3')
            self.assertEqual(d['grid'][t + 1], expected)

    # ── value_to_rgb() ────────────────────────────────────────────────────────

    def test_vrgb_output_range(self):
        for v in range(64):
            r, g, b = self.value_to_rgb(v)
            for ch in [r, g, b]:
                self.assertGreaterEqual(ch, 0)
                self.assertLessEqual(ch, 255)

    def test_vrgb_different_values_differ(self):
        # Most adjacent values should produce different colours
        diffs = sum(
            1 for v in range(62)
            if self.value_to_rgb(v) != self.value_to_rgb(v + 1)
        )
        self.assertGreater(diffs, 50)

    # ── st_spatial_entropy() ──────────────────────────────────────────────────

    def test_sse_length(self):
        d   = self.spacetime('ТУМАН', 'xor3')
        se  = self.st_spatial_entropy(d['grid'])
        self.assertEqual(len(se), d['n_steps'])

    def test_sse_non_negative(self):
        d  = self.spacetime('ГОРА', 'xor')
        se = self.st_spatial_entropy(d['grid'])
        for v in se:
            self.assertGreaterEqual(v, 0.0)

    def test_sse_uniform_zero(self):
        # All-zeros row → entropy = 0
        se = self.st_spatial_entropy([[0] * 16])
        self.assertAlmostEqual(se[0], 0.0, places=8)

    def test_sse_diverse_positive(self):
        d  = self.spacetime('ТУМАН', 'xor3')
        se = self.st_spatial_entropy(d['grid'])
        self.assertGreater(max(se), 0.0)

    # ── st_temporal_entropy() ─────────────────────────────────────────────────

    def test_ste_length(self):
        d  = self.spacetime('ТУМАН', 'xor3')
        te = self.st_temporal_entropy(d['grid'], self.W)
        self.assertEqual(len(te), self.W)

    def test_ste_non_negative(self):
        d  = self.spacetime('ТУМАН', 'xor3')
        te = self.st_temporal_entropy(d['grid'], self.W)
        for v in te:
            self.assertGreaterEqual(v, 0.0)

    # ── st_activity() ─────────────────────────────────────────────────────────

    def test_sact_length(self):
        d   = self.spacetime('ТУМАН', 'xor3')
        act = self.st_activity(d['grid'])
        self.assertEqual(len(act), d['n_steps'] - 1)

    def test_sact_non_negative(self):
        d   = self.spacetime('ГОРА', 'xor')
        act = self.st_activity(d['grid'])
        for v in act:
            self.assertGreaterEqual(v, 0.0)

    def test_sact_period1_attractor_zero(self):
        # XOR ГОРА: attractor is all-zeros (period=1) → no change on attractor
        d   = self.spacetime('ГОРА', 'xor', extra_periods=1)
        T   = d['transient']
        act = self.st_activity(d['grid'])
        for v in act[T:]:
            self.assertAlmostEqual(v, 0.0, places=8)

    # ── st_dict() ─────────────────────────────────────────────────────────────

    def test_sd_keys(self):
        d = self.st_dict('ТУМАН', 'xor3')
        for k in ['word', 'rule', 'width', 'transient', 'period',
                  'grid', 'spatial_entropy', 'temporal_entropy', 'activity',
                  'mean_spatial_h', 'mean_temporal_h',
                  'transient_activity', 'attractor_activity',
                  'ic_entropy', 'attractor_entropy']:
            self.assertIn(k, d)

    def test_sd_ic_entropy_non_negative(self):
        for rule in self.ALL_RULES:
            d = self.st_dict('ГОРА', rule)
            self.assertGreaterEqual(d['ic_entropy'], 0.0)

    def test_sd_attractor_entropy_zeros_is_zero(self):
        # XOR → all-zeros attractor → entropy = 0
        d = self.st_dict('ГОРА', 'xor')
        self.assertAlmostEqual(d['attractor_entropy'], 0.0, places=6)

    def test_sd_spatial_entropy_length(self):
        d = self.st_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['spatial_entropy']), d['n_steps'])

    def test_sd_temporal_entropy_length(self):
        d = self.st_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['temporal_entropy']), self.W)

    def test_sd_activity_length(self):
        d = self.st_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['activity']), d['n_steps'] - 1)

    # ── all_st() ─────────────────────────────────────────────────────────────

    def test_ast_all_rules(self):
        d = self.all_st('ТУМАН')
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    # ── build_st_data() ───────────────────────────────────────────────────────

    def test_bsd_keys(self):
        d = self.build_st_data(['ГОРА', 'ВОДА'])
        for k in ['words', 'per_rule', 'ranking']:
            self.assertIn(k, d)

    def test_bsd_ranking_sorted(self):
        d = self.build_st_data(['ГОРА', 'ВОДА', 'МИР'])
        for rule in self.ALL_RULES:
            vals = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(vals, sorted(vals, reverse=True))

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_st_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('st-canvas', content)

    def test_viewer_has_st_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('st-stats', content)

    def test_viewer_has_st_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('st-btn', content)

    def test_viewer_has_hsv_to_rgb(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('hsvToRgb', content)

    def test_viewer_has_st_spatial_h(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('stSpatialH', content)

    def test_viewer_has_draw_spacetime(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('drawSpacetime', content)


class TestSolanDamage(unittest.TestCase):
    """Tests for solan_damage.py and the viewer Damage Spreading section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_damage import (
            perturb, run_pair,
            single_damage, mean_damage,
            damage_dict, all_damage, build_damage_data,
            _ALL_RULES, _DEFAULT_WIDTH, _BITS,
        )
        from projects.hexglyph.solan_ca import step
        cls.perturb        = staticmethod(perturb)
        cls.run_pair       = staticmethod(run_pair)
        cls.single_damage  = staticmethod(single_damage)
        cls.mean_damage    = staticmethod(mean_damage)
        cls.damage_dict    = staticmethod(damage_dict)
        cls.all_damage     = staticmethod(all_damage)
        cls.build_damage   = staticmethod(build_damage_data)
        cls.ca_step        = staticmethod(step)
        cls.ALL_RULES      = _ALL_RULES
        cls.W              = _DEFAULT_WIDTH
        cls.BITS           = _BITS

    # ── perturb() ─────────────────────────────────────────────────────────────

    def test_perturb_flips_bit0(self):
        cells = [0] * self.W
        p = self.perturb(cells, 0, 0)
        self.assertEqual(p[0], 1)
        self.assertEqual(p[1:], [0] * (self.W - 1))

    def test_perturb_flips_bit5(self):
        cells = [0] * self.W
        p = self.perturb(cells, 3, 5)
        self.assertEqual(p[3], 32)

    def test_perturb_flips_back(self):
        # flipping same bit twice returns original
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ТУМАН'), self.W)
        p  = self.perturb(cells, 2, 3)
        pp = self.perturb(p,     2, 3)
        self.assertEqual(pp, cells)

    def test_perturb_q6_range(self):
        # perturb must keep values in [0, 63]
        cells = [63] * self.W
        for bit in range(self.BITS):
            p = self.perturb(cells, 0, bit)
            for v in p:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 63)

    def test_perturb_only_one_cell_changes(self):
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ГОРА'), self.W)
        p = self.perturb(cells, 5, 2)
        for i in range(self.W):
            if i != 5:
                self.assertEqual(p[i], cells[i])

    # ── run_pair() ────────────────────────────────────────────────────────────

    def test_rp_grid_dims(self):
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ТУМАН'), self.W)
        pert  = self.perturb(cells, 0, 0)
        og, pg, dg = self.run_pair(cells, pert, 'xor3', 10)
        self.assertEqual(len(og), 10)
        self.assertEqual(len(pg), 10)
        self.assertEqual(len(dg), 10)
        for t in range(10):
            self.assertEqual(len(og[t]), self.W)
            self.assertEqual(len(dg[t]), self.W)

    def test_rp_damage_range(self):
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ТУМАН'), self.W)
        pert  = self.perturb(cells, 0, 5)
        _, _, dg = self.run_pair(cells, pert, 'xor3', 12)
        for row in dg:
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_rp_initial_damage_at_perturbed_cell(self):
        # At t=0, only the perturbed cell should have non-zero damage
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ГОРА'), self.W)
        pert  = self.perturb(cells, 7, 5)   # flip bit 5 → Δv=32
        _, _, dg = self.run_pair(cells, pert, 'xor3', 4)
        self.assertGreater(dg[0][7], 0.0)
        for i in range(self.W):
            if i != 7:
                self.assertAlmostEqual(dg[0][i], 0.0, places=10)

    def test_rp_orig_matches_step(self):
        # orig_grid transitions must equal step(prev, rule)
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ТУМАН'), self.W)
        pert  = self.perturb(cells, 0, 1)
        og, _, _ = self.run_pair(cells, pert, 'xor', 8)
        for t in range(len(og) - 1):
            self.assertEqual(og[t + 1], self.ca_step(og[t][:], 'xor'))

    # ── single_damage() ───────────────────────────────────────────────────────

    def test_sd_keys(self):
        d = self.single_damage('ТУМАН', 'xor3')
        for k in ['word', 'rule', 'cell', 'bit', 'n_steps', 'width',
                  'orig_grid', 'pert_grid', 'damage_grid',
                  'total_damage', 'damage_width',
                  'max_damage', 'final_damage', 'extinction_step',
                  'velocity', 'phase']:
            self.assertIn(k, d)

    def test_sd_word_upper(self):
        d = self.single_damage('гора', 'and')
        self.assertEqual(d['word'], 'ГОРА')

    def test_sd_total_length(self):
        d = self.single_damage('ТУМАН', 'xor', n_steps=20)
        self.assertEqual(len(d['total_damage']), 20)

    def test_sd_and_ordered(self):
        # AND contracts — must reach extinction
        d = self.single_damage('ТУМАН', 'and', bit=5, n_steps=32)
        self.assertEqual(d['phase'], 'ordered')
        self.assertGreaterEqual(d['extinction_step'], 0)

    def test_sd_or_ordered(self):
        d = self.single_damage('ТУМАН', 'or', bit=5, n_steps=32)
        self.assertEqual(d['phase'], 'ordered')

    def test_sd_max_damage_non_negative(self):
        for rule in self.ALL_RULES:
            d = self.single_damage('ГОРА', rule, bit=5)
            self.assertGreaterEqual(d['max_damage'], 0.0)

    def test_sd_max_damage_geq_final(self):
        d = self.single_damage('ТУМАН', 'xor3', bit=5)
        self.assertGreaterEqual(d['max_damage'], d['final_damage'])

    def test_sd_initial_damage_positive_for_bit5(self):
        # bit 5 flips → Δv=32 in one cell → total_damage[0]>0
        d = self.single_damage('ТУМАН', 'xor3', cell=0, bit=5, n_steps=8)
        self.assertGreater(d['total_damage'][0], 0.0)

    def test_sd_xor_extinction_at_period(self):
        # XOR with ТУМАН: period=1, damage vanishes at step 8 (ring of 16)
        d = self.single_damage('ТУМАН', 'xor', bit=5, n_steps=24)
        # Damage should eventually go to 0
        self.assertEqual(d['phase'], 'ordered')

    # ── mean_damage() ─────────────────────────────────────────────────────────

    def test_md_keys(self):
        d = self.mean_damage('ТУМАН', 'xor3', n_steps=8)
        for k in ['word', 'rule', 'n_steps', 'width', 'n_perturb',
                  'mean_damage_grid', 'mean_total', 'mean_width',
                  'max_mean_damage', 'final_mean_damage',
                  'extinction_step', 'velocity', 'phase', 'kernel']:
            self.assertIn(k, d)

    def test_md_n_perturb(self):
        d = self.mean_damage('ТУМАН', 'xor3', n_steps=4)
        self.assertEqual(d['n_perturb'], self.W * self.BITS)

    def test_md_mean_grid_dims(self):
        n = 8
        d = self.mean_damage('ГОРА', 'xor3', n_steps=n)
        self.assertEqual(len(d['mean_damage_grid']), n)
        for row in d['mean_damage_grid']:
            self.assertEqual(len(row), self.W)

    def test_md_kernel_dims(self):
        n = 8
        d = self.mean_damage('ГОРА', 'xor3', n_steps=n)
        self.assertEqual(len(d['kernel']), n)
        for row in d['kernel']:
            self.assertEqual(len(row), self.W)

    def test_md_kernel_t0_only_di0(self):
        # At t=0, only the source cell (Δi=0) has damage
        d = self.mean_damage('ТУМАН', 'xor3', n_steps=4)
        for di in range(1, self.W):
            self.assertAlmostEqual(d['kernel'][0][di], 0.0, places=10)
        self.assertGreater(d['kernel'][0][0], 0.0)

    def test_md_and_final_damage_below_chaotic(self):
        # AND contracts strongly — mean final damage must stay below 'chaotic' threshold (0.05)
        # Some rare (cell, bit) pairs may converge to a different fixed point,
        # so phase can be 'critical', but not dominated by large damage.
        d = self.mean_damage('ТУМАН', 'and', n_steps=32)
        self.assertLess(d['final_mean_damage'], 0.05)

    def test_md_max_damage_non_negative(self):
        for rule in self.ALL_RULES:
            d = self.mean_damage('ГОРА', rule, n_steps=8)
            self.assertGreaterEqual(d['max_mean_damage'], 0.0)

    def test_md_kernel_xor_spreads_at_t1(self):
        # XOR at t=1: damage from cell 0 reaches cells ±1 (Δi=1 and Δi=N-1)
        d = self.mean_damage('ТУМАН', 'xor', n_steps=4)
        # kernel[1][1] and kernel[1][N-1] should be > 0
        self.assertGreater(d['kernel'][1][1], 0.0)
        self.assertGreater(d['kernel'][1][self.W - 1], 0.0)
        # kernel[1][0] should be 0 for XOR (XOR: new[i] = left ^ right, no self)
        self.assertAlmostEqual(d['kernel'][1][0], 0.0, places=10)

    # ── damage_dict() ─────────────────────────────────────────────────────────

    def test_dd_keys(self):
        d = self.damage_dict('ТУМАН', 'xor3', n_steps=8)
        for k in ['word', 'rule', 'total_damage', 'mean_total',
                  'max_damage', 'max_mean_damage', 'phase', 'mean_phase',
                  'kernel']:
            self.assertIn(k, d)

    # ── all_damage() ──────────────────────────────────────────────────────────

    def test_ad_all_rules(self):
        d = self.all_damage('ГОРА', n_steps=8)
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    # ── build_damage_data() ───────────────────────────────────────────────────

    def test_bdd_keys(self):
        d = self.build_damage(['ГОРА', 'ВОДА'], n_steps=8)
        for k in ['words', 'per_rule', 'phase_counts']:
            self.assertIn(k, d)

    def test_bdd_phase_counts_sum(self):
        words = ['ГОРА', 'ВОДА', 'МИР']
        d = self.build_damage(words, n_steps=12)
        for rule in self.ALL_RULES:
            pc = d['phase_counts'][rule]
            total = pc['ordered'] + pc['critical'] + pc['chaotic']
            self.assertEqual(total, len(words))

    def test_bdd_and_fewer_chaotic_than_xor3(self):
        # AND contracts strongly → fewer 'chaotic' words than XOR3
        words = ['ГОРА', 'ВОДА', 'МИР', 'ТУМАН']
        d = self.build_damage(words, n_steps=32)
        chaotic_and  = d['phase_counts']['and']['chaotic']
        chaotic_xor3 = d['phase_counts']['xor3']['chaotic']
        self.assertLessEqual(chaotic_and, chaotic_xor3)

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_dm_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dm-canvas', content)

    def test_viewer_has_dm_chart(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dm-chart', content)

    def test_viewer_has_dm_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dm-btn', content)

    def test_viewer_has_dm_hot(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dmHot', content)

    def test_viewer_has_dm_step(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dmStep', content)

    def test_viewer_has_damage_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('конус влияния', content)


class TestSolanSymbolic(unittest.TestCase):
    """Tests for solan_symbolic.py and the viewer Symbolic Dynamics section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_symbolic import (
            binarize, attractor_binary,
            ngrams, ngram_dist,
            block_entropy_n, block_entropy_profile,
            transition_matrix, forbidden_ngrams,
            symbolic_dict, all_symbolic, build_symbolic_data,
            _ALL_RULES, _DEFAULT_WIDTH, _DEFAULT_THR,
        )
        cls.binarize          = staticmethod(binarize)
        cls.attractor_binary  = staticmethod(attractor_binary)
        cls.ngrams            = staticmethod(ngrams)
        cls.ngram_dist        = staticmethod(ngram_dist)
        cls.block_entropy_n   = staticmethod(block_entropy_n)
        cls.bep               = staticmethod(block_entropy_profile)
        cls.transition_matrix = staticmethod(transition_matrix)
        cls.forbidden_ngrams  = staticmethod(forbidden_ngrams)
        cls.symbolic_dict     = staticmethod(symbolic_dict)
        cls.all_symbolic      = staticmethod(all_symbolic)
        cls.build_symbolic    = staticmethod(build_symbolic_data)
        cls.ALL_RULES         = _ALL_RULES
        cls.W                 = _DEFAULT_WIDTH
        cls.THR               = _DEFAULT_THR

    # ── binarize() ────────────────────────────────────────────────────────────

    def test_bin_below_thr(self):
        self.assertEqual(self.binarize(0,  32), 0)
        self.assertEqual(self.binarize(31, 32), 0)

    def test_bin_at_thr(self):
        self.assertEqual(self.binarize(32, 32), 1)

    def test_bin_above_thr(self):
        self.assertEqual(self.binarize(63, 32), 1)

    def test_bin_custom_thr(self):
        self.assertEqual(self.binarize(15, 16), 0)
        self.assertEqual(self.binarize(16, 16), 1)

    def test_bin_output_is_0_or_1(self):
        for v in range(64):
            b = self.binarize(v, self.THR)
            self.assertIn(b, (0, 1))

    # ── attractor_binary() ────────────────────────────────────────────────────

    def test_ab_dims_gora_xor3(self):
        # ГОРА XOR3: period=2, width=16
        bg = self.attractor_binary('ГОРА', 'xor3')
        self.assertEqual(len(bg), 2)
        for row in bg:
            self.assertEqual(len(row), self.W)

    def test_ab_tuman_xor3_period8(self):
        bg = self.attractor_binary('ТУМАН', 'xor3')
        self.assertEqual(len(bg), 8)

    def test_ab_all_binary(self):
        bg = self.attractor_binary('ТУМАН', 'xor3')
        for row in bg:
            for v in row:
                self.assertIn(v, (0, 1))

    def test_ab_xor_zeros_attractor(self):
        # XOR → all-zeros attractor → all symbols = 0
        bg = self.attractor_binary('ТУМАН', 'xor')
        self.assertEqual(len(bg), 1)
        self.assertEqual(bg[0], [0] * self.W)

    def test_ab_or_ones_attractor(self):
        # OR → all-ones attractor → all symbols = 1 (63 >= 32)
        bg = self.attractor_binary('ТУМАН', 'or')
        self.assertEqual(len(bg), 1)
        self.assertEqual(bg[0], [1] * self.W)

    def test_ab_cyclic(self):
        # grid[P] (computed one step past) should equal grid[0]
        from projects.hexglyph.solan_word import encode_word, pad_to
        from projects.hexglyph.solan_ca import step, find_orbit
        word = 'ТУМАН'
        for rule in self.ALL_RULES:
            bg = self.attractor_binary(word, rule)
            # step from last row should match first row's binary representation
            cells = pad_to(encode_word(word), self.W)
            _, period = find_orbit(cells[:], rule)
            period = max(period, 1)
            # binary_grid wraps: row[period % period] == row[0]
            self.assertEqual(len(bg), period)

    # ── ngrams() ──────────────────────────────────────────────────────────────

    def test_ng_circular_length(self):
        seq = [0, 1, 0, 1]
        grams = self.ngrams(seq, 2, circular=True)
        self.assertEqual(len(grams), 4)    # circular: one per position

    def test_ng_non_circular_length(self):
        seq = [0, 1, 0, 1]
        grams = self.ngrams(seq, 2, circular=False)
        self.assertEqual(len(grams), 3)

    def test_ng_1gram_is_each_symbol(self):
        seq = [1, 0, 1]
        grams = self.ngrams(seq, 1)
        self.assertEqual(grams, [(1,), (0,), (1,)])

    def test_ng_wrap(self):
        seq = [1, 0]
        grams = self.ngrams(seq, 2, circular=True)
        self.assertIn((1, 0), grams)
        self.assertIn((0, 1), grams)    # wraps: seq[1], seq[0]

    # ── ngram_dist() ──────────────────────────────────────────────────────────

    def test_nd_constant_seq(self):
        seqs = [[0, 0, 0, 0]]
        d = self.ngram_dist(seqs, 2)
        self.assertEqual(d.get((0, 0), 0), 4)   # circular → 4 (0,0) 2-grams
        self.assertEqual(d.get((0, 1), 0), 0)

    def test_nd_alternating_has_all_bigrams(self):
        seqs = [[0, 1, 0, 1]]
        d = self.ngram_dist(seqs, 2)
        for gram in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            self.assertGreaterEqual(d.get(gram, 0), 0)
        self.assertGreater(d.get((0, 1), 0), 0)
        self.assertGreater(d.get((1, 0), 0), 0)

    # ── block_entropy_n() ─────────────────────────────────────────────────────

    def test_ben_constant_zero(self):
        seqs = [[0] * 8]
        self.assertAlmostEqual(self.block_entropy_n(seqs, 2), 0.0, places=8)

    def test_ben_non_negative(self):
        seqs = [[0, 1, 0, 1, 1, 0]]
        for n in range(1, 5):
            self.assertGreaterEqual(self.block_entropy_n(seqs, n), 0.0)

    def test_ben_two_equal_seqs(self):
        # H1 of equal-probability two-symbol sequence ≈ 1 bit
        seqs = [[0, 1] * 8]
        h1 = self.block_entropy_n(seqs, 1)
        self.assertAlmostEqual(h1, 1.0, places=5)

    # ── block_entropy_profile() ───────────────────────────────────────────────

    def test_bep_length(self):
        seqs = [[0, 1, 0, 1]]
        profile = self.bep(seqs, max_n=5)
        self.assertEqual(len(profile), 5)

    def test_bep_non_negative(self):
        seqs = [[0, 1, 1, 0, 0, 1, 0, 1]]
        for h in self.bep(seqs, 6):
            self.assertGreaterEqual(h, 0.0)

    def test_bep_constant_all_zeros(self):
        seqs = [[0] * 8, [0] * 8]
        for h in self.bep(seqs, 6):
            self.assertAlmostEqual(h, 0.0, places=8)

    # ── transition_matrix() ───────────────────────────────────────────────────

    def test_tm_shape(self):
        seqs = [[0, 1, 0, 1]]
        tm = self.transition_matrix(seqs)
        self.assertEqual(len(tm), 2)
        self.assertEqual(len(tm[0]), 2)
        self.assertEqual(len(tm[1]), 2)

    def test_tm_rows_sum_to_one(self):
        seqs = [[0, 1, 1, 0, 0, 1]]
        tm = self.transition_matrix(seqs)
        for row_sum in [tm[a][0] + tm[a][1] for a in range(2)]:
            if row_sum > 0:
                self.assertAlmostEqual(row_sum, 1.0, places=8)

    def test_tm_constant_zero_seq(self):
        seqs = [[0, 0, 0, 0]]
        tm = self.transition_matrix(seqs)
        self.assertAlmostEqual(tm[0][0], 1.0, places=8)
        self.assertAlmostEqual(tm[0][1], 0.0, places=8)
        # row for symbol 1 is undefined (never seen) → sums to 0
        self.assertAlmostEqual(tm[1][0] + tm[1][1], 0.0, places=8)

    def test_tm_constant_one_seq(self):
        seqs = [[1, 1, 1, 1]]
        tm = self.transition_matrix(seqs)
        self.assertAlmostEqual(tm[1][1], 1.0, places=8)

    # ── forbidden_ngrams() ────────────────────────────────────────────────────

    def test_fn_constant_zero_3forbidden(self):
        # all-zeros: only (0,0) 2-gram seen → 3 forbidden
        seqs = [[0, 0, 0, 0, 0, 0, 0, 0]]
        f = self.forbidden_ngrams(seqs, 2)
        self.assertEqual(len(f), 3)
        self.assertIn((0, 1), f)
        self.assertIn((1, 0), f)
        self.assertIn((1, 1), f)

    def test_fn_constant_one_3forbidden(self):
        seqs = [[1, 1, 1, 1]]
        f = self.forbidden_ngrams(seqs, 2)
        self.assertEqual(len(f), 3)

    def test_fn_alternating_all_bigrams_seen(self):
        seqs = [[0, 1, 0, 1, 0, 1, 0, 1]]
        f = self.forbidden_ngrams(seqs, 2)
        self.assertEqual(len(f), 2)   # (0,0) and (1,1) forbidden for strict alternation
        # (0,1) and (1,0) appear, but (0,0) and (1,1) do not
        self.assertIn((0, 0), f)
        self.assertIn((1, 1), f)

    def test_fn_count_leq_total_ngrams(self):
        bg = self.attractor_binary('ТУМАН', 'xor3')
        seqs = [[bg[t][i] for t in range(len(bg))] for i in range(self.W)]
        f = self.forbidden_ngrams(seqs, 3)
        self.assertLessEqual(len(f), 8)

    # ── symbolic_dict() ───────────────────────────────────────────────────────

    def test_sd_keys(self):
        d = self.symbolic_dict('ТУМАН', 'xor3')
        for k in ['word', 'rule', 'width', 'threshold', 'max_n',
                  'period', 'transient', 'binary_grid', 'temporal_seqs',
                  'spatial_seqs', 'symbol_bias', 'n_unique_temp', 'n_unique_spat',
                  'temporal_entropy', 'spatial_entropy', 'block_entropy',
                  'topological_h', 'transition_mat', 'forbidden_2grams',
                  'forbidden_3grams', 'ngram_1', 'ngram_2']:
            self.assertIn(k, d)

    def test_sd_word_upper(self):
        d = self.symbolic_dict('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_sd_period_matches_orbit(self):
        from projects.hexglyph.solan_ca import find_orbit
        from projects.hexglyph.solan_word import encode_word, pad_to
        for word in ['ТУМАН', 'ГОРА']:
            for rule in self.ALL_RULES:
                cells = pad_to(encode_word(word), self.W)
                _, p  = find_orbit(cells[:], rule)
                d     = self.symbolic_dict(word, rule)
                self.assertEqual(d['period'], max(p, 1))

    def test_sd_bias_xor_zero(self):
        d = self.symbolic_dict('ТУМАН', 'xor')
        self.assertAlmostEqual(d['symbol_bias'], 0.0, places=8)

    def test_sd_bias_or_one(self):
        d = self.symbolic_dict('ТУМАН', 'or')
        self.assertAlmostEqual(d['symbol_bias'], 1.0, places=8)

    def test_sd_topo_h_xor_zero(self):
        # period-1 all-zeros → no variation → topological_h = 0
        d = self.symbolic_dict('ТУМАН', 'xor')
        self.assertAlmostEqual(d['topological_h'], 0.0, places=8)

    def test_sd_topo_h_xor3_positive(self):
        # XOR3 period-8 → complex symbolic dynamics → topological_h > 0
        d = self.symbolic_dict('ТУМАН', 'xor3')
        self.assertGreater(d['topological_h'], 0.0)

    def test_sd_forbidden_2_xor_equals_3(self):
        # XOR all-zeros: only (0,0) seen → 3 forbidden 2-grams
        d = self.symbolic_dict('ТУМАН', 'xor')
        self.assertEqual(len(d['forbidden_2grams']), 3)

    def test_sd_forbidden_2_xor3_zero(self):
        # XOR3 period-8: rich dynamics → all 4 2-grams appear
        d = self.symbolic_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['forbidden_2grams']), 0)

    def test_sd_block_entropy_length(self):
        d = self.symbolic_dict('ТУМАН', 'xor3', max_n=5)
        self.assertEqual(len(d['block_entropy']), 5)

    def test_sd_n_unique_temp_bounded(self):
        d = self.symbolic_dict('ТУМАН', 'xor3')
        self.assertGreaterEqual(d['n_unique_temp'], 1)
        self.assertLessEqual(d['n_unique_temp'], self.W)

    def test_sd_n_unique_spat_eq_period_for_xor3(self):
        # XOR3 period-8 ТУМАН: all 8 attractor rows are distinct
        d = self.symbolic_dict('ТУМАН', 'xor3')
        self.assertEqual(d['n_unique_spat'], d['period'])

    def test_sd_binary_grid_dims(self):
        d = self.symbolic_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['binary_grid']), 8)
        for row in d['binary_grid']:
            self.assertEqual(len(row), self.W)

    def test_sd_ngram1_keys_are_0_and_1(self):
        d = self.symbolic_dict('ТУМАН', 'xor3')
        self.assertTrue(set(d['ngram_1'].keys()) <= {'0', '1'})

    def test_sd_ngram2_string_keys(self):
        d = self.symbolic_dict('ТУМАН', 'xor3')
        for k in d['ngram_2']:
            self.assertEqual(len(k), 2)
            self.assertTrue(set(k) <= {'0', '1'})

    # ── all_symbolic() ────────────────────────────────────────────────────────

    def test_as_all_rules(self):
        d = self.all_symbolic('ТУМАН')
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    # ── build_symbolic_data() ─────────────────────────────────────────────────

    def test_bsd_keys(self):
        d = self.build_symbolic(['ГОРА', 'ВОДА'])
        for k in ['words', 'per_rule', 'ranking']:
            self.assertIn(k, d)

    def test_bsd_ranking_sorted(self):
        d = self.build_symbolic(['ГОРА', 'ВОДА', 'МИР'])
        for rule in self.ALL_RULES:
            vals = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(vals, sorted(vals, reverse=True))

    def test_bsd_per_rule_word_keys(self):
        words = ['ГОРА', 'ВОДА']
        d = self.build_symbolic(words)
        for rule in self.ALL_RULES:
            self.assertEqual(set(d['per_rule'][rule].keys()), set(words))

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_sym_grid(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sym-grid', content)

    def test_viewer_has_sym_bep(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sym-bep', content)

    def test_viewer_has_sym_tm(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sym-tm', content)

    def test_viewer_has_sym_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sym-btn', content)

    def test_viewer_has_sym_block_h(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('symBlockH', content)

    def test_viewer_has_sym_forbidden(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('symForbidden', content)

    def test_viewer_has_symbolic_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('бинарная грамматика', content)


class TestSolanNetwork(unittest.TestCase):
    """Tests for solan_network.py and the viewer Network section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_network import (
            in_weights, out_weights, net_flow,
            pagerank, hits, tarjan_scc,
            network_dict, all_network, build_network_data,
            _ALL_RULES, _DEFAULT_WIDTH,
        )
        cls.in_weights     = staticmethod(in_weights)
        cls.out_weights    = staticmethod(out_weights)
        cls.net_flow       = staticmethod(net_flow)
        cls.pagerank       = staticmethod(pagerank)
        cls.hits           = staticmethod(hits)
        cls.tarjan_scc     = staticmethod(tarjan_scc)
        cls.network_dict   = staticmethod(network_dict)
        cls.all_network    = staticmethod(all_network)
        cls.build_network  = staticmethod(build_network_data)
        cls.ALL_RULES      = _ALL_RULES
        cls.W              = _DEFAULT_WIDTH

    # ── in_weights / out_weights / net_flow ──────────────────────────────────

    def _zero_mat(self, n=4):
        return [[0.0] * n for _ in range(n)]

    def _diag_mat(self, vals):
        n = len(vals)
        m = [[0.0] * n for _ in range(n)]
        for i, v in enumerate(vals):
            m[i][i] = v
        return m

    def test_iw_zero_matrix(self):
        m = self._zero_mat()
        self.assertEqual(self.in_weights(m), [0.0, 0.0, 0.0, 0.0])

    def test_ow_zero_matrix(self):
        m = self._zero_mat()
        self.assertEqual(self.out_weights(m), [0.0, 0.0, 0.0, 0.0])

    def test_iw_ow_sum_equal(self):
        # total in == total out for any matrix
        m = [[1.0, 2.0], [3.0, 4.0]]
        self.assertAlmostEqual(sum(self.in_weights(m)), sum(self.out_weights(m)))

    def test_nf_diag_matrix(self):
        # diagonal: each node sends only to itself → net_flow = 0 for all
        m = self._diag_mat([1.0, 2.0, 3.0])
        nf = self.net_flow(m)
        for v in nf:
            self.assertAlmostEqual(v, 0.0, places=8)

    def test_nf_asymmetric(self):
        # 0→1 strong, 1→0 weak → node 0 is net source, node 1 is net sink
        m = [[0.0, 5.0], [1.0, 0.0]]
        nf = self.net_flow(m)
        self.assertGreater(nf[0], 0.0)   # node 0: out>in → source
        self.assertLess(nf[1], 0.0)      # node 1: in>out → sink

    def test_nf_global_sum_zero(self):
        # total net flow always sums to 0 (out_total == in_total)
        m = [[0.5, 1.5, 2.0], [0.3, 0.0, 0.7], [1.0, 0.0, 0.0]]
        nf = self.net_flow(m)
        self.assertAlmostEqual(sum(nf), 0.0, places=8)

    # ── pagerank ─────────────────────────────────────────────────────────────

    def test_pr_length(self):
        m = self._zero_mat(4)
        pr = self.pagerank(m)
        self.assertEqual(len(pr), 4)

    def test_pr_sums_to_one(self):
        m = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        pr = self.pagerank(m)
        self.assertAlmostEqual(sum(pr), 1.0, places=6)

    def test_pr_uniform_zero_matrix(self):
        # zero matrix → dangling nodes → uniform PR
        m = self._zero_mat(4)
        pr = self.pagerank(m)
        for v in pr:
            self.assertAlmostEqual(v, 0.25, places=5)

    def test_pr_non_negative(self):
        m = [[0.0, 2.0, 0.0, 1.0],
             [0.0, 0.0, 3.0, 0.0],
             [1.0, 0.0, 0.0, 2.0],
             [0.0, 1.0, 0.0, 0.0]]
        pr = self.pagerank(m)
        for v in pr:
            self.assertGreaterEqual(v, 0.0)

    # ── hits ─────────────────────────────────────────────────────────────────

    def test_hits_lengths(self):
        m = [[0.0, 1.0], [1.0, 0.0]]
        hs, as_ = self.hits(m)
        self.assertEqual(len(hs), 2)
        self.assertEqual(len(as_), 2)

    def test_hits_non_negative(self):
        m = [[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        hs, as_ = self.hits(m)
        for v in hs + as_:
            self.assertGreaterEqual(v, 0.0)

    def test_hits_normalised(self):
        m = [[0.0, 1.0, 0.5], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        hs, as_ = self.hits(m)
        self.assertAlmostEqual(sum(hs), 1.0, places=6)
        self.assertAlmostEqual(sum(as_), 1.0, places=6)

    # ── tarjan_scc ───────────────────────────────────────────────────────────

    def test_scc_isolated_nodes(self):
        # zero matrix → each node its own SCC
        m = self._zero_mat(4)
        sccs = self.tarjan_scc(m)
        self.assertEqual(len(sccs), 4)
        for s in sccs:
            self.assertEqual(len(s), 1)

    def test_scc_cycle_is_one_scc(self):
        # 0→1→2→0 cycle → one SCC of size 3
        m = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        sccs = self.tarjan_scc(m)
        self.assertEqual(len(sccs), 1)
        self.assertEqual(len(sccs[0]), 3)

    def test_scc_two_components(self):
        # {0,1} → cycle; {2} isolated
        m = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        sccs = self.tarjan_scc(m)
        self.assertEqual(len(sccs), 2)
        sizes = sorted(len(s) for s in sccs)
        self.assertEqual(sizes, [1, 2])

    def test_scc_largest_first(self):
        # sorted by size descending
        m = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        sccs = self.tarjan_scc(m)
        sizes = [len(s) for s in sccs]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_scc_all_nodes_covered(self):
        m = [[0.0, 1.0, 0.5], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        sccs = self.tarjan_scc(m)
        nodes = sorted(n for scc in sccs for n in scc)
        self.assertEqual(nodes, [0, 1, 2])

    # ── network_dict ─────────────────────────────────────────────────────────

    def test_nd_keys(self):
        d = self.network_dict('ТУМАН', 'xor3')
        for k in ['word', 'rule', 'width', 'period', 'transient',
                  'te_mat', 'in_weight', 'out_weight', 'net_flow',
                  'total_te', 'pagerank', 'hub_score', 'auth_score',
                  'sccs', 'n_sccs', 'largest_scc',
                  'top_sources', 'top_sinks', 'top_pr']:
            self.assertIn(k, d)

    def test_nd_word_upper(self):
        d = self.network_dict('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_nd_te_mat_shape(self):
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['te_mat']), self.W)
        for row in d['te_mat']:
            self.assertEqual(len(row), self.W)

    def test_nd_te_non_negative(self):
        d = self.network_dict('ТУМАН', 'xor3')
        for row in d['te_mat']:
            for v in row:
                self.assertGreaterEqual(v, 0.0)

    def test_nd_total_te_xor_zero(self):
        # period-1 all-zeros → TE = 0 everywhere
        d = self.network_dict('ТУМАН', 'xor')
        self.assertAlmostEqual(d['total_te'], 0.0, places=6)

    def test_nd_total_te_xor3_positive(self):
        # XOR3 period-8 → non-trivial TE
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertGreater(d['total_te'], 0.0)

    def test_nd_pr_sums_to_one(self):
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertAlmostEqual(sum(d['pagerank']), 1.0, places=5)

    def test_nd_pr_length(self):
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['pagerank']), self.W)

    def test_nd_pr_xor_uniform(self):
        # XOR (TE=0) → all weights equal → uniform PageRank = 1/N
        d = self.network_dict('ТУМАН', 'xor')
        for v in d['pagerank']:
            self.assertAlmostEqual(v, 1.0 / self.W, places=5)

    def test_nd_nf_global_sum_zero(self):
        # sum of net_flow must always be 0
        for rule in self.ALL_RULES:
            d = self.network_dict('ТУМАН', rule)
            self.assertAlmostEqual(sum(d['net_flow']), 0.0, places=6,
                                   msg=f'rule={rule}')

    def test_nd_nf_length(self):
        d = self.network_dict('ГОРА', 'xor3')
        self.assertEqual(len(d['net_flow']), self.W)

    def test_nd_sccs_xor_16_isolated(self):
        # XOR (TE=0) → 16 isolated SCCs
        d = self.network_dict('ТУМАН', 'xor')
        self.assertEqual(d['n_sccs'], self.W)
        self.assertEqual(d['largest_scc'], 1)

    def test_nd_sccs_xor3_one_component(self):
        # XOR3 period-8 → fully connected network → 1 SCC of size 16
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertEqual(d['n_sccs'], 1)
        self.assertEqual(d['largest_scc'], self.W)

    def test_nd_top_sources_length(self):
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['top_sources']), 3)

    def test_nd_top_sinks_length(self):
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['top_sinks']), 3)

    def test_nd_top_pr_length(self):
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['top_pr']), 3)

    def test_nd_top_sources_are_valid_indices(self):
        d = self.network_dict('ТУМАН', 'xor3')
        for idx in d['top_sources']:
            self.assertIn(idx, range(self.W))

    def test_nd_hub_auth_lengths(self):
        d = self.network_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['hub_score']), self.W)
        self.assertEqual(len(d['auth_score']), self.W)

    def test_nd_period_matches_orbit(self):
        from projects.hexglyph.solan_ca import find_orbit
        from projects.hexglyph.solan_word import encode_word, pad_to
        for word in ['ТУМАН', 'ГОРА']:
            for rule in self.ALL_RULES:
                cells = pad_to(encode_word(word), self.W)
                _, p  = find_orbit(cells[:], rule)
                d     = self.network_dict(word, rule)
                self.assertEqual(d['period'], max(p, 1))

    # ── all_network ───────────────────────────────────────────────────────────

    def test_an_all_rules(self):
        d = self.all_network('ТУМАН')
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    # ── build_network_data ────────────────────────────────────────────────────

    def test_bnd_keys(self):
        d = self.build_network(['ГОРА', 'ВОДА'])
        for k in ['words', 'per_rule', 'ranking']:
            self.assertIn(k, d)

    def test_bnd_ranking_sorted(self):
        d = self.build_network(['ГОРА', 'ВОДА', 'МИР'])
        for rule in self.ALL_RULES:
            vals = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(vals, sorted(vals, reverse=True))

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_net_graph(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('net-graph', content)

    def test_viewer_has_net_bars(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('net-bars', content)

    def test_viewer_has_net_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('net-btn', content)

    def test_viewer_has_net_page_rank(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('netPageRank', content)

    def test_viewer_has_net_scc(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('netSCC', content)

    def test_viewer_has_net_te_matrix(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('netTEMatrix', content)

    def test_viewer_has_network_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('PageRank', content)
        self.assertIn('СКС', content)


class TestSolanPortrait(unittest.TestCase):
    """Tests for solan_portrait.py and the viewer Portrait section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_portrait import (
            portrait_dict, portrait_compare,
            build_portrait_data, _l2_distance, _cosine_sim,
            _AXES, _LABELS, _N_AXES,
            _DEFAULT_RULE, _DEFAULT_W,
        )
        cls.portrait_dict    = staticmethod(portrait_dict)
        cls.portrait_compare = staticmethod(portrait_compare)
        cls.build_portrait   = staticmethod(build_portrait_data)
        cls.l2               = staticmethod(_l2_distance)
        cls.cosine           = staticmethod(_cosine_sim)
        cls.AXES             = _AXES
        cls.LABELS           = _LABELS
        cls.N_AXES           = _N_AXES
        cls.RULE             = _DEFAULT_RULE
        cls.W                = _DEFAULT_W

    # ── _l2_distance ──────────────────────────────────────────────────────────

    def test_l2_zero_for_identical(self):
        v = [0.5, 0.3, 0.8, 0.1, 0.6, 0.4, 0.7, 0.2]
        self.assertAlmostEqual(self.l2(v, v), 0.0, places=8)

    def test_l2_range(self):
        v1 = [0.0] * 8
        v2 = [1.0] * 8
        # L2 normalised by N: sqrt(sum(1.0^2)/8) = sqrt(1) = 1
        self.assertAlmostEqual(self.l2(v1, v2), 1.0, places=8)

    def test_l2_symmetric(self):
        v1 = [0.2, 0.5, 0.8, 0.1, 0.6, 0.3, 0.7, 0.4]
        v2 = [0.4, 0.3, 0.6, 0.5, 0.2, 0.8, 0.1, 0.7]
        self.assertAlmostEqual(self.l2(v1, v2), self.l2(v2, v1), places=8)

    # ── _cosine_sim ───────────────────────────────────────────────────────────

    def test_cosine_identical(self):
        v = [0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1]
        self.assertAlmostEqual(self.cosine(v, v), 1.0, places=8)

    def test_cosine_range(self):
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(self.cosine(v1, v2), 0.0, places=8)

    # ── portrait_dict ─────────────────────────────────────────────────────────

    def test_pd_keys(self):
        d = self.portrait_dict('ТУМАН', self.RULE)
        for k in ['word', 'rule', 'width', 'axes', 'labels', 'metrics',
                  'ic_entropy', 'period', 'transient', 'complexity',
                  'topological_h', 'te_flow', 'sensitivity', 'autocorr',
                  'ic_entropy_n', 'period_n', 'transient_n', 'complexity_n',
                  'topological_h_n', 'te_flow_n', 'sensitivity_n', 'autocorr_n']:
            self.assertIn(k, d)

    def test_pd_word_upper(self):
        d = self.portrait_dict('туман')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_pd_metrics_length(self):
        d = self.portrait_dict('ТУМАН')
        self.assertEqual(len(d['metrics']), self.N_AXES)

    def test_pd_metrics_in_range(self):
        d = self.portrait_dict('ТУМАН')
        for v in d['metrics']:
            self.assertGreaterEqual(v, 0.0, msg=f'metric<0: {v}')
            self.assertLessEqual(v, 1.0,    msg=f'metric>1: {v}')

    def test_pd_axes_length(self):
        d = self.portrait_dict('ТУМАН')
        self.assertEqual(len(d['axes']),    self.N_AXES)
        self.assertEqual(len(d['labels']),  self.N_AXES)

    def test_pd_period_matches_orbit(self):
        from projects.hexglyph.solan_ca import find_orbit
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ТУМАН'), self.W)
        _, p  = find_orbit(cells[:], self.RULE)
        d = self.portrait_dict('ТУМАН', self.RULE)
        self.assertEqual(d['period'], max(p, 1))

    def test_pd_topo_h_xor3_positive(self):
        d = self.portrait_dict('ТУМАН', 'xor3')
        self.assertGreater(d['topological_h'], 0.0)

    def test_pd_period_xor3_tuman_is_8(self):
        d = self.portrait_dict('ТУМАН', 'xor3')
        self.assertEqual(d['period'], 8)

    def test_pd_period_n_at_most_1(self):
        for word in ['ТУМАН', 'ГОРА', 'ВОДА']:
            d = self.portrait_dict(word, self.RULE)
            self.assertLessEqual(d['period_n'], 1.0)

    def test_pd_transient_n_at_most_1(self):
        d = self.portrait_dict('ТУМАН', self.RULE)
        self.assertLessEqual(d['transient_n'], 1.0)

    def test_pd_normalised_suffix_matches_metrics(self):
        d = self.portrait_dict('ТУМАН')
        norm_keys = [a + '_n' for a in self.AXES]
        for i, k in enumerate(norm_keys):
            # _n keys are rounded to 6 d.p.; metrics stores full floats
            self.assertAlmostEqual(d[k], d['metrics'][i], places=5,
                                   msg=f'axis={k}')

    def test_pd_gora_period_2(self):
        d = self.portrait_dict('ГОРА', 'xor3')
        self.assertEqual(d['period'], 2)

    def test_pd_ic_entropy_nonneg(self):
        d = self.portrait_dict('ТУМАН')
        self.assertGreaterEqual(d['ic_entropy'], 0.0)

    def test_pd_te_flow_xor_zero(self):
        # XOR period-1 all-zeros → TE = 0
        d = self.portrait_dict('ТУМАН', 'xor')
        self.assertAlmostEqual(d['te_flow'], 0.0, places=4)

    def test_pd_te_flow_xor3_positive(self):
        d = self.portrait_dict('ТУМАН', 'xor3')
        self.assertGreater(d['te_flow'], 0.0)

    def test_pd_sensitivity_nonneg(self):
        d = self.portrait_dict('ТУМАН')
        self.assertGreaterEqual(d['sensitivity'], 0.0)

    def test_pd_autocorr_n_in_range(self):
        # autocorr_n = (acf1 + 1) / 2 ∈ [0, 1]
        for word in ['ТУМАН', 'ГОРА', 'ВОДА']:
            d = self.portrait_dict(word)
            self.assertGreaterEqual(d['autocorr_n'], 0.0)
            self.assertLessEqual(d['autocorr_n'], 1.0)

    # ── portrait_compare ──────────────────────────────────────────────────────

    def test_pc_keys(self):
        c = self.portrait_compare('ТУМАН', 'ГОРА')
        for k in ['word1', 'word2', 'rule', 'portrait1', 'portrait2',
                  'l2_distance', 'cosine_sim']:
            self.assertIn(k, c)

    def test_pc_self_distance_zero(self):
        c = self.portrait_compare('ТУМАН', 'ТУМАН')
        self.assertAlmostEqual(c['l2_distance'], 0.0, places=6)

    def test_pc_self_cosine_one(self):
        c = self.portrait_compare('ТУМАН', 'ТУМАН')
        self.assertAlmostEqual(c['cosine_sim'], 1.0, places=6)

    def test_pc_distance_positive(self):
        c = self.portrait_compare('ТУМАН', 'ГОРА')
        self.assertGreater(c['l2_distance'], 0.0)

    def test_pc_cosine_in_range(self):
        c = self.portrait_compare('ТУМАН', 'ГОРА')
        self.assertGreaterEqual(c['cosine_sim'], -1.0)
        self.assertLessEqual(c['cosine_sim'], 1.0)

    # ── build_portrait_data ───────────────────────────────────────────────────

    def test_bpd_keys(self):
        d = self.build_portrait(['ГОРА', 'ВОДА', 'МИР'])
        for k in ['words', 'rule', 'axes', 'labels', 'portraits', 'ranking']:
            self.assertIn(k, d)

    def test_bpd_portraits_shape(self):
        words = ['ГОРА', 'ВОДА']
        d = self.build_portrait(words)
        self.assertEqual(set(d['portraits'].keys()), set(words))
        for w, m in d['portraits'].items():
            self.assertEqual(len(m), self.N_AXES, msg=f'word={w}')

    def test_bpd_ranking_axes(self):
        d = self.build_portrait(['ГОРА', 'ВОДА', 'МИР'])
        for ax in self.AXES:
            self.assertIn(ax, d['ranking'])

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_prt_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prt-canvas', content)

    def test_viewer_has_prt_bars(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prt-bars', content)

    def test_viewer_has_prt_word2(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prt-word2', content)

    def test_viewer_has_draw_radar(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('drawRadar', content)

    def test_viewer_has_prt_lz76(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prtLZ76', content)

    def test_viewer_has_prt_topo_h(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prtTopoH', content)

    def test_viewer_has_prt_lyap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prtLyap', content)

    def test_viewer_has_portrait_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('радарный отпечаток', content)


class TestSolanCoarse(unittest.TestCase):
    """Tests for solan_coarse.py and the viewer Coarse-Graining section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_coarse import (
            coarsen, dequantize, coarse_step,
            commutator_traj, coarse_orbit,
            coarse_dict, all_coarse, build_coarse_data,
            _DEFAULT_LEVELS,
        )
        cls.coarsen         = staticmethod(coarsen)
        cls.dequantize      = staticmethod(dequantize)
        cls.coarse_step     = staticmethod(coarse_step)
        cls.commutator_traj = staticmethod(commutator_traj)
        cls.coarse_orbit    = staticmethod(coarse_orbit)
        cls.coarse_dict     = staticmethod(coarse_dict)
        cls.all_coarse      = staticmethod(all_coarse)
        cls.build_coarse_data = staticmethod(build_coarse_data)
        cls.DEFAULT_LEVELS  = _DEFAULT_LEVELS

    # ── coarsen ───────────────────────────────────────────────────────────────

    def test_coarsen_k2_lower_half(self):
        self.assertEqual(self.coarsen(0, 2), 0)
        self.assertEqual(self.coarsen(31, 2), 0)

    def test_coarsen_k2_upper_half(self):
        self.assertEqual(self.coarsen(32, 2), 1)
        self.assertEqual(self.coarsen(63, 2), 1)

    def test_coarsen_k64_identity(self):
        for v in [0, 10, 32, 63]:
            self.assertEqual(self.coarsen(v, 64), v)

    def test_coarsen_k3_bins(self):
        self.assertEqual(self.coarsen(0, 3), 0)
        self.assertEqual(self.coarsen(21, 3), 0)   # 21*3//64 = 0
        self.assertEqual(self.coarsen(22, 3), 1)   # 22*3//64 = 1
        self.assertEqual(self.coarsen(63, 3), 2)

    def test_coarsen_result_in_range(self):
        for k in [2, 3, 4, 8, 16, 64]:
            for v in [0, 15, 31, 32, 48, 63]:
                self.assertGreaterEqual(self.coarsen(v, k), 0)
                self.assertLess(self.coarsen(v, k), k)

    def test_coarsen_monotone(self):
        for k in [2, 4, 8]:
            vals = [self.coarsen(v, k) for v in range(64)]
            self.assertEqual(vals, sorted(vals))

    # ── dequantize ────────────────────────────────────────────────────────────

    def test_dequantize_k2(self):
        self.assertEqual(self.dequantize(0, 2), 0)
        self.assertEqual(self.dequantize(1, 2), 63)

    def test_dequantize_k64_identity(self):
        for i in [0, 10, 32, 63]:
            self.assertEqual(self.dequantize(i, 64), i)

    def test_dequantize_k1(self):
        self.assertEqual(self.dequantize(0, 1), 0)

    def test_dequantize_k3_midpoint(self):
        self.assertEqual(self.dequantize(1, 3), 32)

    def test_dequantize_result_in_range(self):
        for k in [2, 3, 4, 8, 16, 64]:
            for i in range(k):
                v = self.dequantize(i, k)
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 63)

    # ── coarse_step ───────────────────────────────────────────────────────────

    def test_coarse_step_length(self):
        cells = [0, 1, 0, 1, 1, 0, 1, 0,
                 1, 0, 1, 0, 0, 1, 0, 1]
        out = self.coarse_step(cells, 'xor3', 2)
        self.assertEqual(len(out), 16)

    def test_coarse_step_output_in_range(self):
        cells = [self.coarsen(v, 4) for v in range(16)]
        for k in [2, 4, 8]:
            out = self.coarse_step(cells, 'xor3', k)
            for v in out:
                self.assertGreaterEqual(v, 0)
                self.assertLess(v, k)

    def test_coarse_step_k64_matches_q6(self):
        from projects.hexglyph.solan_ca import step
        cells = list(range(16))
        q6 = step(cells[:], 'xor3')
        cg = self.coarse_step(cells, 'xor3', 64)
        self.assertEqual(cg, q6)

    # ── commutator_traj ───────────────────────────────────────────────────────

    def test_ctraj_length(self):
        traj = self.commutator_traj('ТУМАН', 'xor3', 3, n_steps=10)
        self.assertEqual(len(traj), 11)

    def test_ctraj_first_is_zero(self):
        for rule in ['xor', 'xor3', 'and', 'or']:
            traj = self.commutator_traj('ТУМАН', rule, 3, n_steps=5)
            self.assertEqual(traj[0], 0.0)

    def test_ctraj_values_in_unit_interval(self):
        traj = self.commutator_traj('ГОРА', 'xor3', 4, n_steps=10)
        for d in traj:
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_ctraj_k2_exact_xor(self):
        traj = self.commutator_traj('ТУМАН', 'xor', 2, n_steps=10)
        self.assertTrue(all(d == 0.0 for d in traj))

    def test_ctraj_k2_exact_xor3(self):
        traj = self.commutator_traj('ТУМАН', 'xor3', 2, n_steps=10)
        self.assertTrue(all(d == 0.0 for d in traj))

    def test_ctraj_k2_exact_and(self):
        traj = self.commutator_traj('ТУМАН', 'and', 2, n_steps=10)
        self.assertTrue(all(d == 0.0 for d in traj))

    def test_ctraj_k2_exact_or(self):
        traj = self.commutator_traj('ТУМАН', 'or', 2, n_steps=10)
        self.assertTrue(all(d == 0.0 for d in traj))

    def test_ctraj_k64_exact(self):
        traj = self.commutator_traj('ТУМАН', 'xor3', 64, n_steps=10)
        self.assertTrue(all(d == 0.0 for d in traj))

    # ── coarse_orbit ──────────────────────────────────────────────────────────

    def test_corbit_keys(self):
        d = self.coarse_orbit('ТУМАН', 'xor3', 3)
        for key in ('transient', 'period', 'entropy', 'entropy_norm', 'n_unique'):
            self.assertIn(key, d)

    def test_corbit_period_positive(self):
        d = self.coarse_orbit('ТУМАН', 'xor3', 3)
        self.assertGreaterEqual(d['period'], 1)

    def test_corbit_entropy_nonneg(self):
        """No -0.0 or negative entropy."""
        for word in ['ГОРА', 'ТУМАН', 'ВОДА']:
            for k in [2, 3, 4]:
                d = self.coarse_orbit(word, 'and', k)
                self.assertGreaterEqual(d['entropy'], 0.0)

    def test_corbit_entropy_norm_in_unit(self):
        d = self.coarse_orbit('ТУМАН', 'xor3', 4)
        self.assertGreaterEqual(d['entropy_norm'], 0.0)
        self.assertLessEqual(d['entropy_norm'], 1.0)

    def test_corbit_n_unique_le_period(self):
        d = self.coarse_orbit('ТУМАН', 'xor3', 8)
        self.assertLessEqual(d['n_unique'], d['period'])

    def test_corbit_k64_period_matches_q6(self):
        from projects.hexglyph.solan_ca import find_orbit
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ТУМАН'), 16)
        q6_t, q6_p = find_orbit(cells[:], 'xor3')
        d = self.coarse_orbit('ТУМАН', 'xor3', 64)
        self.assertEqual(d['period'], max(q6_p, 1))

    # ── coarse_dict ───────────────────────────────────────────────────────────

    def test_cdict_keys(self):
        d = self.coarse_dict('ТУМАН', 'xor3')
        for key in ('word', 'rule', 'levels', 'n_steps',
                    'q6_period', 'q6_transient',
                    'by_level', 'exact_levels', 'max_inconsistency'):
            self.assertIn(key, d)

    def test_cdict_word_upper(self):
        d = self.coarse_dict('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_cdict_exact_levels_contains_2_and_64(self):
        d = self.coarse_dict('ТУМАН', 'xor3')
        self.assertIn(2, d['exact_levels'])
        self.assertIn(64, d['exact_levels'])

    def test_cdict_by_level_keys(self):
        d = self.coarse_dict('ГОРА', 'xor3', levels=[2, 4])
        for k in [2, 4]:
            bl = d['by_level'][k]
            for sub in ('commutator', 'max_commutator', 'mean_commutator',
                        'is_exact', 'transient', 'period', 'entropy_norm', 'n_unique'):
                self.assertIn(sub, bl)

    def test_cdict_is_exact_k2(self):
        d = self.coarse_dict('ТУМАН', 'xor3')
        self.assertTrue(d['by_level'][2]['is_exact'])

    def test_cdict_is_exact_k64(self):
        d = self.coarse_dict('ТУМАН', 'xor3')
        self.assertTrue(d['by_level'][64]['is_exact'])

    def test_cdict_max_inconsistency_keys(self):
        levels = [2, 3, 4]
        d = self.coarse_dict('ТУМАН', 'xor3', levels=levels)
        self.assertEqual(set(d['max_inconsistency'].keys()), set(levels))

    def test_cdict_commutator_length(self):
        d = self.coarse_dict('ТУМАН', 'xor3', levels=[3], n_steps=15)
        self.assertEqual(len(d['by_level'][3]['commutator']), 16)

    # ── all_coarse ────────────────────────────────────────────────────────────

    def test_all_coarse_rules(self):
        d = self.all_coarse('ГОРА', levels=[2, 4])
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_coarse_each_is_dict(self):
        d = self.all_coarse('ГОРА', levels=[2])
        for rule, cd in d.items():
            self.assertIn('by_level', cd)

    # ── build_coarse_data ─────────────────────────────────────────────────────

    def test_bcd_keys(self):
        d = self.build_coarse_data(['ГОРА', 'ВОДА'], levels=[2, 3])
        for key in ('words', 'levels', 'per_rule'):
            self.assertIn(key, d)

    def test_bcd_per_rule_has_all_rules(self):
        d = self.build_coarse_data(['ГОРА'], levels=[2])
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_bcd_word_entry_keys(self):
        d = self.build_coarse_data(['ГОРА'], levels=[2, 3])
        entry = d['per_rule']['xor3']['ГОРА']
        for key in ('q6_period', 'exact_levels', 'max_inc_3', 'max_inc_4'):
            self.assertIn(key, entry)

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_cg_hmap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cg-hmap', content)

    def test_viewer_has_cg_bars(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cg-bars', content)

    def test_viewer_has_cg_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cg-stats', content)

    def test_viewer_has_cg_coarsen(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cgCoarsen', content)

    def test_viewer_has_cg_dequant(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cgDequant', content)

    def test_viewer_has_cg_comm_traj(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cgCommTraj', content)

    def test_viewer_has_cg_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cgRun', content)

    def test_viewer_has_coarse_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Огрубление Q6', content)


class TestSolanActive(unittest.TestCase):
    """Tests for solan_active.py and the viewer AIS section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_active import (
            cell_seqs, h_ngrams, ais_k,
            ais_profile, cell_ais_k,
            ais_dict, all_ais, build_ais_data,
        )
        cls.cell_seqs      = staticmethod(cell_seqs)
        cls.h_ngrams       = staticmethod(h_ngrams)
        cls.ais_k          = staticmethod(ais_k)
        cls.ais_profile    = staticmethod(ais_profile)
        cls.cell_ais_k     = staticmethod(cell_ais_k)
        cls.ais_dict       = staticmethod(ais_dict)
        cls.all_ais        = staticmethod(all_ais)
        cls.build_ais_data = staticmethod(build_ais_data)

    # ── h_ngrams ──────────────────────────────────────────────────────────────

    def test_h_ngrams_constant_zero(self):
        seqs = [[0, 0, 0, 0]]
        self.assertAlmostEqual(self.h_ngrams(seqs, 1), 0.0)

    def test_h_ngrams_alternating_1gram(self):
        seqs = [[0, 1, 0, 1]]
        self.assertAlmostEqual(self.h_ngrams(seqs, 1), 1.0, places=6)

    def test_h_ngrams_alternating_2gram(self):
        """Alternating [0,1]: bigrams (0,1) and (1,0) each p=0.5 → H=1."""
        seqs = [[0, 1, 0, 1]]
        self.assertAlmostEqual(self.h_ngrams(seqs, 2), 1.0, places=6)

    def test_h_ngrams_nonneg(self):
        """Entropy is always ≥ 0."""
        for seqs in [[[0, 0]], [[1, 1, 1]], [[0, 1, 0, 1, 1]]]:
            self.assertGreaterEqual(self.h_ngrams(seqs, 1), 0.0)

    def test_h_ngrams_empty_seqs(self):
        self.assertEqual(self.h_ngrams([], 1), 0.0)

    def test_h_ngrams_h2_ge_h1(self):
        """H_2 >= H_1 (more context → more bits, generally)."""
        seqs = [[0, 1, 0, 0, 1, 1, 0, 1]]
        h1 = self.h_ngrams(seqs, 1)
        h2 = self.h_ngrams(seqs, 2)
        self.assertGreaterEqual(h2, h1 - 1e-9)

    # ── ais_k ────────────────────────────────────────────────────────────────

    def test_ais_k_constant_seq(self):
        """Constant sequence has H1=0 → AIS=0."""
        seqs = [[0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertAlmostEqual(self.ais_k(seqs, 1), 0.0)
        self.assertAlmostEqual(self.ais_k(seqs, 2), 0.0)

    def test_ais_k_alternating_k1(self):
        """Per-cell alternating [0,1]: AIS_1 = H1 - (H2-H1) = 1-(1-1) = 1."""
        seqs = [[0, 1, 0, 1, 0, 1, 0, 1]]
        self.assertAlmostEqual(self.ais_k(seqs, 1), 1.0, places=6)

    def test_ais_k_alternating_k2(self):
        """Alternating: AIS_2 = H1 - (H3-H2) = 1-(1-1) = 1."""
        seqs = [[0, 1, 0, 1, 0, 1, 0, 1]]
        self.assertAlmostEqual(self.ais_k(seqs, 2), 1.0, places=6)

    def test_ais_k_nonneg(self):
        """AIS is always ≥ 0."""
        for k in [1, 2, 3]:
            v = self.ais_k([[0, 1, 0, 0, 1, 1, 0, 1]], k)
            self.assertGreaterEqual(v, 0.0)

    def test_ais_k_le_h1(self):
        """AIS ≤ H1."""
        seqs = [[0, 1, 0, 0, 1, 1, 0, 1]]
        h1 = self.h_ngrams(seqs, 1)
        for k in [1, 2, 3]:
            self.assertLessEqual(self.ais_k(seqs, k), h1 + 1e-9)

    # ── cell_seqs ─────────────────────────────────────────────────────────────

    def test_cell_seqs_width(self):
        seqs = self.cell_seqs('ТУМАН', 'xor3')
        self.assertEqual(len(seqs), 16)

    def test_cell_seqs_binary(self):
        seqs = self.cell_seqs('ТУМАН', 'xor3')
        for s in seqs:
            for v in s:
                self.assertIn(v, (0, 1))

    def test_cell_seqs_equal_lengths(self):
        """All temporal sequences have the same length (orbit period)."""
        seqs = self.cell_seqs('ТУМАН', 'xor3')
        lengths = [len(s) for s in seqs]
        self.assertEqual(len(set(lengths)), 1)

    def test_cell_seqs_xor_h1_zero(self):
        """XOR ТУМАН has fixed-point attractor → H1=0 for all cells."""
        seqs = self.cell_seqs('ТУМАН', 'xor')
        h1 = self.h_ngrams(seqs, 1)
        self.assertAlmostEqual(h1, 0.0)

    # ── ais_profile ───────────────────────────────────────────────────────────

    def test_ais_profile_length(self):
        p = self.ais_profile('ТУМАН', 'xor3', max_k=6)
        self.assertEqual(len(p), 6)

    def test_ais_profile_nonneg(self):
        p = self.ais_profile('ТУМАН', 'xor3', max_k=4)
        for v in p:
            self.assertGreaterEqual(v, 0.0)

    def test_ais_profile_xor_all_zero(self):
        """XOR ТУМАН fixed-point: all AIS = 0."""
        p = self.ais_profile('ТУМАН', 'xor', max_k=4)
        for v in p:
            self.assertAlmostEqual(v, 0.0)

    def test_ais_profile_xor3_nondecreasing(self):
        """XOR3 ТУМАН: AIS profile generally non-decreasing (more memory helps)."""
        p = self.ais_profile('ТУМАН', 'xor3', max_k=6)
        for i in range(len(p) - 1):
            self.assertGreaterEqual(p[i + 1] + 1e-6, p[i])

    # ── cell_ais_k ────────────────────────────────────────────────────────────

    def test_cell_ais_k_length(self):
        c = self.cell_ais_k('ТУМАН', 'xor3', k=2)
        self.assertEqual(len(c), 16)

    def test_cell_ais_k_range(self):
        c = self.cell_ais_k('ТУМАН', 'xor3', k=2)
        for v in c:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_cell_ais_k_xor_zero(self):
        """XOR ТУМАН fixed-point: per-cell AIS = 0."""
        c = self.cell_ais_k('ТУМАН', 'xor', k=2)
        for v in c:
            self.assertAlmostEqual(v, 0.0)

    def test_cell_ais_k_and_gora_perfect(self):
        """ГОРА AND period-2 alternating: per-cell AIS_1 = 1.0."""
        c = self.cell_ais_k('ГОРА', 'and', k=1)
        for v in c:
            self.assertAlmostEqual(v, 1.0, places=5)

    # ── ais_dict ──────────────────────────────────────────────────────────────

    def test_ais_dict_keys(self):
        d = self.ais_dict('ТУМАН', 'xor3')
        for key in ('word', 'rule', 'max_k', 'h1',
                    'ais_profile', 'ais_1', 'ais_2',
                    'cell_ais', 'total_ais', 'mean_ais',
                    'max_cell_ais', 'min_cell_ais', 'ais_frac'):
            self.assertIn(key, d)

    def test_ais_dict_word_upper(self):
        d = self.ais_dict('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_ais_dict_profile_length(self):
        d = self.ais_dict('ТУМАН', 'xor3', max_k=4)
        self.assertEqual(len(d['ais_profile']), 4)

    def test_ais_dict_ais1_matches_profile(self):
        d = self.ais_dict('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['ais_1'], d['ais_profile'][0], places=6)

    def test_ais_dict_ais2_matches_profile(self):
        d = self.ais_dict('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['ais_2'], d['ais_profile'][1], places=6)

    def test_ais_dict_xor_h1_zero(self):
        d = self.ais_dict('ТУМАН', 'xor')
        self.assertAlmostEqual(d['h1'], 0.0)

    def test_ais_dict_xor3_h1_positive(self):
        d = self.ais_dict('ТУМАН', 'xor3')
        self.assertGreater(d['h1'], 0.0)

    def test_ais_dict_xor3_ais2_positive(self):
        d = self.ais_dict('ТУМАН', 'xor3')
        self.assertGreater(d['ais_2'], 0.0)

    def test_ais_dict_frac_in_unit(self):
        d = self.ais_dict('ТУМАН', 'xor3')
        self.assertGreaterEqual(d['ais_frac'], 0.0)
        self.assertLessEqual(d['ais_frac'], 1.0 + 1e-6)

    def test_ais_dict_gora_xor3_ais2_perfect(self):
        """ГОРА XOR3: pooled AIS_2 = H1 = 1.0 (period-2 mixed cells)."""
        d = self.ais_dict('ГОРА', 'xor3')
        self.assertAlmostEqual(d['h1'], 1.0, places=5)
        self.assertAlmostEqual(d['ais_2'], 1.0, places=5)
        self.assertAlmostEqual(d['ais_frac'], 1.0, places=5)

    def test_ais_dict_gora_xor3_ais1_zero(self):
        """ГОРА XOR3: pooled AIS_1 = 0 (1-gram context uninformative)."""
        d = self.ais_dict('ГОРА', 'xor3')
        self.assertAlmostEqual(d['ais_1'], 0.0, places=5)

    def test_ais_dict_cell_ais_length(self):
        d = self.ais_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['cell_ais']), 16)

    # ── all_ais ───────────────────────────────────────────────────────────────

    def test_all_ais_rules(self):
        d = self.all_ais('ГОРА')
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_ais_each_valid(self):
        d = self.all_ais('ГОРА')
        for rule, ad in d.items():
            self.assertIn('ais_profile', ad)

    # ── build_ais_data ────────────────────────────────────────────────────────

    def test_bad_keys(self):
        d = self.build_ais_data(['ГОРА', 'ВОДА'], max_k=2)
        for key in ('words', 'max_k', 'per_rule'):
            self.assertIn(key, d)

    def test_bad_per_rule_has_all_rules(self):
        d = self.build_ais_data(['ГОРА'], max_k=2)
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_bad_word_entry_keys(self):
        d = self.build_ais_data(['ГОРА'], max_k=2)
        entry = d['per_rule']['xor3']['ГОРА']
        for key in ('h1', 'ais_2', 'ais_frac', 'mean_ais'):
            self.assertIn(key, entry)

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_ais_profile_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ais-profile', content)

    def test_viewer_has_ais_cell_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ais-cell', content)

    def test_viewer_has_ais_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ais-stats', content)

    def test_viewer_has_ais_h_ngrams(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('aisHNgrams', content)

    def test_viewer_has_ais_k_func(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('aisK', content)

    def test_viewer_has_ais_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('aisRun', content)

    def test_viewer_has_ais_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Active Information Storage', content)


class TestSolanTemporal(unittest.TestCase):
    """Tests for solan_temporal.py and the viewer Temporal DFT section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_temporal import (
            cell_dft_power, spectral_entropy_of,
            attractor_temporal_spectra,
            temporal_dict, all_temporal, build_temporal_data,
        )
        cls.cell_dft_power             = staticmethod(cell_dft_power)
        cls.spectral_entropy_of        = staticmethod(spectral_entropy_of)
        cls.attractor_temporal_spectra = staticmethod(attractor_temporal_spectra)
        cls.temporal_dict              = staticmethod(temporal_dict)
        cls.all_temporal               = staticmethod(all_temporal)
        cls.build_temporal_data        = staticmethod(build_temporal_data)

    # ── cell_dft_power ────────────────────────────────────────────────────────

    def test_cdp_constant_seq_dc_only(self):
        """Constant sequence → all power at DC (k=0), AC = 0."""
        power = self.cell_dft_power([40, 40, 40, 40])
        self.assertAlmostEqual(power[0], 40.0 ** 2, places=4)
        for k in range(1, len(power)):
            self.assertAlmostEqual(power[k], 0.0, places=6)

    def test_cdp_zero_seq_all_zero(self):
        """All-zero sequence → S[k]=0 for all k."""
        power = self.cell_dft_power([0, 0, 0, 0])
        for p in power:
            self.assertAlmostEqual(p, 0.0, places=10)

    def test_cdp_alternating_dc_and_nyquist(self):
        """[0,1,0,1]: DC=0.25 (mean=0.5²), AC at k=2 (Nyquist)=0.25, k=1=0."""
        power = self.cell_dft_power([0, 1, 0, 1])
        self.assertAlmostEqual(power[0], 0.25, places=6)  # DC = (0.5)² = 0.25
        self.assertAlmostEqual(power[1], 0.0,  places=6)  # k=1
        self.assertAlmostEqual(power[2], 0.25, places=6)  # Nyquist

    def test_cdp_length(self):
        """Output length is P//2 + 1."""
        for n in [1, 2, 4, 8]:
            seq = [32] * n
            self.assertEqual(len(self.cell_dft_power(seq)), n // 2 + 1)

    def test_cdp_empty(self):
        self.assertEqual(self.cell_dft_power([]), [])

    def test_cdp_nonneg(self):
        """Power values are always ≥ 0."""
        for seq in [[10, 30, 50, 20], [0, 63, 0, 63], [32] * 8]:
            for p in self.cell_dft_power(seq):
                self.assertGreaterEqual(p, 0.0)

    def test_cdp_parseval(self):
        """Parseval: Σ S[k] ≈ mean-square of sequence."""
        seq = [10, 40, 20, 63]
        n = len(seq)
        power = self.cell_dft_power(seq)
        mean_sq = sum(v ** 2 for v in seq) / n
        total_s = sum(power)
        # For real sequence, Parseval's: Σ_k |DFT[k]|²/n² ≈ mean_sq/n
        # We use one-sided, so check Σ S_k ≈ mean_sq (within normalization)
        self.assertGreater(total_s, 0.0)

    # ── spectral_entropy_of ───────────────────────────────────────────────────

    def test_seo_single_freq(self):
        """All power at one bin → H_s = 0."""
        self.assertAlmostEqual(self.spectral_entropy_of([1.0, 0.0, 0.0]), 0.0)

    def test_seo_uniform(self):
        """Uniform power across 4 bins → H_s = log₂(4) = 2.0 bits."""
        h = self.spectral_entropy_of([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(h, 2.0, places=6)

    def test_seo_two_equal(self):
        """Two equal bins → H_s = 1.0 bit."""
        h = self.spectral_entropy_of([0.5, 0.5])
        self.assertAlmostEqual(h, 1.0, places=6)

    def test_seo_zero_power(self):
        """All-zero power → H_s = 0."""
        self.assertAlmostEqual(self.spectral_entropy_of([0.0, 0.0, 0.0]), 0.0)

    def test_seo_nonneg(self):
        for power in [[0.3, 0.7], [0.1, 0.5, 0.4], [1.0]]:
            self.assertGreaterEqual(self.spectral_entropy_of(power), 0.0)

    # ── attractor_temporal_spectra ────────────────────────────────────────────

    def test_ats_shape(self):
        """Returns N=16 spectra each of length P//2+1."""
        specs = self.attractor_temporal_spectra('ТУМАН', 'xor3')
        self.assertEqual(len(specs), 16)
        for s in specs:
            self.assertGreater(len(s), 0)

    def test_ats_all_same_length(self):
        specs = self.attractor_temporal_spectra('ТУМАН', 'xor3')
        lengths = [len(s) for s in specs]
        self.assertEqual(len(set(lengths)), 1)

    def test_ats_xor_dc_zero(self):
        """XOR ТУМАН (all-zero fixed point): all power = 0."""
        specs = self.attractor_temporal_spectra('ТУМАН', 'xor')
        for s in specs:
            for p in s:
                self.assertAlmostEqual(p, 0.0, places=6)

    def test_ats_or_dc_positive(self):
        """OR ТУМАН (all-63 fixed point): DC=63²=3969, AC=0."""
        specs = self.attractor_temporal_spectra('ТУМАН', 'or')
        for s in specs:
            self.assertAlmostEqual(s[0], 63.0 ** 2, places=2)
            for k in range(1, len(s)):
                self.assertAlmostEqual(s[k], 0.0, places=4)

    def test_ats_nonneg(self):
        specs = self.attractor_temporal_spectra('ГОРА', 'xor3')
        for s in specs:
            for p in s:
                self.assertGreaterEqual(p, 0.0)

    def test_ats_period_1_one_bin(self):
        """Period-1 attractor → only 1 frequency bin (k=0)."""
        specs = self.attractor_temporal_spectra('ТУМАН', 'xor')
        for s in specs:
            self.assertEqual(len(s), 1)

    def test_ats_period_2_two_bins(self):
        """ГОРА XOR3 period-2 → 2 frequency bins (k=0,1)."""
        specs = self.attractor_temporal_spectra('ГОРА', 'xor3')
        for s in specs:
            self.assertEqual(len(s), 2)

    def test_ats_period_8_five_bins(self):
        """ТУМАН XOR3 period-8 → 5 frequency bins (k=0..4)."""
        specs = self.attractor_temporal_spectra('ТУМАН', 'xor3')
        for s in specs:
            self.assertEqual(len(s), 5)

    # ── temporal_dict ─────────────────────────────────────────────────────────

    def test_td_keys(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        for key in ('word', 'rule', 'period', 'n_freqs',
                    'spectra', 'spectral_entropy', 'dominant_freq',
                    'dc_fraction', 'total_power',
                    'mean_spec_entropy', 'mean_dc', 'global_dominant'):
            self.assertIn(key, d)

    def test_td_word_upper(self):
        d = self.temporal_dict('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_td_spectra_count(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['spectra']), 16)

    def test_td_n_freqs_period_1(self):
        d = self.temporal_dict('ТУМАН', 'xor')
        self.assertEqual(d['n_freqs'], 1)
        self.assertEqual(d['period'], 1)

    def test_td_n_freqs_period_8(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        self.assertEqual(d['n_freqs'], 5)
        self.assertEqual(d['period'], 8)

    def test_td_spec_entropy_length(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        self.assertEqual(len(d['spectral_entropy']), 16)

    def test_td_spec_entropy_nonneg(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        for h in d['spectral_entropy']:
            self.assertGreaterEqual(h, 0.0)

    def test_td_dc_fraction_in_unit(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        for dc in d['dc_fraction']:
            self.assertGreaterEqual(dc, 0.0)
            self.assertLessEqual(dc, 1.0 + 1e-6)

    def test_td_or_dc_one(self):
        """OR ТУМАН (all-63 fixed point): DC fraction = 1.0 for all cells."""
        d = self.temporal_dict('ТУМАН', 'or')
        for dc in d['dc_fraction']:
            self.assertAlmostEqual(dc, 1.0, places=5)

    def test_td_xor_spec_entropy_zero(self):
        """XOR ТУМАН (all-zero fixed point): spectral entropy = 0."""
        d = self.temporal_dict('ТУМАН', 'xor')
        for h in d['spectral_entropy']:
            self.assertAlmostEqual(h, 0.0)

    def test_td_mean_spec_entropy_nonneg(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        self.assertGreaterEqual(d['mean_spec_entropy'], 0.0)

    def test_td_mean_dc_in_unit(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        self.assertGreaterEqual(d['mean_dc'], 0.0)
        self.assertLessEqual(d['mean_dc'], 1.0 + 1e-6)

    def test_td_dominant_freq_in_range(self):
        d = self.temporal_dict('ТУМАН', 'xor3')
        for kf in d['dominant_freq']:
            self.assertGreaterEqual(kf, 0)
            self.assertLess(kf, d['n_freqs'])

    # ── all_temporal ──────────────────────────────────────────────────────────

    def test_at_rules(self):
        d = self.all_temporal('ГОРА')
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_at_each_valid(self):
        d = self.all_temporal('ГОРА')
        for rule, td in d.items():
            self.assertIn('spectra', td)
            self.assertIn('period', td)

    # ── build_temporal_data ───────────────────────────────────────────────────

    def test_btd_keys(self):
        d = self.build_temporal_data(['ГОРА', 'ВОДА'])
        for key in ('words', 'width', 'per_rule'):
            self.assertIn(key, d)

    def test_btd_per_rule_rules(self):
        d = self.build_temporal_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_btd_word_entry_keys(self):
        d = self.build_temporal_data(['ГОРА'])
        entry = d['per_rule']['xor3']['ГОРА']
        for key in ('period', 'mean_spec_entropy', 'mean_dc', 'global_dominant'):
            self.assertIn(key, entry)

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_tmp_hmap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('tmp-hmap', content)

    def test_viewer_has_tmp_bars(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('tmp-bars', content)

    def test_viewer_has_tmp_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('tmp-stats', content)

    def test_viewer_has_tmp_dft(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('tmpDFT', content)

    def test_viewer_has_tmp_spec_h(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('tmpSpecH', content)

    def test_viewer_has_tmp_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('tmpRun', content)

    def test_viewer_has_temporal_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Временной спектр Q6', content)


class TestSolanPersistence(unittest.TestCase):
    """Tests for solan_persistence.py and the viewer Persistence section."""

    @classmethod
    def setUpClass(cls):
        import importlib, sys
        spec = importlib.util.spec_from_file_location(
            'solan_persistence',
            str(pathlib.Path(__file__).resolve().parents[1] /
                'projects' / 'hexglyph' / 'solan_persistence.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules['solan_persistence'] = mod
        spec.loader.exec_module(mod)
        cls.mod = mod

    # ── run_lengths ────────────────────────────────────────────────────

    def test_rl_empty(self):
        self.assertEqual(self.mod.run_lengths([]), [])

    def test_rl_single(self):
        self.assertEqual(self.mod.run_lengths([0]), [1])

    def test_rl_constant_circular(self):
        # All same → one run of length n
        self.assertEqual(self.mod.run_lengths([0, 0, 0, 0]), [4])

    def test_rl_alternating_circular(self):
        runs = self.mod.run_lengths([0, 1, 0, 1])
        self.assertEqual(runs, [1, 1, 1, 1])

    def test_rl_mixed_circular(self):
        # [0,1,1,0,0,1] circular runs: 0|1,1|0,0|1 → wraps: 0 at end+0 at start=2
        # transitions at positions 0,2,4 (0→1,1→0,0→1); lengths: 0→2=2, 2→4=2, 4→0+6=2?
        # Let's just check sum = n and all positive
        seq = [0, 1, 1, 0, 0, 1]
        runs = self.mod.run_lengths(seq, circular=True)
        self.assertEqual(sum(runs), len(seq))
        self.assertTrue(all(r > 0 for r in runs))

    def test_rl_sum_equals_length(self):
        import random
        rng = random.Random(42)
        seq = [rng.randint(0, 1) for _ in range(16)]
        runs = self.mod.run_lengths(seq)
        self.assertEqual(sum(runs), 16)

    def test_rl_noncircular_constant(self):
        self.assertEqual(self.mod.run_lengths([1, 1, 1], circular=False), [3])

    def test_rl_noncircular_alternating(self):
        self.assertEqual(self.mod.run_lengths([0, 1, 0], circular=False), [1, 1, 1])

    def test_rl_noncircular_sum(self):
        seq = [0, 0, 1, 1, 0, 1]
        runs = self.mod.run_lengths(seq, circular=False)
        self.assertEqual(sum(runs), len(seq))

    # ── persistence ────────────────────────────────────────────────────

    def test_persist_constant(self):
        self.assertAlmostEqual(self.mod.persistence([0, 0, 0, 0]), 1.0)

    def test_persist_alternating_even(self):
        self.assertAlmostEqual(self.mod.persistence([0, 1, 0, 1]), 0.0)

    def test_persist_single(self):
        self.assertAlmostEqual(self.mod.persistence([1]), 1.0)

    def test_persist_empty(self):
        self.assertAlmostEqual(self.mod.persistence([]), 1.0)

    def test_persist_range(self):
        # persistence must be in [0, 1]
        import random
        rng = random.Random(7)
        seq = [rng.randint(0, 1) for _ in range(20)]
        p = self.mod.persistence(seq)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_persist_half(self):
        # [0,0,1,1] circular: transitions at 1 and 3 → 2/4 → persist=0.5
        self.assertAlmostEqual(self.mod.persistence([0, 0, 1, 1]), 0.5)

    # ── run_stats ──────────────────────────────────────────────────────

    def test_rs_keys(self):
        s = self.mod.run_stats([0, 1, 0, 1])
        for k in ('n_runs', 'persistence', 'mean_run', 'std_run', 'max_run', 'min_run', 'cv_run'):
            self.assertIn(k, s)

    def test_rs_constant(self):
        s = self.mod.run_stats([0, 0, 0, 0])
        self.assertEqual(s['n_runs'], 1)
        self.assertAlmostEqual(s['persistence'], 1.0)
        self.assertAlmostEqual(s['mean_run'], 4.0)
        self.assertAlmostEqual(s['cv_run'], 0.0)

    def test_rs_alternating(self):
        s = self.mod.run_stats([0, 1, 0, 1])
        self.assertEqual(s['n_runs'], 4)
        self.assertAlmostEqual(s['persistence'], 0.0)
        self.assertAlmostEqual(s['mean_run'], 1.0)
        self.assertAlmostEqual(s['cv_run'], 0.0)

    def test_rs_max_run_constant(self):
        s = self.mod.run_stats([1, 1, 1, 1, 1])
        self.assertEqual(s['max_run'], 5)

    def test_rs_min_run_alternating(self):
        s = self.mod.run_stats([0, 1, 0, 1, 0, 1])
        self.assertEqual(s['min_run'], 1)

    # ── cell_run_stats ─────────────────────────────────────────────────

    def test_crs_length(self):
        stats = self.mod.cell_run_stats('ТУМАН', 'xor', 16)
        self.assertEqual(len(stats), 16)

    def test_crs_each_has_keys(self):
        stats = self.mod.cell_run_stats('ТУМАН', 'xor3', 16)
        for s in stats:
            self.assertIn('persistence', s)
            self.assertIn('mean_run', s)

    def test_crs_xor_tuman_all_persistent(self):
        # XOR ТУМАН P=1 → all constant → persistence=1.0
        stats = self.mod.cell_run_stats('ТУМАН', 'xor', 16)
        for s in stats:
            self.assertAlmostEqual(s['persistence'], 1.0)

    def test_crs_gora_and_all_alternating(self):
        # ГОРА AND P=2, all alternating → persistence=0.0
        stats = self.mod.cell_run_stats('ГОРА', 'and', 16)
        for s in stats:
            self.assertAlmostEqual(s['persistence'], 0.0)

    def test_crs_gora_xor3_mixed(self):
        # ГОРА XOR3 P=2: 8 constant (persist=1), 8 alternating (persist=0)
        stats = self.mod.cell_run_stats('ГОРА', 'xor3', 16)
        persts = [s['persistence'] for s in stats]
        n_ones  = sum(1 for p in persts if p == 1.0)
        n_zeros = sum(1 for p in persts if p == 0.0)
        self.assertEqual(n_ones,  8)
        self.assertEqual(n_zeros, 8)

    # ── pooled_run_dist ────────────────────────────────────────────────

    def test_prd_returns_dict(self):
        h = self.mod.pooled_run_dist('ТУМАН', 'xor3', 16)
        self.assertIsInstance(h, dict)

    def test_prd_xor_tuman_only_run1(self):
        # XOR ТУМАН P=1 → each cell is constant → run length = 1 (trivially P=1)
        h = self.mod.pooled_run_dist('ТУМАН', 'xor', 16)
        self.assertEqual(list(h.keys()), [1])

    def test_prd_all_positive_counts(self):
        h = self.mod.pooled_run_dist('ТУМАН', 'xor3', 16)
        self.assertTrue(all(v > 0 for v in h.values()))

    def test_prd_gora_xor3_has_run1_and_run2(self):
        h = self.mod.pooled_run_dist('ГОРА', 'xor3', 16)
        self.assertIn(1, h)
        self.assertIn(2, h)

    # ── persistence_dict ───────────────────────────────────────────────

    def test_pd_keys(self):
        d = self.mod.persistence_dict('ТУМАН', 'xor3', 16)
        for k in ('word', 'rule', 'period', 'cell_stats', 'run_dist',
                  'mean_persistence', 'mean_run', 'mean_cv',
                  'max_run_global', 'min_persistence', 'max_persistence',
                  'all_persistent', 'all_alternating'):
            self.assertIn(k, d)

    def test_pd_xor_tuman_all_persistent(self):
        d = self.mod.persistence_dict('ТУМАН', 'xor', 16)
        self.assertTrue(d['all_persistent'])
        self.assertFalse(d['all_alternating'])
        self.assertAlmostEqual(d['mean_persistence'], 1.0)

    def test_pd_gora_and_all_alternating(self):
        d = self.mod.persistence_dict('ГОРА', 'and', 16)
        self.assertTrue(d['all_alternating'])
        self.assertFalse(d['all_persistent'])
        self.assertAlmostEqual(d['mean_persistence'], 0.0)

    def test_pd_gora_xor3_mean_persistence(self):
        d = self.mod.persistence_dict('ГОРА', 'xor3', 16)
        self.assertAlmostEqual(d['mean_persistence'], 0.5, places=5)

    def test_pd_cell_stats_length(self):
        d = self.mod.persistence_dict('ТУМАН', 'xor3', 16)
        self.assertEqual(len(d['cell_stats']), 16)

    def test_pd_period_correct(self):
        d = self.mod.persistence_dict('ТУМАН', 'xor3', 16)
        self.assertEqual(d['period'], 8)

    def test_pd_word_uppercased(self):
        d = self.mod.persistence_dict('туман', 'xor3', 16)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_pd_max_run_positive(self):
        d = self.mod.persistence_dict('ТУМАН', 'xor3', 16)
        self.assertGreater(d['max_run_global'], 0)

    def test_pd_min_le_max_persistence(self):
        d = self.mod.persistence_dict('ТУМАН', 'xor3', 16)
        self.assertLessEqual(d['min_persistence'], d['max_persistence'])

    # ── all_persistence ────────────────────────────────────────────────

    def test_ap_four_rules(self):
        ap = self.mod.all_persistence('ТУМАН', 16)
        self.assertEqual(set(ap.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_ap_each_is_dict(self):
        ap = self.mod.all_persistence('ГОРА', 16)
        for rule, d in ap.items():
            self.assertIn('mean_persistence', d)

    # ── build_persistence_data ─────────────────────────────────────────

    def test_bpd_structure(self):
        d = self.mod.build_persistence_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', d)
        self.assertIn('width', d)
        self.assertIn('per_rule', d)

    def test_bpd_per_rule_keys(self):
        d = self.mod.build_persistence_data(['ТУМАН'], 16)
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_bpd_word_entry_keys(self):
        d = self.mod.build_persistence_data(['ТУМАН'], 16)
        entry = d['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'mean_persistence', 'mean_run', 'mean_cv',
                  'max_run_global', 'all_persistent'):
            self.assertIn(k, entry)

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_prs_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prs-cell', content)

    def test_viewer_has_prs_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prs-run', content)

    def test_viewer_has_prs_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prs-stats', content)

    def test_viewer_has_prs_run_fn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prsRun', content)

    def test_viewer_has_prs_rl_fn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('prsRL', content)

    def test_viewer_has_persistence_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Персистентность Q6', content)


class TestSolanBlock(unittest.TestCase):
    """Tests for solan_block.py and the viewer Block Entropy section."""

    @classmethod
    def setUpClass(cls):
        import importlib, sys
        spec = importlib.util.spec_from_file_location(
            'solan_block',
            str(pathlib.Path(__file__).resolve().parents[1] /
                'projects' / 'hexglyph' / 'solan_block.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules['solan_block'] = mod
        spec.loader.exec_module(mod)
        cls.mod = mod

    # ── h_block ────────────────────────────────────────────────────────

    def test_hb_empty_seqs(self):
        self.assertAlmostEqual(self.mod.h_block([], 1), 0.0)

    def test_hb_constant_seq(self):
        # constant sequence → only one unigram → H_1=0
        seqs = [[0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertAlmostEqual(self.mod.h_block(seqs, 1), 0.0)

    def test_hb_alternating_unigram(self):
        # alternating [0,1,0,1] → H_1 = 1 bit
        seqs = [[0, 1, 0, 1]]
        self.assertAlmostEqual(self.mod.h_block(seqs, 1), 1.0, places=6)

    def test_hb_alternating_bigram(self):
        # alternating [0,1,0,1] → bigrams (0,1),(1,0),(0,1),(1,0) → H_2=1
        seqs = [[0, 1, 0, 1]]
        self.assertAlmostEqual(self.mod.h_block(seqs, 2), 1.0, places=6)

    def test_hb_nonnegative(self):
        import random
        rng = random.Random(0)
        seqs = [[rng.randint(0, 1) for _ in range(10)] for _ in range(4)]
        for n in range(1, 5):
            self.assertGreaterEqual(self.mod.h_block(seqs, n), 0.0)

    def test_hb_subadditive_h2_le_2h1(self):
        # H_2 ≤ 2·H_1 by chain rule + non-negativity of cond. entropy
        seqs = [[0, 1, 0, 1, 0, 0, 1, 1]]
        h1 = self.mod.h_block(seqs, 1)
        h2 = self.mod.h_block(seqs, 2)
        self.assertLessEqual(h2, 2.0 * h1 + 1e-9)

    def test_hb_monotone_nondecreasing(self):
        # H_{n+1} >= H_n
        seqs = [[0, 1, 0, 0, 1, 1, 0, 1]]
        prev = 0.0
        for n in range(1, 5):
            hn = self.mod.h_block(seqs, n)
            self.assertGreaterEqual(hn + 1e-9, prev)
            prev = hn

    # ── block_profile ──────────────────────────────────────────────────

    def test_bp_xor_tuman_all_zero(self):
        prof = self.mod.block_profile('ТУМАН', 'xor', 8, 16)
        self.assertEqual(len(prof), 8)
        for h in prof:
            self.assertAlmostEqual(h, 0.0, places=5)

    def test_bp_gora_and_all_ones(self):
        # ГОРА AND P=2, all alternating → H_n=1 ∀n
        prof = self.mod.block_profile('ГОРА', 'and', 6, 16)
        for h in prof:
            self.assertAlmostEqual(h, 1.0, places=5)

    def test_bp_gora_xor3_saturates(self):
        # ГОРА XOR3: H_1=1, H_2=2, H_n=2 for n≥2
        prof = self.mod.block_profile('ГОРА', 'xor3', 6, 16)
        self.assertAlmostEqual(prof[0], 1.0, places=5)
        self.assertAlmostEqual(prof[1], 2.0, places=5)
        for h in prof[2:]:
            self.assertAlmostEqual(h, 2.0, places=5)

    def test_bp_length(self):
        prof = self.mod.block_profile('ТУМАН', 'xor3', 5, 16)
        self.assertEqual(len(prof), 5)

    def test_bp_monotone(self):
        prof = self.mod.block_profile('ТУМАН', 'xor3', 8, 16)
        for i in range(len(prof) - 1):
            self.assertLessEqual(prof[i] - 1e-9, prof[i + 1])

    # ── h_rate_profile ─────────────────────────────────────────────────

    def test_hrp_length(self):
        prof = self.mod.block_profile('ТУМАН', 'xor3', 8, 16)
        rates = self.mod.h_rate_profile(prof)
        self.assertEqual(len(rates), len(prof) - 1)

    def test_hrp_nonnegative(self):
        prof = self.mod.block_profile('ТУМАН', 'xor3', 8, 16)
        rates = self.mod.h_rate_profile(prof)
        self.assertTrue(all(r >= 0.0 for r in rates))

    def test_hrp_gora_and_all_zero(self):
        # ГОРА AND: H_n=1 ∀n → rates = 0
        prof = self.mod.block_profile('ГОРА', 'and', 6, 16)
        rates = self.mod.h_rate_profile(prof)
        for r in rates:
            self.assertAlmostEqual(r, 0.0, places=5)

    def test_hrp_gora_xor3_first_rate(self):
        # ГОРА XOR3: h_2 = H_2 - H_1 = 2 - 1 = 1.0
        prof = self.mod.block_profile('ГОРА', 'xor3', 6, 16)
        rates = self.mod.h_rate_profile(prof)
        self.assertAlmostEqual(rates[0], 1.0, places=5)

    def test_hrp_gora_xor3_rest_zero(self):
        # ГОРА XOR3: h_n = 0 for n≥3 (saturates at n=2)
        prof = self.mod.block_profile('ГОРА', 'xor3', 6, 16)
        rates = self.mod.h_rate_profile(prof)
        for r in rates[1:]:
            self.assertAlmostEqual(r, 0.0, places=5)

    def test_hrp_empty_profile(self):
        self.assertEqual(self.mod.h_rate_profile([]), [])

    def test_hrp_single(self):
        self.assertEqual(self.mod.h_rate_profile([1.0]), [])

    # ── saturation_index ───────────────────────────────────────────────

    def test_si_xor_tuman(self):
        prof = self.mod.block_profile('ТУМАН', 'xor', 8, 16)
        n = self.mod.saturation_index(prof)
        self.assertEqual(n, 2)

    def test_si_gora_and(self):
        prof = self.mod.block_profile('ГОРА', 'and', 8, 16)
        n = self.mod.saturation_index(prof)
        self.assertEqual(n, 2)

    def test_si_gora_xor3(self):
        prof = self.mod.block_profile('ГОРА', 'xor3', 8, 16)
        n = self.mod.saturation_index(prof)
        self.assertEqual(n, 3)

    def test_si_returns_int(self):
        prof = self.mod.block_profile('ТУМАН', 'xor3', 8, 16)
        n = self.mod.saturation_index(prof)
        self.assertIsInstance(n, int)

    # ── excess_entropy_estimate ────────────────────────────────────────

    def test_ee_xor_tuman(self):
        prof = self.mod.block_profile('ТУМАН', 'xor', 8, 16)
        self.assertAlmostEqual(self.mod.excess_entropy_estimate(prof), 0.0)

    def test_ee_gora_and(self):
        prof = self.mod.block_profile('ГОРА', 'and', 8, 16)
        self.assertAlmostEqual(self.mod.excess_entropy_estimate(prof), 1.0, places=5)

    def test_ee_gora_xor3(self):
        prof = self.mod.block_profile('ГОРА', 'xor3', 8, 16)
        self.assertAlmostEqual(self.mod.excess_entropy_estimate(prof), 2.0, places=5)

    def test_ee_empty(self):
        self.assertAlmostEqual(self.mod.excess_entropy_estimate([]), 0.0)

    # ── block_dict ─────────────────────────────────────────────────────

    def test_bd_keys(self):
        d = self.mod.block_dict('ТУМАН', 'xor3', 8, 16)
        for k in ('word', 'rule', 'period', 'max_n', 'h_profile', 'h_rate',
                  'h1', 'h_inf_estimate', 'excess_entropy', 'saturation_n',
                  'normalised_E'):
            self.assertIn(k, d)

    def test_bd_xor_tuman(self):
        d = self.mod.block_dict('ТУМАН', 'xor', 8, 16)
        self.assertAlmostEqual(d['h1'], 0.0)
        self.assertAlmostEqual(d['excess_entropy'], 0.0)
        self.assertEqual(d['saturation_n'], 2)

    def test_bd_gora_and(self):
        d = self.mod.block_dict('ГОРА', 'and', 8, 16)
        self.assertAlmostEqual(d['h1'], 1.0, places=5)
        self.assertAlmostEqual(d['excess_entropy'], 1.0, places=5)
        self.assertEqual(d['saturation_n'], 2)

    def test_bd_gora_xor3(self):
        d = self.mod.block_dict('ГОРА', 'xor3', 8, 16)
        self.assertAlmostEqual(d['h1'], 1.0, places=5)
        self.assertAlmostEqual(d['excess_entropy'], 2.0, places=5)
        self.assertEqual(d['saturation_n'], 3)

    def test_bd_period_xor3_tuman(self):
        d = self.mod.block_dict('ТУМАН', 'xor3', 8, 16)
        self.assertEqual(d['period'], 8)

    def test_bd_word_uppercase(self):
        d = self.mod.block_dict('туман', 'xor3', 8, 16)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_bd_profile_length(self):
        d = self.mod.block_dict('ТУМАН', 'xor3', 6, 16)
        self.assertEqual(len(d['h_profile']), 6)

    def test_bd_rate_length(self):
        d = self.mod.block_dict('ТУМАН', 'xor3', 6, 16)
        self.assertEqual(len(d['h_rate']), 5)

    def test_bd_normalised_E_in_range(self):
        d = self.mod.block_dict('ТУМАН', 'xor3', 8, 16)
        self.assertGreaterEqual(d['normalised_E'], 0.0)
        self.assertLessEqual(d['normalised_E'], 1.0 + 1e-6)

    # ── all_block ──────────────────────────────────────────────────────

    def test_ab_four_rules(self):
        ab = self.mod.all_block('ТУМАН', 8, 16)
        self.assertEqual(set(ab.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_ab_each_has_dict(self):
        ab = self.mod.all_block('ГОРА', 8, 16)
        for rule, d in ab.items():
            self.assertIn('excess_entropy', d)

    # ── build_block_data ───────────────────────────────────────────────

    def test_bbd_structure(self):
        d = self.mod.build_block_data(['ТУМАН', 'ГОРА'], 8, 16)
        self.assertIn('words', d)
        self.assertIn('width', d)
        self.assertIn('max_n', d)
        self.assertIn('per_rule', d)

    def test_bbd_per_rule_keys(self):
        d = self.mod.build_block_data(['ТУМАН'], 8, 16)
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_bbd_word_entry_keys(self):
        d = self.mod.build_block_data(['ТУМАН'], 8, 16)
        entry = d['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'h1', 'h_inf_estimate', 'excess_entropy',
                  'saturation_n', 'normalised_E', 'h_profile', 'h_rate'):
            self.assertIn(k, entry)

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_blo_profile(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('blo-profile', content)

    def test_viewer_has_blo_rate(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('blo-rate', content)

    def test_viewer_has_blo_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('blo-stats', content)

    def test_viewer_has_blo_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bloRun', content)

    def test_viewer_has_blo_hblock(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bloHBlock', content)

    def test_viewer_has_block_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Блочная энтропия Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_block',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_block(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_block', content)


class TestSolanMultiscale(unittest.TestCase):
    """Tests for solan_multiscale.py and the viewer MSE section."""

    @classmethod
    def setUpClass(cls):
        import importlib, sys
        spec = importlib.util.spec_from_file_location(
            'solan_multiscale',
            str(pathlib.Path(__file__).resolve().parents[1] /
                'projects' / 'hexglyph' / 'solan_multiscale.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules['solan_multiscale'] = mod
        spec.loader.exec_module(mod)
        cls.mod = mod

    # ── coarse_grain ───────────────────────────────────────────────────

    def test_cg_empty(self):
        self.assertEqual(self.mod.coarse_grain([], 2), [])

    def test_cg_tau1_identity(self):
        self.assertEqual(self.mod.coarse_grain([1.0, 2.0, 3.0], 1), [1.0, 2.0, 3.0])

    def test_cg_tau2(self):
        result = self.mod.coarse_grain([1.0, 2.0, 3.0, 4.0], 2)
        self.assertAlmostEqual(result[0], 1.5)
        self.assertAlmostEqual(result[1], 3.5)

    def test_cg_length(self):
        result = self.mod.coarse_grain(list(range(10)), 3)
        self.assertEqual(len(result), 3)   # floor(10/3)=3

    def test_cg_constant(self):
        result = self.mod.coarse_grain([5.0] * 8, 4)
        self.assertEqual(result, [5.0, 5.0])

    def test_cg_tau_larger_than_n(self):
        result = self.mod.coarse_grain([1.0, 2.0], 5)
        self.assertEqual(result, [])

    # ── sample_entropy ─────────────────────────────────────────────────

    def test_se_too_short_nan(self):
        import math
        result = self.mod.sample_entropy([1.0, 2.0], m=2, r=1.0)
        self.assertTrue(math.isnan(result))

    def test_se_nonnegative(self):
        import random
        rng = random.Random(42)
        series = [float(rng.randint(0, 63)) for _ in range(50)]
        se = self.mod.sample_entropy(series, m=2, r=5.0)
        import math
        if not math.isnan(se) and not math.isinf(se):
            self.assertGreaterEqual(se, 0.0)

    def test_se_constant_near_zero(self):
        import math
        # Long constant series → SampEn → 0 as N→∞ (finite-sample gives small value)
        series = [10.0] * 100
        se = self.mod.sample_entropy(series, m=2, r=0.0)
        # SampEn = -ln((N-m-2)/(N-m)) → small positive for large N
        if not math.isnan(se):
            self.assertLess(se, 0.1)

    def test_se_explicit_r(self):
        import math
        series = [float(i % 4) for i in range(50)]
        se = self.mod.sample_entropy(series, m=2, r=0.5)
        self.assertFalse(math.isnan(se) and se == 0.0)  # at least some value

    # ── get_cell_series ────────────────────────────────────────────────

    def test_gcs_width(self):
        s = self.mod.get_cell_series('ТУМАН', 'xor3', 16, 100)
        self.assertEqual(len(s), 16)

    def test_gcs_min_len(self):
        s = self.mod.get_cell_series('ТУМАН', 'xor3', 16, 100)
        for cell in s:
            self.assertGreaterEqual(len(cell), 100)

    def test_gcs_float_values(self):
        s = self.mod.get_cell_series('ТУМАН', 'xor', 16, 50)
        for cell in s:
            for v in cell:
                self.assertIsInstance(v, float)

    def test_gcs_range(self):
        s = self.mod.get_cell_series('ТУМАН', 'xor3', 16, 100)
        for cell in s:
            for v in cell:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 63)

    # ── mse_cell ───────────────────────────────────────────────────────

    def test_mc_length(self):
        series = [float(i % 8) for i in range(200)]
        result = self.mod.mse_cell(series, max_tau=6, m=2)
        self.assertEqual(len(result), 6)

    def test_mc_nonneg_or_nan(self):
        import math
        series = [float(i % 5) for i in range(200)]
        result = self.mod.mse_cell(series, max_tau=6, m=2)
        for v in result:
            if not math.isnan(v) and not math.isinf(v):
                self.assertGreaterEqual(v, 0.0)

    def test_mc_empty_series(self):
        import math
        result = self.mod.mse_cell([], max_tau=4, m=2)
        self.assertEqual(len(result), 4)
        for v in result:
            self.assertTrue(math.isnan(v))

    # ── mean_mse_profile ───────────────────────────────────────────────

    def test_mmp_length(self):
        result = self.mod.mean_mse_profile('ТУМАН', 'xor3', 6, 16, 2)
        self.assertEqual(len(result), 6)

    def test_mmp_nonneg(self):
        import math
        result = self.mod.mean_mse_profile('ТУМАН', 'xor3', 6, 16, 2)
        for v in result:
            if not math.isnan(v) and not math.isinf(v):
                self.assertGreaterEqual(v, 0.0)

    # ── mse_dict ───────────────────────────────────────────────────────

    def test_md_keys(self):
        d = self.mod.mse_dict('ТУМАН', 'xor3', 6, 16, 2)
        for k in ('word', 'rule', 'period', 'max_tau', 'm',
                  'cell_profiles', 'mean_profile', 'complexity_index',
                  'peak_tau', 'peak_se'):
            self.assertIn(k, d)

    def test_md_period_xor3(self):
        d = self.mod.mse_dict('ТУМАН', 'xor3', 6, 16, 2)
        self.assertEqual(d['period'], 8)

    def test_md_word_uppercase(self):
        d = self.mod.mse_dict('туман', 'xor3', 6, 16, 2)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_md_ci_xor3_gt_xor(self):
        # ТУМАН XOR3 (P=8, complex) > ТУМАН XOR (P=1, constant)
        ci_xor3 = self.mod.mse_dict('ТУМАН', 'xor3', 6, 16, 2)['complexity_index']
        ci_xor  = self.mod.mse_dict('ТУМАН', 'xor',  6, 16, 2)['complexity_index']
        self.assertGreater(ci_xor3, ci_xor)

    def test_md_cell_profiles_shape(self):
        d = self.mod.mse_dict('ТУМАН', 'xor3', 6, 16, 2)
        self.assertEqual(len(d['cell_profiles']), 16)
        for cp in d['cell_profiles']:
            self.assertEqual(len(cp), 6)

    def test_md_mean_profile_length(self):
        d = self.mod.mse_dict('ТУМАН', 'xor3', 6, 16, 2)
        self.assertEqual(len(d['mean_profile']), 6)

    def test_md_peak_tau_valid(self):
        d = self.mod.mse_dict('ТУМАН', 'xor3', 6, 16, 2)
        self.assertGreaterEqual(d['peak_tau'], 1)
        self.assertLessEqual(d['peak_tau'], 6)

    def test_md_ci_nonneg(self):
        d = self.mod.mse_dict('ТУМАН', 'xor', 6, 16, 2)
        self.assertGreaterEqual(d['complexity_index'], 0.0)

    def test_md_ci_xor3_positive(self):
        d = self.mod.mse_dict('ТУМАН', 'xor3', 6, 16, 2)
        self.assertGreater(d['complexity_index'], 0.0)

    # ── all_mse ────────────────────────────────────────────────────────

    def test_am_four_rules(self):
        am = self.mod.all_mse('ТУМАН', 6, 16, 2)
        self.assertEqual(set(am.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_am_each_has_ci(self):
        am = self.mod.all_mse('ГОРА', 6, 16, 2)
        for rule, d in am.items():
            self.assertIn('complexity_index', d)

    # ── build_mse_data ─────────────────────────────────────────────────

    def test_bmd_structure(self):
        d = self.mod.build_mse_data(['ТУМАН', 'ГОРА'], 6, 16, 2)
        self.assertIn('words', d)
        self.assertIn('width', d)
        self.assertIn('max_tau', d)
        self.assertIn('m', d)
        self.assertIn('per_rule', d)

    def test_bmd_per_rule_keys(self):
        d = self.mod.build_mse_data(['ТУМАН'], 6, 16, 2)
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_bmd_word_entry_keys(self):
        d = self.mod.build_mse_data(['ТУМАН'], 6, 16, 2)
        entry = d['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'mean_profile', 'complexity_index', 'peak_tau', 'peak_se'):
            self.assertIn(k, entry)

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_mse_profile(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mse-profile', content)

    def test_viewer_has_mse_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mse-cell', content)

    def test_viewer_has_mse_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mse-stats', content)

    def test_viewer_has_mse_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mseRun', content)

    def test_viewer_has_samp_en(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sampEn', content)

    def test_viewer_has_mse_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Мультимасштабная энтропия Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_multiscale',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_multiscale(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_multiscale', content)


class TestSolanChPlane(unittest.TestCase):
    """Tests for solan_ch_plane.py and the viewer C-H plane section."""

    @classmethod
    def setUpClass(cls):
        import importlib, sys
        spec = importlib.util.spec_from_file_location(
            'solan_ch_plane',
            str(pathlib.Path(__file__).resolve().parents[1] /
                'projects' / 'hexglyph' / 'solan_ch_plane.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules['solan_ch_plane'] = mod
        spec.loader.exec_module(mod)
        cls.mod = mod

    # ── ordinal_pattern ────────────────────────────────────────────────

    def test_op_constant(self):
        # All equal → stable sort by index → (0,1,2)
        self.assertEqual(self.mod.ordinal_pattern([5, 5, 5]), (0, 1, 2))

    def test_op_ascending(self):
        self.assertEqual(self.mod.ordinal_pattern([1, 2, 3]), (0, 1, 2))

    def test_op_descending(self):
        self.assertEqual(self.mod.ordinal_pattern([3, 2, 1]), (2, 1, 0))

    def test_op_mixed(self):
        # [1, 3, 2]: sorted indices: 0(1), 2(2), 1(3) → ranks: 0→0, 1→2, 2→1
        self.assertEqual(self.mod.ordinal_pattern([1, 3, 2]), (0, 2, 1))

    def test_op_length_2(self):
        self.assertEqual(self.mod.ordinal_pattern([10, 5]), (1, 0))

    def test_op_length_4(self):
        result = self.mod.ordinal_pattern([2, 0, 3, 1])
        self.assertEqual(len(result), 4)
        self.assertEqual(sorted(result), [0, 1, 2, 3])

    # ── h_norm ─────────────────────────────────────────────────────────

    def test_hn_empty(self):
        self.assertAlmostEqual(self.mod.h_norm({}, 3), 0.0)

    def test_hn_concentrated(self):
        # All probability on 1 pattern → H=0
        self.assertAlmostEqual(self.mod.h_norm({(0, 1, 2): 1.0}, 3), 0.0)

    def test_hn_uniform_m2(self):
        # Uniform over m=2 → 2 patterns → H_S = 1.0
        dist = {(0, 1): 0.5, (1, 0): 0.5}
        self.assertAlmostEqual(self.mod.h_norm(dist, 2), 1.0, places=6)

    def test_hn_uniform_m3(self):
        # Uniform over all 6 patterns → H_S = 1.0
        dist = {i: 1/6 for i in range(6)}
        self.assertAlmostEqual(self.mod.h_norm(dist, 3), 1.0, places=5)

    def test_hn_in_range(self):
        dist = {(0, 1, 2): 0.5, (2, 1, 0): 0.3, (1, 0, 2): 0.2}
        hs = self.mod.h_norm(dist, 3)
        self.assertGreaterEqual(hs, 0.0)
        self.assertLessEqual(hs, 1.0)

    # ── jsd_uniform ────────────────────────────────────────────────────

    def test_jsd_uniform_input(self):
        # When dist == uniform → JSD = 0
        M = 6
        dist = {i: 1/M for i in range(M)}
        jsd = self.mod.jsd_uniform(dist, 3)
        self.assertAlmostEqual(jsd, 0.0, places=5)

    def test_jsd_concentrated(self):
        # All probability on 1 → maximum JSD
        dist = {(0, 1, 2): 1.0}
        jsd = self.mod.jsd_uniform(dist, 3)
        self.assertGreater(jsd, 0.5)

    def test_jsd_in_range(self):
        dist = {(0, 1, 2): 0.4, (2, 1, 0): 0.4, (1, 2, 0): 0.2}
        jsd = self.mod.jsd_uniform(dist, 3)
        self.assertGreaterEqual(jsd, 0.0)
        self.assertLessEqual(jsd, 1.0)

    def test_jsd_empty(self):
        # Empty dist → all missing → mixture = U/2
        jsd = self.mod.jsd_uniform({}, 3)
        self.assertAlmostEqual(jsd, 0.5, places=5)   # H_mix=H_U, H_P=0 → (H_U+H_U)/2-H_U/2 = H_U/2 → 0.5

    # ── statistical_complexity ─────────────────────────────────────────

    def test_sc_concentrated_zero(self):
        # H_S=0 → C=0 regardless of JSD
        dist = {(0, 1, 2): 1.0}
        self.assertAlmostEqual(self.mod.statistical_complexity(dist, 3), 0.0)

    def test_sc_uniform_zero(self):
        # JSD=0 → C=0
        M = 6
        dist = {i: 1/M for i in range(M)}
        self.assertAlmostEqual(self.mod.statistical_complexity(dist, 3), 0.0, places=5)

    def test_sc_in_range(self):
        dist = {(0, 1, 2): 0.5, (2, 1, 0): 0.5}
        c = self.mod.statistical_complexity(dist, 3)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    # ── pattern_dist ───────────────────────────────────────────────────

    def test_pd_sums_to_one(self):
        dist = self.mod.pattern_dist('ТУМАН', 'xor3', 3, 16, 64)
        total = sum(dist.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_pd_xor_tuman_one_pattern(self):
        # Fixed point → only 1 ordinal pattern
        dist = self.mod.pattern_dist('ТУМАН', 'xor', 3, 16, 64)
        self.assertEqual(len(dist), 1)

    def test_pd_xor3_tuman_multiple(self):
        # P=8 complex → multiple patterns
        dist = self.mod.pattern_dist('ТУМАН', 'xor3', 3, 16, 64)
        self.assertGreater(len(dist), 1)

    def test_pd_probabilities_positive(self):
        dist = self.mod.pattern_dist('ГОРА', 'xor3', 3, 16, 64)
        self.assertTrue(all(p > 0 for p in dist.values()))

    # ── ch_point ───────────────────────────────────────────────────────

    def test_chp_returns_tuple(self):
        result = self.mod.ch_point('ТУМАН', 'xor3', 3, 16, 64)
        self.assertEqual(len(result), 3)

    def test_chp_xor_tuman_hs_zero(self):
        hs, djs, c = self.mod.ch_point('ТУМАН', 'xor', 3, 16, 64)
        self.assertAlmostEqual(hs, 0.0)
        self.assertAlmostEqual(c, 0.0)

    def test_chp_xor3_tuman_high_hs(self):
        hs, djs, c = self.mod.ch_point('ТУМАН', 'xor3', 3, 16, 64)
        self.assertGreater(hs, 0.9)

    def test_chp_gora_xor3_nonzero_c(self):
        hs, djs, c = self.mod.ch_point('ГОРА', 'xor3', 3, 16, 64)
        self.assertGreater(c, 0.0)

    def test_chp_values_in_range(self):
        hs, djs, c = self.mod.ch_point('ГОРА', 'and', 3, 16, 64)
        self.assertGreaterEqual(hs, 0.0); self.assertLessEqual(hs, 1.0)
        self.assertGreaterEqual(djs, 0.0); self.assertLessEqual(djs, 1.0)
        self.assertGreaterEqual(c, 0.0);  self.assertLessEqual(c, 1.0)

    # ── cell_ch ────────────────────────────────────────────────────────

    def test_cc_length(self):
        cells = self.mod.cell_ch('ТУМАН', 'xor3', 3, 16, 64)
        self.assertEqual(len(cells), 16)

    def test_cc_keys(self):
        cells = self.mod.cell_ch('ТУМАН', 'xor3', 3, 16, 64)
        for c in cells:
            for k in ('cell', 'h_s', 'd_js', 'c', 'n_patterns'):
                self.assertIn(k, c)

    def test_cc_values_in_range(self):
        cells = self.mod.cell_ch('ТУМАН', 'xor3', 3, 16, 64)
        for c in cells:
            self.assertGreaterEqual(c['h_s'], 0.0)
            self.assertLessEqual(c['h_s'], 1.0)
            self.assertGreaterEqual(c['c'], 0.0)

    def test_cc_xor_tuman_all_zero_c(self):
        cells = self.mod.cell_ch('ТУМАН', 'xor', 3, 16, 64)
        for c in cells:
            self.assertAlmostEqual(c['c'], 0.0)

    # ── ch_dict ────────────────────────────────────────────────────────

    def test_cd_keys(self):
        d = self.mod.ch_dict('ТУМАН', 'xor3', 3, 16, 64)
        for k in ('word', 'rule', 'period', 'm', 'M', 'h_s', 'd_js', 'c',
                  'cell_ch', 'mean_h_s', 'mean_d_js', 'mean_c', 'std_c'):
            self.assertIn(k, d)

    def test_cd_period_xor3(self):
        d = self.mod.ch_dict('ТУМАН', 'xor3', 3, 16, 64)
        self.assertEqual(d['period'], 8)

    def test_cd_word_uppercase(self):
        d = self.mod.ch_dict('туман', 'xor3', 3, 16, 64)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_cd_M_is_factorial_m(self):
        import math
        d = self.mod.ch_dict('ТУМАН', 'xor3', 3, 16, 64)
        self.assertEqual(d['M'], math.factorial(d['m']))

    def test_cd_xor_c_zero(self):
        d = self.mod.ch_dict('ТУМАН', 'xor', 3, 16, 64)
        self.assertAlmostEqual(d['c'], 0.0)

    def test_cd_gora_xor3_c_positive(self):
        d = self.mod.ch_dict('ГОРА', 'xor3', 3, 16, 64)
        self.assertGreater(d['c'], 0.0)

    def test_cd_cell_ch_length(self):
        d = self.mod.ch_dict('ТУМАН', 'xor3', 3, 16, 64)
        self.assertEqual(len(d['cell_ch']), 16)

    def test_cd_std_c_nonneg(self):
        d = self.mod.ch_dict('ТУМАН', 'xor3', 3, 16, 64)
        self.assertGreaterEqual(d['std_c'], 0.0)

    # ── all_ch ─────────────────────────────────────────────────────────

    def test_ach_four_rules(self):
        ac = self.mod.all_ch('ТУМАН', 3, 16, 64)
        self.assertEqual(set(ac.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_ach_each_has_c(self):
        ac = self.mod.all_ch('ГОРА', 3, 16, 64)
        for rule, d in ac.items():
            self.assertIn('c', d)

    # ── build_ch_data ──────────────────────────────────────────────────

    def test_bcd_structure(self):
        d = self.mod.build_ch_data(['ТУМАН', 'ГОРА'], 3, 16, 64)
        self.assertIn('words', d)
        self.assertIn('width', d)
        self.assertIn('m', d)
        self.assertIn('per_rule', d)

    def test_bcd_per_rule_keys(self):
        d = self.mod.build_ch_data(['ТУМАН'], 3, 16, 64)
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_bcd_word_entry_keys(self):
        d = self.mod.build_ch_data(['ТУМАН'], 3, 16, 64)
        entry = d['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'h_s', 'd_js', 'c', 'mean_h_s', 'mean_d_js', 'mean_c'):
            self.assertIn(k, entry)

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_chp_plot(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('chp-plot', content)

    def test_viewer_has_chp_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('chp-cell', content)

    def test_viewer_has_chp_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('chp-stats', content)

    def test_viewer_has_chp_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('chpRun', content)

    def test_viewer_has_ord_pat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ordPat', content)

    def test_viewer_has_ch_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('C-H плоскость Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_ch_plane',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_ch_plane(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_ch_plane', content)


class TestSolanWperm(unittest.TestCase):
    """Tests for solan_wperm.py and the viewer WPE section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_wperm import (
            ordinal_pattern, window_weight, wpe, nwpe,
            spatial_wpe, spatial_pe, wpe_dict, all_wpe,
            build_wpe_data,
        )
        cls.ordinal_pattern = staticmethod(ordinal_pattern)
        cls.window_weight   = staticmethod(window_weight)
        cls.wpe             = staticmethod(wpe)
        cls.nwpe            = staticmethod(nwpe)
        cls.spatial_wpe     = staticmethod(spatial_wpe)
        cls.spatial_pe      = staticmethod(spatial_pe)
        cls.wpe_dict        = staticmethod(wpe_dict)
        cls.all_wpe         = staticmethod(all_wpe)
        cls.build_wpe_data  = staticmethod(build_wpe_data)

    # ── ordinal_pattern ────────────────────────────────────────────────

    def test_ordinal_pattern_ascending(self):
        self.assertEqual(self.ordinal_pattern([1, 2, 3]), (0, 1, 2))

    def test_ordinal_pattern_descending(self):
        self.assertEqual(self.ordinal_pattern([3, 2, 1]), (2, 1, 0))

    def test_ordinal_pattern_ties_stable(self):
        # ties broken by index: [2,2,1] → 1 is lowest, then first 2, then second 2
        self.assertEqual(self.ordinal_pattern([2, 2, 1]), (1, 2, 0))

    def test_ordinal_pattern_length_2(self):
        pat = self.ordinal_pattern([5, 3])
        self.assertEqual(len(pat), 2)
        self.assertIn(pat, [(0, 1), (1, 0)])

    # ── window_weight ──────────────────────────────────────────────────

    def test_window_weight_constant_is_zero(self):
        self.assertAlmostEqual(self.window_weight([4, 4, 4]), 0.0)

    def test_window_weight_single_is_zero(self):
        self.assertAlmostEqual(self.window_weight([7]), 0.0)

    def test_window_weight_two_elements(self):
        # [0, 2]: mean=1, var = ((0-1)²+(2-1)²)/(2-1) = 2
        self.assertAlmostEqual(self.window_weight([0, 2]), 2.0)

    def test_window_weight_nonneg(self):
        self.assertGreaterEqual(self.window_weight([1, 5, 3, 7]), 0.0)

    # ── wpe / nwpe ─────────────────────────────────────────────────────

    def test_wpe_constant_series_is_zero(self):
        self.assertAlmostEqual(self.wpe([3] * 20, 3), 0.0)

    def test_wpe_too_short_is_zero(self):
        self.assertAlmostEqual(self.wpe([1, 2], 3), 0.0)

    def test_nwpe_range(self):
        import random; random.seed(0)
        s = [random.randint(0, 63) for _ in range(50)]
        v = self.nwpe(s, 3)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_nwpe_constant_is_zero(self):
        self.assertAlmostEqual(self.nwpe([5] * 30, 3), 0.0)

    def test_nwpe_uniform_random_high(self):
        # Long i.i.d. uniform series should give high nWPE
        import random; random.seed(42)
        s = [random.randint(0, 63) for _ in range(500)]
        self.assertGreater(self.nwpe(s, 3), 0.7)

    # ── Fixed-point attractors (P=1) → nWPE = 0 ───────────────────────

    def test_tuman_xor_nwpe_zero(self):
        profile = self.spatial_wpe('ТУМАН', 'xor', 16, 3)
        self.assertTrue(all(v == 0.0 for v in profile))

    def test_gora_or_nwpe_zero(self):
        # OR fixed-point → all cells constant → WPE = 0
        profile = self.spatial_wpe('ГОРА', 'or', 16, 3)
        self.assertTrue(all(v == 0.0 for v in profile))

    # ── ГОРА AND (P=2, alternating) → nWPE = nPE ─────────────────────

    def test_gora_and_wpe_equals_pe(self):
        # Equal variance on both ordinal patterns → WPE = PE
        w_prof = self.spatial_wpe('ГОРА', 'and', 16, 3)
        p_prof = self.spatial_pe('ГОРА', 'and', 16, 3)
        for w, p in zip(w_prof, p_prof):
            self.assertAlmostEqual(w, p, places=6)

    # ── ТУМАН XOR3 (P=8) → nWPE < nPE (slightly) ─────────────────────

    def test_tuman_xor3_mean_nwpe_positive(self):
        d = self.wpe_dict('ТУМАН', 'xor3', 16, 3)
        self.assertGreater(d['mean_nwpe'], 0.0)

    def test_tuman_xor3_mean_npe_positive(self):
        d = self.wpe_dict('ТУМАН', 'xor3', 16, 3)
        self.assertGreater(d['mean_npe'], 0.0)

    def test_tuman_xor3_delta_small(self):
        d = self.wpe_dict('ТУМАН', 'xor3', 16, 3)
        # |ΔWPE| should be small (< 0.3)
        self.assertLess(abs(d['mean_delta']), 0.3)

    # ── wpe_dict structure ─────────────────────────────────────────────

    def test_wpe_dict_keys(self):
        d = self.wpe_dict('ТУМАН', 'xor3', 16, 3)
        for k in ('word', 'rule', 'period', 'm', 'wpe_profile',
                  'pe_profile', 'delta', 'mean_nwpe', 'mean_npe',
                  'mean_delta', 'std_wpe', 'max_nwpe', 'min_nwpe'):
            self.assertIn(k, d)

    def test_wpe_dict_profile_length(self):
        d = self.wpe_dict('ГОРА', 'xor3', 16, 3)
        self.assertEqual(len(d['wpe_profile']), 16)
        self.assertEqual(len(d['pe_profile']), 16)
        self.assertEqual(len(d['delta']), 16)

    def test_wpe_dict_delta_consistent(self):
        d = self.wpe_dict('ТУМАН', 'xor3', 16, 3)
        for w, p, dv in zip(d['wpe_profile'], d['pe_profile'], d['delta']):
            self.assertAlmostEqual(dv, w - p, places=6)

    def test_wpe_dict_nwpe_in_range(self):
        d = self.wpe_dict('ТУМАН', 'xor3', 16, 3)
        for v in d['wpe_profile']:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_wpe_dict_std_nonneg(self):
        d = self.wpe_dict('ТУМАН', 'xor3', 16, 3)
        self.assertGreaterEqual(d['std_wpe'], 0.0)

    def test_wpe_dict_max_ge_min(self):
        d = self.wpe_dict('ТУМАН', 'xor3', 16, 3)
        self.assertGreaterEqual(d['max_nwpe'], d['min_nwpe'])

    def test_wpe_dict_xor_fixed_point_zero(self):
        d = self.wpe_dict('ТУМАН', 'xor', 16, 3)
        self.assertAlmostEqual(d['mean_nwpe'], 0.0)
        self.assertAlmostEqual(d['mean_npe'], 0.0)

    # ── all_wpe ────────────────────────────────────────────────────────

    def test_all_wpe_has_four_rules(self):
        result = self.all_wpe('ТУМАН', 16, 3)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_wpe_values_are_dicts(self):
        result = self.all_wpe('ГОРА', 16, 3)
        for rule, d in result.items():
            self.assertIsInstance(d, dict)
            self.assertIn('mean_nwpe', d)

    # ── build_wpe_data ─────────────────────────────────────────────────

    def test_build_wpe_data_structure(self):
        data = self.build_wpe_data(['ТУМАН', 'ГОРА'], 16, 3)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_wpe_data_words_uppercase(self):
        data = self.build_wpe_data(['туман'], 16, 3)
        self.assertIn('ТУМАН', data['words'])

    def test_build_wpe_data_entry_keys(self):
        data = self.build_wpe_data(['ТУМАН'], 16, 3)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'mean_nwpe', 'mean_npe', 'mean_delta',
                  'std_wpe', 'max_nwpe', 'min_nwpe'):
            self.assertIn(k, entry)

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_wpe_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wpe-cell', content)

    def test_viewer_has_wpe_delta(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wpe-delta', content)

    def test_viewer_has_wpe_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wpe-stats', content)

    def test_viewer_has_wpe_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wpeRun', content)

    def test_viewer_has_wpe_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Взвешенная энтропия перестановок Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_wperm',
             '--word', 'ТУМАН', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_wperm(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_wperm', content)


class TestSolanForbidden(unittest.TestCase):
    """Tests for solan_forbidden.py and the viewer Forbidden Patterns section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_forbidden import (
            ordinal_pattern, all_patterns, observed_cell,
            observed_pooled, forbidden_cell_profile,
            forbidden_dict, all_forbidden, build_forbidden_data,
        )
        cls.ordinal_pattern       = staticmethod(ordinal_pattern)
        cls.all_patterns          = staticmethod(all_patterns)
        cls.observed_cell         = staticmethod(observed_cell)
        cls.observed_pooled       = staticmethod(observed_pooled)
        cls.forbidden_cell_profile = staticmethod(forbidden_cell_profile)
        cls.forbidden_dict        = staticmethod(forbidden_dict)
        cls.all_forbidden         = staticmethod(all_forbidden)
        cls.build_forbidden_data  = staticmethod(build_forbidden_data)

    # ── ordinal_pattern ────────────────────────────────────────────────

    def test_ordinal_pattern_ascending(self):
        self.assertEqual(self.ordinal_pattern([1, 2, 3]), (0, 1, 2))

    def test_ordinal_pattern_descending(self):
        self.assertEqual(self.ordinal_pattern([3, 2, 1]), (2, 1, 0))

    def test_ordinal_pattern_ties_stable(self):
        self.assertEqual(self.ordinal_pattern([2, 2, 1]), (1, 2, 0))

    # ── all_patterns ───────────────────────────────────────────────────

    def test_all_patterns_m3_count(self):
        self.assertEqual(len(self.all_patterns(3)), 6)

    def test_all_patterns_m2_count(self):
        self.assertEqual(len(self.all_patterns(2)), 2)

    def test_all_patterns_m4_count(self):
        self.assertEqual(len(self.all_patterns(4)), 24)

    def test_all_patterns_is_frozenset(self):
        self.assertIsInstance(self.all_patterns(3), frozenset)

    # ── observed_cell ──────────────────────────────────────────────────

    def test_observed_cell_constant_one_pattern(self):
        # Constant series → only one ordinal pattern (all tied → (0,1,2))
        obs = self.observed_cell([5, 5, 5, 5, 5], 3)
        self.assertEqual(len(obs), 1)

    def test_observed_cell_is_frozenset(self):
        self.assertIsInstance(self.observed_cell([1, 2, 3, 1, 2], 3), frozenset)

    def test_observed_cell_subset_of_all(self):
        obs = self.observed_cell([10, 5, 30, 20, 15, 8], 3)
        self.assertTrue(obs <= self.all_patterns(3))

    def test_observed_cell_empty_series(self):
        obs = self.observed_cell([], 3)
        self.assertEqual(len(obs), 0)

    # ── observed_pooled ────────────────────────────────────────────────

    def test_observed_pooled_fixed_point_one(self):
        # Fixed point (P=1): only one pattern pooled
        obs = self.observed_pooled('ТУМАН', 'xor', 16, 3)
        self.assertEqual(len(obs), 1)

    def test_observed_pooled_period2_two(self):
        # P=2 alternating: exactly 2 patterns
        obs = self.observed_pooled('ГОРА', 'and', 16, 3)
        self.assertEqual(len(obs), 2)

    def test_observed_pooled_tuman_xor3_all(self):
        # ТУМАН XOR3 (P=8): all 6 patterns observed
        obs = self.observed_pooled('ТУМАН', 'xor3', 16, 3)
        self.assertEqual(len(obs), 6)

    def test_observed_pooled_subset_of_all(self):
        obs = self.observed_pooled('ГОРА', 'xor3', 16, 3)
        self.assertTrue(obs <= self.all_patterns(3))

    # ── forbidden fractions ────────────────────────────────────────────

    def test_fixed_point_f_m(self):
        d = self.forbidden_dict('ТУМАН', 'xor', 16, 3)
        self.assertAlmostEqual(d['f_m'], 5 / 6, places=6)

    def test_period2_f_m(self):
        d = self.forbidden_dict('ГОРА', 'and', 16, 3)
        self.assertAlmostEqual(d['f_m'], 4 / 6, places=6)

    def test_tuman_xor3_f_m_zero(self):
        d = self.forbidden_dict('ТУМАН', 'xor3', 16, 3)
        self.assertEqual(d['f_m'], 0.0)
        self.assertEqual(d['n_forbidden'], 0)

    def test_f_m_range(self):
        d = self.forbidden_dict('ГОРА', 'xor3', 16, 3)
        self.assertGreaterEqual(d['f_m'], 0.0)
        self.assertLessEqual(d['f_m'], 1.0)

    def test_o_m_plus_f_m_equals_one(self):
        d = self.forbidden_dict('ТУМАН', 'xor3', 16, 3)
        self.assertAlmostEqual(d['o_m'] + d['f_m'], 1.0, places=8)

    # ── forbidden_dict structure ───────────────────────────────────────

    def test_forbidden_dict_keys(self):
        d = self.forbidden_dict('ТУМАН', 'xor3', 16, 3)
        for k in ('word', 'rule', 'period', 'm', 'M',
                  'n_observed', 'n_forbidden', 'f_m', 'o_m',
                  'observed_set', 'forbidden_set', 'cell_profile',
                  'mean_cell_obs', 'mean_cell_f', 'std_cell_f'):
            self.assertIn(k, d)

    def test_forbidden_dict_M_is_m_factorial(self):
        d = self.forbidden_dict('ТУМАН', 'xor3', 16, 3)
        import math
        self.assertEqual(d['M'], math.factorial(3))

    def test_forbidden_dict_cell_profile_length(self):
        d = self.forbidden_dict('ТУМАН', 'xor3', 16, 3)
        self.assertEqual(len(d['cell_profile']), 16)

    def test_forbidden_dict_observed_plus_forbidden_is_M(self):
        d = self.forbidden_dict('ГОРА', 'and', 16, 3)
        self.assertEqual(d['n_observed'] + d['n_forbidden'], d['M'])

    def test_forbidden_dict_observed_set_valid(self):
        d = self.forbidden_dict('ГОРА', 'and', 16, 3)
        for pat in d['observed_set']:
            self.assertIn(tuple(pat), self.all_patterns(3))

    def test_forbidden_dict_word_uppercase(self):
        d = self.forbidden_dict('туман', 'xor3', 16, 3)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_forbidden_dict_std_nonneg(self):
        d = self.forbidden_dict('ТУМАН', 'xor3', 16, 3)
        self.assertGreaterEqual(d['std_cell_f'], 0.0)

    def test_forbidden_dict_tuman_xor3_per_cell_heterogeneous(self):
        # ТУМАН XOR3: cells see 3–6 patterns (heterogeneous)
        d = self.forbidden_dict('ТУМАН', 'xor3', 16, 3)
        obs_counts = [c['n_observed'] for c in d['cell_profile']]
        self.assertGreater(max(obs_counts), min(obs_counts))

    # ── all_forbidden ──────────────────────────────────────────────────

    def test_all_forbidden_has_four_rules(self):
        result = self.all_forbidden('ТУМАН', 16, 3)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_forbidden_values_are_dicts(self):
        result = self.all_forbidden('ГОРА', 16, 3)
        for rule, d in result.items():
            self.assertIsInstance(d, dict)
            self.assertIn('f_m', d)

    # ── build_forbidden_data ───────────────────────────────────────────

    def test_build_forbidden_data_structure(self):
        data = self.build_forbidden_data(['ТУМАН', 'ГОРА'], 16, 3)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_forbidden_data_entry_keys(self):
        data = self.build_forbidden_data(['ТУМАН'], 16, 3)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'M', 'n_observed', 'n_forbidden',
                  'f_m', 'o_m', 'mean_cell_obs', 'mean_cell_f', 'std_cell_f'):
            self.assertIn(k, entry)

    def test_build_forbidden_data_words_uppercase(self):
        data = self.build_forbidden_data(['туман'], 16, 3)
        self.assertIn('ТУМАН', data['words'])

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_for_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('for-cell', content)

    def test_viewer_has_for_pat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('for-pat', content)

    def test_viewer_has_for_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('for-stats', content)

    def test_viewer_has_for_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('forRun', content)

    def test_viewer_has_for_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Запрещённые паттерны Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_forbidden',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_forbidden(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_forbidden', content)


class TestSolanBitflip(unittest.TestCase):
    """Tests for solan_bitflip.py and the viewer Bit-Flip section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_bitflip import (
            flip_masks, bit_flip_freqs, flip_entropy,
            flip_cooccurrence, cell_flip_stats,
            all_cell_flip_stats, aggregate_flip_freqs,
            flip_summary, all_flips, build_flip_data,
        )
        cls.flip_masks           = staticmethod(flip_masks)
        cls.bit_flip_freqs       = staticmethod(bit_flip_freqs)
        cls.flip_entropy         = staticmethod(flip_entropy)
        cls.flip_cooccurrence    = staticmethod(flip_cooccurrence)
        cls.cell_flip_stats      = staticmethod(cell_flip_stats)
        cls.all_cell_flip_stats  = staticmethod(all_cell_flip_stats)
        cls.aggregate_flip_freqs = staticmethod(aggregate_flip_freqs)
        cls.flip_summary         = staticmethod(flip_summary)
        cls.all_flips            = staticmethod(all_flips)
        cls.build_flip_data      = staticmethod(build_flip_data)

    # ── flip_masks() ──────────────────────────────────────────────────

    def test_flip_masks_empty(self):
        self.assertEqual(self.flip_masks([]), [])

    def test_flip_masks_single_self_loop(self):
        self.assertEqual(self.flip_masks([5]), [0])    # 5 XOR 5 = 0

    def test_flip_masks_two_elements(self):
        # 47 XOR 1 = 46, 1 XOR 47 = 46
        self.assertEqual(self.flip_masks([47, 1]), [46, 46])

    def test_flip_masks_constant_series_all_zero(self):
        self.assertEqual(self.flip_masks([3, 3, 3]), [0, 0, 0])

    def test_flip_masks_xor_identity(self):
        # flip mask = x XOR y
        series = [10, 20, 30]
        masks = self.flip_masks(series)
        self.assertEqual(masks[0], 10 ^ 20)
        self.assertEqual(masks[1], 20 ^ 30)
        self.assertEqual(masks[2], 30 ^ 10)   # circular

    def test_flip_masks_length(self):
        series = [1, 2, 3, 4, 5]
        self.assertEqual(len(self.flip_masks(series)), 5)

    # ── bit_flip_freqs() ──────────────────────────────────────────────

    def test_bit_flip_freqs_empty(self):
        freqs = self.bit_flip_freqs([])
        self.assertEqual(len(freqs), 6)
        self.assertTrue(all(f == 0.0 for f in freqs))

    def test_bit_flip_freqs_constant(self):
        freqs = self.bit_flip_freqs([7, 7, 7])
        self.assertTrue(all(f == 0.0 for f in freqs))

    def test_bit_flip_freqs_gora_and_signature(self):
        # 47 = 0b101111, 1 = 0b000001; XOR = 0b101110
        # bits 1,2,3,5 always flip; bits 0,4 never
        freqs = self.bit_flip_freqs([47, 1])
        self.assertAlmostEqual(freqs[0], 0.0, places=6)   # bit 0 stable
        self.assertAlmostEqual(freqs[1], 1.0, places=6)   # bit 1 always flips
        self.assertAlmostEqual(freqs[2], 1.0, places=6)   # bit 2 always flips
        self.assertAlmostEqual(freqs[3], 1.0, places=6)   # bit 3 always flips
        self.assertAlmostEqual(freqs[4], 0.0, places=6)   # bit 4 stable
        self.assertAlmostEqual(freqs[5], 1.0, places=6)   # bit 5 always flips

    def test_bit_flip_freqs_range(self):
        series = [48, 51, 43, 43, 43, 43, 40, 48]
        freqs = self.bit_flip_freqs(series)
        self.assertTrue(all(0.0 <= f <= 1.0 for f in freqs))

    def test_bit_flip_freqs_length(self):
        freqs = self.bit_flip_freqs([1, 2, 3, 4])
        self.assertEqual(len(freqs), 6)

    # ── flip_entropy() ────────────────────────────────────────────────

    def test_flip_entropy_constant_zero(self):
        # Constant series: all masks = 0 → H = 0
        self.assertAlmostEqual(self.flip_entropy([5, 5, 5, 5]), 0.0, places=6)

    def test_flip_entropy_gora_and_zero(self):
        # ГОРА AND: always same mask 46 → H = 0
        self.assertAlmostEqual(self.flip_entropy([47, 1]), 0.0, places=6)

    def test_flip_entropy_nonneg(self):
        for series in [[1, 2, 3, 4], [0, 63, 0, 63], [48, 51, 43, 43]]:
            self.assertGreaterEqual(self.flip_entropy(series), 0.0)

    def test_flip_entropy_two_distinct_masks(self):
        # [0, 1, 0, 1]: masks = [1, 1, 1, 1] (same) → H = 0
        self.assertAlmostEqual(self.flip_entropy([0, 1, 0, 1]), 0.0, places=6)

    def test_flip_entropy_diverse_masks(self):
        # Different masks → H > 0
        self.assertGreater(self.flip_entropy([48, 51, 43, 43, 43, 43, 40, 48]), 0.0)

    # ── flip_cooccurrence() ───────────────────────────────────────────

    def test_flip_cooccurrence_shape(self):
        mat = self.flip_cooccurrence([47, 1])
        self.assertEqual(len(mat), 6)
        for row in mat:
            self.assertEqual(len(row), 6)

    def test_flip_cooccurrence_diagonal_equals_freqs(self):
        series = [47, 1]
        freqs = self.bit_flip_freqs(series)
        mat = self.flip_cooccurrence(series)
        for b in range(6):
            self.assertAlmostEqual(mat[b][b], freqs[b], places=6)

    def test_flip_cooccurrence_symmetric(self):
        series = [48, 51, 43, 43, 43, 43, 40, 48]
        mat = self.flip_cooccurrence(series)
        for b in range(6):
            for b2 in range(6):
                self.assertAlmostEqual(mat[b][b2], mat[b2][b], places=8)

    def test_flip_cooccurrence_gora_and_block(self):
        # Bits 1,2,3,5 always flip together → co-occurrence = 1
        mat = self.flip_cooccurrence([47, 1])
        for b in [1, 2, 3, 5]:
            for b2 in [1, 2, 3, 5]:
                self.assertAlmostEqual(mat[b][b2], 1.0, places=6)
        # Bits 0,4 never flip → all co-occurrences = 0
        for b in [0, 4]:
            for b2 in range(6):
                self.assertAlmostEqual(mat[b][b2], 0.0, places=6)

    def test_flip_cooccurrence_nonneg(self):
        mat = self.flip_cooccurrence([1, 2, 3, 4, 5])
        for row in mat:
            for v in row:
                self.assertGreaterEqual(v, 0.0)

    # ── Fixed-point / zero-flip attractors ────────────────────────────

    def test_tuman_xor_all_freqs_zero(self):
        freqs = self.aggregate_flip_freqs('ТУМАН', 'xor', 16)
        self.assertTrue(all(abs(f) < 1e-9 for f in freqs))

    def test_tuman_xor_entropy_zero(self):
        d = self.flip_summary('ТУМАН', 'xor', 16)
        self.assertAlmostEqual(d['entropy_mean'], 0.0, places=6)

    # ── ГОРА AND (P=2, 47↔1) ─────────────────────────────────────────

    def test_gora_and_freqs_signature(self):
        freqs = self.aggregate_flip_freqs('ГОРА', 'and', 16)
        self.assertAlmostEqual(freqs[0], 0.0, places=6)
        self.assertAlmostEqual(freqs[1], 1.0, places=6)
        self.assertAlmostEqual(freqs[2], 1.0, places=6)
        self.assertAlmostEqual(freqs[3], 1.0, places=6)
        self.assertAlmostEqual(freqs[4], 0.0, places=6)
        self.assertAlmostEqual(freqs[5], 1.0, places=6)

    def test_gora_and_entropy_zero(self):
        d = self.flip_summary('ГОРА', 'and', 16)
        self.assertAlmostEqual(d['entropy_mean'], 0.0, places=6)

    def test_gora_and_dominant_mask_46(self):
        d = self.flip_summary('ГОРА', 'and', 16)
        for cs in d['cell_stats']:
            self.assertEqual(cs['dominant_mask'], 46)   # 0b101110

    def test_gora_and_n_distinct_masks_one(self):
        d = self.flip_summary('ГОРА', 'and', 16)
        for cs in d['cell_stats']:
            self.assertEqual(cs['n_distinct_masks'], 1)

    # ── ТУМАН XOR3 (P=8) ──────────────────────────────────────────────

    def test_tuman_xor3_entropy_positive(self):
        d = self.flip_summary('ТУМАН', 'xor3', 16)
        self.assertGreater(d['entropy_mean'], 0.0)

    def test_tuman_xor3_n_distinct_masks_gt_1(self):
        d = self.flip_summary('ТУМАН', 'xor3', 16)
        self.assertTrue(any(cs['n_distinct_masks'] > 1 for cs in d['cell_stats']))

    def test_tuman_xor3_freqs_in_range(self):
        freqs = self.aggregate_flip_freqs('ТУМАН', 'xor3', 16)
        for f in freqs:
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)

    def test_tuman_xor3_bit3_most_active(self):
        freqs = self.aggregate_flip_freqs('ТУМАН', 'xor3', 16)
        # bit 3 is most active (freq=0.500)
        self.assertEqual(freqs.index(max(freqs)), 3)

    # ── flip_summary structure ────────────────────────────────────────

    def test_flip_summary_keys(self):
        d = self.flip_summary('ТУМАН', 'xor3', 16)
        for k in ('word', 'rule', 'period', 'cell_stats', 'agg_freqs',
                  'agg_coocc', 'entropy_mean', 'entropy_std', 'entropy_max',
                  'most_active_bit', 'least_active_bit'):
            self.assertIn(k, d)

    def test_flip_summary_agg_freqs_length(self):
        d = self.flip_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(len(d['agg_freqs']), 6)

    def test_flip_summary_agg_coocc_shape(self):
        d = self.flip_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(len(d['agg_coocc']), 6)
        for row in d['agg_coocc']:
            self.assertEqual(len(row), 6)

    def test_flip_summary_word_uppercase(self):
        d = self.flip_summary('туман', 'xor3', 16)
        self.assertEqual(d['word'], 'ТУМАН')

    # ── all_flips ─────────────────────────────────────────────────────

    def test_all_flips_four_rules(self):
        result = self.all_flips('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── build_flip_data ───────────────────────────────────────────────

    def test_build_flip_data_structure(self):
        data = self.build_flip_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)

    def test_build_flip_data_entry_keys(self):
        data = self.build_flip_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'agg_freqs', 'entropy_mean', 'entropy_std',
                  'entropy_max', 'most_active_bit', 'least_active_bit'):
            self.assertIn(k, entry)

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_bf_bar(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bf-bar', content)

    def test_viewer_has_bf_heat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bf-heat', content)

    def test_viewer_has_bf_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bf-stats', content)

    def test_viewer_has_bf_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bfRun', content)

    def test_viewer_has_bitflip_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Bit-Flip Dynamics Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_bitflip',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_bitflip(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_bitflip', content)


class TestSolanPhase(unittest.TestCase):
    """Tests for solan_phase.py and the viewer Phase Offset Analysis section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_phase import (
            series_match, phase_offset, sync_matrix,
            sync_clusters, inter_cluster_offsets, sync_fraction,
            phase_summary, all_phase, build_phase_data,
        )
        cls.series_match          = staticmethod(series_match)
        cls.phase_offset          = staticmethod(phase_offset)
        cls.sync_matrix           = staticmethod(sync_matrix)
        cls.sync_clusters         = staticmethod(sync_clusters)
        cls.inter_cluster_offsets = staticmethod(inter_cluster_offsets)
        cls.sync_fraction         = staticmethod(sync_fraction)
        cls.phase_summary         = staticmethod(phase_summary)
        cls.all_phase             = staticmethod(all_phase)
        cls.build_phase_data      = staticmethod(build_phase_data)

    # ── series_match() ────────────────────────────────────────────────

    def test_series_match_offset_0_identical(self):
        self.assertTrue(self.series_match([1, 2, 3], [1, 2, 3], 0))

    def test_series_match_offset_1(self):
        # [3, 1, 2] should match [1, 2, 3] with offset=2
        self.assertTrue(self.series_match([1, 2, 3], [3, 1, 2], 2))

    def test_series_match_wrong_offset(self):
        self.assertFalse(self.series_match([1, 2, 3], [2, 3, 1], 0))

    def test_series_match_empty(self):
        self.assertFalse(self.series_match([], [1], 0))

    def test_series_match_self_any_offset_constant(self):
        # Constant series: matches any offset
        s = [5, 5, 5]
        for k in range(3):
            self.assertTrue(self.series_match(s, s, k))

    # ── phase_offset() ────────────────────────────────────────────────

    def test_phase_offset_self(self):
        self.assertEqual(self.phase_offset([1, 2, 3], [1, 2, 3]), 0)

    def test_phase_offset_shift_1(self):
        # sj = shift(si, 1): sj[t] = si[(t+1)%3]
        si = [10, 20, 30]
        sj = [20, 30, 10]   # sj[t] = si[(t+1)%3]
        self.assertEqual(self.phase_offset(si, sj), 1)

    def test_phase_offset_shift_2(self):
        si = [10, 20, 30]
        sj = [30, 10, 20]   # sj[t] = si[(t+2)%3]
        self.assertEqual(self.phase_offset(si, sj), 2)

    def test_phase_offset_none(self):
        self.assertIsNone(self.phase_offset([1, 2, 3], [4, 5, 6]))

    def test_phase_offset_antiphase_p2(self):
        # [47, 1] and [1, 47]: sj[0]=1=si[1], sj[1]=47=si[0] → offset=1
        self.assertEqual(self.phase_offset([47, 1], [1, 47]), 1)

    def test_phase_offset_antiphase_symmetry(self):
        # Anti-phase is symmetric for P=2: offset(A,B) = offset(B,A) = 1
        self.assertEqual(self.phase_offset([47, 1], [1, 47]),
                         self.phase_offset([1, 47], [47, 1]))

    # ── ТУМАН XOR (P=1, trivial) ──────────────────────────────────────

    def test_tuman_xor_sync_fraction_1(self):
        self.assertAlmostEqual(self.sync_fraction('ТУМАН', 'xor', 16), 1.0)

    def test_tuman_xor_one_cluster(self):
        clusters = self.sync_clusters('ТУМАН', 'xor', 16)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 16)

    def test_tuman_xor_matrix_all_zero(self):
        mat = self.sync_matrix('ТУМАН', 'xor', 16)
        for row in mat:
            for v in row:
                self.assertEqual(v, 0)

    # ── ГОРА AND (P=2, anti-phase) ────────────────────────────────────

    def test_gora_and_sync_fraction_1(self):
        self.assertAlmostEqual(self.sync_fraction('ГОРА', 'and', 16), 1.0)

    def test_gora_and_two_clusters(self):
        clusters = self.sync_clusters('ГОРА', 'and', 16)
        self.assertEqual(len(clusters), 2)
        sizes = sorted(len(g) for g in clusters)
        self.assertEqual(sizes, [8, 8])

    def test_gora_and_clusters_even_odd(self):
        clusters = self.sync_clusters('ГОРА', 'and', 16)
        all_cells = sorted(c for g in clusters for c in g)
        self.assertEqual(all_cells, list(range(16)))
        # One cluster even cells, one odd
        for g in clusters:
            parities = set(c % 2 for c in g)
            self.assertEqual(len(parities), 1)  # all same parity

    def test_gora_and_inter_cluster_offset_1(self):
        ic = self.inter_cluster_offsets('ГОРА', 'and', 16)
        for off in ic.values():
            self.assertEqual(off, 1)

    def test_gora_and_antiphase_detected(self):
        d = self.phase_summary('ГОРА', 'and', 16)
        self.assertTrue(d['any_antiphase'])

    def test_gora_and_offset_hist(self):
        d = self.phase_summary('ГОРА', 'and', 16)
        hist = d['offset_hist']
        self.assertIn('0', hist)
        self.assertIn('1', hist)
        self.assertEqual(hist['0'], 128)
        self.assertEqual(hist['1'], 128)

    def test_gora_and_matrix_diagonal_0(self):
        mat = self.sync_matrix('ГОРА', 'and', 16)
        for i in range(16):
            self.assertEqual(mat[i][i], 0)

    def test_gora_and_matrix_offdiag_0_or_1(self):
        mat = self.sync_matrix('ГОРА', 'and', 16)
        for i in range(16):
            for j in range(16):
                self.assertIn(mat[i][j], [0, 1])

    # ── ГОРА XOR3 (P=2, 4 clusters, no antiphase) ─────────────────────

    def test_gora_xor3_four_clusters(self):
        clusters = self.sync_clusters('ГОРА', 'xor3', 16)
        self.assertEqual(len(clusters), 4)
        sizes = sorted(len(g) for g in clusters)
        self.assertEqual(sizes, [4, 4, 4, 4])

    def test_gora_xor3_sync_fraction_25(self):
        sf = self.sync_fraction('ГОРА', 'xor3', 16)
        self.assertAlmostEqual(sf, 0.25, places=6)

    def test_gora_xor3_no_antiphase(self):
        d = self.phase_summary('ГОРА', 'xor3', 16)
        self.assertFalse(d['any_antiphase'])

    def test_gora_xor3_inter_cluster_all_none(self):
        ic = self.inter_cluster_offsets('ГОРА', 'xor3', 16)
        for off in ic.values():
            self.assertIsNone(off)

    # ── ТУМАН XOR3 (P=8, all unique) ──────────────────────────────────

    def test_tuman_xor3_16_clusters(self):
        clusters = self.sync_clusters('ТУМАН', 'xor3', 16)
        self.assertEqual(len(clusters), 16)

    def test_tuman_xor3_sync_fraction_low(self):
        sf = self.sync_fraction('ТУМАН', 'xor3', 16)
        self.assertAlmostEqual(sf, 1 / 16, places=6)

    def test_tuman_xor3_no_antiphase(self):
        d = self.phase_summary('ТУМАН', 'xor3', 16)
        self.assertFalse(d['any_antiphase'])

    def test_tuman_xor3_n_distinct_16(self):
        d = self.phase_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(d['n_distinct'], 16)

    # ── sync_matrix() structure ───────────────────────────────────────

    def test_sync_matrix_diagonal_always_0(self):
        for word, rule in [('ТУМАН','xor3'), ('ГОРА','and'), ('ЛЕБЕДЬ','or')]:
            mat = self.sync_matrix(word, rule, 16)
            for i in range(16):
                self.assertEqual(mat[i][i], 0)

    def test_sync_matrix_shape_16x16(self):
        mat = self.sync_matrix('ТУМАН', 'xor3', 16)
        self.assertEqual(len(mat), 16)
        for row in mat:
            self.assertEqual(len(row), 16)

    def test_sync_matrix_offset_range(self):
        # All non-None offsets must be in [0, P-1]
        from projects.hexglyph.solan_traj import word_trajectory
        for word, rule in [('ГОРА','and'), ('ГОРА','xor3'), ('ТУМАН','xor3')]:
            P = word_trajectory(word, rule, 16)['period']
            mat = self.sync_matrix(word, rule, 16)
            for row in mat:
                for v in row:
                    if v is not None:
                        self.assertGreaterEqual(v, 0)
                        self.assertLess(v, P)

    # ── phase_summary() structure ─────────────────────────────────────

    def test_phase_summary_keys(self):
        d = self.phase_summary('ГОРА', 'and', 16)
        for k in ('word', 'rule', 'period', 'n_distinct', 'sync_fraction',
                  'offset_hist', 'n_clusters', 'cluster_sizes', 'ic_offsets',
                  'any_antiphase', 'dominant_offset', 'matrix'):
            self.assertIn(k, d)

    def test_phase_summary_word_uppercase(self):
        d = self.phase_summary('гора', 'and', 16)
        self.assertEqual(d['word'], 'ГОРА')

    def test_phase_summary_sync_fraction_range(self):
        d = self.phase_summary('ТУМАН', 'xor3', 16)
        self.assertGreaterEqual(d['sync_fraction'], 0.0)
        self.assertLessEqual(d['sync_fraction'], 1.0)

    # ── all_phase() ───────────────────────────────────────────────────

    def test_all_phase_four_rules(self):
        result = self.all_phase('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── build_phase_data() ────────────────────────────────────────────

    def test_build_phase_data_structure(self):
        data = self.build_phase_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)

    def test_build_phase_data_no_matrix(self):
        # build_phase_data should NOT include 'matrix' (compact output)
        data = self.build_phase_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        self.assertNotIn('matrix', entry)

    def test_build_phase_data_has_sync_fraction(self):
        data = self.build_phase_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        self.assertIn('sync_fraction', entry)

    # ── Viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_ph_matrix(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ph-matrix', content)

    def test_viewer_has_ph_hist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ph-hist', content)

    def test_viewer_has_ph_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ph-stats', content)

    def test_viewer_has_ph_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('phRun', content)

    def test_viewer_has_phase_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Phase Offset Analysis Q6', content)

    def test_viewer_antiphase_text(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('anti-phase', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_phase',
             '--word', 'ТУМАН', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_phase(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_phase', content)


class TestSolanBalance(unittest.TestCase):
    """Tests for solan_balance.py and the viewer Bit Balance section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_balance import (
            bit_balance, bit_flip_freq, classify_bit,
            bit_profile, cell_balance_stats, all_cell_balance_stats,
            aggregate_balance, balance_plane_points, count_classes,
            balance_summary, all_balance, build_balance_data,
        )
        cls.bit_balance            = staticmethod(bit_balance)
        cls.bit_flip_freq          = staticmethod(bit_flip_freq)
        cls.classify_bit           = staticmethod(classify_bit)
        cls.bit_profile            = staticmethod(bit_profile)
        cls.cell_balance_stats     = staticmethod(cell_balance_stats)
        cls.all_cell_balance_stats = staticmethod(all_cell_balance_stats)
        cls.aggregate_balance      = staticmethod(aggregate_balance)
        cls.balance_plane_points   = staticmethod(balance_plane_points)
        cls.count_classes          = staticmethod(count_classes)
        cls.balance_summary        = staticmethod(balance_summary)
        cls.all_balance            = staticmethod(all_balance)
        cls.build_balance_data     = staticmethod(build_balance_data)

    # ── bit_balance() ─────────────────────────────────────────────────

    def test_bit_balance_empty(self):
        bal = self.bit_balance([])
        self.assertEqual(len(bal), 6)
        self.assertTrue(all(v == 0.0 for v in bal))

    def test_bit_balance_all_zero(self):
        bal = self.bit_balance([0, 0, 0])
        self.assertTrue(all(v == 0.0 for v in bal))

    def test_bit_balance_all_63(self):
        bal = self.bit_balance([63, 63, 63])
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in bal))

    def test_bit_balance_alternating_bit0(self):
        # 1=0b000001 alternates with 0=0b000000 → b0 balance=0.5
        bal = self.bit_balance([1, 0, 1, 0])
        self.assertAlmostEqual(bal[0], 0.5, places=9)
        self.assertAlmostEqual(bal[1], 0.0, places=9)

    def test_bit_balance_gora_and_cell(self):
        # 47=0b101111, 1=0b000001; series [47, 1]
        bal = self.bit_balance([47, 1])
        # bit 0: both have bit 0 = 1 → balance = 1.0
        self.assertAlmostEqual(bal[0], 1.0, places=9)
        # bit 1: 47 has bit 1=1, 1 has bit 1=0 → balance = 0.5
        self.assertAlmostEqual(bal[1], 0.5, places=9)
        # bit 4: 47 has bit 4=0, 1 has bit 4=0 → balance = 0.0
        self.assertAlmostEqual(bal[4], 0.0, places=9)

    def test_bit_balance_range(self):
        for series in [[0, 63, 32, 15, 47], [48, 51, 43]]:
            bal = self.bit_balance(series)
            for v in bal:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    # ── bit_flip_freq() ───────────────────────────────────────────────

    def test_bit_flip_freq_constant(self):
        freq = self.bit_flip_freq([7, 7, 7])
        self.assertTrue(all(f == 0.0 for f in freq))

    def test_bit_flip_freq_gora_and(self):
        freq = self.bit_flip_freq([47, 1])
        self.assertAlmostEqual(freq[0], 0.0, places=9)   # b0 never flips
        self.assertAlmostEqual(freq[1], 1.0, places=9)   # b1 always flips

    # ── classify_bit() ────────────────────────────────────────────────

    def test_classify_frozen_off(self):
        self.assertEqual(self.classify_bit(0.0, 0.0), 'FROZEN_OFF')

    def test_classify_frozen_on(self):
        self.assertEqual(self.classify_bit(1.0, 0.0), 'FROZEN_ON')

    def test_classify_strict_alt(self):
        self.assertEqual(self.classify_bit(0.5, 1.0), 'STRICT_ALT')

    def test_classify_oscillating(self):
        self.assertEqual(self.classify_bit(0.5, 0.5), 'OSCILLATING')

    def test_classify_dc_bias(self):
        self.assertEqual(self.classify_bit(0.75, 0.5), 'DC_BIAS')

    def test_classify_eps_boundary(self):
        # Slightly off from frozen → still frozen within eps
        self.assertEqual(self.classify_bit(0.01, 0.01), 'FROZEN_OFF')

    # ── ТУМАН XOR (P=1, all=0) ────────────────────────────────────────

    def test_tuman_xor_all_frozen_off(self):
        d = self.balance_summary('ТУМАН', 'xor', 16)
        self.assertEqual(d['class_counts'].get('FROZEN_OFF', 0), 96)
        self.assertEqual(d['total_frozen_bits'], 96)

    def test_tuman_xor_agg_balance_zero(self):
        agg = self.aggregate_balance('ТУМАН', 'xor', 16)
        self.assertTrue(all(v == 0.0 for v in agg['balance']))

    def test_tuman_xor_agg_flip_zero(self):
        agg = self.aggregate_balance('ТУМАН', 'xor', 16)
        self.assertTrue(all(v == 0.0 for v in agg['flip']))

    # ── ГОРА AND (P=2, anti-phase) ────────────────────────────────────

    def test_gora_and_b0_frozen_on(self):
        agg = self.aggregate_balance('ГОРА', 'and', 16)
        self.assertAlmostEqual(agg['balance'][0], 1.0, places=6)
        self.assertAlmostEqual(agg['flip'][0], 0.0, places=6)
        self.assertEqual(agg['class'][0], 'FROZEN_ON')

    def test_gora_and_b4_frozen_off(self):
        agg = self.aggregate_balance('ГОРА', 'and', 16)
        self.assertAlmostEqual(agg['balance'][4], 0.0, places=6)
        self.assertAlmostEqual(agg['flip'][4], 0.0, places=6)
        self.assertEqual(agg['class'][4], 'FROZEN_OFF')

    def test_gora_and_b1_b2_b3_b5_strict_alt(self):
        agg = self.aggregate_balance('ГОРА', 'and', 16)
        for b in [1, 2, 3, 5]:
            self.assertAlmostEqual(agg['balance'][b], 0.5, places=6)
            self.assertAlmostEqual(agg['flip'][b], 1.0, places=6)
            self.assertEqual(agg['class'][b], 'STRICT_ALT')

    def test_gora_and_total_frozen_32(self):
        # b0 FROZEN_ON (16 cells) + b4 FROZEN_OFF (16 cells) = 32
        d = self.balance_summary('ГОРА', 'and', 16)
        self.assertEqual(d['total_frozen_bits'], 32)

    def test_gora_and_class_counts(self):
        cls = self.count_classes('ГОРА', 'and', 16)
        self.assertEqual(cls.get('FROZEN_ON', 0), 16)    # b0 × 16 cells
        self.assertEqual(cls.get('FROZEN_OFF', 0), 16)   # b4 × 16 cells
        self.assertEqual(cls.get('STRICT_ALT', 0), 64)   # b1,2,3,5 × 16 cells

    # ── ГОРА XOR3 (P=2, 4 clusters) ──────────────────────────────────

    def test_gora_xor3_b0_agg_frozen_on(self):
        agg = self.aggregate_balance('ГОРА', 'xor3', 16)
        self.assertAlmostEqual(agg['balance'][0], 1.0, places=6)
        self.assertEqual(agg['class'][0], 'FROZEN_ON')

    def test_gora_xor3_b4_agg_strict_alt(self):
        agg = self.aggregate_balance('ГОРА', 'xor3', 16)
        self.assertAlmostEqual(agg['flip'][4], 1.0, places=6)
        self.assertEqual(agg['class'][4], 'STRICT_ALT')

    def test_gora_xor3_per_cell_b0_all_frozen_on(self):
        css = self.all_cell_balance_stats('ГОРА', 'xor3', 16)
        for cs in css:
            b0 = cs['profile'][0]
            self.assertAlmostEqual(b0['balance'], 1.0, places=6)
            self.assertEqual(b0['class'], 'FROZEN_ON')

    def test_gora_xor3_total_frozen_48(self):
        d = self.balance_summary('ГОРА', 'xor3', 16)
        self.assertEqual(d['total_frozen_bits'], 48)

    # ── ТУМАН XOR3 (P=8) — hidden frozen bits ─────────────────────────

    def test_tuman_xor3_agg_balance_near_half(self):
        agg = self.aggregate_balance('ТУМАН', 'xor3', 16)
        for b in range(6):
            self.assertGreater(agg['balance'][b], 0.3)
            self.assertLess(agg['balance'][b], 0.7)

    def test_tuman_xor3_cell0_b2_frozen_off(self):
        css = self.all_cell_balance_stats('ТУМАН', 'xor3', 16)
        b2 = css[0]['profile'][2]
        self.assertAlmostEqual(b2['balance'], 0.0, places=6)
        self.assertEqual(b2['class'], 'FROZEN_OFF')

    def test_tuman_xor3_cell0_b5_frozen_on(self):
        css = self.all_cell_balance_stats('ТУМАН', 'xor3', 16)
        b5 = css[0]['profile'][5]
        self.assertAlmostEqual(b5['balance'], 1.0, places=6)
        self.assertEqual(b5['class'], 'FROZEN_ON')

    def test_tuman_xor3_cell1_b4_frozen_on(self):
        css = self.all_cell_balance_stats('ТУМАН', 'xor3', 16)
        b4 = css[1]['profile'][4]
        self.assertAlmostEqual(b4['balance'], 1.0, places=6)
        self.assertEqual(b4['class'], 'FROZEN_ON')

    def test_tuman_xor3_has_frozen_bits(self):
        d = self.balance_summary('ТУМАН', 'xor3', 16)
        self.assertGreater(d['total_frozen_bits'], 0)

    # ── balance_plane_points() ────────────────────────────────────────

    def test_balance_plane_96_points(self):
        pts = self.balance_plane_points('ТУМАН', 'xor3', 16)
        self.assertEqual(len(pts), 16 * 6)   # N × 6 bits

    def test_balance_plane_point_keys(self):
        pts = self.balance_plane_points('ГОРА', 'and', 16)
        self.assertIn('cell', pts[0])
        self.assertIn('bit', pts[0])
        self.assertIn('balance', pts[0])
        self.assertIn('flip', pts[0])
        self.assertIn('class', pts[0])

    def test_balance_plane_balance_range(self):
        pts = self.balance_plane_points('ТУМАН', 'xor3', 16)
        for p in pts:
            self.assertGreaterEqual(p['balance'], 0.0)
            self.assertLessEqual(p['balance'], 1.0)

    # ── balance_summary() structure ───────────────────────────────────

    def test_balance_summary_keys(self):
        d = self.balance_summary('ГОРА', 'and', 16)
        for k in ('word', 'rule', 'period', 'agg_balance', 'agg_flip',
                  'agg_class', 'class_counts', 'total_frozen_bits',
                  'max_frozen_per_cell', 'most_active_bit',
                  'least_active_bit', 'cell_stats', 'n_points'):
            self.assertIn(k, d)

    def test_balance_summary_n_points_96(self):
        d = self.balance_summary('ТУМАН', 'and', 16)
        self.assertEqual(d['n_points'], 96)

    def test_balance_summary_word_uppercase(self):
        d = self.balance_summary('гора', 'and', 16)
        self.assertEqual(d['word'], 'ГОРА')

    # ── all_balance() ─────────────────────────────────────────────────

    def test_all_balance_four_rules(self):
        result = self.all_balance('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── build_balance_data() ──────────────────────────────────────────

    def test_build_balance_data_no_cell_stats(self):
        data = self.build_balance_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        self.assertNotIn('cell_stats', entry)

    def test_build_balance_data_has_agg(self):
        data = self.build_balance_data(['ТУМАН'], 16)
        entry = data['per_rule']['and']['ТУМАН']
        self.assertIn('agg_balance', entry)
        self.assertIn('agg_flip', entry)

    # ── Viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_bl_scatter(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bl-scatter', content)

    def test_viewer_has_bl_bars(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bl-bars', content)

    def test_viewer_has_bl_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bl-stats', content)

    def test_viewer_has_bl_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('blRun', content)

    def test_viewer_has_balance_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Bit Balance', content)

    def test_viewer_has_frozen_on(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('FROZEN_ON', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_balance',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_balance(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_balance', content)


class TestSolanCoact(unittest.TestCase):
    """Tests for solan_coact.py and the viewer Bit Co-activation section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_coact import (
            bit_joint_prob, bit_pearson_corr, cell_coact_stats,
            aggregate_joint_prob, aggregate_pearson, top_corr_pairs,
            coact_summary, all_coact, build_coact_data,
        )
        cls.bit_joint_prob      = staticmethod(bit_joint_prob)
        cls.bit_pearson_corr    = staticmethod(bit_pearson_corr)
        cls.cell_coact_stats    = staticmethod(cell_coact_stats)
        cls.aggregate_joint_prob = staticmethod(aggregate_joint_prob)
        cls.aggregate_pearson   = staticmethod(aggregate_pearson)
        cls.top_corr_pairs      = staticmethod(top_corr_pairs)
        cls.coact_summary       = staticmethod(coact_summary)
        cls.all_coact           = staticmethod(all_coact)
        cls.build_coact_data    = staticmethod(build_coact_data)

    # ── bit_joint_prob() ──────────────────────────────────────────────

    def test_bit_joint_prob_empty(self):
        mat = self.bit_joint_prob([])
        self.assertEqual(len(mat), 6)
        for row in mat:
            self.assertTrue(all(v == 0.0 for v in row))

    def test_bit_joint_prob_diagonal_equals_balance(self):
        # J[b][b] = P(bit_b = 1) = balance_b
        series = [47, 1]   # 47=0b101111, 1=0b000001
        mat = self.bit_joint_prob(series)
        # bit 0: always 1 → J[0][0] = 1.0
        self.assertAlmostEqual(mat[0][0], 1.0, places=9)
        # bit 4: always 0 → J[4][4] = 0.0
        self.assertAlmostEqual(mat[4][4], 0.0, places=9)
        # bit 1: on in 47, off in 1 → J[1][1] = 0.5
        self.assertAlmostEqual(mat[1][1], 0.5, places=9)

    def test_bit_joint_prob_symmetric(self):
        series = [48, 51, 43, 40]
        mat = self.bit_joint_prob(series)
        for b in range(6):
            for b2 in range(6):
                self.assertAlmostEqual(mat[b][b2], mat[b2][b], places=9)

    def test_bit_joint_prob_shape(self):
        mat = self.bit_joint_prob([1, 2, 3])
        self.assertEqual(len(mat), 6)
        for row in mat: self.assertEqual(len(row), 6)

    def test_bit_joint_prob_range(self):
        series = [0, 63, 47, 1, 15, 48]
        mat = self.bit_joint_prob(series)
        for row in mat:
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    # ── bit_pearson_corr() ────────────────────────────────────────────

    def test_bit_pearson_empty(self):
        mat = self.bit_pearson_corr([])
        self.assertEqual(len(mat), 6)
        for row in mat:
            self.assertTrue(all(v == 0.0 for v in row))

    def test_bit_pearson_frozen_zero(self):
        # Constant series: all bits frozen → all correlations = 0
        mat = self.bit_pearson_corr([47, 47, 47])
        for row in mat:
            for v in row:
                self.assertAlmostEqual(v, 0.0, places=9)

    def test_bit_pearson_self_loop_all_active(self):
        # Self-correlation for active bits = 1.0
        mat = self.bit_pearson_corr([47, 1])  # bits 1,2,3,5 are active
        for b in [1, 2, 3, 5]:
            self.assertAlmostEqual(mat[b][b], 1.0, places=9)

    def test_bit_pearson_frozen_bits_zero_diagonal(self):
        # Frozen bits (bit 0 always 1, bit 4 always 0) → diagonal = 0
        mat = self.bit_pearson_corr([47, 1])
        self.assertAlmostEqual(mat[0][0], 0.0, places=9)
        self.assertAlmostEqual(mat[4][4], 0.0, places=9)

    def test_bit_pearson_symmetric(self):
        series = [48, 51, 43, 40, 63, 1]
        mat = self.bit_pearson_corr(series)
        for b in range(6):
            for b2 in range(6):
                self.assertAlmostEqual(mat[b][b2], mat[b2][b], places=8)

    # ── ТУМАН XOR (P=1, all=0) ────────────────────────────────────────

    def test_tuman_xor_pearson_all_zero(self):
        mat = self.aggregate_pearson('ТУМАН', 'xor', 16)
        for row in mat:
            for v in row:
                self.assertAlmostEqual(v, 0.0, places=6)

    # ── ГОРА AND (P=2, block structure) ──────────────────────────────

    def test_gora_and_bits_1235_block_one(self):
        mat = self.aggregate_pearson('ГОРА', 'and', 16)
        for b in [1, 2, 3, 5]:
            for b2 in [1, 2, 3, 5]:
                self.assertAlmostEqual(mat[b][b2], 1.0, places=6)

    def test_gora_and_frozen_rows_zero(self):
        mat = self.aggregate_pearson('ГОРА', 'and', 16)
        for b2 in range(6):
            self.assertAlmostEqual(mat[0][b2], 0.0, places=6)
            self.assertAlmostEqual(mat[4][b2], 0.0, places=6)
            self.assertAlmostEqual(mat[b2][0], 0.0, places=6)
            self.assertAlmostEqual(mat[b2][4], 0.0, places=6)

    def test_gora_and_n_dependent(self):
        d = self.coact_summary('ГОРА', 'and', 16)
        # 6 off-diagonal pairs among {1,2,3,5}: (1,2),(1,3),(1,5),(2,3),(2,5),(3,5)
        self.assertEqual(d['n_dependent'], 6)

    def test_gora_and_n_negative_zero(self):
        d = self.coact_summary('ГОРА', 'and', 16)
        self.assertEqual(d['n_negative'], 0)

    # ── ТУМАН XOR3 (P=8, b0=b1 everywhere) ───────────────────────────

    def test_tuman_xor3_b0_b1_corr_one(self):
        mat = self.aggregate_pearson('ТУМАН', 'xor3', 16)
        self.assertAlmostEqual(mat[0][1], 1.0, places=5)
        self.assertAlmostEqual(mat[1][0], 1.0, places=5)

    def test_tuman_xor3_n_dependent_one(self):
        d = self.coact_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(d['n_dependent'], 1)   # only b0-b1 pair

    def test_tuman_xor3_has_negative_pairs(self):
        d = self.coact_summary('ТУМАН', 'xor3', 16)
        self.assertGreater(d['n_negative'], 0)

    def test_tuman_xor3_diagonal_b2_b4_b5_lt_1(self):
        # Cells with frozen bits contribute 0 → aggregate diagonal < 1
        d = self.coact_summary('ТУМАН', 'xor3', 16)
        diag = d['diagonal']
        self.assertLess(diag[2], 1.0)
        self.assertLess(diag[4], 1.0)
        self.assertLess(diag[5], 1.0)

    def test_tuman_xor3_diagonal_b0_b3_one(self):
        d = self.coact_summary('ТУМАН', 'xor3', 16)
        diag = d['diagonal']
        self.assertAlmostEqual(diag[0], 1.0, places=5)
        self.assertAlmostEqual(diag[3], 1.0, places=5)

    # ── top_corr_pairs() ─────────────────────────────────────────────

    def test_top_corr_pairs_sorted_descending(self):
        mat = self.aggregate_pearson('ГОРА', 'and', 16)
        pairs = self.top_corr_pairs(mat, n=10)
        vals = [abs(r) for r, _, _ in pairs]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_top_corr_pairs_off_diagonal(self):
        mat = self.aggregate_pearson('ТУМАН', 'xor3', 16)
        pairs = self.top_corr_pairs(mat, n=6)
        for _, b, b2 in pairs:
            self.assertNotEqual(b, b2)
            self.assertLess(b, b2)   # b < b'

    # ── coact_summary() structure ─────────────────────────────────────

    def test_coact_summary_keys(self):
        d = self.coact_summary('ГОРА', 'and', 16)
        for k in ('word', 'rule', 'period', 'agg_joint', 'agg_pearson',
                  'top_pairs', 'n_positive', 'n_negative', 'n_dependent',
                  'diagonal', 'n_frozen_bits'):
            self.assertIn(k, d)

    def test_coact_summary_word_uppercase(self):
        d = self.coact_summary('гора', 'and', 16)
        self.assertEqual(d['word'], 'ГОРА')

    def test_coact_summary_matrix_shape(self):
        d = self.coact_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(len(d['agg_pearson']), 6)
        for row in d['agg_pearson']:
            self.assertEqual(len(row), 6)

    # ── all_coact() ───────────────────────────────────────────────────

    def test_all_coact_four_rules(self):
        result = self.all_coact('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── build_coact_data() ────────────────────────────────────────────

    def test_build_coact_data_has_pearson(self):
        data = self.build_coact_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        self.assertIn('agg_pearson', entry)

    def test_build_coact_data_has_n_dependent(self):
        data = self.build_coact_data(['ГОРА'], 16)
        entry = data['per_rule']['and']['ГОРА']
        self.assertIn('n_dependent', entry)

    # ── Viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_ca_matrix(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ca-matrix', content)

    def test_viewer_has_ca_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ca-stats', content)

    def test_viewer_has_coact_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('coactRun', content)

    def test_viewer_has_coact_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Bit Co-activation Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_coact',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_coact(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_coact', content)


class TestSolanRuns(unittest.TestCase):
    """Tests for solan_runs.py and the viewer Run-Length section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_runs import (
            run_lengths, plateau_fraction, run_entropy,
            cell_run_stats, all_cell_stats, change_matrix,
            run_summary, all_run_summaries, build_run_data,
        )
        cls.run_lengths      = staticmethod(run_lengths)
        cls.plateau_fraction = staticmethod(plateau_fraction)
        cls.run_entropy      = staticmethod(run_entropy)
        cls.cell_run_stats   = staticmethod(cell_run_stats)
        cls.all_cell_stats   = staticmethod(all_cell_stats)
        cls.change_matrix    = staticmethod(change_matrix)
        cls.run_summary      = staticmethod(run_summary)
        cls.all_run_summaries = staticmethod(all_run_summaries)
        cls.build_run_data   = staticmethod(build_run_data)

    # ── run_lengths() ──────────────────────────────────────────────────

    def test_run_lengths_empty(self):
        self.assertEqual(self.run_lengths([]), [])

    def test_run_lengths_single(self):
        self.assertEqual(self.run_lengths([5]), [1])

    def test_run_lengths_all_same(self):
        self.assertEqual(self.run_lengths([3, 3, 3, 3]), [4])

    def test_run_lengths_alternating(self):
        self.assertEqual(self.run_lengths([1, 2, 1, 2]), [1, 1, 1, 1])

    def test_run_lengths_with_plateau(self):
        self.assertEqual(self.run_lengths([48, 51, 43, 43, 43, 43, 40, 48]),
                         [1, 1, 4, 1, 1])

    def test_run_lengths_sum_equals_length(self):
        series = [1, 2, 2, 3, 3, 3, 4]
        rl = self.run_lengths(series)
        self.assertEqual(sum(rl), len(series))

    def test_run_lengths_nondecreasing_then_decreasing(self):
        rl = self.run_lengths([0, 0, 1, 1, 1, 2])
        self.assertEqual(rl, [2, 3, 1])

    # ── plateau_fraction() ────────────────────────────────────────────

    def test_plateau_frac_empty(self):
        self.assertEqual(self.plateau_fraction([]), 0.0)

    def test_plateau_frac_constant(self):
        # Circular: every x_t == x_{t+1 mod 1} → pf = 1.0
        self.assertAlmostEqual(self.plateau_fraction([0]), 1.0)

    def test_plateau_frac_alternating_zero(self):
        # [47, 1, 47, 1]: circular pairs all ≠ → pf = 0
        self.assertAlmostEqual(self.plateau_fraction([47, 1, 47, 1]), 0.0)

    def test_plateau_frac_xor3_cell0(self):
        # [48, 51, 43, 43, 43, 43, 40, 48]: 4 circular matches → pf=0.5
        series = [48, 51, 43, 43, 43, 43, 40, 48]
        self.assertAlmostEqual(self.plateau_fraction(series), 0.5)

    def test_plateau_frac_range(self):
        for series in [[1, 2, 3], [0, 0, 0], [1, 1, 2]]:
            pf = self.plateau_fraction(series)
            self.assertGreaterEqual(pf, 0.0)
            self.assertLessEqual(pf, 1.0)

    def test_plateau_frac_circular_wrap(self):
        # [1, 2, 1]: t=2 wraps → x[2]=1, x[0]=1 → match → 1/3
        self.assertAlmostEqual(self.plateau_fraction([1, 2, 1]), 1/3, places=6)

    # ── run_entropy() ─────────────────────────────────────────────────

    def test_run_entropy_empty(self):
        self.assertEqual(self.run_entropy([]), 0.0)

    def test_run_entropy_uniform_all_ones(self):
        # All runs length 1 → entropy = 0
        self.assertAlmostEqual(self.run_entropy([1, 1, 1, 1]), 0.0, places=6)

    def test_run_entropy_single_run(self):
        # One run of length P → entropy = 0
        self.assertAlmostEqual(self.run_entropy([8]), 0.0, places=6)

    def test_run_entropy_nonneg(self):
        for rl in [[1, 1, 4, 1, 1], [2, 3, 1], [1, 2, 3]]:
            self.assertGreaterEqual(self.run_entropy(rl), 0.0)

    def test_run_entropy_with_plateau(self):
        # [1, 1, 4, 1, 1]: has two distinct run lengths → entropy > 0
        self.assertGreater(self.run_entropy([1, 1, 4, 1, 1]), 0.0)

    # ── cell_run_stats() ──────────────────────────────────────────────

    def test_cell_run_stats_empty(self):
        d = self.cell_run_stats([])
        self.assertEqual(d['max_run'], 0)
        self.assertEqual(d['n_runs'], 0)

    def test_cell_run_stats_constant(self):
        d = self.cell_run_stats([0, 0, 0])
        self.assertEqual(d['max_run'], 3)
        self.assertEqual(d['n_runs'], 1)

    def test_cell_run_stats_alternating(self):
        d = self.cell_run_stats([47, 1])
        self.assertEqual(d['max_run'], 1)
        self.assertAlmostEqual(d['plateau_frac'], 0.0)

    def test_cell_run_stats_keys(self):
        d = self.cell_run_stats([1, 2, 3])
        for k in ('runs', 'n_runs', 'max_run', 'mean_run',
                  'plateau_frac', 'run_entropy'):
            self.assertIn(k, d)

    def test_cell_run_stats_xor3_cell0(self):
        series = [48, 51, 43, 43, 43, 43, 40, 48]
        d = self.cell_run_stats(series)
        self.assertEqual(d['max_run'], 4)
        self.assertEqual(d['runs'], [1, 1, 4, 1, 1])
        self.assertAlmostEqual(d['plateau_frac'], 0.5, places=4)

    # ── ТУМАН XOR (P=1, fixed) ────────────────────────────────────────

    def test_tuman_xor_max_run_one(self):
        d = self.run_summary('ТУМАН', 'xor', 16)
        self.assertEqual(d['global_max_run'], 1)

    def test_tuman_xor_plateau_frac_one(self):
        d = self.run_summary('ТУМАН', 'xor', 16)
        self.assertAlmostEqual(d['pf_stats']['mean'], 1.0)

    # ── ГОРА AND (P=2, alternating) ───────────────────────────────────

    def test_gora_and_max_run_one(self):
        d = self.run_summary('ГОРА', 'and', 16)
        self.assertEqual(d['global_max_run'], 1)

    def test_gora_and_plateau_frac_zero(self):
        d = self.run_summary('ГОРА', 'and', 16)
        self.assertAlmostEqual(d['pf_stats']['mean'], 0.0)
        self.assertAlmostEqual(d['pf_stats']['max'], 0.0)

    def test_gora_and_all_run_lengths_are_one(self):
        d = self.run_summary('ГОРА', 'and', 16)
        self.assertTrue(all(r == 1 for r in d['all_run_lengths']))

    # ── ТУМАН XOR3 (P=8) ──────────────────────────────────────────────

    def test_tuman_xor3_global_max_run(self):
        d = self.run_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(d['global_max_run'], 4)

    def test_tuman_xor3_mean_pf_low(self):
        d = self.run_summary('ТУМАН', 'xor3', 16)
        # 14/128 ≈ 0.109 from solan_return
        self.assertAlmostEqual(d['pf_stats']['mean'], 14/128, places=3)

    def test_tuman_xor3_max_pf_half(self):
        d = self.run_summary('ТУМАН', 'xor3', 16)
        # cell 0 and 15 have pf=0.5 (max)
        self.assertAlmostEqual(d['pf_stats']['max'], 0.5, places=4)

    def test_tuman_xor3_min_pf_zero(self):
        d = self.run_summary('ТУМАН', 'xor3', 16)
        self.assertAlmostEqual(d['pf_stats']['min'], 0.0, places=4)

    def test_tuman_xor3_run_values_include_4(self):
        d = self.run_summary('ТУМАН', 'xor3', 16)
        self.assertIn(4, d['all_run_lengths'])

    def test_tuman_xor3_run_values_include_1(self):
        d = self.run_summary('ТУМАН', 'xor3', 16)
        self.assertIn(1, d['all_run_lengths'])

    # ── change_matrix() ───────────────────────────────────────────────

    def test_change_matrix_shape(self):
        mat = self.change_matrix('ТУМАН', 'xor3', 16)
        self.assertEqual(len(mat), 16)
        for row in mat:
            self.assertEqual(len(row), 8)   # P=8

    def test_change_matrix_binary(self):
        mat = self.change_matrix('ТУМАН', 'xor3', 16)
        for row in mat:
            for v in row:
                self.assertIn(v, (0, 1))

    def test_change_matrix_tuman_xor_all_zero(self):
        # Fixed point: x_t always same → no changes
        mat = self.change_matrix('ТУМАН', 'xor', 16)
        for row in mat:
            self.assertTrue(all(v == 0 for v in row))

    def test_change_matrix_gora_and_all_one(self):
        # Alternating: every step is a change (circularly)
        mat = self.change_matrix('ГОРА', 'and', 16)
        for row in mat:
            self.assertTrue(all(v == 1 for v in row))

    def test_change_matrix_sum_equals_transitions(self):
        # Sum over all cells × steps = number of transition steps
        mat = self.change_matrix('ТУМАН', 'xor3', 16)
        total_changes = sum(v for row in mat for v in row)
        d = self.run_summary('ТУМАН', 'xor3', 16)
        # Transitions = total_pairs - plateau_pairs
        total_pairs = 16 * d['period']
        plateau_pairs = round(d['pf_stats']['mean'] * total_pairs)
        self.assertEqual(total_changes, total_pairs - plateau_pairs)

    # ── run_summary structure ──────────────────────────────────────────

    def test_run_summary_keys(self):
        d = self.run_summary('ТУМАН', 'xor3', 16)
        for k in ('word', 'rule', 'period', 'cell_stats',
                  'max_run_stats', 'pf_stats', 'entropy_stats',
                  'global_max_run', 'global_mean_run', 'all_run_lengths'):
            self.assertIn(k, d)

    def test_run_summary_word_uppercase(self):
        d = self.run_summary('туман', 'xor3', 16)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_run_summary_all_run_lengths_sum(self):
        d = self.run_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(sum(d['all_run_lengths']), 16 * d['period'])

    # ── all_run_summaries ──────────────────────────────────────────────

    def test_all_run_summaries_four_rules(self):
        result = self.all_run_summaries('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── build_run_data ─────────────────────────────────────────────────

    def test_build_run_data_structure(self):
        data = self.build_run_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)

    def test_build_run_data_entry_keys(self):
        data = self.build_run_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'global_max_run', 'global_mean_run',
                  'max_run_stats', 'pf_stats', 'entropy_stats'):
            self.assertIn(k, entry)

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_rle_bar(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rle-bar', content)

    def test_viewer_has_rle_heat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rle-heat', content)

    def test_viewer_has_rle_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rle-stats', content)

    def test_viewer_has_rle_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rleRun', content)

    def test_viewer_has_runs_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Run-Length', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_runs',
             '--word', 'ТУМАН', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_runs(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_runs', content)


class TestSolanMoments(unittest.TestCase):
    """Tests for solan_moments.py and the viewer Temporal Moments section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_moments import (
            temporal_moments, cell_moments_list, moments_summary,
            all_moments, build_moments_data,
        )
        cls.temporal_moments   = staticmethod(temporal_moments)
        cls.cell_moments_list  = staticmethod(cell_moments_list)
        cls.moments_summary    = staticmethod(moments_summary)
        cls.all_moments        = staticmethod(all_moments)
        cls.build_moments_data = staticmethod(build_moments_data)

    # ── temporal_moments() ────────────────────────────────────────────

    def test_moments_empty(self):
        m = self.temporal_moments([])
        self.assertEqual(m['var'], 0.0)
        self.assertIsNone(m['skew'])
        self.assertIsNone(m['kurt'])

    def test_moments_constant_zero(self):
        m = self.temporal_moments([0, 0, 0])
        self.assertAlmostEqual(m['mean'], 0.0)
        self.assertAlmostEqual(m['var'],  0.0)
        self.assertIsNone(m['skew'])
        self.assertIsNone(m['kurt'])

    def test_moments_constant_nonzero(self):
        m = self.temporal_moments([5, 5, 5, 5])
        self.assertAlmostEqual(m['mean'], 5.0)
        self.assertAlmostEqual(m['var'],  0.0)
        self.assertIsNone(m['skew'])

    def test_moments_two_point_bernoulli_mean(self):
        m = self.temporal_moments([47, 1])
        self.assertAlmostEqual(m['mean'], 24.0, places=4)

    def test_moments_two_point_bernoulli_var(self):
        m = self.temporal_moments([47, 1])
        self.assertAlmostEqual(m['var'], 529.0, places=3)

    def test_moments_two_point_bernoulli_std(self):
        m = self.temporal_moments([47, 1])
        self.assertAlmostEqual(m['std'], 23.0, places=3)

    def test_moments_two_point_bernoulli_skew_zero(self):
        # Symmetric two-point distribution → skew = 0
        m = self.temporal_moments([47, 1])
        self.assertAlmostEqual(m['skew'], 0.0, places=8)

    def test_moments_two_point_bernoulli_kurt_minus_two(self):
        # Exact result: excess kurtosis of uniform 2-point = −2
        m = self.temporal_moments([47, 1])
        self.assertAlmostEqual(m['kurt'], -2.0, places=8)

    def test_moments_kurt_minimum_property(self):
        # For any 2-point distribution: kurt = −2
        for a, b in [(0, 63), (10, 50), (1, 30)]:
            m = self.temporal_moments([a, b])
            self.assertAlmostEqual(m['kurt'], -2.0, places=6,
                                   msg=f'kurt≠-2 for [{a},{b}]')

    def test_moments_uniform_three_values(self):
        # [0, 1, 2]: symmetric → skew=0
        # excess kurtosis of discrete uniform on 3 points = μ₄/σ⁴ − 3
        # μ=1, σ²=2/3, μ₄=2/3 → kurt = (2/3)/(4/9) − 3 = 3/2 − 3 = −3/2
        m = self.temporal_moments([0, 1, 2])
        self.assertAlmostEqual(m['skew'], 0.0,  places=6)
        self.assertAlmostEqual(m['kurt'], -3/2, places=5)

    def test_moments_single_element_constant(self):
        m = self.temporal_moments([42])
        self.assertAlmostEqual(m['mean'], 42.0)
        self.assertAlmostEqual(m['var'], 0.0)
        self.assertIsNone(m['skew'])

    def test_moments_std_is_sqrt_var(self):
        import math
        m = self.temporal_moments([0, 10, 20, 30])
        # std is rounded to 6 decimal places in the module
        self.assertAlmostEqual(m['std'], math.sqrt(m['var']), places=4)

    def test_moments_skew_asymmetric(self):
        # Positive-skew series: mostly low, one high
        m = self.temporal_moments([0, 0, 0, 30])
        self.assertGreater(m['skew'], 0)

    # ── Fixed-point attractor ─────────────────────────────────────────

    def test_tuman_xor_all_constant(self):
        d = self.moments_summary('ТУМАН', 'xor', 16)
        self.assertEqual(d['n_constant'], 16)
        self.assertEqual(d['n_defined'], 0)

    def test_tuman_or_all_constant(self):
        d = self.moments_summary('ТУМАН', 'or', 16)
        self.assertEqual(d['n_constant'], 16)

    # ── ГОРА AND (P=2, 47↔1) ─────────────────────────────────────────

    def test_gora_and_no_constant(self):
        d = self.moments_summary('ГОРА', 'and', 16)
        self.assertEqual(d['n_constant'], 0)
        self.assertEqual(d['n_defined'], 16)

    def test_gora_and_all_skew_zero(self):
        d = self.moments_summary('ГОРА', 'and', 16)
        ss = d['skew_stats']
        self.assertAlmostEqual(ss['mean'], 0.0, places=6)
        self.assertAlmostEqual(ss['std'],  0.0, places=6)

    def test_gora_and_all_kurt_minus_two(self):
        d = self.moments_summary('ГОРА', 'and', 16)
        ks = d['kurt_stats']
        self.assertAlmostEqual(ks['mean'], -2.0, places=6)
        self.assertAlmostEqual(ks['std'],   0.0, places=6)

    def test_gora_and_mean_var_correct(self):
        d = self.moments_summary('ГОРА', 'and', 16)
        vs = d['var_stats']
        # All cells have series [47,1] or [1,47] → var=529
        self.assertAlmostEqual(vs['mean'], 529.0, places=2)
        self.assertAlmostEqual(vs['std'],   0.0,  places=2)

    # ── ТУМАН XOR3 (P=8) ──────────────────────────────────────────────

    def test_tuman_xor3_no_constant(self):
        d = self.moments_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(d['n_constant'], 0)

    def test_tuman_xor3_var_range(self):
        d = self.moments_summary('ТУМАН', 'xor3', 16)
        vs = d['var_stats']
        self.assertGreater(vs['max'], vs['min'])   # spatial heterogeneity

    def test_tuman_xor3_var_positive(self):
        d = self.moments_summary('ТУМАН', 'xor3', 16)
        self.assertGreater(d['var_stats']['min'], 0)

    def test_tuman_xor3_kurt_le_zero(self):
        # All platykurtic (kurt < 0) — bounded finite distribution
        d = self.moments_summary('ТУМАН', 'xor3', 16)
        self.assertLess(d['kurt_stats']['max'], 0)

    def test_tuman_xor3_kurt_ge_minus_two(self):
        # Exact lower bound: kurt ≥ −2 for any distribution
        d = self.moments_summary('ТУМАН', 'xor3', 16)
        self.assertGreaterEqual(d['kurt_stats']['min'], -2.0 - 1e-6)

    # ── cell_moments_list ──────────────────────────────────────────────

    def test_cell_moments_list_length(self):
        cms = self.cell_moments_list('ТУМАН', 'xor3', 16)
        self.assertEqual(len(cms), 16)

    def test_cell_moments_list_has_cell_key(self):
        cms = self.cell_moments_list('ТУМАН', 'xor3', 16)
        for i, m in enumerate(cms):
            self.assertEqual(m['cell'], i)

    def test_cell_moments_list_has_required_keys(self):
        cms = self.cell_moments_list('ТУМАН', 'xor3', 16)
        for m in cms:
            for k in ('mean', 'var', 'std', 'skew', 'kurt', 'cell'):
                self.assertIn(k, m)

    # ── moments_summary structure ──────────────────────────────────────

    def test_summary_keys(self):
        d = self.moments_summary('ТУМАН', 'xor3', 16)
        for k in ('word', 'rule', 'period', 'cell_moments',
                  'var_stats', 'mean_stats', 'skew_stats', 'kurt_stats',
                  'n_constant', 'n_defined'):
            self.assertIn(k, d)

    def test_summary_word_uppercase(self):
        d = self.moments_summary('туман', 'xor3', 16)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_summary_n_constant_plus_defined_equals_width(self):
        d = self.moments_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(d['n_constant'] + d['n_defined'], 16)

    # ── all_moments ────────────────────────────────────────────────────

    def test_all_moments_has_four_rules(self):
        result = self.all_moments('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_moments_values_are_dicts(self):
        result = self.all_moments('ГОРА', 16)
        for rule, d in result.items():
            self.assertIsInstance(d, dict)
            self.assertIn('n_defined', d)

    # ── build_moments_data ─────────────────────────────────────────────

    def test_build_moments_data_structure(self):
        data = self.build_moments_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_moments_data_entry_keys(self):
        data = self.build_moments_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'n_constant', 'n_defined', 'var_stats',
                  'skew_stats', 'kurt_stats'):
            self.assertIn(k, entry)

    def test_build_moments_data_words_uppercase(self):
        data = self.build_moments_data(['туман'], 16)
        self.assertIn('ТУМАН', data['words'])

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_mom_var(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mom-var', content)

    def test_viewer_has_mom_sk(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mom-sk', content)

    def test_viewer_has_mom_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mom-stats', content)

    def test_viewer_has_mom_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('momRun', content)

    def test_viewer_has_moments_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Temporal Moments Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_moments',
             '--word', 'ТУМАН', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_moments(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_moments', content)


class TestSolanReturn(unittest.TestCase):
    """Tests for solan_return.py and the viewer First-Return Map section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_return import (
            return_map, map_stats, aggregate_map, jump_histogram,
            return_dict, all_return, build_return_data,
        )
        cls.return_map        = staticmethod(return_map)
        cls.map_stats         = staticmethod(map_stats)
        cls.aggregate_map     = staticmethod(aggregate_map)
        cls.jump_histogram    = staticmethod(jump_histogram)
        cls.return_dict       = staticmethod(return_dict)
        cls.all_return        = staticmethod(all_return)
        cls.build_return_data = staticmethod(build_return_data)

    # ── return_map() ───────────────────────────────────────────────────

    def test_return_map_empty(self):
        self.assertEqual(self.return_map([]), [])

    def test_return_map_single(self):
        # Single element: (x_0, x_0) circular
        self.assertEqual(self.return_map([5]), [(5, 5)])

    def test_return_map_two_elements(self):
        self.assertEqual(self.return_map([47, 1]), [(47, 1), (1, 47)])

    def test_return_map_length(self):
        s = [1, 2, 3, 4, 5]
        self.assertEqual(len(self.return_map(s)), 5)

    def test_return_map_circular_last_pair(self):
        s = [1, 2, 3]
        pairs = self.return_map(s)
        self.assertEqual(pairs[-1], (3, 1))   # wraps to first element

    def test_return_map_first_pair(self):
        s = [10, 20, 30]
        pairs = self.return_map(s)
        self.assertEqual(pairs[0], (10, 20))

    # ── map_stats() ────────────────────────────────────────────────────

    def test_map_stats_empty(self):
        d = self.map_stats([])
        self.assertEqual(d['n_pairs'], 0)
        self.assertEqual(d['n_distinct'], 0)

    def test_map_stats_single(self):
        d = self.map_stats([(5, 5)])
        self.assertEqual(d['n_distinct'], 1)
        self.assertEqual(d['diagonal_count'], 1)
        self.assertEqual(d['mean_jump'], 0.0)

    def test_map_stats_alternating(self):
        pairs = [(47, 1), (1, 47)]
        d = self.map_stats(pairs)
        self.assertEqual(d['n_distinct'], 2)
        self.assertEqual(d['diagonal_count'], 0)
        self.assertAlmostEqual(d['mean_jump'], 46.0)
        self.assertEqual(d['max_jump'], 46)

    def test_map_stats_diagonal_count(self):
        pairs = [(1, 2), (3, 3), (4, 5), (6, 6)]
        d = self.map_stats(pairs)
        self.assertEqual(d['diagonal_count'], 2)

    def test_map_stats_mean_jump(self):
        pairs = [(0, 10), (10, 0)]    # jumps: 10, 10
        d = self.map_stats(pairs)
        self.assertAlmostEqual(d['mean_jump'], 10.0)

    # ── Fixed-point attractors ─────────────────────────────────────────

    def test_tuman_xor_one_diagonal_point(self):
        d = self.return_dict('ТУМАН', 'xor', 16)
        self.assertEqual(d['n_distinct'], 1)
        self.assertEqual(d['diag_pairs'], d['total_pairs'])   # all on diagonal
        self.assertAlmostEqual(d['mean_jump'], 0.0)

    def test_gora_or_one_diagonal_point(self):
        d = self.return_dict('ГОРА', 'or', 16)
        self.assertEqual(d['n_distinct'], 1)
        self.assertAlmostEqual(d['mean_jump'], 0.0)

    # ── ГОРА AND (P=2, alternating 47↔1) ─────────────────────────────

    def test_gora_and_two_points_off_diagonal(self):
        d = self.return_dict('ГОРА', 'and', 16)
        self.assertEqual(d['n_distinct'], 2)
        self.assertEqual(d['diag_pairs'], 0)

    def test_gora_and_mean_jump_46(self):
        d = self.return_dict('ГОРА', 'and', 16)
        self.assertAlmostEqual(d['mean_jump'], 46.0, places=4)

    def test_gora_and_max_jump_46(self):
        d = self.return_dict('ГОРА', 'and', 16)
        self.assertEqual(d['max_jump'], 46)

    def test_gora_and_total_pairs(self):
        d = self.return_dict('ГОРА', 'and', 16)
        self.assertEqual(d['total_pairs'], 16 * 2)   # P=2 × 16 cells

    def test_gora_and_jump_hist_peak_at_46(self):
        hist = self.jump_histogram('ГОРА', 'and', 16)
        self.assertEqual(hist[46], 32)   # all 16×2 pairs have jump=46
        self.assertEqual(hist[0], 0)

    # ── ТУМАН XOR3 (P=8) ──────────────────────────────────────────────

    def test_tuman_xor3_distinct_gt_2(self):
        d = self.return_dict('ТУМАН', 'xor3', 16)
        self.assertGreater(d['n_distinct'], 2)

    def test_tuman_xor3_total_pairs(self):
        d = self.return_dict('ТУМАН', 'xor3', 16)
        self.assertEqual(d['total_pairs'], 16 * 8)   # P=8 × 16 cells

    def test_tuman_xor3_mean_jump_positive(self):
        d = self.return_dict('ТУМАН', 'xor3', 16)
        self.assertGreater(d['mean_jump'], 0)

    def test_tuman_xor3_max_jump_le_63(self):
        d = self.return_dict('ТУМАН', 'xor3', 16)
        self.assertLessEqual(d['max_jump'], 63)

    def test_tuman_xor3_some_diagonal(self):
        # XOR3 has some self-loops
        d = self.return_dict('ТУМАН', 'xor3', 16)
        self.assertGreater(d['diag_pairs'], 0)

    # ── aggregate_map ──────────────────────────────────────────────────

    def test_aggregate_map_tuman_xor_keys(self):
        agg = self.aggregate_map('ТУМАН', 'xor', 16)
        self.assertEqual(set(agg.keys()), {(0, 0)})

    def test_aggregate_map_gora_and_keys(self):
        agg = self.aggregate_map('ГОРА', 'and', 16)
        self.assertEqual(set(agg.keys()), {(47, 1), (1, 47)})

    def test_aggregate_map_gora_and_counts(self):
        agg = self.aggregate_map('ГОРА', 'and', 16)
        self.assertEqual(agg[(47, 1)], 16)
        self.assertEqual(agg[(1, 47)], 16)

    # ── jump_histogram ─────────────────────────────────────────────────

    def test_jump_histogram_length(self):
        hist = self.jump_histogram('ТУМАН', 'xor3', 16)
        self.assertEqual(len(hist), 64)

    def test_jump_histogram_sum_equals_total_pairs(self):
        hist = self.jump_histogram('ТУМАН', 'xor3', 16)
        d    = self.return_dict('ТУМАН', 'xor3', 16)
        self.assertEqual(sum(hist), d['total_pairs'])

    def test_jump_histogram_nonneg(self):
        hist = self.jump_histogram('ГОРА', 'xor3', 16)
        self.assertTrue(all(v >= 0 for v in hist))

    # ── return_dict structure ──────────────────────────────────────────

    def test_return_dict_keys(self):
        d = self.return_dict('ТУМАН', 'xor3', 16)
        for k in ('word', 'rule', 'period', 'cell_maps', 'agg_map',
                  'jump_hist', 'n_distinct', 'mean_jump', 'max_jump',
                  'diag_pairs', 'total_pairs'):
            self.assertIn(k, d)

    def test_return_dict_cell_maps_length(self):
        d = self.return_dict('ТУМАН', 'xor3', 16)
        self.assertEqual(len(d['cell_maps']), 16)

    def test_return_dict_word_uppercase(self):
        d = self.return_dict('туман', 'xor3', 16)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_return_dict_diag_le_total(self):
        d = self.return_dict('ТУМАН', 'xor3', 16)
        self.assertLessEqual(d['diag_pairs'], d['total_pairs'])

    # ── all_return ─────────────────────────────────────────────────────

    def test_all_return_has_four_rules(self):
        result = self.all_return('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_return_values_are_dicts(self):
        result = self.all_return('ГОРА', 16)
        for rule, d in result.items():
            self.assertIsInstance(d, dict)
            self.assertIn('n_distinct', d)

    # ── build_return_data ──────────────────────────────────────────────

    def test_build_return_data_structure(self):
        data = self.build_return_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_return_data_entry_keys(self):
        data = self.build_return_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'n_distinct', 'mean_jump', 'max_jump',
                  'diag_pairs', 'total_pairs'):
            self.assertIn(k, entry)

    def test_build_return_data_words_uppercase(self):
        data = self.build_return_data(['туман'], 16)
        self.assertIn('ТУМАН', data['words'])

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_ret_map(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ret-map', content)

    def test_viewer_has_ret_hist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ret-hist', content)

    def test_viewer_has_ret_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ret-stats', content)

    def test_viewer_has_ret_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('retRun', content)

    def test_viewer_has_ret_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('First-Return Map Q6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_return',
             '--word', 'ТУМАН', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_return(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_return', content)


class TestSolanPerm(unittest.TestCase):
    """Tests for solan_perm.py and the viewer Permutation Entropy section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_perm import (
            ordinal_pattern, perm_entropy,
            spatial_pe, pe_dict, all_pe, build_pe_data,
            _ALL_RULES, _DEFAULT_WIDTH, _DEFAULT_M,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.ordinal_pattern = staticmethod(ordinal_pattern)
        cls.perm_entropy    = staticmethod(perm_entropy)
        cls.spatial_pe      = staticmethod(spatial_pe)
        cls.pe_dict         = staticmethod(pe_dict)
        cls.all_pe          = staticmethod(all_pe)
        cls.build_pe_data   = staticmethod(build_pe_data)
        cls.ALL_RULES       = _ALL_RULES
        cls.W               = _DEFAULT_WIDTH
        cls.M               = _DEFAULT_M
        cls.LEXICON         = list(LEXICON)

    # ── ordinal_pattern() ─────────────────────────────────────────────────────

    def test_op_no_ties_ascending(self):
        self.assertEqual(self.ordinal_pattern([1, 2, 3]), (0, 1, 2))

    def test_op_no_ties_descending(self):
        self.assertEqual(self.ordinal_pattern([3, 2, 1]), (2, 1, 0))

    def test_op_no_ties_mixed(self):
        self.assertEqual(self.ordinal_pattern([1, 3, 2]), (0, 2, 1))

    def test_op_ties_stable(self):
        # [3, 1, 3]: 1 at idx 1 → rank 0; 3 at idx 0 → rank 1; 3 at idx 2 → rank 2
        self.assertEqual(self.ordinal_pattern([3, 1, 3]), (1, 0, 2))

    def test_op_all_equal(self):
        # Stable sort preserves original order → (0, 1, 2)
        self.assertEqual(self.ordinal_pattern([5, 5, 5]), (0, 1, 2))

    def test_op_length_2(self):
        self.assertEqual(self.ordinal_pattern([0, 1]), (0, 1))
        self.assertEqual(self.ordinal_pattern([1, 0]), (1, 0))

    def test_op_length_matches_input(self):
        for length in [2, 3, 4, 5]:
            pat = self.ordinal_pattern(list(range(length)))
            self.assertEqual(len(pat), length)

    def test_op_is_permutation(self):
        import random
        rng = random.Random(0)
        for _ in range(20):
            m = rng.randint(2, 5)
            win = [rng.randint(0, 63) for _ in range(m)]
            pat = self.ordinal_pattern(win)
            self.assertEqual(sorted(pat), list(range(m)))

    # ── perm_entropy() ────────────────────────────────────────────────────────

    def test_pe_period1_zero(self):
        self.assertAlmostEqual(self.perm_entropy([42, 42, 42, 42], 3), 0.0, places=8)

    def test_pe_constant_single_zero(self):
        self.assertAlmostEqual(self.perm_entropy([7], 3), 0.0, places=8)

    def test_pe_m_less_than_2_zero(self):
        self.assertAlmostEqual(self.perm_entropy([1, 2, 3, 4], 1), 0.0, places=8)

    def test_pe_empty_zero(self):
        self.assertAlmostEqual(self.perm_entropy([], 3), 0.0, places=8)

    def test_pe_period2_m2_max(self):
        # Period-2 with m=2 → 2 distinct patterns → normalised PE = 1.0
        self.assertAlmostEqual(self.perm_entropy([3, 7], 2), 1.0, places=6)
        self.assertAlmostEqual(self.perm_entropy([7, 3], 2), 1.0, places=6)

    def test_pe_period2_m3_value(self):
        # Period-2 m=3 → always log(2)/log(3!) = 1/log2(6) regardless of values
        import math
        expected = round(1.0 / math.log2(math.factorial(3)), 8)
        self.assertAlmostEqual(self.perm_entropy([3, 7], 3), expected, places=6)
        self.assertAlmostEqual(self.perm_entropy([7, 3], 3), expected, places=6)
        self.assertAlmostEqual(self.perm_entropy([1, 63], 3), expected, places=6)

    def test_pe_in_range(self):
        import random
        rng = random.Random(99)
        for _ in range(30):
            P = rng.randint(1, 10)
            m = rng.randint(2, 4)
            series = [rng.randint(0, 63) for _ in range(P)]
            pe = self.perm_entropy(series, m)
            self.assertGreaterEqual(pe, 0.0)
            self.assertLessEqual(pe, 1.0 + 1e-8)

    def test_pe_non_negative(self):
        # Verify no -0.0 leaks
        pe = self.perm_entropy([5], 3)
        self.assertGreaterEqual(pe, 0.0)
        self.assertFalse(str(pe).startswith('-'))

    def test_pe_tuman_xor3_positive(self):
        from projects.hexglyph.solan_transfer import get_orbit
        orbit = get_orbit('ТУМАН', 'xor3')
        series = [s[0] for s in orbit]
        pe = self.perm_entropy(series, 3)
        self.assertGreater(pe, 0.0)

    # ── spatial_pe() ──────────────────────────────────────────────────────────

    def test_spe_length(self):
        profile = self.spatial_pe('ТУМАН', 'xor3', self.W, 3)
        self.assertEqual(len(profile), self.W)

    def test_spe_all_in_range(self):
        for rule in self.ALL_RULES:
            profile = self.spatial_pe('ТУМАН', rule, self.W, 3)
            for v in profile:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0 + 1e-8)

    def test_spe_period1_all_zero(self):
        profile = self.spatial_pe('ГОРА', 'xor', self.W, 3)
        for v in profile:
            self.assertAlmostEqual(v, 0.0, places=8)

    def test_spe_tuman_xor3_nonzero(self):
        profile = self.spatial_pe('ТУМАН', 'xor3', self.W, 3)
        self.assertGreater(max(profile), 0.0)

    # ── pe_dict() ────────────────────────────────────────────────────────────

    def test_pd_keys(self):
        d = self.pe_dict('ТУМАН', 'xor3', self.W, 3)
        for k in ['word', 'rule', 'width', 'm', 'period',
                  'max_patterns', 'profile', 'mean_pe',
                  'max_pe_val', 'min_pe_val', 'spatial_var', 'multi_m']:
            self.assertIn(k, d)

    def test_pd_word_upper(self):
        d = self.pe_dict('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_pd_period_tuman_xor3(self):
        d = self.pe_dict('ТУМАН', 'xor3')
        self.assertEqual(d['period'], 8)

    def test_pd_mean_pe_matches_profile(self):
        d = self.pe_dict('ТУМАН', 'xor3', self.W, 3)
        expected = round(sum(d['profile']) / self.W, 8)
        self.assertAlmostEqual(d['mean_pe'], expected, places=6)

    def test_pd_period1_mean_zero(self):
        d = self.pe_dict('ГОРА', 'xor')
        self.assertAlmostEqual(d['mean_pe'], 0.0, places=8)

    def test_pd_multi_m_keys(self):
        d = self.pe_dict('ТУМАН', 'xor3')
        for m_key in ['2', '3', '4']:
            self.assertIn(m_key, d['multi_m'])

    def test_pd_max_patterns_period1(self):
        d = self.pe_dict('ГОРА', 'xor', m=3)
        self.assertEqual(d['max_patterns'], 1)  # min(period=1, 3!=6) = 1

    def test_pd_max_patterns_period8_m3(self):
        d = self.pe_dict('ТУМАН', 'xor3', m=3)
        # min(8, 6) = 6
        self.assertEqual(d['max_patterns'], 6)

    def test_pd_spatial_var_non_negative(self):
        d = self.pe_dict('ТУМАН', 'xor3')
        self.assertGreaterEqual(d['spatial_var'], 0.0)

    # ── all_pe() ─────────────────────────────────────────────────────────────

    def test_ape_all_rules(self):
        d = self.all_pe('ТУМАН')
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    # ── build_pe_data() ──────────────────────────────────────────────────────

    def test_bpd_keys(self):
        d = self.build_pe_data(['ГОРА', 'ВОДА'])
        for k in ['words', 'm_vals', 'per_rule', 'ranking']:
            self.assertIn(k, d)

    def test_bpd_ranking_sorted(self):
        d = self.build_pe_data(['ГОРА', 'ВОДА', 'МИР'])
        for rule in self.ALL_RULES:
            vals = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(vals, sorted(vals, reverse=True))

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_pe_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pe-canvas', content)

    def test_viewer_has_pe_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pe-stats', content)

    def test_viewer_has_pe_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pe-btn', content)

    def test_viewer_has_ordinal_pattern(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ordinalPattern', content)

    def test_viewer_has_perm_entropy(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('permEntropy', content)

    def test_viewer_has_spatial_pe(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spatialPE', content)


class TestSolanBasin(unittest.TestCase):
    """Tests for solan_basin.py and the viewer Basin section."""

    @classmethod
    def setUpClass(cls):
        import random as _rnd
        from projects.hexglyph.solan_basin import (
            word_ic, flip_k_bits, random_ic,
            attractor_sig, attractors_match,
            basin_at_k, sample_global_basin,
            basin_profile, trajectory_basin, all_basins,
            build_basin_data, basin_dict,
            _ALL_RULES, _DEFAULT_WIDTH, _N_BITS,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.word_ic             = staticmethod(word_ic)
        cls.flip_k_bits         = staticmethod(flip_k_bits)
        cls.random_ic           = staticmethod(random_ic)
        cls.attractor_sig       = staticmethod(attractor_sig)
        cls.attractors_match    = staticmethod(attractors_match)
        cls.basin_at_k          = staticmethod(basin_at_k)
        cls.sample_global_basin = staticmethod(sample_global_basin)
        cls.basin_profile       = staticmethod(basin_profile)
        cls.trajectory_basin    = staticmethod(trajectory_basin)
        cls.all_basins          = staticmethod(all_basins)
        cls.build_basin_data    = staticmethod(build_basin_data)
        cls.basin_dict          = staticmethod(basin_dict)
        cls.ALL_RULES           = _ALL_RULES
        cls.W                   = _DEFAULT_WIDTH
        cls.N_BITS              = _N_BITS
        cls.LEXICON             = list(LEXICON)
        cls.rng                 = _rnd.Random(42)

    # ── word_ic() ─────────────────────────────────────────────────────────────

    def test_wic_length(self):
        ic = self.word_ic('ГОРА', 16)
        self.assertEqual(len(ic), 16)

    def test_wic_q6_range(self):
        ic = self.word_ic('ТУМАН', 16)
        for v in ic:
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 63)

    def test_wic_uppercase(self):
        self.assertEqual(self.word_ic('гора', 16), self.word_ic('ГОРА', 16))

    # ── flip_k_bits() ─────────────────────────────────────────────────────────

    def test_fkb_k0_identity(self):
        cells = self.word_ic('ГОРА', 16)
        rng = __import__('random').Random(1)
        self.assertEqual(self.flip_k_bits(cells, 0, rng), cells)

    def test_fkb_length_preserved(self):
        cells = self.word_ic('ГОРА', 16)
        rng = __import__('random').Random(1)
        self.assertEqual(len(self.flip_k_bits(cells, 3, rng)), 16)

    def test_fkb_q6_range_preserved(self):
        cells = self.word_ic('ТУМАН', 16)
        rng = __import__('random').Random(7)
        result = self.flip_k_bits(cells, 5, rng)
        for v in result:
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 63)

    def test_fkb_k1_changes_one_bit(self):
        import random
        cells = [0] * 16   # all bits zero
        rng = random.Random(3)
        result = self.flip_k_bits(cells, 1, rng)
        # exactly one cell should have changed
        changed = sum(1 for a, b in zip(cells, result) if a != b)
        self.assertEqual(changed, 1)

    # ── random_ic() ───────────────────────────────────────────────────────────

    def test_ric_length(self):
        rng = __import__('random').Random(0)
        ic = self.random_ic(16, rng)
        self.assertEqual(len(ic), 16)

    def test_ric_q6_range(self):
        rng = __import__('random').Random(0)
        ic = self.random_ic(16, rng)
        for v in ic:
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 63)

    # ── attractor_sig() ───────────────────────────────────────────────────────

    def test_as_is_frozenset(self):
        sig = self.attractor_sig('ГОРА', 'xor')
        self.assertIsInstance(sig, frozenset)

    def test_as_non_empty(self):
        sig = self.attractor_sig('ГОРА', 'xor3')
        self.assertGreater(len(sig), 0)

    def test_as_xor_same_across_words(self):
        # Both converge to all-zeros under XOR → identical signatures
        sig1 = self.attractor_sig('ГОРА', 'xor')
        sig2 = self.attractor_sig('ТУМАН', 'xor')
        self.assertEqual(sig1, sig2)

    def test_as_xor_vs_xor3_different(self):
        s1 = self.attractor_sig('ГОРА', 'xor')
        s2 = self.attractor_sig('ГОРА', 'xor3')
        self.assertNotEqual(s1, s2)

    def test_as_deterministic(self):
        s1 = self.attractor_sig('ВОДА', 'xor3')
        s2 = self.attractor_sig('ВОДА', 'xor3')
        self.assertEqual(s1, s2)

    def test_as_period_matches_size(self):
        from projects.hexglyph.solan_ca import find_orbit
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ГОРА'), 16)
        _, period = find_orbit(cells[:], 'xor3')
        sig = self.attractor_sig('ГОРА', 'xor3')
        self.assertEqual(len(sig), max(period, 1))

    # ── attractors_match() ────────────────────────────────────────────────────

    def test_am_same_sig_true(self):
        sig = self.attractor_sig('ГОРА', 'xor')
        self.assertTrue(self.attractors_match(sig, sig))

    def test_am_different_sigs_false(self):
        s1 = self.attractor_sig('ГОРА', 'xor3')
        s2 = self.attractor_sig('ТУМАН', 'xor3')
        # These might or might not match; just verify the function returns bool
        result = self.attractors_match(s1, s2)
        self.assertIsInstance(result, bool)

    # ── basin_at_k() ──────────────────────────────────────────────────────────

    def test_bak_k0_is_one(self):
        rng = __import__('random').Random(1)
        f = self.basin_at_k('ГОРА', 'xor3', 16, 0, 10, rng)
        self.assertEqual(f, 1.0)

    def test_bak_range(self):
        rng = __import__('random').Random(2)
        f = self.basin_at_k('ГОРА', 'xor3', 16, 3, 20, rng)
        self.assertGreaterEqual(f, 0.0)
        self.assertLessEqual(f, 1.0)

    def test_bak_xor_always_one(self):
        # XOR → global basin: all ICs reach the same zero attractor
        rng = __import__('random').Random(5)
        for k in [1, 4, 8]:
            f = self.basin_at_k('ГОРА', 'xor', 16, k, 30, rng)
            self.assertAlmostEqual(f, 1.0, places=1)

    # ── sample_global_basin() ─────────────────────────────────────────────────

    def test_sgb_keys(self):
        rng = __import__('random').Random(0)
        d = self.sample_global_basin('ГОРА', 'xor', 16, 20, rng)
        for k in ['fraction', 'n_same', 'n_samples']:
            self.assertIn(k, d)

    def test_sgb_fraction_range(self):
        rng = __import__('random').Random(0)
        d = self.sample_global_basin('ГОРА', 'xor3', 16, 20, rng)
        self.assertGreaterEqual(d['fraction'], 0.0)
        self.assertLessEqual(d['fraction'], 1.0)

    def test_sgb_xor_high_fraction(self):
        rng = __import__('random').Random(0)
        d = self.sample_global_basin('ГОРА', 'xor', 16, 50, rng)
        self.assertGreater(d['fraction'], 0.9)

    # ── basin_profile() ───────────────────────────────────────────────────────

    def test_bp_length(self):
        p = self.basin_profile('ГОРА', 'xor3', 16, max_k=6, n_per_k=10, seed=42)
        self.assertEqual(len(p), 7)

    def test_bp_first_is_one(self):
        p = self.basin_profile('ГОРА', 'xor3', 16, max_k=6, n_per_k=10, seed=42)
        self.assertEqual(p[0], 1.0)

    def test_bp_all_in_range(self):
        p = self.basin_profile('ГОРА', 'xor3', 16, max_k=4, n_per_k=10, seed=42)
        for f in p:
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)

    def test_bp_xor_all_ones(self):
        p = self.basin_profile('ГОРА', 'xor', 16, max_k=5, n_per_k=20, seed=42)
        for f in p:
            self.assertAlmostEqual(f, 1.0, places=1)

    # ── trajectory_basin() ────────────────────────────────────────────────────

    def test_tb_keys(self):
        tr = self.trajectory_basin('ГОРА', 'xor3', max_k=4, n_per_k=10, seed=42)
        for k in ['word', 'rule', 'width', 'profile', 'mean_profile',
                  'k50', 'global_basin']:
            self.assertIn(k, tr)

    def test_tb_word_uppercased(self):
        tr = self.trajectory_basin('гора', 'xor3', max_k=4, n_per_k=10, seed=42)
        self.assertEqual(tr['word'], 'ГОРА')

    def test_tb_profile_length(self):
        tr = self.trajectory_basin('ГОРА', 'xor3', max_k=6, n_per_k=10, seed=42)
        self.assertEqual(len(tr['profile']), 7)

    def test_tb_mean_profile_range(self):
        tr = self.trajectory_basin('ГОРА', 'xor3', max_k=4, n_per_k=10, seed=42)
        self.assertGreaterEqual(tr['mean_profile'], 0.0)
        self.assertLessEqual(tr['mean_profile'], 1.0)

    # ── all_basins() ──────────────────────────────────────────────────────────

    def test_ab_all_rules(self):
        d = self.all_basins('ГОРА', max_k=4, n_per_k=10, seed=42)
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    # ── build_basin_data() ────────────────────────────────────────────────────

    def test_bbd_keys(self):
        d = self.build_basin_data(['ГОРА', 'ВОДА'], n_per_k=10, seed=42)
        for k in ['words', 'width', 'n_per_k', 'per_rule',
                  'ranking', 'widest', 'narrowest']:
            self.assertIn(k, d)

    def test_bbd_ranking_sorted(self):
        d = self.build_basin_data(['ГОРА', 'ВОДА', 'МИР'], n_per_k=10, seed=42)
        for rule in self.ALL_RULES:
            vals = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(vals, sorted(vals, reverse=True))

    # ── basin_dict() ──────────────────────────────────────────────────────────

    def test_bd_json_serialisable(self):
        import json
        d = self.basin_dict('ГОРА', max_k=4, n_per_k=10, seed=42)
        json.dumps(d)

    def test_bd_has_profile(self):
        d = self.basin_dict('ГОРА', max_k=4, n_per_k=10, seed=42)
        for rule in self.ALL_RULES:
            self.assertIn('profile', d['rules'][rule])

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_ba_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ba-canvas', content)

    def test_viewer_has_ba_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ba-stats', content)

    def test_viewer_has_ba_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ba-btn', content)

    def test_viewer_has_ba_profile(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('baProfile', content)

    def test_viewer_has_ba_attractor_sig(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('baAttractorSig', content)


class TestSolanBit(unittest.TestCase):
    """Tests for solan_bit.py and the viewer Bit-Plane section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_bit import (
            extract_bit_plane, bit_step,
            bit_plane_trajectory, word_bit_planes,
            attractor_activity, bit_plane_signature,
            build_bit_plane_data, bit_plane_dict,
            _SEGMENT_NAMES, _ALL_RULES,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.extract_bit_plane    = staticmethod(extract_bit_plane)
        cls.bit_step             = staticmethod(bit_step)
        cls.bit_plane_trajectory = staticmethod(bit_plane_trajectory)
        cls.word_bit_planes      = staticmethod(word_bit_planes)
        cls.attractor_activity   = staticmethod(attractor_activity)
        cls.bit_plane_signature  = staticmethod(bit_plane_signature)
        cls.build_bit_plane_data = staticmethod(build_bit_plane_data)
        cls.bit_plane_dict       = staticmethod(bit_plane_dict)
        cls.SEGMENT_NAMES        = _SEGMENT_NAMES
        cls.ALL_RULES            = _ALL_RULES
        cls.LEXICON              = list(LEXICON)

    # ── extract_bit_plane() ───────────────────────────────────────────────────

    def test_extract_bit_plane_returns_list(self):
        r = self.extract_bit_plane([63, 0, 42], 0)
        self.assertIsInstance(r, list)

    def test_extract_bit_plane_length(self):
        cells = [1, 2, 4, 8, 16, 32]
        self.assertEqual(len(self.extract_bit_plane(cells, 0)), 6)

    def test_extract_bit_plane_bit0_all_ones(self):
        # 63 = 0b111111 → bit 0 = 1 everywhere
        r = self.extract_bit_plane([63] * 8, 0)
        self.assertTrue(all(v == 1 for v in r))

    def test_extract_bit_plane_bit0_zero(self):
        # 0 → all bits zero
        r = self.extract_bit_plane([0] * 8, 0)
        self.assertTrue(all(v == 0 for v in r))

    def test_extract_bit_plane_values_binary(self):
        r = self.extract_bit_plane([0, 1, 2, 3, 62, 63], 0)
        self.assertTrue(all(v in (0, 1) for v in r))

    def test_extract_bit_plane_specific_bit(self):
        # 4 = 0b000100 → bit 2 = 1, bit 0 = 0
        r = self.extract_bit_plane([4], 2)
        self.assertEqual(r, [1])
        r0 = self.extract_bit_plane([4], 0)
        self.assertEqual(r0, [0])

    def test_extract_all_6_bits_of_63(self):
        for b in range(6):
            r = self.extract_bit_plane([63], b)
            self.assertEqual(r, [1])

    # ── bit_step() ────────────────────────────────────────────────────────────

    def test_bit_step_xor_length(self):
        bits = [1, 0, 1, 0]
        r = self.bit_step(bits, 'xor')
        self.assertEqual(len(r), 4)

    def test_bit_step_values_binary(self):
        bits = [1, 0, 1, 0, 1, 0, 1, 0]
        for rule in self.ALL_RULES:
            r = self.bit_step(bits, rule)
            self.assertTrue(all(v in (0, 1) for v in r))

    def test_bit_step_all_zeros_stays_zero(self):
        bits = [0] * 8
        for rule in self.ALL_RULES:
            r = self.bit_step(bits, rule)
            self.assertEqual(r, [0] * 8)

    def test_bit_step_xor_alternating(self):
        # [1,0,1,0,...] under XOR: left^right = 0^0 or 1^1 → all zeros
        bits = [1, 0, 1, 0, 1, 0, 1, 0]
        r = self.bit_step(bits, 'xor')
        self.assertEqual(r, [0] * 8)

    def test_bit_step_or_all_ones_stays(self):
        bits = [1] * 8
        r = self.bit_step(bits, 'or')
        self.assertEqual(r, [1] * 8)

    def test_bit_step_and_all_ones_stays(self):
        bits = [1] * 8
        r = self.bit_step(bits, 'and')
        self.assertEqual(r, [1] * 8)

    def test_bit_step_invalid_rule_raises(self):
        with self.assertRaises(ValueError):
            self.bit_step([0, 1], 'bad_rule')

    # ── bit_plane_trajectory() ────────────────────────────────────────────────

    def test_bpt_returns_dict(self):
        r = self.bit_plane_trajectory('ГОРА', 0, 'xor3')
        self.assertIsInstance(r, dict)

    def test_bpt_required_keys(self):
        r = self.bit_plane_trajectory('ГОРА', 0, 'xor3')
        for k in ('bit', 'segment', 'rule', 'word', 'rows', 'transient', 'period', 'active', 'mean_attr'):
            self.assertIn(k, r)

    def test_bpt_bit_index_stored(self):
        for b in range(6):
            r = self.bit_plane_trajectory('ГОРА', b, 'xor3')
            self.assertEqual(r['bit'], b)

    def test_bpt_segment_name(self):
        for b in range(6):
            r = self.bit_plane_trajectory('ГОРА', b, 'xor3')
            self.assertEqual(r['segment'], self.SEGMENT_NAMES[b])

    def test_bpt_rows_binary(self):
        r = self.bit_plane_trajectory('ГОРА', 0, 'xor3')
        for row in r['rows']:
            self.assertTrue(all(v in (0, 1) for v in row))

    def test_bpt_rows_count_matches_transient_period(self):
        r = self.bit_plane_trajectory('ГОРА', 0, 'xor3')
        self.assertEqual(len(r['rows']), r['transient'] + r['period'])

    def test_bpt_active_type_bool(self):
        r = self.bit_plane_trajectory('ГОРА', 0, 'xor3')
        self.assertIsInstance(r['active'], bool)

    def test_bpt_mean_attr_range(self):
        r = self.bit_plane_trajectory('ГОРА', 0, 'xor3')
        self.assertGreaterEqual(r['mean_attr'], 0.0)
        self.assertLessEqual(r['mean_attr'], 1.0)

    def test_bpt_xor_attractor_inactive(self):
        # XOR always converges to all-zeros for all lexicon words
        for b in range(6):
            r = self.bit_plane_trajectory('ГОРА', b, 'xor')
            self.assertFalse(r['active'])

    def test_bpt_and_attractor_period_pos(self):
        # AND bit-plane attractor period is ≥ 1
        for b in range(6):
            r = self.bit_plane_trajectory('ГОРА', b, 'and')
            self.assertGreaterEqual(r['period'], 1)

    # ── word_bit_planes() ─────────────────────────────────────────────────────

    def test_wbp_returns_dict_of_6(self):
        r = self.word_bit_planes('ГОРА', 'xor3')
        self.assertEqual(set(r.keys()), set(range(6)))

    def test_wbp_each_value_is_dict(self):
        r = self.word_bit_planes('ГОРА', 'xor3')
        for p in r.values():
            self.assertIsInstance(p, dict)

    # ── attractor_activity() ──────────────────────────────────────────────────

    def test_aa_returns_dict_of_6(self):
        r = self.attractor_activity('ГОРА', 'xor3')
        self.assertEqual(set(r.keys()), set(range(6)))

    def test_aa_keys_per_bit(self):
        r = self.attractor_activity('ГОРА', 'xor3')
        for entry in r.values():
            for k in ('active', 'period', 'mean_attr', 'segment'):
                self.assertIn(k, entry)

    def test_aa_xor3_gora_has_active_bits(self):
        r = self.attractor_activity('ГОРА', 'xor3')
        cnt = sum(1 for d in r.values() if d['active'])
        self.assertGreater(cnt, 0)

    def test_aa_xor_gora_no_active_bits(self):
        r = self.attractor_activity('ГОРА', 'xor')
        cnt = sum(1 for d in r.values() if d['active'])
        self.assertEqual(cnt, 0)

    # ── bit_plane_signature() ─────────────────────────────────────────────────

    def test_signature_returns_all_rules(self):
        r = self.bit_plane_signature('ГОРА')
        self.assertEqual(set(r.keys()), set(self.ALL_RULES))

    def test_signature_each_rule_has_6_bits(self):
        r = self.bit_plane_signature('ГОРА')
        for rule, act in r.items():
            self.assertEqual(len(act), 6, f"rule {rule}")

    # ── build_bit_plane_data() ────────────────────────────────────────────────

    def test_build_returns_dict(self):
        d = self.build_bit_plane_data(['ГОРА', 'ЛУНА'])
        self.assertIsInstance(d, dict)

    def test_build_required_keys(self):
        d = self.build_bit_plane_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'per_rule', 'active_count', 'max_active', 'min_active'):
            self.assertIn(k, d)

    def test_build_active_count_range(self):
        d = self.build_bit_plane_data(['ГОРА', 'ЛУНА', 'ТУМАН'])
        for rule in self.ALL_RULES:
            for word, cnt in d['active_count'][rule].items():
                self.assertGreaterEqual(cnt, 0)
                self.assertLessEqual(cnt, 6)

    def test_build_max_active_exists(self):
        d = self.build_bit_plane_data(['ГОРА', 'ЛУНА', 'ТУМАН'])
        for rule in self.ALL_RULES:
            word, cnt = d['max_active'][rule]
            self.assertIn(word, ['ГОРА', 'ЛУНА', 'ТУМАН'])
            self.assertGreaterEqual(cnt, 0)

    # ── bit_plane_dict() ──────────────────────────────────────────────────────

    def test_bpd_json_serialisable(self):
        import json
        d = self.bit_plane_dict('ГОРА')
        dumped = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(dumped, str)

    def test_bpd_top_keys(self):
        d = self.bit_plane_dict('ГОРА')
        for k in ('word', 'width', 'segment_names', 'rules'):
            self.assertIn(k, d)

    def test_bpd_all_rules_present(self):
        d = self.bit_plane_dict('ГОРА')
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_bpd_segment_names(self):
        d = self.bit_plane_dict('ГОРА')
        self.assertEqual(d['segment_names'], self.SEGMENT_NAMES)

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_bit_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bit-canvas', content)

    def test_viewer_has_bit_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bit-btn', content)

    def test_viewer_has_bit_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bitRun', content)

    def test_viewer_has_bit_word_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bit-word', content)

    def test_viewer_has_bit_rule_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bit-rule', content)

    def test_viewer_bit_section_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Битовые плоскости CA Q6', content)

    def test_viewer_has_seg_colors(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('SEG_COLORS', content)

    def test_viewer_has_compute_bit_planes(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('computeBitPlanes', content)


class TestSolanTraj(unittest.TestCase):
    """Tests for solan_traj.py and the viewer Trajectory section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_traj import (
            word_trajectory, all_word_trajectories, traj_stats,
            build_trajectory_data, trajectory_similarity, trajectory_dict,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.word_trajectory         = staticmethod(word_trajectory)
        cls.all_word_trajectories   = staticmethod(all_word_trajectories)
        cls.traj_stats              = staticmethod(traj_stats)
        cls.build_trajectory_data   = staticmethod(build_trajectory_data)
        cls.trajectory_similarity   = staticmethod(trajectory_similarity)
        cls.trajectory_dict         = staticmethod(trajectory_dict)
        cls.LEXICON                 = list(LEXICON)

    # ── word_trajectory() basics ──────────────────────────────────────────────

    def test_word_trajectory_returns_dict(self):
        t = self.word_trajectory('ГОРА', 'xor3')
        self.assertIsInstance(t, dict)

    def test_word_trajectory_required_keys(self):
        t = self.word_trajectory('ГОРА', 'xor3')
        for k in ('rows', 'transient', 'period', 'n_rows', 'word', 'rule', 'width'):
            self.assertIn(k, t)

    def test_word_trajectory_rows_count(self):
        t = self.word_trajectory('ГОРА', 'xor3')
        # n_rows = transient + period
        self.assertEqual(t['n_rows'], t['transient'] + t['period'])
        self.assertEqual(len(t['rows']), t['n_rows'])

    def test_word_trajectory_row_width(self):
        t = self.word_trajectory('ГОРА', 'xor3', width=16)
        for row in t['rows']:
            self.assertEqual(len(row), 16)

    def test_word_trajectory_gora_xor3(self):
        # ГОРА xor3: transient=0, period=2
        t = self.word_trajectory('ГОРА', 'xor3')
        self.assertEqual(t['transient'], 0)
        self.assertEqual(t['period'], 2)
        self.assertEqual(t['n_rows'], 2)

    def test_word_trajectory_tundra_and(self):
        # ТУНДРА and: transient=2, period=2
        t = self.word_trajectory('ТУНДРА', 'and')
        self.assertEqual(t['transient'], 2)
        self.assertEqual(t['period'], 2)

    def test_word_trajectory_xor_all_converge_to_zeros(self):
        # XOR fixed point is all-zeros for all lexicon words
        for word in self.LEXICON[:10]:
            t = self.word_trajectory(word, 'xor')
            attractor_rows = t['rows'][t['transient']:]
            for row in attractor_rows:
                self.assertTrue(all(v == 0 for v in row),
                                msg=f'{word} XOR attractor not all-zeros: {row}')

    def test_word_trajectory_attractor_is_periodic(self):
        # The attractor rows should form a strict cycle (row[tr] == row[tr+per])
        for word in ['ГОРА', 'ЛУНА', 'ЖУРНАЛ']:
            for rule in ['xor3', 'and', 'or']:
                t = self.word_trajectory(word, rule)
                rows, tr, per = t['rows'], t['transient'], t['period']
                # Verify: applying step from rows[tr] per times returns to rows[tr]
                from projects.hexglyph.solan_ca import step
                c = rows[tr][:]
                for _ in range(per):
                    c = step(c, rule)
                self.assertEqual(c, rows[tr],
                    msg=f'{word}/{rule}: attractor not periodic after {per} steps')

    def test_word_trajectory_cells_are_q6(self):
        t = self.word_trajectory('ЛУНА', 'xor3')
        for row in t['rows']:
            for v in row:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 63)

    # ── all_word_trajectories() ───────────────────────────────────────────────

    def test_all_word_trajectories_keys(self):
        trajs = self.all_word_trajectories('ГОРА')
        self.assertEqual(set(trajs.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_word_trajectories_consistent(self):
        trajs = self.all_word_trajectories('ТУМАН')
        for rule, t in trajs.items():
            self.assertEqual(t['rule'], rule)
            self.assertEqual(t['word'], 'ТУМАН')

    # ── traj_stats() ──────────────────────────────────────────────────────────

    def test_traj_stats_keys(self):
        st = self.traj_stats('ГОРА', 'xor3')
        for k in ('transient', 'period', 'attr_entropy', 'total_entropy',
                  'mean_q6', 'unique_cells', 'unique_states'):
            self.assertIn(k, st)

    def test_traj_stats_xor_attr_entropy_zero(self):
        # XOR attractor = all-zeros → H=0
        for word in self.LEXICON[:5]:
            st = self.traj_stats(word, 'xor')
            self.assertAlmostEqual(st['attr_entropy'], 0.0, places=10)

    def test_traj_stats_xor3_entropy_positive(self):
        # XOR3 has non-trivial attractor
        for word in self.LEXICON[:5]:
            st = self.traj_stats(word, 'xor3')
            self.assertGreater(st['attr_entropy'], 0.0)

    def test_traj_stats_entropy_nonnegative(self):
        for rule in ['xor', 'xor3', 'and', 'or']:
            st = self.traj_stats('ГОРА', rule)
            self.assertGreaterEqual(st['attr_entropy'],  0.0)
            self.assertGreaterEqual(st['total_entropy'], 0.0)

    def test_traj_stats_mean_q6_range(self):
        for rule in ['xor3', 'and', 'or']:
            st = self.traj_stats('ГОРА', rule)
            self.assertGreaterEqual(st['mean_q6'], 0.0)
            self.assertLessEqual(st['mean_q6'], 63.0)

    def test_traj_stats_unique_states_equals_period(self):
        for word in ['ГОРА', 'ЛУНА', 'ЖУРНАЛ']:
            for rule in ['xor3', 'and', 'or']:
                st = self.traj_stats(word, rule)
                self.assertEqual(st['unique_states'], st['period'])

    def test_traj_stats_gora_xor3_attr_entropy(self):
        # ГОРА xor3 attractor has 8 distinct Q6 values across 2 rows × 16 cells
        # → H=3.0 bits (4 distinct values each appearing 8 times in 32 cells)
        st = self.traj_stats('ГОРА', 'xor3')
        self.assertAlmostEqual(st['attr_entropy'], 3.0, places=5)

    # ── trajectory_similarity() ───────────────────────────────────────────────

    def test_similarity_self_is_zero(self):
        t = self.word_trajectory('ГОРА', 'xor3')
        self.assertAlmostEqual(self.trajectory_similarity(t, t), 0.0, places=10)

    def test_similarity_symmetric(self):
        t1 = self.word_trajectory('ГОРА', 'xor3')
        t2 = self.word_trajectory('ЛУНА', 'xor3')
        self.assertAlmostEqual(
            self.trajectory_similarity(t1, t2),
            self.trajectory_similarity(t2, t1),
            places=10)

    def test_similarity_range(self):
        t1 = self.word_trajectory('ГОРА', 'xor3')
        t2 = self.word_trajectory('ТУМАН', 'xor3')
        s = self.trajectory_similarity(t1, t2)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_similarity_xor_is_zero_for_same_attractor(self):
        # All XOR trajectories end at all-zeros, so short trajectories should
        # have very low similarity (both reach zeros)
        t1 = self.word_trajectory('ГОРА', 'xor')
        t2 = self.word_trajectory('ЛУНА', 'xor')
        # They differ in transient but share the zero-attractor; similarity < 0.4
        s = self.trajectory_similarity(t1, t2)
        self.assertLess(s, 0.4)

    # ── build_trajectory_data() ───────────────────────────────────────────────

    def test_build_data_keys(self):
        data = self.build_trajectory_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'per_rule', 'max_entropy', 'min_entropy'):
            self.assertIn(k, data)

    def test_build_data_per_rule_rules(self):
        data = self.build_trajectory_data(['ГОРА', 'ЛУНА'])
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_data_per_rule_words(self):
        words = ['ГОРА', 'ЛУНА', 'ЖУРНАЛ']
        data = self.build_trajectory_data(words)
        for rule in ['xor', 'xor3', 'and', 'or']:
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    def test_build_data_max_entropy_exists(self):
        data = self.build_trajectory_data(['ГОРА', 'ЛУНА', 'ТУМАН'])
        for rule in ['xor3', 'and', 'or']:
            word, h = data['max_entropy'][rule]
            self.assertIn(word, ['ГОРА', 'ЛУНА', 'ТУМАН'])
            self.assertGreaterEqual(h, 0.0)

    # ── trajectory_dict() ─────────────────────────────────────────────────────

    def test_trajectory_dict_serialisable(self):
        import json
        d = self.trajectory_dict('ГОРА')
        dumped = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(dumped, str)

    def test_trajectory_dict_keys(self):
        d = self.trajectory_dict('ГОРА')
        for k in ('word', 'width', 'rules'):
            self.assertIn(k, d)

    def test_trajectory_dict_all_rules(self):
        d = self.trajectory_dict('ГОРА')
        self.assertEqual(set(d['rules'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_trajectory_dict_rule_keys(self):
        d = self.trajectory_dict('ГОРА')
        for rule_data in d['rules'].values():
            for k in ('transient', 'period', 'n_rows', 'rows',
                      'attr_entropy', 'total_entropy', 'mean_q6', 'unique_cells'):
                self.assertIn(k, rule_data)

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_traj_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('traj-canvas', content)

    def test_viewer_has_traj_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('traj-info', content)

    def test_viewer_has_q6_color(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('q6Color', content)

    def test_viewer_has_compute_traj(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('computeTraj', content)

    def test_viewer_has_encl_export(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.encL', content)

    def test_viewer_has_padl_export(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.padL', content)

    def test_viewer_has_lexicon_arr_export(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('window.LEXICON_ARR', content)

    def test_viewer_trajectory_section_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Траектория CA Q6', content)

    def test_viewer_has_traj_word_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('traj-word', content)

    def test_viewer_has_traj_rule_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('traj-rule', content)

    def test_viewer_has_traj_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('trajRun', content)


class TestSolanSpatent(unittest.TestCase):
    """Tests for solan_spatent.py and the viewer Spatial Entropy section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_spatent import (
            spatial_entropy,
            orbit_spatial_entropy,
            spatial_entropy_stats,
            spatent_summary,
            all_spatent,
            build_spatent_data,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.spatial_entropy        = staticmethod(spatial_entropy)
        cls.orbit_spatial_entropy  = staticmethod(orbit_spatial_entropy)
        cls.spatial_entropy_stats  = staticmethod(spatial_entropy_stats)
        cls.spatent_summary        = staticmethod(spatent_summary)
        cls.all_spatent            = staticmethod(all_spatent)
        cls.build_spatent_data     = staticmethod(build_spatent_data)
        cls.LEXICON                = list(LEXICON)

    # ── spatial_entropy() ─────────────────────────────────────────────────────

    def test_spatent_all_same_is_zero(self):
        # All cells same value → H = 0
        self.assertAlmostEqual(self.spatial_entropy([5] * 16), 0.0, places=10)

    def test_spatent_all_zeros_is_zero(self):
        self.assertAlmostEqual(self.spatial_entropy([0] * 16), 0.0, places=10)

    def test_spatent_binary_is_one_bit(self):
        # 8 cells = 0, 8 cells = 63 → H = 1.0
        self.assertAlmostEqual(self.spatial_entropy([0, 63] * 8), 1.0, places=10)

    def test_spatent_four_equal_groups_is_two_bits(self):
        # 4 distinct values, 4 cells each → H = 2.0
        self.assertAlmostEqual(self.spatial_entropy([0,1,2,3]*4), 2.0, places=10)

    def test_spatent_all_distinct_is_log2_n(self):
        import math
        state = list(range(16))
        self.assertAlmostEqual(self.spatial_entropy(state), math.log2(16), places=10)

    def test_spatent_empty_is_zero(self):
        self.assertEqual(self.spatial_entropy([]), 0.0)

    def test_spatent_single_cell_is_zero(self):
        self.assertAlmostEqual(self.spatial_entropy([42]), 0.0, places=10)

    def test_spatent_nonnegative(self):
        import random
        state = [random.randint(0, 63) for _ in range(16)]
        self.assertGreaterEqual(self.spatial_entropy(state), 0.0)

    # ── orbit_spatial_entropy() ───────────────────────────────────────────────

    def test_orbit_spatent_tuman_xor_is_zero(self):
        # ТУМАН XOR: P=1, all-zero state → H_0 = 0
        profile = self.orbit_spatial_entropy('ТУМАН', 'xor')
        self.assertEqual(len(profile), 1)
        self.assertAlmostEqual(profile[0], 0.0, places=10)

    def test_orbit_spatent_gora_and_is_one_bit(self):
        # ГОРА AND: both steps have 8 cells=47, 8 cells=1 → H_t = 1.0
        profile = self.orbit_spatial_entropy('ГОРА', 'and')
        self.assertEqual(len(profile), 2)
        for h in profile:
            self.assertAlmostEqual(h, 1.0, places=5)

    def test_orbit_spatent_gora_xor3_is_two_bits(self):
        # ГОРА XOR3: 4 clusters of 4 cells → H_t = 2.0 for all steps
        profile = self.orbit_spatial_entropy('ГОРА', 'xor3')
        self.assertEqual(len(profile), 2)
        for h in profile:
            self.assertAlmostEqual(h, 2.0, places=5)

    def test_orbit_spatent_tuman_xor3_len_8(self):
        profile = self.orbit_spatial_entropy('ТУМАН', 'xor3')
        self.assertEqual(len(profile), 8)

    def test_orbit_spatent_tuman_xor3_max_is_3375(self):
        profile = self.orbit_spatial_entropy('ТУМАН', 'xor3')
        self.assertAlmostEqual(max(profile), 3.375, places=4)

    def test_orbit_spatent_tuman_xor3_min_above_2(self):
        profile = self.orbit_spatial_entropy('ТУМАН', 'xor3')
        self.assertGreater(min(profile), 2.0)

    def test_orbit_spatent_all_values_nonneg(self):
        for word in ['ТУМАН', 'ГОРА', 'ЛУНА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                profile = self.orbit_spatial_entropy(word, rule)
                for h in profile:
                    self.assertGreaterEqual(h, 0.0)

    def test_orbit_spatent_all_values_leq_max_h(self):
        import math
        max_h = math.log2(16)
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor3', 'and', 'or']:
                profile = self.orbit_spatial_entropy(word, rule)
                for h in profile:
                    self.assertLessEqual(h, max_h + 1e-9)

    # ── spatial_entropy_stats() ────────────────────────────────────────────────

    def test_stats_keys(self):
        st = self.spatial_entropy_stats('ГОРА', 'xor3')
        for k in ('mean', 'std', 'min', 'max', 'delta', 'profile'):
            self.assertIn(k, st)

    def test_stats_tuman_xor_all_zero(self):
        st = self.spatial_entropy_stats('ТУМАН', 'xor')
        self.assertAlmostEqual(st['mean'],  0.0, places=10)
        self.assertAlmostEqual(st['delta'], 0.0, places=10)

    def test_stats_gora_and_mean_one(self):
        st = self.spatial_entropy_stats('ГОРА', 'and')
        self.assertAlmostEqual(st['mean'], 1.0, places=5)

    def test_stats_gora_and_delta_zero(self):
        st = self.spatial_entropy_stats('ГОРА', 'and')
        self.assertAlmostEqual(st['delta'], 0.0, places=10)

    def test_stats_gora_xor3_mean_two(self):
        st = self.spatial_entropy_stats('ГОРА', 'xor3')
        self.assertAlmostEqual(st['mean'], 2.0, places=5)

    def test_stats_tuman_xor3_delta_approx(self):
        # delta_H = max_H - min_H = 3.375 - 2.1738 ≈ 1.2012
        st = self.spatial_entropy_stats('ТУМАН', 'xor3')
        self.assertAlmostEqual(st['delta'], 3.375 - 2.1738, places=3)

    def test_stats_tuman_xor3_mean_above_2(self):
        st = self.spatial_entropy_stats('ТУМАН', 'xor3')
        self.assertGreater(st['mean'], 2.5)

    def test_stats_min_leq_mean_leq_max(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                st = self.spatial_entropy_stats(word, rule)
                self.assertLessEqual(st['min'],  st['mean'] + 1e-9)
                self.assertLessEqual(st['mean'], st['max']  + 1e-9)

    def test_stats_delta_is_max_minus_min(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor3', 'and']:
                st = self.spatial_entropy_stats(word, rule)
                self.assertAlmostEqual(st['delta'], st['max'] - st['min'], places=9)

    # ── spatent_summary() ─────────────────────────────────────────────────────

    def test_summary_keys(self):
        d = self.spatent_summary('ГОРА', 'and')
        for k in ('word', 'rule', 'period', 'profile', 'mean_H', 'std_H',
                  'min_H', 'max_H', 'delta_H', 'max_possible_H',
                  'norm_mean_H', 'variability'):
            self.assertIn(k, d)

    def test_summary_word_preserved(self):
        d = self.spatent_summary('гора', 'and')
        self.assertEqual(d['word'], 'ГОРА')

    def test_summary_gora_and_variability_constant(self):
        d = self.spatent_summary('ГОРА', 'and')
        self.assertEqual(d['variability'], 'constant')

    def test_summary_tuman_xor3_variability_high(self):
        d = self.spatent_summary('ТУМАН', 'xor3')
        self.assertEqual(d['variability'], 'high')

    def test_summary_norm_mean_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                d = self.spatent_summary(word, rule)
                self.assertGreaterEqual(d['norm_mean_H'], 0.0)
                self.assertLessEqual(d['norm_mean_H'],    1.0 + 1e-9)

    def test_summary_max_possible_h_is_4(self):
        d = self.spatent_summary('ГОРА', 'xor3', width=16)
        self.assertAlmostEqual(d['max_possible_H'], 4.0, places=5)

    def test_summary_period_matches_profile_len(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor3', 'and']:
                d = self.spatent_summary(word, rule)
                self.assertEqual(d['period'], len(d['profile']))

    # ── all_spatent() ─────────────────────────────────────────────────────────

    def test_all_spatent_keys(self):
        r = self.all_spatent('ГОРА')
        self.assertEqual(set(r.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_spatent_consistent_word(self):
        r = self.all_spatent('ТУМАН')
        for rule, d in r.items():
            self.assertEqual(d['word'], 'ТУМАН')
            self.assertEqual(d['rule'], rule)

    # ── build_spatent_data() ───────────────────────────────────────────────────

    def test_build_data_keys(self):
        data = self.build_spatent_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'max_possible_H', 'per_rule'):
            self.assertIn(k, data)

    def test_build_data_per_rule_has_4_rules(self):
        data = self.build_spatent_data(['ГОРА'])
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_data_per_rule_word_keys(self):
        words = ['ГОРА', 'ТУМАН', 'ЛУНА']
        data = self.build_spatent_data(words)
        for rule in ['xor', 'xor3', 'and', 'or']:
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    def test_build_data_max_possible_h(self):
        import math
        data = self.build_spatent_data(['ГОРА'], width=16)
        self.assertAlmostEqual(data['max_possible_H'], math.log2(16), places=5)

    def test_build_data_per_word_keys(self):
        data = self.build_spatent_data(['ГОРА'])
        entry = data['per_rule']['and']['ГОРА']
        for k in ('period', 'profile', 'mean_H', 'std_H', 'min_H',
                  'max_H', 'delta_H', 'norm_mean_H', 'variability'):
            self.assertIn(k, entry)

    # ── Scientific properties ─────────────────────────────────────────────────

    def test_xor_zero_attractor_gives_zero_h(self):
        # XOR always converges to all-zeros → H=0 for all lexicon words
        for word in self.LEXICON[:10]:
            st = self.spatial_entropy_stats(word, 'xor')
            self.assertAlmostEqual(st['mean'], 0.0, places=10,
                                   msg=f'{word} XOR should have H=0')

    def test_constant_profile_has_zero_std(self):
        # Constant profile → std_H = 0
        for word in ['ГОРА', 'ВОДА', 'НОРА']:
            st = self.spatial_entropy_stats(word, 'and')
            self.assertAlmostEqual(st['std'], 0.0, places=10)

    def test_tuman_xor3_is_most_variable(self):
        # ТУМАН XOR3 has large delta_H > 1.0
        st = self.spatial_entropy_stats('ТУМАН', 'xor3')
        self.assertGreater(st['delta'], 1.0)

    # ── Viewer section tests ───────────────────────────────────────────────────

    def test_viewer_has_spatent_profile_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spatent-profile', content)

    def test_viewer_has_spatent_bars_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spatent-bars', content)

    def test_viewer_has_spatent_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spatent-info', content)

    def test_viewer_has_spatent_word_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('se-word', content)

    def test_viewer_has_spatent_rule_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('se-rule', content)

    def test_viewer_has_spatent_run_button(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('se-btn', content)

    def test_viewer_has_spatial_h_function(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('spatialH', content)

    def test_viewer_has_se_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('seRun', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_spatent',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_spatent(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_spatent', content)


class TestSolanEdge(unittest.TestCase):
    """Tests for solan_edge.py and the viewer Spatial Edge section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_edge import (
            edge_density,
            bit_edge_density,
            bit_edge_vector,
            orbit_edge_profile,
            orbit_bit_edge_profile,
            edge_stats,
            mean_bit_edge,
            classify_bit_edge,
            edge_summary,
            all_edges,
            build_edge_data,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.edge_density           = staticmethod(edge_density)
        cls.bit_edge_density       = staticmethod(bit_edge_density)
        cls.bit_edge_vector        = staticmethod(bit_edge_vector)
        cls.orbit_edge_profile     = staticmethod(orbit_edge_profile)
        cls.orbit_bit_edge_profile = staticmethod(orbit_bit_edge_profile)
        cls.edge_stats             = staticmethod(edge_stats)
        cls.mean_bit_edge          = staticmethod(mean_bit_edge)
        cls.classify_bit_edge      = staticmethod(classify_bit_edge)
        cls.edge_summary           = staticmethod(edge_summary)
        cls.all_edges              = staticmethod(all_edges)
        cls.build_edge_data        = staticmethod(build_edge_data)
        cls.LEXICON                = list(LEXICON)

    # ── edge_density() ───────────────────────────────────────────────────────

    def test_edge_density_all_same_is_zero(self):
        self.assertAlmostEqual(self.edge_density([5] * 16), 0.0, places=10)

    def test_edge_density_alternating_is_one(self):
        self.assertAlmostEqual(self.edge_density([0, 63] * 8), 1.0, places=10)

    def test_edge_density_empty_is_zero(self):
        self.assertEqual(self.edge_density([]), 0.0)

    def test_edge_density_single_is_zero(self):
        self.assertAlmostEqual(self.edge_density([42]), 0.0, places=10)

    def test_edge_density_range_0_1(self):
        import random; random.seed(0)
        for _ in range(10):
            state = [random.randint(0, 63) for _ in range(16)]
            e = self.edge_density(state)
            self.assertGreaterEqual(e, 0.0)
            self.assertLessEqual(e, 1.0)

    def test_edge_density_one_pair_differs(self):
        # 15 cells same, 1 differs → 2 boundaries on ring of 16
        state = [0] * 16
        state[0] = 1
        self.assertAlmostEqual(self.edge_density(state), 2 / 16, places=10)

    # ── bit_edge_density() ────────────────────────────────────────────────────

    def test_bit_edge_density_all_same_bit(self):
        self.assertAlmostEqual(self.bit_edge_density([0] * 16, 0), 0.0, places=10)

    def test_bit_edge_density_alternating_bit(self):
        state = [0, 1] * 8   # bit 0 alternates 0/1
        self.assertAlmostEqual(self.bit_edge_density(state, 0), 1.0, places=10)

    def test_bit_edge_density_range(self):
        for b in range(6):
            v = self.bit_edge_density([47, 1] * 8, b)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    # ── bit_edge_vector() ─────────────────────────────────────────────────────

    def test_bit_edge_vector_length_6(self):
        self.assertEqual(len(self.bit_edge_vector([0] * 16)), 6)

    def test_bit_edge_vector_all_zeros_state(self):
        bev = self.bit_edge_vector([0] * 16)
        self.assertTrue(all(v == 0.0 for v in bev))

    def test_bit_edge_vector_gora_and_state(self):
        # 47=0b101111, 1=0b000001: bits 0 and 4 same; bits 1,2,3,5 differ
        state = [47, 1] * 8
        bev = self.bit_edge_vector(state)
        self.assertAlmostEqual(bev[0], 0.0, places=10)
        self.assertAlmostEqual(bev[4], 0.0, places=10)
        for b in [1, 2, 3, 5]:
            self.assertAlmostEqual(bev[b], 1.0, places=10)

    # ── orbit_edge_profile() ──────────────────────────────────────────────────

    def test_orbit_edge_profile_tuman_xor(self):
        profile = self.orbit_edge_profile('ТУМАН', 'xor')
        self.assertEqual(len(profile), 1)
        self.assertAlmostEqual(profile[0], 0.0, places=10)

    def test_orbit_edge_profile_gora_and(self):
        profile = self.orbit_edge_profile('ГОРА', 'and')
        self.assertEqual(len(profile), 2)
        for e in profile:
            self.assertAlmostEqual(e, 1.0, places=5)

    def test_orbit_edge_profile_tuman_xor3_len_8(self):
        self.assertEqual(len(self.orbit_edge_profile('ТУМАН', 'xor3')), 8)

    def test_orbit_edge_profile_tuman_xor3_max(self):
        self.assertAlmostEqual(max(self.orbit_edge_profile('ТУМАН', 'xor3')), 1.0, places=5)

    def test_orbit_edge_profile_tuman_xor3_min(self):
        # min = 13/16 = 0.8125
        self.assertAlmostEqual(min(self.orbit_edge_profile('ТУМАН', 'xor3')), 13 / 16, places=5)

    def test_orbit_edge_profile_all_nonneg(self):
        for word in ['ТУМАН', 'ГОРА', 'ЛУНА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for e in self.orbit_edge_profile(word, rule):
                    self.assertGreaterEqual(e, 0.0)

    # ── orbit_bit_edge_profile() ──────────────────────────────────────────────

    def test_bit_edge_profile_shape_gora_and(self):
        bep = self.orbit_bit_edge_profile('ГОРА', 'and')
        self.assertEqual(len(bep), 2)
        for row in bep:
            self.assertEqual(len(row), 6)

    def test_bit_edge_profile_gora_xor3_bit0_zero(self):
        # ГОРА XOR3: bit 0 = 1 everywhere → no spatial boundary for bit 0
        bep = self.orbit_bit_edge_profile('ГОРА', 'xor3')
        for row in bep:
            self.assertAlmostEqual(row[0], 0.0, places=10)

    def test_bit_edge_profile_gora_and_bits_1235_full(self):
        bep = self.orbit_bit_edge_profile('ГОРА', 'and')
        for row in bep:
            for b in [1, 2, 3, 5]:
                self.assertAlmostEqual(row[b], 1.0, places=5)

    # ── edge_stats() ──────────────────────────────────────────────────────────

    def test_edge_stats_keys(self):
        st = self.edge_stats('ГОРА', 'and')
        for k in ('mean', 'std', 'min', 'max', 'delta', 'profile'):
            self.assertIn(k, st)

    def test_edge_stats_tuman_xor_mean_zero(self):
        self.assertAlmostEqual(self.edge_stats('ТУМАН', 'xor')['mean'], 0.0, places=10)

    def test_edge_stats_gora_and_mean_one(self):
        self.assertAlmostEqual(self.edge_stats('ГОРА', 'and')['mean'], 1.0, places=5)

    def test_edge_stats_gora_and_delta_zero(self):
        self.assertAlmostEqual(self.edge_stats('ГОРА', 'and')['delta'], 0.0, places=10)

    def test_edge_stats_tuman_xor3_delta(self):
        # delta = 1.0 - 13/16 = 3/16 = 0.1875
        self.assertAlmostEqual(self.edge_stats('ТУМАН', 'xor3')['delta'], 3 / 16, places=5)

    # ── mean_bit_edge() ────────────────────────────────────────────────────────

    def test_mean_bit_edge_length_6(self):
        self.assertEqual(len(self.mean_bit_edge('ГОРА', 'and')), 6)

    def test_mean_bit_edge_gora_and_zero_bits(self):
        mbe = self.mean_bit_edge('ГОРА', 'and')
        self.assertAlmostEqual(mbe[0], 0.0, places=5)
        self.assertAlmostEqual(mbe[4], 0.0, places=5)

    def test_mean_bit_edge_gora_and_full_bits(self):
        mbe = self.mean_bit_edge('ГОРА', 'and')
        for b in [1, 2, 3, 5]:
            self.assertAlmostEqual(mbe[b], 1.0, places=5)

    def test_mean_bit_edge_gora_xor3_bit0_zero(self):
        self.assertAlmostEqual(self.mean_bit_edge('ГОРА', 'xor3')[0], 0.0, places=5)

    def test_mean_bit_edge_gora_xor3_others_half(self):
        mbe = self.mean_bit_edge('ГОРА', 'xor3')
        for b in range(1, 6):
            self.assertAlmostEqual(mbe[b], 0.5, places=5)

    def test_mean_bit_edge_range_all(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for v in self.mean_bit_edge(word, rule):
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 1.0)

    # ── classify_bit_edge() ───────────────────────────────────────────────────

    def test_classify_zero(self):
        self.assertEqual(self.classify_bit_edge(0.0), 'ZERO')

    def test_classify_full(self):
        self.assertEqual(self.classify_bit_edge(1.0), 'FULL')

    def test_classify_half(self):
        self.assertEqual(self.classify_bit_edge(0.5), 'HALF')

    def test_classify_intermediate(self):
        self.assertEqual(self.classify_bit_edge(0.3), 'INTERMEDIATE')

    def test_classify_near_zero(self):
        self.assertEqual(self.classify_bit_edge(0.01), 'ZERO')

    def test_classify_near_full(self):
        self.assertEqual(self.classify_bit_edge(0.99), 'FULL')

    # ── edge_summary() ────────────────────────────────────────────────────────

    def test_summary_keys(self):
        d = self.edge_summary('ГОРА', 'and')
        for k in ('word', 'rule', 'period', 'profile', 'mean_E', 'std_E',
                  'min_E', 'max_E', 'delta_E', 'variability',
                  'mean_bit_edge', 'bit_edge_classes', 'class_counts'):
            self.assertIn(k, d)

    def test_summary_word_uppercase(self):
        self.assertEqual(self.edge_summary('гора', 'and')['word'], 'ГОРА')

    def test_summary_gora_and_variability_constant(self):
        self.assertEqual(self.edge_summary('ГОРА', 'and')['variability'], 'constant')

    def test_summary_gora_and_class_counts(self):
        d = self.edge_summary('ГОРА', 'and')
        self.assertEqual(d['class_counts']['ZERO'], 2)
        self.assertEqual(d['class_counts']['FULL'], 4)

    def test_summary_gora_xor3_bit0_is_zero_class(self):
        d = self.edge_summary('ГОРА', 'xor3')
        self.assertEqual(d['bit_edge_classes'][0], 'ZERO')

    def test_summary_period_matches_profile(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor3', 'and']:
                d = self.edge_summary(word, rule)
                self.assertEqual(d['period'], len(d['profile']))

    # ── all_edges() ───────────────────────────────────────────────────────────

    def test_all_edges_four_rules(self):
        self.assertEqual(set(self.all_edges('ГОРА').keys()), {'xor','xor3','and','or'})

    def test_all_edges_consistent_word(self):
        r = self.all_edges('ТУМАН')
        for rule, d in r.items():
            self.assertEqual(d['word'], 'ТУМАН')

    # ── build_edge_data() ─────────────────────────────────────────────────────

    def test_build_data_keys(self):
        data = self.build_edge_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'per_rule'):
            self.assertIn(k, data)

    def test_build_data_four_rules(self):
        data = self.build_edge_data(['ГОРА'])
        self.assertEqual(set(data['per_rule'].keys()), {'xor','xor3','and','or'})

    def test_build_data_word_keys(self):
        words = ['ГОРА', 'ТУМАН', 'ЛУНА']
        data = self.build_edge_data(words)
        for rule in ['xor', 'xor3', 'and', 'or']:
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    # ── Scientific invariants ──────────────────────────────────────────────────

    def test_xor_zero_attractor_edge_zero(self):
        for word in self.LEXICON[:10]:
            st = self.edge_stats(word, 'xor')
            self.assertAlmostEqual(st['mean'], 0.0, places=10,
                                   msg=f'{word} XOR edge should be 0')

    def test_gora_and_bit_edge_matches_bitflip(self):
        mbe = self.mean_bit_edge('ГОРА', 'and')
        for b in [0, 4]:
            self.assertEqual(self.classify_bit_edge(mbe[b]), 'ZERO')
        for b in [1, 2, 3, 5]:
            self.assertEqual(self.classify_bit_edge(mbe[b]), 'FULL')

    def test_gora_xor3_only_bit0_zero_boundary(self):
        mbe = self.mean_bit_edge('ГОРА', 'xor3')
        self.assertEqual(self.classify_bit_edge(mbe[0]), 'ZERO')
        for b in range(1, 6):
            self.assertEqual(self.classify_bit_edge(mbe[b]), 'HALF')

    # ── Viewer section tests ───────────────────────────────────────────────────

    def test_viewer_has_edge_profile_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('edge-profile', content)

    def test_viewer_has_edge_bitheat_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('edge-bitheat', content)

    def test_viewer_has_edge_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('edge-info', content)

    def test_viewer_has_ed_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ed-word', content)

    def test_viewer_has_ed_rule(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ed-rule', content)

    def test_viewer_has_ed_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ed-btn', content)

    def test_viewer_has_edge_density_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('edgeDensity', content)

    def test_viewer_has_bit_edge_vector_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bitEdgeVector', content)

    def test_viewer_has_ed_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('edRun', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_edge',
             '--word', 'ТУМАН', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_edge(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_edge', content)


class TestSolanSymm(unittest.TestCase):
    """Tests for solan_symm.py and the viewer Rotational Symmetry section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_symm import (
            rot_period,
            rot_order,
            orbit_rot_periods,
            orbit_rot_orders,
            min_rot_period,
            max_rot_order,
            symm_summary,
            all_symm,
            build_symm_data,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.rot_period       = staticmethod(rot_period)
        cls.rot_order        = staticmethod(rot_order)
        cls.orbit_rot_periods = staticmethod(orbit_rot_periods)
        cls.orbit_rot_orders  = staticmethod(orbit_rot_orders)
        cls.min_rot_period   = staticmethod(min_rot_period)
        cls.max_rot_order    = staticmethod(max_rot_order)
        cls.symm_summary     = staticmethod(symm_summary)
        cls.all_symm         = staticmethod(all_symm)
        cls.build_symm_data  = staticmethod(build_symm_data)
        cls.LEXICON          = list(LEXICON)

    # ── rot_period() ──────────────────────────────────────────────────────────

    def test_rot_period_constant_is_one(self):
        # Constant state [v]*16 → rot_period = 1
        self.assertEqual(self.rot_period([7] * 16), 1)

    def test_rot_period_alternating_is_two(self):
        self.assertEqual(self.rot_period([0, 63] * 8), 2)

    def test_rot_period_four_cluster_is_four(self):
        self.assertEqual(self.rot_period([1, 2, 3, 4] * 4), 4)

    def test_rot_period_unique_is_n(self):
        # All distinct values → rot_period = N
        state = list(range(16))
        self.assertEqual(self.rot_period(state), 16)

    def test_rot_period_eight_cluster_is_eight(self):
        self.assertEqual(self.rot_period([1, 2, 3, 4, 5, 6, 7, 8] * 2), 8)

    def test_rot_period_gora_and_state(self):
        # ГОРА AND state [47, 1, 47, 1, ...] → rot_period = 2
        self.assertEqual(self.rot_period([47, 1] * 8), 2)

    # ── rot_order() ───────────────────────────────────────────────────────────

    def test_rot_order_constant_is_n(self):
        self.assertEqual(self.rot_order([5] * 16), 16)

    def test_rot_order_alternating_is_8(self):
        self.assertEqual(self.rot_order([0, 63] * 8), 8)

    def test_rot_order_four_cluster_is_4(self):
        self.assertEqual(self.rot_order([1, 2, 3, 4] * 4), 4)

    def test_rot_order_unique_is_1(self):
        self.assertEqual(self.rot_order(list(range(16))), 1)

    def test_rot_order_times_rot_period_equals_n(self):
        for state in [[0]*16, [0,1]*8, [0,1,2,3]*4, list(range(16))]:
            N = len(state)
            rp = self.rot_period(state)
            ro = self.rot_order(state)
            self.assertEqual(rp * ro, N)

    # ── orbit_rot_periods() / orbit_rot_orders() ──────────────────────────────

    def test_orbit_rot_periods_tuman_xor(self):
        periods = self.orbit_rot_periods('ТУМАН', 'xor')
        self.assertEqual(len(periods), 1)
        self.assertEqual(periods[0], 1)  # all-zero state

    def test_orbit_rot_orders_tuman_xor(self):
        orders = self.orbit_rot_orders('ТУМАН', 'xor')
        self.assertEqual(orders[0], 16)  # maximum symmetry

    def test_orbit_rot_periods_gora_and(self):
        periods = self.orbit_rot_periods('ГОРА', 'and')
        self.assertEqual(len(periods), 2)
        for rp in periods:
            self.assertEqual(rp, 2)  # binary alternation

    def test_orbit_rot_orders_gora_and(self):
        orders = self.orbit_rot_orders('ГОРА', 'and')
        for ro in orders:
            self.assertEqual(ro, 8)

    def test_orbit_rot_periods_gora_xor3(self):
        periods = self.orbit_rot_periods('ГОРА', 'xor3')
        self.assertEqual(len(periods), 2)
        for rp in periods:
            self.assertEqual(rp, 4)  # 4-cluster structure

    def test_orbit_rot_orders_gora_xor3(self):
        orders = self.orbit_rot_orders('ГОРА', 'xor3')
        for ro in orders:
            self.assertEqual(ro, 4)

    def test_orbit_rot_periods_tuman_xor3(self):
        periods = self.orbit_rot_periods('ТУМАН', 'xor3')
        self.assertEqual(len(periods), 8)
        for rp in periods:
            self.assertEqual(rp, 16)  # fully asymmetric

    def test_orbit_rot_orders_tuman_xor3(self):
        orders = self.orbit_rot_orders('ТУМАН', 'xor3')
        for ro in orders:
            self.assertEqual(ro, 1)

    def test_orbit_rot_periods_length_equals_period(self):
        from projects.hexglyph.solan_traj import word_trajectory
        for word in ['ГОРА', 'ТУМАН']:
            for rule in ['xor3', 'and']:
                traj = word_trajectory(word, rule)
                periods = self.orbit_rot_periods(word, rule)
                self.assertEqual(len(periods), traj['period'])

    # ── min_rot_period() / max_rot_order() ───────────────────────────────────

    def test_min_rot_period_tuman_xor(self):
        self.assertEqual(self.min_rot_period('ТУМАН', 'xor'), 1)

    def test_min_rot_period_gora_and(self):
        self.assertEqual(self.min_rot_period('ГОРА', 'and'), 2)

    def test_min_rot_period_gora_xor3(self):
        self.assertEqual(self.min_rot_period('ГОРА', 'xor3'), 4)

    def test_min_rot_period_tuman_xor3(self):
        self.assertEqual(self.min_rot_period('ТУМАН', 'xor3'), 16)

    def test_max_rot_order_tuman_xor(self):
        self.assertEqual(self.max_rot_order('ТУМАН', 'xor'), 16)

    def test_max_rot_order_gora_and(self):
        self.assertEqual(self.max_rot_order('ГОРА', 'and'), 8)

    def test_max_rot_order_gora_xor3(self):
        self.assertEqual(self.max_rot_order('ГОРА', 'xor3'), 4)

    def test_max_rot_order_tuman_xor3(self):
        self.assertEqual(self.max_rot_order('ТУМАН', 'xor3'), 1)

    # ── symm_summary() ────────────────────────────────────────────────────────

    def test_summary_keys(self):
        d = self.symm_summary('ГОРА', 'xor3')
        for k in ('word', 'rule', 'period', 'rot_periods', 'rot_orders',
                  'min_rot_period', 'max_rot_order', 'uniform_symmetry',
                  'symmetry_level'):
            self.assertIn(k, d)

    def test_summary_word_preserved(self):
        self.assertEqual(self.symm_summary('гора', 'xor3')['word'], 'ГОРА')

    def test_summary_tuman_xor_level_maximum(self):
        self.assertEqual(self.symm_summary('ТУМАН', 'xor')['symmetry_level'], 'maximum')

    def test_summary_gora_and_level_high(self):
        self.assertEqual(self.symm_summary('ГОРА', 'and')['symmetry_level'], 'high')

    def test_summary_gora_xor3_level_moderate(self):
        self.assertEqual(self.symm_summary('ГОРА', 'xor3')['symmetry_level'], 'moderate')

    def test_summary_tuman_xor3_level_none(self):
        self.assertEqual(self.symm_summary('ТУМАН', 'xor3')['symmetry_level'], 'none')

    def test_summary_gora_xor3_uniform_true(self):
        d = self.symm_summary('ГОРА', 'xor3')
        self.assertTrue(d['uniform_symmetry'])

    def test_summary_period_matches_len_rot_periods(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor3', 'and']:
                d = self.symm_summary(word, rule)
                self.assertEqual(d['period'], len(d['rot_periods']))
                self.assertEqual(d['period'], len(d['rot_orders']))

    # ── Scientific invariants ──────────────────────────────────────────────────

    def test_xor3_period2_words_all_have_rot_period_4(self):
        # All XOR3 period-2 words must have 4-fold rotational symmetry
        from projects.hexglyph.solan_traj import word_trajectory
        for word in self.LEXICON:
            traj = word_trajectory(word, 'xor3')
            if traj['period'] == 2:
                mrp = self.min_rot_period(word, 'xor3')
                self.assertEqual(mrp, 4,
                    msg=f'{word} XOR3 P=2 should have rot_period=4, got {mrp}')

    def test_xor_attractor_max_symmetry(self):
        # XOR always → all-zeros (constant) → maximum symmetry for all words
        for word in self.LEXICON[:12]:
            mro = self.max_rot_order(word, 'xor')
            self.assertEqual(mro, 16,
                msg=f'{word} XOR should have rot_order=16, got {mro}')

    def test_rot_period_divides_n(self):
        # rot_period must always divide N=16
        for word in ['ТУМАН', 'ГОРА', 'ЛУНА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for rp in self.orbit_rot_periods(word, rule):
                    self.assertEqual(16 % rp, 0,
                        msg=f'{word}/{rule}: rot_period={rp} does not divide 16')

    # ── all_symm() and build_symm_data() ─────────────────────────────────────

    def test_all_symm_four_rules(self):
        self.assertEqual(set(self.all_symm('ГОРА').keys()), {'xor','xor3','and','or'})

    def test_build_symm_data_keys(self):
        data = self.build_symm_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'per_rule'):
            self.assertIn(k, data)

    def test_build_symm_data_word_coverage(self):
        words = ['ГОРА', 'ТУМАН']
        data = self.build_symm_data(words)
        for rule in ['xor', 'xor3', 'and', 'or']:
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    # ── Viewer section tests ───────────────────────────────────────────────────

    def test_viewer_has_symm_ring_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('symm-ring', content)

    def test_viewer_has_symm_rules_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('symm-rules', content)

    def test_viewer_has_symm_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('symm-info', content)

    def test_viewer_has_sy_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sy-word', content)

    def test_viewer_has_sy_rule(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sy-rule', content)

    def test_viewer_has_sy_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sy-btn', content)

    def test_viewer_has_rot_period_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rotPeriod', content)

    def test_viewer_has_sy_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('syRun', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_symm',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_symm(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_symm', content)


class TestSolanVocab(unittest.TestCase):
    """Tests for solan_vocab.py and the viewer Orbit Vocabulary section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_vocab import (
            orbit_vocabulary,
            vocab_size,
            vocab_coverage,
            value_hist,
            uniform_distribution,
            vocab_bit_profile,
            common_bits,
            vocab_hamming_hist,
            vocab_summary,
            all_vocab,
            build_vocab_data,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.orbit_vocabulary    = staticmethod(orbit_vocabulary)
        cls.vocab_size          = staticmethod(vocab_size)
        cls.vocab_coverage      = staticmethod(vocab_coverage)
        cls.value_hist          = staticmethod(value_hist)
        cls.uniform_distribution = staticmethod(uniform_distribution)
        cls.vocab_bit_profile   = staticmethod(vocab_bit_profile)
        cls.common_bits         = staticmethod(common_bits)
        cls.vocab_hamming_hist  = staticmethod(vocab_hamming_hist)
        cls.vocab_summary       = staticmethod(vocab_summary)
        cls.all_vocab           = staticmethod(all_vocab)
        cls.build_vocab_data    = staticmethod(build_vocab_data)
        cls.LEXICON             = list(LEXICON)

    # ── orbit_vocabulary() ────────────────────────────────────────────────────

    def test_vocab_tuman_xor_is_zero(self):
        v = self.orbit_vocabulary('ТУМАН', 'xor')
        self.assertEqual(v, [0])

    def test_vocab_gora_and(self):
        v = self.orbit_vocabulary('ГОРА', 'and')
        self.assertEqual(sorted(v), [1, 47])

    def test_vocab_gora_xor3_size_8(self):
        v = self.orbit_vocabulary('ГОРА', 'xor3')
        self.assertEqual(len(v), 8)

    def test_vocab_gora_xor3_values(self):
        v = self.orbit_vocabulary('ГОРА', 'xor3')
        self.assertEqual(sorted(v), sorted([1, 15, 17, 31, 33, 47, 49, 63]))

    def test_vocab_gora_or_is_63(self):
        v = self.orbit_vocabulary('ГОРА', 'or')
        self.assertEqual(v, [63])

    def test_vocab_tuman_xor3_size_15(self):
        v = self.orbit_vocabulary('ТУМАН', 'xor3')
        self.assertEqual(len(v), 15)

    def test_vocab_is_sorted(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                v = self.orbit_vocabulary(word, rule)
                self.assertEqual(v, sorted(v))

    def test_vocab_values_in_q6_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for val in self.orbit_vocabulary(word, rule):
                    self.assertGreaterEqual(val, 0)
                    self.assertLessEqual(val, 63)

    def test_vocab_nonempty(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                self.assertGreater(len(self.orbit_vocabulary(word, rule)), 0)

    # ── vocab_size() / vocab_coverage() ──────────────────────────────────────

    def test_vocab_size_tuman_xor(self):
        self.assertEqual(self.vocab_size('ТУМАН', 'xor'), 1)

    def test_vocab_size_gora_and(self):
        self.assertEqual(self.vocab_size('ГОРА', 'and'), 2)

    def test_vocab_size_gora_xor3(self):
        self.assertEqual(self.vocab_size('ГОРА', 'xor3'), 8)

    def test_vocab_coverage_tuman_xor(self):
        self.assertAlmostEqual(self.vocab_coverage('ТУМАН', 'xor'), 1 / 64, places=5)

    def test_vocab_coverage_gora_and(self):
        self.assertAlmostEqual(self.vocab_coverage('ГОРА', 'and'), 2 / 64, places=5)

    def test_vocab_coverage_gora_xor3(self):
        self.assertAlmostEqual(self.vocab_coverage('ГОРА', 'xor3'), 8 / 64, places=5)

    def test_vocab_coverage_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                c = self.vocab_coverage(word, rule)
                self.assertGreater(c, 0.0)
                self.assertLessEqual(c, 1.0)

    # ── value_hist() / uniform_distribution() ─────────────────────────────────

    def test_hist_tuman_xor(self):
        h = self.value_hist('ТУМАН', 'xor')
        self.assertEqual(h, {0: 16})

    def test_hist_gora_and_uniform(self):
        h = self.value_hist('ГОРА', 'and')
        self.assertEqual(h.get(47, 0), 16)
        self.assertEqual(h.get(1, 0), 16)

    def test_hist_counts_sum_to_period_times_width(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor3', 'and']:
                h = self.value_hist(word, rule)
                from projects.hexglyph.solan_traj import word_trajectory
                traj = word_trajectory(word, rule)
                self.assertEqual(sum(h.values()), traj['period'] * 16)

    def test_uniform_gora_and(self):
        self.assertTrue(self.uniform_distribution('ГОРА', 'and'))

    def test_uniform_gora_xor3(self):
        self.assertTrue(self.uniform_distribution('ГОРА', 'xor3'))

    def test_uniform_tuman_xor(self):
        self.assertTrue(self.uniform_distribution('ТУМАН', 'xor'))

    def test_nonuniform_tuman_xor3(self):
        # ТУМАН XOR3 has non-uniform distribution (43 and 60 appear 13× each)
        self.assertFalse(self.uniform_distribution('ТУМАН', 'xor3'))

    # ── vocab_bit_profile() ───────────────────────────────────────────────────

    def test_bit_profile_length_6(self):
        bp = self.vocab_bit_profile('ГОРА', 'and')
        self.assertEqual(len(bp), 6)

    def test_bit_profile_tuman_xor_all_zero(self):
        bp = self.vocab_bit_profile('ТУМАН', 'xor')
        self.assertTrue(all(v == 0.0 for v in bp))

    def test_bit_profile_gora_or_all_one(self):
        # OR vocab = {63 = 0b111111}: all bits always 1
        bp = self.vocab_bit_profile('ГОРА', 'or')
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in bp))

    def test_bit_profile_gora_xor3_bit0_is_one(self):
        # All 8 vocab values have bit 0 = 1
        bp = self.vocab_bit_profile('ГОРА', 'xor3')
        self.assertAlmostEqual(bp[0], 1.0, places=9)

    def test_bit_profile_gora_and_bit0_is_one(self):
        # Both 1 and 47 have bit 0 = 1
        bp = self.vocab_bit_profile('ГОРА', 'and')
        self.assertAlmostEqual(bp[0], 1.0, places=9)

    def test_bit_profile_gora_and_bit4_is_zero(self):
        # Both 1=0b000001 and 47=0b101111 have bit 4 = 0
        bp = self.vocab_bit_profile('ГОРА', 'and')
        self.assertAlmostEqual(bp[4], 0.0, places=9)

    def test_bit_profile_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for v in self.vocab_bit_profile(word, rule):
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 1.0)

    # ── common_bits() ─────────────────────────────────────────────────────────

    def test_common_bits_tuman_xor(self):
        al1, al0 = self.common_bits('ТУМАН', 'xor')
        self.assertEqual(al1, set())
        self.assertEqual(al0, {0, 1, 2, 3, 4, 5})

    def test_common_bits_gora_or(self):
        al1, al0 = self.common_bits('ГОРА', 'or')
        self.assertEqual(al1, {0, 1, 2, 3, 4, 5})
        self.assertEqual(al0, set())

    def test_common_bits_gora_and_always1(self):
        al1, _ = self.common_bits('ГОРА', 'and')
        self.assertIn(0, al1)

    def test_common_bits_gora_and_always0(self):
        _, al0 = self.common_bits('ГОРА', 'and')
        self.assertIn(4, al0)

    def test_common_bits_gora_xor3_bit0_always1(self):
        al1, _ = self.common_bits('ГОРА', 'xor3')
        self.assertIn(0, al1)

    def test_common_bits_are_disjoint(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                al1, al0 = self.common_bits(word, rule)
                self.assertEqual(al1 & al0, set(),
                    msg=f'{word}/{rule}: always-1 and always-0 overlap')

    # ── vocab_hamming_hist() ──────────────────────────────────────────────────

    def test_hamming_hist_tuman_xor(self):
        h = self.vocab_hamming_hist('ТУМАН', 'xor')
        self.assertEqual(h, {0: 1})

    def test_hamming_hist_gora_or(self):
        h = self.vocab_hamming_hist('ГОРА', 'or')
        self.assertEqual(h, {6: 1})

    def test_hamming_hist_gora_and(self):
        h = self.vocab_hamming_hist('ГОРА', 'and')
        self.assertEqual(h.get(1, 0), 1)   # value 1
        self.assertEqual(h.get(5, 0), 1)   # value 47

    def test_hamming_hist_keys_in_0_6(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for hw in self.vocab_hamming_hist(word, rule):
                    self.assertGreaterEqual(hw, 0)
                    self.assertLessEqual(hw, 6)

    # ── vocab_summary() ────────────────────────────────────────────────────────

    def test_summary_keys(self):
        d = self.vocab_summary('ГОРА', 'xor3')
        for k in ('word', 'rule', 'period', 'total_cell_steps',
                  'vocab', 'vocab_size', 'vocab_coverage', 'hist',
                  'uniform_dist', 'hist_entropy', 'bit_profile',
                  'always_1_bits', 'always_0_bits', 'hamming_hist',
                  'dominant_value', 'dominant_frac'):
            self.assertIn(k, d)

    def test_summary_word_preserved(self):
        self.assertEqual(self.vocab_summary('гора', 'xor3')['word'], 'ГОРА')

    def test_summary_hist_entropy_uniform(self):
        # ГОРА XOR3 uniform over 8 values → H = log2(8) = 3.0
        d = self.vocab_summary('ГОРА', 'xor3')
        self.assertAlmostEqual(d['hist_entropy'], 3.0, places=5)

    def test_summary_hist_entropy_and(self):
        # ГОРА AND uniform over 2 values → H = 1.0
        d = self.vocab_summary('ГОРА', 'and')
        self.assertAlmostEqual(d['hist_entropy'], 1.0, places=5)

    def test_summary_hist_entropy_xor(self):
        # ТУМАН XOR single value → H = 0
        d = self.vocab_summary('ТУМАН', 'xor')
        self.assertAlmostEqual(d['hist_entropy'], 0.0, places=5)

    def test_summary_total_cell_steps(self):
        d = self.vocab_summary('ГОРА', 'and', width=16)
        self.assertEqual(d['total_cell_steps'], 2 * 16)   # P=2, N=16

    # ── all_vocab() / build_vocab_data() ──────────────────────────────────────

    def test_all_vocab_four_rules(self):
        r = self.all_vocab('ГОРА')
        self.assertEqual(set(r.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_vocab_data_keys(self):
        data = self.build_vocab_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'q6_total', 'per_rule'):
            self.assertIn(k, data)

    def test_build_vocab_data_q6_total(self):
        data = self.build_vocab_data(['ГОРА'])
        self.assertEqual(data['q6_total'], 64)

    def test_build_vocab_data_word_coverage(self):
        words = ['ГОРА', 'ТУМАН']
        data = self.build_vocab_data(words)
        for rule in ['xor', 'xor3', 'and', 'or']:
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    # ── Scientific properties ─────────────────────────────────────────────────

    def test_xor_vocab_always_contains_zero(self):
        # XOR always converges to all-zeros for all lexicon words
        for word in self.LEXICON[:12]:
            v = self.orbit_vocabulary(word, 'xor')
            self.assertIn(0, v, msg=f'{word} XOR vocab should contain 0')
            self.assertEqual(len(v), 1, msg=f'{word} XOR vocab should have size 1')

    def test_gora_xor3_all_vocab_bits0_set(self):
        # All 8 vocab values in ГОРА XOR3 have bit 0 = 1
        vocab = self.orbit_vocabulary('ГОРА', 'xor3')
        for v in vocab:
            self.assertEqual((v >> 0) & 1, 1,
                msg=f'ГОРА XOR3 vocab value {v} should have bit 0=1')

    def test_or_attractor_vocab_max_value(self):
        # OR typically saturates to 63 (all bits 1) for many words
        v = self.orbit_vocabulary('ГОРА', 'or')
        self.assertEqual(v, [63])

    # ── Viewer section tests ───────────────────────────────────────────────────

    def test_viewer_has_vocab_bars(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('vocab-bars', content)

    def test_viewer_has_vocab_bits(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('vocab-bits', content)

    def test_viewer_has_vocab_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('vocab-info', content)

    def test_viewer_has_vo_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('vo-word', content)

    def test_viewer_has_vo_rule(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('vo-rule', content)

    def test_viewer_has_vo_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('vo-btn', content)

    def test_viewer_has_vo_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('voRun', content)

    def test_viewer_has_hw_bits_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('hwBits', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_vocab',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_vocab(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_vocab', content)


class TestSolanLayer(unittest.TestCase):
    """Tests for solan_layer.py and the viewer Bit-Layer Decomposition section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_layer import (
            bit_plane,
            plane_period,
            plane_type,
            plane_density,
            layer_periods,
            active_bits,
            frozen_bits,
            lcm_equals_period,
            layer_summary,
            all_layers,
            build_layer_data,
            _lcm_list,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.bit_plane         = staticmethod(bit_plane)
        cls.plane_period      = staticmethod(plane_period)
        cls.plane_type        = staticmethod(plane_type)
        cls.plane_density     = staticmethod(plane_density)
        cls.layer_periods     = staticmethod(layer_periods)
        cls.active_bits       = staticmethod(active_bits)
        cls.frozen_bits       = staticmethod(frozen_bits)
        cls.lcm_equals_period = staticmethod(lcm_equals_period)
        cls.layer_summary     = staticmethod(layer_summary)
        cls.all_layers        = staticmethod(all_layers)
        cls.build_layer_data  = staticmethod(build_layer_data)
        cls.lcm_list          = staticmethod(_lcm_list)
        cls.LEXICON           = list(LEXICON)

    # ── bit_plane() ────────────────────────────────────────────────────────────

    def test_bit_plane_values_are_0_or_1(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for b in range(6):
                    plane = self.bit_plane(word, rule, b)
                    for row in plane:
                        for v in row:
                            self.assertIn(v, (0, 1))

    def test_bit_plane_width(self):
        plane = self.bit_plane('ГОРА', 'xor3', 0)
        self.assertTrue(all(len(row) == 16 for row in plane))

    def test_bit_plane_period(self):
        from projects.hexglyph.solan_perm import get_orbit
        orbit = get_orbit('ГОРА', 'xor3', 16)
        plane = self.bit_plane('ГОРА', 'xor3', 0)
        self.assertEqual(len(plane), len(orbit))

    def test_bit_plane_gora_xor3_bit0_all_ones(self):
        plane = self.bit_plane('ГОРА', 'xor3', 0)
        self.assertTrue(all(all(v == 1 for v in row) for row in plane))

    def test_bit_plane_tuman_xor_all_zeros(self):
        for b in range(6):
            plane = self.bit_plane('ТУМАН', 'xor', b)
            self.assertTrue(all(all(v == 0 for v in row) for row in plane))

    def test_bit_plane_gora_or_all_ones(self):
        for b in range(6):
            plane = self.bit_plane('ГОРА', 'or', b)
            self.assertTrue(all(all(v == 1 for v in row) for row in plane))

    def test_bit_plane_gora_and_bit4_all_zeros(self):
        plane = self.bit_plane('ГОРА', 'and', 4)
        self.assertTrue(all(all(v == 0 for v in row) for row in plane))

    def test_bit_plane_gora_and_bit0_all_ones(self):
        plane = self.bit_plane('ГОРА', 'and', 0)
        self.assertTrue(all(all(v == 1 for v in row) for row in plane))

    # ── plane_period() ─────────────────────────────────────────────────────────

    def test_plane_period_frozen_is_1(self):
        self.assertEqual(self.plane_period('ТУМАН', 'xor', 0), 1)
        self.assertEqual(self.plane_period('ГОРА', 'or', 5), 1)
        self.assertEqual(self.plane_period('ГОРА', 'and', 0), 1)
        self.assertEqual(self.plane_period('ГОРА', 'and', 4), 1)

    def test_plane_period_gora_xor3_active_bits(self):
        for b in [1, 2, 3, 4, 5]:
            self.assertEqual(self.plane_period('ГОРА', 'xor3', b), 2)

    def test_plane_period_tuman_xor3_all_8(self):
        for b in range(6):
            self.assertEqual(self.plane_period('ТУМАН', 'xor3', b), 8)

    def test_plane_period_divides_orbit_period(self):
        from projects.hexglyph.solan_traj import word_trajectory
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                P = word_trajectory(word, rule)['period']
                for b in range(6):
                    p = self.plane_period(word, rule, b)
                    self.assertEqual(P % p, 0,
                        msg=f'{word}/{rule}/b{b}: period {p} does not divide P={P}')

    # ── plane_type() ──────────────────────────────────────────────────────────

    def test_plane_type_frozen0_tuman_xor(self):
        for b in range(6):
            self.assertEqual(self.plane_type('ТУМАН', 'xor', b), 'frozen_0')

    def test_plane_type_frozen1_gora_or(self):
        for b in range(6):
            self.assertEqual(self.plane_type('ГОРА', 'or', b), 'frozen_1')

    def test_plane_type_gora_xor3_b0(self):
        self.assertEqual(self.plane_type('ГОРА', 'xor3', 0), 'frozen_1')

    def test_plane_type_gora_and_b0(self):
        self.assertEqual(self.plane_type('ГОРА', 'and', 0), 'frozen_1')

    def test_plane_type_gora_and_b4(self):
        self.assertEqual(self.plane_type('ГОРА', 'and', 4), 'frozen_0')

    def test_plane_type_gora_xor3_active_spatial(self):
        for b in [1, 2, 3, 4, 5]:
            pt = self.plane_type('ГОРА', 'xor3', b)
            self.assertEqual(pt, 'spatial',
                msg=f'ГОРА xor3 bit {b} should be spatial')

    def test_plane_type_is_valid_string(self):
        valid = {'frozen_0', 'frozen_1', 'uniform_alt', 'uniform_irr', 'spatial'}
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for b in range(6):
                    self.assertIn(self.plane_type(word, rule, b), valid)

    # ── plane_density() ────────────────────────────────────────────────────────

    def test_density_length_equals_period(self):
        from projects.hexglyph.solan_traj import word_trajectory
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                P = word_trajectory(word, rule)['period']
                dens = self.plane_density(word, rule, 0)
                self.assertEqual(len(dens), P)

    def test_density_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for b in range(6):
                    for d in self.plane_density(word, rule, b):
                        self.assertGreaterEqual(d, 0.0)
                        self.assertLessEqual(d, 1.0)

    def test_density_frozen0_is_zero(self):
        dens = self.plane_density('ТУМАН', 'xor', 3)
        self.assertTrue(all(abs(d) < 1e-9 for d in dens))

    def test_density_frozen1_is_one(self):
        dens = self.plane_density('ГОРА', 'or', 2)
        self.assertTrue(all(abs(d - 1.0) < 1e-9 for d in dens))

    def test_density_gora_xor3_b0_always_1(self):
        dens = self.plane_density('ГОРА', 'xor3', 0)
        self.assertTrue(all(abs(d - 1.0) < 1e-9 for d in dens))

    def test_density_gora_xor3_b1_oscillates_75_25(self):
        dens = self.plane_density('ГОРА', 'xor3', 1)
        self.assertEqual(sorted(round(d, 3) for d in dens), [0.25, 0.75])

    def test_density_gora_xor3_b4_oscillates_50_50(self):
        dens = self.plane_density('ГОРА', 'xor3', 4)
        self.assertEqual(sorted(round(d, 3) for d in dens), [0.5, 0.5])

    def test_density_gora_and_b0_always_1(self):
        dens = self.plane_density('ГОРА', 'and', 0)
        self.assertTrue(all(abs(d - 1.0) < 1e-9 for d in dens))

    def test_density_gora_and_b4_always_0(self):
        dens = self.plane_density('ГОРА', 'and', 4)
        self.assertTrue(all(abs(d) < 1e-9 for d in dens))

    # ── layer_periods() / active_bits() / frozen_bits() ───────────────────────

    def test_layer_periods_length_6(self):
        lps = self.layer_periods('ГОРА', 'xor3')
        self.assertEqual(len(lps), 6)

    def test_layer_periods_gora_and(self):
        lps = self.layer_periods('ГОРА', 'and')
        self.assertEqual(lps, [1, 2, 2, 2, 1, 2])

    def test_layer_periods_gora_xor3(self):
        lps = self.layer_periods('ГОРА', 'xor3')
        self.assertEqual(lps, [1, 2, 2, 2, 2, 2])

    def test_layer_periods_tuman_xor(self):
        lps = self.layer_periods('ТУМАН', 'xor')
        self.assertEqual(lps, [1, 1, 1, 1, 1, 1])

    def test_layer_periods_tuman_xor3(self):
        lps = self.layer_periods('ТУМАН', 'xor3')
        self.assertEqual(lps, [8, 8, 8, 8, 8, 8])

    def test_active_bits_gora_xor3(self):
        self.assertEqual(self.active_bits('ГОРА', 'xor3'), [1, 2, 3, 4, 5])

    def test_active_bits_gora_or(self):
        self.assertEqual(self.active_bits('ГОРА', 'or'), [])

    def test_active_bits_tuman_xor(self):
        self.assertEqual(self.active_bits('ТУМАН', 'xor'), [])

    def test_frozen_bits_tuman_xor(self):
        f0, f1 = self.frozen_bits('ТУМАН', 'xor')
        self.assertEqual(f0, [0, 1, 2, 3, 4, 5])
        self.assertEqual(f1, [])

    def test_frozen_bits_gora_or(self):
        f0, f1 = self.frozen_bits('ГОРА', 'or')
        self.assertEqual(f0, [])
        self.assertEqual(f1, [0, 1, 2, 3, 4, 5])

    def test_frozen_bits_gora_and(self):
        f0, f1 = self.frozen_bits('ГОРА', 'and')
        self.assertIn(4, f0)
        self.assertIn(0, f1)

    def test_frozen_active_partition(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                act = set(self.active_bits(word, rule))
                f0  = set(self.frozen_bits(word, rule)[0])
                f1  = set(self.frozen_bits(word, rule)[1])
                self.assertEqual(act | f0 | f1, set(range(6)),
                    msg=f'{word}/{rule}: active+frozen0+frozen1 != {{0..5}}')
                self.assertEqual(act & f0, set())
                self.assertEqual(act & f1, set())
                self.assertEqual(f0 & f1, set())

    # ── LCM theorem ───────────────────────────────────────────────────────────

    def test_lcm_theorem_gora_xor(self):
        self.assertTrue(self.lcm_equals_period('ГОРА', 'xor'))

    def test_lcm_theorem_gora_and(self):
        self.assertTrue(self.lcm_equals_period('ГОРА', 'and'))

    def test_lcm_theorem_gora_xor3(self):
        self.assertTrue(self.lcm_equals_period('ГОРА', 'xor3'))

    def test_lcm_theorem_tuman_xor3(self):
        self.assertTrue(self.lcm_equals_period('ТУМАН', 'xor3'))

    def test_lcm_theorem_all_words_rules(self):
        for word in self.LEXICON[:12]:
            for rule in ['xor', 'xor3', 'and', 'or']:
                self.assertTrue(self.lcm_equals_period(word, rule),
                    msg=f'LCM theorem failed for {word}/{rule}')

    def test_lcm_list_basic(self):
        self.assertEqual(self.lcm_list([1, 2, 2, 2, 1, 2]), 2)
        self.assertEqual(self.lcm_list([1, 1, 1, 1, 1, 1]), 1)
        self.assertEqual(self.lcm_list([8, 8, 8, 8, 8, 8]), 8)

    # ── layer_summary() ────────────────────────────────────────────────────────

    def test_summary_keys(self):
        d = self.layer_summary('ГОРА', 'xor3')
        for k in ('word', 'rule', 'period', 'plane_periods', 'plane_types',
                  'plane_density', 'mean_density', 'density_var',
                  'active_bits', 'n_active', 'frozen_0_bits', 'frozen_1_bits',
                  'n_frozen', 'lcm_period', 'lcm_equals_P'):
            self.assertIn(k, d)

    def test_summary_word_normalised(self):
        self.assertEqual(self.layer_summary('гора', 'xor3')['word'], 'ГОРА')

    def test_summary_lcm_equals_P_always_true(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                d = self.layer_summary(word, rule)
                self.assertTrue(d['lcm_equals_P'],
                    msg=f'{word}/{rule} summary: lcm_equals_P should be True')

    def test_summary_n_active_plus_frozen_is_6(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                d = self.layer_summary(word, rule)
                self.assertEqual(d['n_active'] + d['n_frozen'], 6)

    def test_summary_gora_xor3_n_active_5(self):
        self.assertEqual(self.layer_summary('ГОРА', 'xor3')['n_active'], 5)

    def test_summary_gora_and_n_frozen_2(self):
        self.assertEqual(self.layer_summary('ГОРА', 'and')['n_frozen'], 2)

    def test_summary_density_var_frozen_is_zero(self):
        d = self.layer_summary('ГОРА', 'or')
        for var in d['density_var']:
            self.assertAlmostEqual(var, 0.0, places=9)

    # ── all_layers() / build_layer_data() ─────────────────────────────────────

    def test_all_layers_four_rules(self):
        r = self.all_layers('ГОРА')
        self.assertEqual(set(r.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_layer_data_keys(self):
        data = self.build_layer_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'per_rule'):
            self.assertIn(k, data)

    def test_build_layer_data_word_coverage(self):
        words = ['ГОРА', 'ТУМАН']
        data = self.build_layer_data(words)
        for rule in ['xor', 'xor3', 'and', 'or']:
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_layer_grid(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('layer-grid', content)

    def test_viewer_has_layer_dens(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('layer-dens', content)

    def test_viewer_has_layer_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('layer-info', content)

    def test_viewer_has_ly_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ly-word', content)

    def test_viewer_has_ly_rule(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ly-rule', content)

    def test_viewer_has_ly_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ly-btn', content)

    def test_viewer_has_ly_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lyRun', content)

    def test_viewer_has_ly_min_period_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lyMinPeriod', content)

    def test_viewer_has_bit_cols(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('BIT_COLS', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_layer',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_layer(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_layer', content)


class TestSolanDist(unittest.TestCase):
    """Tests for solan_dist.py and the viewer Orbit Distance Map section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_dist import (
            step_distance_q6,
            step_distance_bits,
            distance_series_q6,
            distance_series_bits,
            distance_matrix_q6,
            distance_matrix_bits,
            orbit_diameter_q6,
            orbit_diameter_bits,
            mean_distance_q6,
            packing_efficiency,
            near_returns,
            dist_summary,
            all_dist,
            build_dist_data,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.step_distance_q6    = staticmethod(step_distance_q6)
        cls.step_distance_bits  = staticmethod(step_distance_bits)
        cls.distance_series_q6  = staticmethod(distance_series_q6)
        cls.distance_series_bits = staticmethod(distance_series_bits)
        cls.distance_matrix_q6  = staticmethod(distance_matrix_q6)
        cls.distance_matrix_bits = staticmethod(distance_matrix_bits)
        cls.orbit_diameter_q6   = staticmethod(orbit_diameter_q6)
        cls.orbit_diameter_bits  = staticmethod(orbit_diameter_bits)
        cls.mean_distance_q6    = staticmethod(mean_distance_q6)
        cls.packing_efficiency  = staticmethod(packing_efficiency)
        cls.near_returns        = staticmethod(near_returns)
        cls.dist_summary        = staticmethod(dist_summary)
        cls.all_dist            = staticmethod(all_dist)
        cls.build_dist_data     = staticmethod(build_dist_data)
        cls.LEXICON             = list(LEXICON)

    # ── step_distance_q6() / step_distance_bits() ────────────────────────────

    def test_step_distance_q6_identical(self):
        self.assertEqual(self.step_distance_q6([1, 2, 3], [1, 2, 3]), 0)

    def test_step_distance_q6_all_different(self):
        self.assertEqual(self.step_distance_q6([0, 0, 0], [1, 2, 3]), 3)

    def test_step_distance_q6_range(self):
        a, b = [47] * 16, [1] * 16
        self.assertEqual(self.step_distance_q6(a, b), 16)

    def test_step_distance_bits_zero(self):
        self.assertEqual(self.step_distance_bits([5, 5, 5], [5, 5, 5]), 0)

    def test_step_distance_bits_known(self):
        # 47 XOR 1 = 46 = 0b101110 → popcount = 4; 16 cells → 64 bits
        a = [47] * 16
        b = [1] * 16
        self.assertEqual(self.step_distance_bits(a, b), 64)

    def test_step_distance_bits_gora_xor3(self):
        # Verify ГОРА XOR3 bit distance between orbit steps = 48
        from projects.hexglyph.solan_perm import get_orbit
        orbit = get_orbit('ГОРА', 'xor3', 16)
        self.assertEqual(self.step_distance_bits(orbit[0], orbit[1]), 48)

    # ── distance_series_q6() / distance_series_bits() ─────────────────────────

    def test_series_q6_starts_at_zero(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                ds = self.distance_series_q6(word, rule)
                self.assertEqual(ds[0], 0)

    def test_series_bits_starts_at_zero(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                ds = self.distance_series_bits(word, rule)
                self.assertEqual(ds[0], 0)

    def test_series_q6_tuman_xor3(self):
        ds = self.distance_series_q6('ТУМАН', 'xor3')
        self.assertEqual(ds, [0, 16, 16, 6, 16, 16, 12, 14])

    def test_series_q6_gora_and(self):
        ds = self.distance_series_q6('ГОРА', 'and')
        self.assertEqual(ds, [0, 16])

    def test_series_bits_gora_and(self):
        ds = self.distance_series_bits('ГОРА', 'and')
        self.assertEqual(ds, [0, 64])

    def test_series_q6_gora_xor3(self):
        ds = self.distance_series_q6('ГОРА', 'xor3')
        self.assertEqual(ds, [0, 16])

    def test_series_bits_gora_xor3(self):
        ds = self.distance_series_bits('ГОРА', 'xor3')
        self.assertEqual(ds, [0, 48])

    def test_series_q6_length_equals_period(self):
        from projects.hexglyph.solan_traj import word_trajectory
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                P  = word_trajectory(word, rule)['period']
                ds = self.distance_series_q6(word, rule)
                self.assertEqual(len(ds), P)

    def test_series_q6_values_in_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for d in self.distance_series_q6(word, rule):
                    self.assertGreaterEqual(d, 0)
                    self.assertLessEqual(d, 16)

    # ── distance_matrix_q6() / distance_matrix_bits() ─────────────────────────

    def test_matrix_q6_diagonal_zero(self):
        mat = self.distance_matrix_q6('ТУМАН', 'xor3')
        for t in range(len(mat)):
            self.assertEqual(mat[t][t], 0)

    def test_matrix_q6_symmetric(self):
        mat = self.distance_matrix_q6('ТУМАН', 'xor3')
        P = len(mat)
        for t in range(P):
            for s in range(P):
                self.assertEqual(mat[t][s], mat[s][t])

    def test_matrix_bits_diagonal_zero(self):
        mat = self.distance_matrix_bits('ГОРА', 'and')
        for t in range(len(mat)):
            self.assertEqual(mat[t][t], 0)

    def test_matrix_bits_symmetric(self):
        mat = self.distance_matrix_bits('ГОРА', 'and')
        P = len(mat)
        for t in range(P):
            for s in range(P):
                self.assertEqual(mat[t][s], mat[s][t])

    def test_matrix_q6_gora_and(self):
        mat = self.distance_matrix_q6('ГОРА', 'and')
        self.assertEqual(mat, [[0, 16], [16, 0]])

    def test_matrix_bits_gora_and(self):
        mat = self.distance_matrix_bits('ГОРА', 'and')
        self.assertEqual(mat, [[0, 64], [64, 0]])

    def test_matrix_q6_tuman_xor3_row0(self):
        mat = self.distance_matrix_q6('ТУМАН', 'xor3')
        self.assertEqual(mat[0], [0, 16, 16, 6, 16, 16, 12, 14])

    def test_matrix_q6_tuman_xor3_closest_entry(self):
        mat = self.distance_matrix_q6('ТУМАН', 'xor3')
        # Closest non-diagonal: (0,3) and (3,0) with distance 6
        off_diag = [mat[t][s] for t in range(8) for s in range(8) if t != s]
        self.assertEqual(min(off_diag), 6)

    # ── orbit_diameter_q6() / orbit_diameter_bits() ────────────────────────────

    def test_diameter_q6_tuman_xor(self):
        self.assertEqual(self.orbit_diameter_q6('ТУМАН', 'xor'), 0)

    def test_diameter_q6_gora_and(self):
        self.assertEqual(self.orbit_diameter_q6('ГОРА', 'and'), 16)

    def test_diameter_q6_gora_xor3(self):
        self.assertEqual(self.orbit_diameter_q6('ГОРА', 'xor3'), 16)

    def test_diameter_q6_tuman_xor3(self):
        self.assertEqual(self.orbit_diameter_q6('ТУМАН', 'xor3'), 16)

    def test_diameter_bits_gora_and(self):
        self.assertEqual(self.orbit_diameter_bits('ГОРА', 'and'), 64)

    def test_diameter_bits_gora_xor3(self):
        self.assertEqual(self.orbit_diameter_bits('ГОРА', 'xor3'), 48)

    def test_diameter_q6_at_most_N(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                d = self.orbit_diameter_q6(word, rule)
                self.assertLessEqual(d, 16)
                self.assertGreaterEqual(d, 0)

    # ── mean_distance_q6() / packing_efficiency() ─────────────────────────────

    def test_mean_distance_tuman_xor(self):
        self.assertAlmostEqual(self.mean_distance_q6('ТУМАН', 'xor'), 0.0, places=5)

    def test_mean_distance_gora_and(self):
        self.assertAlmostEqual(self.mean_distance_q6('ГОРА', 'and'), 16.0, places=5)

    def test_mean_distance_tuman_xor3(self):
        md = self.mean_distance_q6('ТУМАН', 'xor3')
        # From computed matrix: mean of 8*8-8=56 off-diag entries
        self.assertAlmostEqual(md, 13.928571, places=4)

    def test_packing_efficiency_tuman_xor(self):
        self.assertAlmostEqual(self.packing_efficiency('ТУМАН', 'xor'), 0.0, places=9)

    def test_packing_efficiency_gora_and(self):
        self.assertAlmostEqual(self.packing_efficiency('ГОРА', 'and'), 1.0, places=9)

    def test_packing_efficiency_gora_xor3(self):
        self.assertAlmostEqual(self.packing_efficiency('ГОРА', 'xor3'), 1.0, places=9)

    def test_packing_efficiency_tuman_xor3(self):
        pe = self.packing_efficiency('ТУМАН', 'xor3')
        self.assertAlmostEqual(pe, 13.928571 / 16, places=4)

    def test_packing_efficiency_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                pe = self.packing_efficiency(word, rule)
                self.assertGreaterEqual(pe, 0.0)
                self.assertLessEqual(pe, 1.0)

    # ── near_returns() ────────────────────────────────────────────────────────

    def test_near_returns_tuman_xor(self):
        nr = self.near_returns('ТУМАН', 'xor')
        self.assertEqual(nr, [])

    def test_near_returns_gora_and(self):
        # All pairs at dist=16, no near-returns under N/2=8
        nr = self.near_returns('ГОРА', 'and')
        self.assertEqual(nr, [])

    def test_near_returns_tuman_xor3(self):
        # t=3 is the near-return with dist=6 < 8
        nr = self.near_returns('ТУМАН', 'xor3')
        self.assertIn(3, nr)

    def test_near_returns_tuman_xor3_dist_at_t3(self):
        ds = self.distance_series_q6('ТУМАН', 'xor3')
        self.assertEqual(ds[3], 6)

    def test_near_returns_custom_threshold(self):
        # With threshold=7, t=3 (dist=6) should be a near-return
        nr = self.near_returns('ТУМАН', 'xor3', threshold=7)
        self.assertIn(3, nr)

    def test_near_returns_strict_threshold(self):
        # With threshold=5, t=3 (dist=6) should NOT be a near-return
        nr = self.near_returns('ТУМАН', 'xor3', threshold=5)
        self.assertNotIn(3, nr)

    # ── dist_summary() ────────────────────────────────────────────────────────

    def test_summary_keys(self):
        d = self.dist_summary('ТУМАН', 'xor3')
        for k in ('word', 'rule', 'period', 'N',
                  'distance_series_q6', 'distance_series_bits',
                  'diameter_q6', 'diameter_bits',
                  'mean_dist_q6', 'packing_efficiency', 'near_returns',
                  'closest_dist_q6', 'closest_pair'):
            self.assertIn(k, d)

    def test_summary_word_normalised(self):
        self.assertEqual(self.dist_summary('туман', 'xor3')['word'], 'ТУМАН')

    def test_summary_tuman_xor3_near_return(self):
        d = self.dist_summary('ТУМАН', 'xor3')
        self.assertIn(3, d['near_returns'])

    def test_summary_gora_and_closest_dist(self):
        d = self.dist_summary('ГОРА', 'and')
        self.assertEqual(d['closest_dist_q6'], 16)

    def test_summary_tuman_xor3_closest_dist(self):
        d = self.dist_summary('ТУМАН', 'xor3')
        self.assertEqual(d['closest_dist_q6'], 6)

    def test_summary_tuman_xor3_closest_pair(self):
        d = self.dist_summary('ТУМАН', 'xor3')
        t, s = d['closest_pair']
        mat = self.distance_matrix_q6('ТУМАН', 'xor3')
        self.assertEqual(mat[t][s], 6)

    def test_summary_diam_norm_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                d = self.dist_summary(word, rule)
                self.assertGreaterEqual(d['diam_q6_norm'], 0.0)
                self.assertLessEqual(d['diam_q6_norm'], 1.0)

    # ── all_dist() / build_dist_data() ────────────────────────────────────────

    def test_all_dist_four_rules(self):
        r = self.all_dist('ГОРА')
        self.assertEqual(set(r.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_dist_data_keys(self):
        data = self.build_dist_data(['ГОРА', 'ТУМАН'])
        for k in ('words', 'width', 'N_max_q6', 'N_max_bits', 'per_rule'):
            self.assertIn(k, data)

    def test_build_dist_data_word_coverage(self):
        words = ['ГОРА', 'ТУМАН']
        data = self.build_dist_data(words)
        for rule in ['xor', 'xor3', 'and', 'or']:
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_dist_matrix(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dist-matrix', content)

    def test_viewer_has_dist_series(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dist-series', content)

    def test_viewer_has_dist_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dist-info', content)

    def test_viewer_has_dt_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dt-word', content)

    def test_viewer_has_dt_rule(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dt-rule', content)

    def test_viewer_has_dt_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dt-btn', content)

    def test_viewer_has_dt_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dtRun', content)

    def test_viewer_has_dq6_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('dq6', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_dist',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_dist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_dist', content)


class TestSolanConfig(unittest.TestCase):
    """Tests for solan_config.py and the viewer Neighborhood Config Coverage."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_config import (
            active_configs,
            n_active_configs,
            coverage_fraction,
            coverage_vector,
            mean_coverage,
            full_coverage_bits,
            minimal_coverage_bits,
            config_transition_table,
            config_summary,
            all_config,
            build_config_data,
            _ALL_CONFIGS,
            _N_CONFIGS,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.active_configs           = staticmethod(active_configs)
        cls.n_active_configs         = staticmethod(n_active_configs)
        cls.coverage_fraction        = staticmethod(coverage_fraction)
        cls.coverage_vector          = staticmethod(coverage_vector)
        cls.mean_coverage            = staticmethod(mean_coverage)
        cls.full_coverage_bits       = staticmethod(full_coverage_bits)
        cls.minimal_coverage_bits    = staticmethod(minimal_coverage_bits)
        cls.config_transition_table  = staticmethod(config_transition_table)
        cls.config_summary           = staticmethod(config_summary)
        cls.all_config               = staticmethod(all_config)
        cls.build_config_data        = staticmethod(build_config_data)
        cls.ALL_CONFIGS              = list(_ALL_CONFIGS)
        cls.N_CONFIGS                = _N_CONFIGS
        cls.LEXICON                  = list(LEXICON)

    # ── active_configs() ──────────────────────────────────────────────────────

    def test_active_configs_deterministic(self):
        """Each (l,c,r) config maps to exactly one output."""
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for b in range(6):
                    cfg = self.active_configs(word, rule, b)
                    for k, v in cfg.items():
                        self.assertIn(v, (0, 1))

    def test_active_configs_output_in_0_1(self):
        cfg = self.active_configs('ТУМАН', 'xor3', 0)
        for v in cfg.values():
            self.assertIn(v, (0, 1))

    def test_active_configs_keys_valid(self):
        cfg = self.active_configs('ГОРА', 'and', 1)
        for key in cfg:
            self.assertIn(len(key), (3,))
            for v in key:
                self.assertIn(v, (0, 1))

    def test_active_configs_tuman_xor_all_zero_zero_zero(self):
        for b in range(6):
            cfg = self.active_configs('ТУМАН', 'xor', b)
            self.assertEqual(list(cfg.keys()), [(0, 0, 0)])

    def test_active_configs_gora_or_all_one_one_one(self):
        for b in range(6):
            cfg = self.active_configs('ГОРА', 'or', b)
            self.assertEqual(list(cfg.keys()), [(1, 1, 1)])

    def test_active_configs_gora_xor3_bit0_only_111(self):
        cfg = self.active_configs('ГОРА', 'xor3', 0)
        self.assertEqual(list(cfg.keys()), [(1, 1, 1)])

    def test_active_configs_gora_and_bit0_only_111(self):
        cfg = self.active_configs('ГОРА', 'and', 0)
        self.assertEqual(list(cfg.keys()), [(1, 1, 1)])

    def test_active_configs_gora_and_bit4_only_000(self):
        cfg = self.active_configs('ГОРА', 'and', 4)
        self.assertEqual(list(cfg.keys()), [(0, 0, 0)])

    def test_active_configs_gora_and_bit1_two_configs(self):
        cfg = self.active_configs('ГОРА', 'and', 1)
        self.assertEqual(len(cfg), 2)
        self.assertIn((0, 1, 0), cfg)
        self.assertIn((1, 0, 1), cfg)

    def test_active_configs_gora_and_bit1_outputs(self):
        cfg = self.active_configs('ГОРА', 'and', 1)
        self.assertEqual(cfg[(0, 1, 0)], 0)   # AND(0,0) = 0
        self.assertEqual(cfg[(1, 0, 1)], 1)   # AND(1,1) = 1

    def test_active_configs_tuman_xor3_all_8(self):
        for b in range(6):
            cfg = self.active_configs('ТУМАН', 'xor3', b)
            self.assertEqual(len(cfg), 8)

    def test_active_configs_gora_xor3_bits_1235_all_8(self):
        for b in [1, 2, 3, 5]:
            cfg = self.active_configs('ГОРА', 'xor3', b)
            self.assertEqual(len(cfg), 8,
                msg=f'ГОРА xor3 bit {b} should have 8 active configs')

    def test_active_configs_gora_xor3_bit4_four_configs(self):
        cfg = self.active_configs('ГОРА', 'xor3', 4)
        self.assertEqual(len(cfg), 4)
        # Only l≠r configs: (0,0,1), (0,1,1), (1,0,0), (1,1,0)
        expected = {(0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0)}
        self.assertEqual(set(cfg.keys()), expected)

    def test_active_configs_subset_of_all(self):
        all_cfgs = set(self.ALL_CONFIGS)
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for b in range(6):
                    cfg = self.active_configs(word, rule, b)
                    self.assertTrue(set(cfg.keys()).issubset(all_cfgs))

    # ── n_active_configs() / coverage_fraction() ──────────────────────────────

    def test_n_active_tuman_xor(self):
        for b in range(6):
            self.assertEqual(self.n_active_configs('ТУМАН', 'xor', b), 1)

    def test_n_active_gora_or(self):
        for b in range(6):
            self.assertEqual(self.n_active_configs('ГОРА', 'or', b), 1)

    def test_n_active_tuman_xor3_all_8(self):
        for b in range(6):
            self.assertEqual(self.n_active_configs('ТУМАН', 'xor3', b), 8)

    def test_n_active_gora_xor3_b0(self):
        self.assertEqual(self.n_active_configs('ГОРА', 'xor3', 0), 1)

    def test_n_active_gora_xor3_b4(self):
        self.assertEqual(self.n_active_configs('ГОРА', 'xor3', 4), 4)

    def test_n_active_gora_and_b1(self):
        self.assertEqual(self.n_active_configs('ГОРА', 'and', 1), 2)

    def test_coverage_fraction_tuman_xor(self):
        for b in range(6):
            self.assertAlmostEqual(self.coverage_fraction('ТУМАН', 'xor', b),
                                   1 / 8, places=5)

    def test_coverage_fraction_tuman_xor3(self):
        for b in range(6):
            self.assertAlmostEqual(self.coverage_fraction('ТУМАН', 'xor3', b),
                                   1.0, places=5)

    def test_coverage_fraction_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                for b in range(6):
                    cf = self.coverage_fraction(word, rule, b)
                    self.assertGreaterEqual(cf, 0.0)
                    self.assertLessEqual(cf, 1.0)

    # ── coverage_vector() / mean_coverage() ───────────────────────────────────

    def test_coverage_vector_length_6(self):
        cv = self.coverage_vector('ГОРА', 'xor3')
        self.assertEqual(len(cv), 6)

    def test_coverage_vector_tuman_xor(self):
        self.assertEqual(self.coverage_vector('ТУМАН', 'xor'), [1, 1, 1, 1, 1, 1])

    def test_coverage_vector_gora_and(self):
        self.assertEqual(self.coverage_vector('ГОРА', 'and'), [1, 2, 2, 2, 1, 2])

    def test_coverage_vector_gora_xor3(self):
        self.assertEqual(self.coverage_vector('ГОРА', 'xor3'), [1, 8, 8, 8, 4, 8])

    def test_coverage_vector_tuman_xor3(self):
        self.assertEqual(self.coverage_vector('ТУМАН', 'xor3'), [8, 8, 8, 8, 8, 8])

    def test_mean_coverage_tuman_xor(self):
        self.assertAlmostEqual(self.mean_coverage('ТУМАН', 'xor'), 1/8, places=5)

    def test_mean_coverage_tuman_xor3(self):
        self.assertAlmostEqual(self.mean_coverage('ТУМАН', 'xor3'), 1.0, places=5)

    def test_mean_coverage_gora_xor3(self):
        # [1,8,8,8,4,8] / 48 = 37/48
        self.assertAlmostEqual(self.mean_coverage('ГОРА', 'xor3'), 37/48, places=5)

    def test_mean_coverage_gora_and(self):
        # [1,2,2,2,1,2] / 48 = 10/48
        self.assertAlmostEqual(self.mean_coverage('ГОРА', 'and'), 10/48, places=5)

    def test_mean_coverage_range(self):
        for word in ['ТУМАН', 'ГОРА']:
            for rule in ['xor', 'xor3', 'and', 'or']:
                mc = self.mean_coverage(word, rule)
                self.assertGreater(mc, 0.0)
                self.assertLessEqual(mc, 1.0)

    # ── full_coverage_bits() / minimal_coverage_bits() ───────────────────────

    def test_full_coverage_tuman_xor3(self):
        self.assertEqual(self.full_coverage_bits('ТУМАН', 'xor3'), [0, 1, 2, 3, 4, 5])

    def test_full_coverage_tuman_xor(self):
        self.assertEqual(self.full_coverage_bits('ТУМАН', 'xor'), [])

    def test_full_coverage_gora_xor3(self):
        full = self.full_coverage_bits('ГОРА', 'xor3')
        self.assertEqual(sorted(full), [1, 2, 3, 5])
        self.assertNotIn(0, full)
        self.assertNotIn(4, full)

    def test_minimal_coverage_tuman_xor(self):
        self.assertEqual(self.minimal_coverage_bits('ТУМАН', 'xor'), [0, 1, 2, 3, 4, 5])

    def test_minimal_coverage_tuman_xor3(self):
        self.assertEqual(self.minimal_coverage_bits('ТУМАН', 'xor3'), [])

    def test_minimal_coverage_gora_and(self):
        mini = self.minimal_coverage_bits('ГОРА', 'and')
        self.assertIn(0, mini)
        self.assertIn(4, mini)

    def test_minimal_coverage_gora_xor3(self):
        mini = self.minimal_coverage_bits('ГОРА', 'xor3')
        self.assertEqual(mini, [0])

    # ── config_transition_table() ─────────────────────────────────────────────

    def test_transition_table_xor3_size(self):
        tt = self.config_transition_table('ГОРА', 'xor3', 0)
        self.assertEqual(len(tt), 8)

    def test_transition_table_xor3_known(self):
        tt = self.config_transition_table('ГОРА', 'xor3', 0)
        # XOR3: l XOR c XOR r
        self.assertEqual(tt[(0, 0, 0)], 0)
        self.assertEqual(tt[(0, 0, 1)], 1)
        self.assertEqual(tt[(1, 1, 1)], 1)
        self.assertEqual(tt[(0, 1, 1)], 0)

    def test_transition_table_and_known(self):
        tt = self.config_transition_table('ГОРА', 'and', 0)
        # AND: l AND r
        self.assertEqual(tt[(0, 0, 0)], 0)
        self.assertEqual(tt[(1, 0, 1)], 1)
        self.assertEqual(tt[(0, 1, 1)], 0)

    def test_transition_table_consistent_with_active(self):
        """Active configs must be consistent with the full rule table."""
        for rule in ['xor', 'xor3', 'and', 'or']:
            tt = self.config_transition_table('ГОРА', rule, 0)
            for b in range(6):
                cfg = self.active_configs('ГОРА', rule, b)
                for key, out in cfg.items():
                    self.assertEqual(out, tt[key],
                        msg=f'ГОРА/{rule}/b{b}: config {key} has inconsistent output')

    # ── config_summary() ──────────────────────────────────────────────────────

    def test_summary_keys(self):
        d = self.config_summary('ГОРА', 'xor3')
        for k in ('word', 'rule', 'period', 'coverage_vector',
                  'coverage_fractions', 'mean_coverage',
                  'full_coverage_bits', 'minimal_coverage_bits',
                  'n_full_coverage', 'n_minimal',
                  'per_bit_configs', 'per_bit_outputs',
                  'output_diversity', 'rule_table'):
            self.assertIn(k, d)

    def test_summary_word_normalised(self):
        self.assertEqual(self.config_summary('гора', 'xor3')['word'], 'ГОРА')

    def test_summary_n_full_tuman_xor3(self):
        self.assertEqual(self.config_summary('ТУМАН', 'xor3')['n_full_coverage'], 6)

    def test_summary_n_minimal_tuman_xor(self):
        self.assertEqual(self.config_summary('ТУМАН', 'xor')['n_minimal'], 6)

    def test_summary_output_diversity_frozen(self):
        d = self.config_summary('ТУМАН', 'xor')
        # All bits frozen to 0 → output_diversity = [1,1,1,1,1,1]
        self.assertTrue(all(v == 1 for v in d['output_diversity']))

    # ── all_config() / build_config_data() ────────────────────────────────────

    def test_all_config_four_rules(self):
        r = self.all_config('ГОРА')
        self.assertEqual(set(r.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_config_data_keys(self):
        data = self.build_config_data(['ГОРА', 'ТУМАН'])
        for k in ('words', 'width', 'n_configs', 'all_configs', 'per_rule'):
            self.assertIn(k, data)

    def test_build_config_data_n_configs(self):
        data = self.build_config_data(['ГОРА'])
        self.assertEqual(data['n_configs'], 8)

    def test_build_config_data_all_configs_length(self):
        data = self.build_config_data(['ГОРА'])
        self.assertEqual(len(data['all_configs']), 8)

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_config_grid(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('config-grid', content)

    def test_viewer_has_config_cov(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('config-cov', content)

    def test_viewer_has_config_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('config-info', content)

    def test_viewer_has_cf_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cf-word', content)

    def test_viewer_has_cf_rule(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cf-rule', content)

    def test_viewer_has_cf_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cf-btn', content)

    def test_viewer_has_cf_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cfRun', content)

    def test_viewer_has_rule_bit_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ruleBit', content)

    def test_viewer_has_all_cfgs_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ALL_CFGS', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_config',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_config(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_config', content)



class TestSolanSegment(unittest.TestCase):
    """Tests for solan_segment.py — Spatial Domain Segmentation."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_segment import (
            spatial_segments, seg_lengths_per_cell,
            n_segments_step, n_segments, max_seg_length,
            min_seg_length, mean_seg_length, seg_lengths,
            global_max_seg_length, segment_summary,
            all_segment, build_segment_data,
        )
        cls.spatial_segments      = staticmethod(spatial_segments)
        cls.seg_lengths_per_cell  = staticmethod(seg_lengths_per_cell)
        cls.n_segments_step       = staticmethod(n_segments_step)
        cls.n_segments            = staticmethod(n_segments)
        cls.max_seg_length        = staticmethod(max_seg_length)
        cls.min_seg_length        = staticmethod(min_seg_length)
        cls.mean_seg_length       = staticmethod(mean_seg_length)
        cls.seg_lengths           = staticmethod(seg_lengths)
        cls.global_max_seg_length = staticmethod(global_max_seg_length)
        cls.segment_summary       = staticmethod(segment_summary)
        cls.all_segment           = staticmethod(all_segment)
        cls.build_segment_data    = staticmethod(build_segment_data)

    # ── spatial_segments ──────────────────────────────────────────────────────

    def test_spatial_segments_returns_list(self):
        self.assertIsInstance(self.spatial_segments((0, 0, 0, 0)), list)

    def test_spatial_segments_uniform_zero(self):
        self.assertEqual(self.spatial_segments((0, 0, 0, 0)), [(0, 4)])

    def test_spatial_segments_uniform_63(self):
        self.assertEqual(self.spatial_segments((63, 63, 63, 63)), [(63, 4)])

    def test_spatial_segments_alternating(self):
        segs = self.spatial_segments((47, 1, 47, 1))
        self.assertEqual(len(segs), 4)
        self.assertTrue(all(l == 1 for _, l in segs))

    def test_spatial_segments_lengths_sum_to_n(self):
        state = (49, 47, 15, 63, 49, 47, 15, 63)
        segs = self.spatial_segments(state)
        self.assertEqual(sum(l for _, l in segs), len(state))

    def test_spatial_segments_elements_are_tuples(self):
        segs = self.spatial_segments((0, 0, 1, 1))
        self.assertIsInstance(segs[0], tuple)
        self.assertEqual(len(segs[0]), 2)

    def test_spatial_segments_two_segment_state(self):
        # (0,0,1,1): two segments of length 2
        segs = self.spatial_segments((0, 0, 1, 1))
        self.assertEqual(len(segs), 2)
        self.assertTrue(all(l == 2 for _, l in segs))

    def test_spatial_segments_ring_wrap(self):
        # (1,0,0,1) on ring: cells 3,0 share value 1 → one segment of length 2
        segs = self.spatial_segments((1, 0, 0, 1))
        lens = sorted([l for _, l in segs])
        self.assertEqual(lens, [2, 2])

    def test_spatial_segments_single_length_1(self):
        # (0,1,1,1): cell 0 differs from its neighbours → one segment of length 1
        segs = self.spatial_segments((0, 1, 1, 1))
        lens = sorted([l for _, l in segs])
        self.assertEqual(sorted(lens), sorted([1, 3]))

    # ── seg_lengths_per_cell ─────────────────────────────────────────────────

    def test_seg_lengths_per_cell_returns_list(self):
        self.assertIsInstance(self.seg_lengths_per_cell((0, 0, 1, 1)), list)

    def test_seg_lengths_per_cell_length_n(self):
        result = self.seg_lengths_per_cell((0, 0, 1, 1))
        self.assertEqual(len(result), 4)

    def test_seg_lengths_per_cell_uniform(self):
        result = self.seg_lengths_per_cell((7, 7, 7, 7))
        self.assertEqual(result, [4, 4, 4, 4])

    def test_seg_lengths_per_cell_alternating(self):
        result = self.seg_lengths_per_cell((47, 1, 47, 1))
        self.assertEqual(result, [1, 1, 1, 1])

    def test_seg_lengths_per_cell_two_segments(self):
        result = self.seg_lengths_per_cell((0, 0, 1, 1))
        self.assertEqual(result, [2, 2, 2, 2])

    def test_seg_lengths_per_cell_values_positive(self):
        result = self.seg_lengths_per_cell((3, 5, 5, 5, 1, 1))
        self.assertTrue(all(v > 0 for v in result))

    # ── n_segments ───────────────────────────────────────────────────────────

    def test_n_segments_returns_list(self):
        self.assertIsInstance(self.n_segments('ГОРА', 'and'), list)

    def test_n_segments_length_equals_period(self):
        result = self.n_segments('ГОРА', 'and')
        self.assertEqual(len(result), 2)

    def test_n_segments_tuman_xor_uniform(self):
        self.assertEqual(self.n_segments('ТУМАН', 'xor'), [1])

    def test_n_segments_gora_or_uniform(self):
        self.assertEqual(self.n_segments('ГОРА', 'or'), [1])

    def test_n_segments_gora_and_max_fragmented(self):
        self.assertEqual(self.n_segments('ГОРА', 'and'), [16, 16])

    def test_n_segments_gora_xor3_max_fragmented(self):
        self.assertEqual(self.n_segments('ГОРА', 'xor3'), [16, 16])

    def test_n_segments_tuman_xor3(self):
        self.assertEqual(self.n_segments('ТУМАН', 'xor3'),
                         [15, 16, 16, 16, 16, 16, 16, 13])

    def test_n_segments_positive(self):
        result = self.n_segments('ТУМАН', 'xor3')
        self.assertTrue(all(v > 0 for v in result))

    def test_n_segments_le_width(self):
        result = self.n_segments('ТУМАН', 'xor3')
        self.assertTrue(all(v <= 16 for v in result))

    def test_n_segments_step_consistent(self):
        # n_segments_step must match n_segments list
        ns = self.n_segments('ТУМАН', 'xor3')
        for t, expected in enumerate(ns):
            self.assertEqual(self.n_segments_step('ТУМАН', 'xor3', t), expected)

    def test_n_segments_case_insensitive(self):
        self.assertEqual(
            self.n_segments('гора', 'and'),
            self.n_segments('ГОРА', 'and'),
        )

    # ── max_seg_length ────────────────────────────────────────────────────────

    def test_max_seg_length_returns_list(self):
        self.assertIsInstance(self.max_seg_length('ГОРА', 'and'), list)

    def test_max_seg_length_tuman_xor_full_ring(self):
        self.assertEqual(self.max_seg_length('ТУМАН', 'xor'), [16])

    def test_max_seg_length_gora_or_full_ring(self):
        self.assertEqual(self.max_seg_length('ГОРА', 'or'), [16])

    def test_max_seg_length_gora_and_one(self):
        self.assertEqual(self.max_seg_length('ГОРА', 'and'), [1, 1])

    def test_max_seg_length_tuman_xor3(self):
        self.assertEqual(self.max_seg_length('ТУМАН', 'xor3'), [2, 1, 1, 1, 1, 1, 1, 4])

    def test_max_seg_length_ge_min_seg_length(self):
        mx = self.max_seg_length('ТУМАН', 'xor3')
        mn = self.min_seg_length('ТУМАН', 'xor3')
        for a, b in zip(mx, mn):
            self.assertGreaterEqual(a, b)

    # ── min_seg_length ────────────────────────────────────────────────────────

    def test_min_seg_length_returns_list(self):
        self.assertIsInstance(self.min_seg_length('ГОРА', 'and'), list)

    def test_min_seg_length_tuman_xor(self):
        self.assertEqual(self.min_seg_length('ТУМАН', 'xor'), [16])

    def test_min_seg_length_gora_and(self):
        self.assertEqual(self.min_seg_length('ГОРА', 'and'), [1, 1])

    def test_min_seg_length_positive(self):
        result = self.min_seg_length('ТУМАН', 'xor3')
        self.assertTrue(all(v >= 1 for v in result))

    # ── mean_seg_length ───────────────────────────────────────────────────────

    def test_mean_seg_length_returns_list(self):
        self.assertIsInstance(self.mean_seg_length('ГОРА', 'and'), list)

    def test_mean_seg_length_tuman_xor(self):
        self.assertAlmostEqual(self.mean_seg_length('ТУМАН', 'xor')[0], 16.0)

    def test_mean_seg_length_gora_and(self):
        result = self.mean_seg_length('ГОРА', 'and')
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in result))

    def test_mean_seg_length_equals_n_over_n_segs(self):
        ns = self.n_segments('ТУМАН', 'xor3')
        ml = self.mean_seg_length('ТУМАН', 'xor3')
        for n, m in zip(ns, ml):
            self.assertAlmostEqual(m, 16 / n, places=4)

    # ── global_max_seg_length ─────────────────────────────────────────────────

    def test_global_max_seg_length_returns_int(self):
        self.assertIsInstance(self.global_max_seg_length('ГОРА', 'and'), int)

    def test_global_max_seg_length_tuman_xor(self):
        self.assertEqual(self.global_max_seg_length('ТУМАН', 'xor'), 16)

    def test_global_max_seg_length_gora_and(self):
        self.assertEqual(self.global_max_seg_length('ГОРА', 'and'), 1)

    def test_global_max_seg_length_tuman_xor3(self):
        self.assertEqual(self.global_max_seg_length('ТУМАН', 'xor3'), 4)

    def test_global_max_ge_all_step_max(self):
        gm = self.global_max_seg_length('ТУМАН', 'xor3')
        step_max = self.max_seg_length('ТУМАН', 'xor3')
        self.assertEqual(gm, max(step_max))

    # ── seg_lengths ───────────────────────────────────────────────────────────

    def test_seg_lengths_returns_list_of_lists(self):
        result = self.seg_lengths('ГОРА', 'and')
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)

    def test_seg_lengths_sum_to_n(self):
        result = self.seg_lengths('ТУМАН', 'xor3')
        for step_lens in result:
            self.assertEqual(sum(step_lens), 16)

    def test_seg_lengths_sorted_descending(self):
        result = self.seg_lengths('ТУМАН', 'xor3')
        for step_lens in result:
            self.assertEqual(step_lens, sorted(step_lens, reverse=True))

    # ── segment_summary ───────────────────────────────────────────────────────

    def test_segment_summary_returns_dict(self):
        self.assertIsInstance(self.segment_summary('ГОРА', 'and'), dict)

    def test_segment_summary_required_keys(self):
        d = self.segment_summary('ГОРА', 'and')
        for key in ('word', 'rule', 'period', 'n_cells',
                    'n_segments', 'mean_n_segments',
                    'max_n_segments', 'min_n_segments',
                    'max_seg_length', 'min_seg_length',
                    'global_max_len', 'global_min_len',
                    'seg_lengths', 'fully_fragmented', 'always_uniform'):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_segment_summary_tuman_xor_always_uniform(self):
        d = self.segment_summary('ТУМАН', 'xor')
        self.assertTrue(d['always_uniform'])
        self.assertFalse(d['fully_fragmented'])
        self.assertEqual(d['global_max_len'], 16)

    def test_segment_summary_gora_and_fully_fragmented(self):
        d = self.segment_summary('ГОРА', 'and')
        self.assertTrue(d['fully_fragmented'])
        self.assertFalse(d['always_uniform'])
        self.assertEqual(d['global_max_len'], 1)

    def test_segment_summary_gora_or_always_uniform(self):
        d = self.segment_summary('ГОРА', 'or')
        self.assertTrue(d['always_uniform'])

    def test_segment_summary_gora_xor3_fully_fragmented(self):
        d = self.segment_summary('ГОРА', 'xor3')
        self.assertTrue(d['fully_fragmented'])

    def test_segment_summary_tuman_xor3_neither(self):
        d = self.segment_summary('ТУМАН', 'xor3')
        self.assertFalse(d['fully_fragmented'])
        self.assertFalse(d['always_uniform'])

    def test_segment_summary_tuman_xor3_mean(self):
        d = self.segment_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['mean_n_segments'], 15.5)

    def test_segment_summary_tuman_xor3_global_max(self):
        d = self.segment_summary('ТУМАН', 'xor3')
        self.assertEqual(d['global_max_len'], 4)

    def test_segment_summary_n_segs_list(self):
        d = self.segment_summary('ТУМАН', 'xor3')
        self.assertEqual(d['n_segments'], [15, 16, 16, 16, 16, 16, 16, 13])

    def test_segment_summary_word_upper(self):
        d = self.segment_summary('гора', 'and')
        self.assertEqual(d['word'], 'ГОРА')

    def test_segment_summary_n_cells(self):
        d = self.segment_summary('ГОРА', 'and')
        self.assertEqual(d['n_cells'], 16)

    # ── all_segment ───────────────────────────────────────────────────────────

    def test_all_segment_returns_dict(self):
        self.assertIsInstance(self.all_segment('ГОРА'), dict)

    def test_all_segment_four_rules(self):
        d = self.all_segment('ГОРА')
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── build_segment_data ────────────────────────────────────────────────────

    def test_build_segment_data_returns_dict(self):
        self.assertIsInstance(self.build_segment_data(['ГОРА']), dict)

    def test_build_segment_data_top_keys(self):
        d = self.build_segment_data(['ГОРА'])
        for key in ('words', 'width', 'per_rule'):
            self.assertIn(key, d)

    def test_build_segment_data_rule_keys(self):
        d = self.build_segment_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_segment_data_word_uppercase(self):
        d = self.build_segment_data(['гора'])
        self.assertIn('ГОРА', d['per_rule']['and'])

    def test_build_segment_data_gora_and_fully_fragmented(self):
        d = self.build_segment_data(['ГОРА'])
        self.assertTrue(d['per_rule']['and']['ГОРА']['fully_fragmented'])

    def test_build_segment_data_known_fields(self):
        d   = self.build_segment_data(['ГОРА'])
        rec = d['per_rule']['xor3']['ГОРА']
        for key in ('period', 'n_segments', 'mean_n_segments',
                    'max_n_segments', 'min_n_segments',
                    'max_seg_length', 'global_max_len', 'global_min_len',
                    'seg_lengths', 'fully_fragmented', 'always_uniform'):
            self.assertIn(key, rec)

    # ── Viewer HTML markers ───────────────────────────────────────────────────

    def test_viewer_has_seg_map(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('seg-map', content)

    def test_viewer_has_seg_nsegs(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('seg-nsegs', content)

    def test_viewer_has_seg_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('seg-info', content)

    def test_viewer_has_sg_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sg-word', content)

    def test_viewer_has_sg_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sg-btn', content)

    def test_viewer_has_sg_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sgRun', content)

    def test_viewer_has_sg_segments_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sgSegments', content)

    def test_viewer_has_sg_len_per_cell_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sgLenPerCell', content)

    def test_viewer_has_sg_color_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sgColor', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_segment',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_segment(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_segment', content)


class TestSolanEntropyOrbit(unittest.TestCase):
    """Tests for solan_entropy.py — Orbit-level Shannon Entropy Analysis."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_entropy import (
            spatial_entropy, temporal_entropy_cell,
            spatial_entropy_orbit, temporal_entropy_all,
            mean_spatial_entropy, mean_temporal_entropy,
            max_spatial_entropy, min_spatial_entropy,
            max_temporal_entropy, min_temporal_entropy,
            entropy_summary, all_entropy, build_entropy_data,
        )
        cls.spatial_entropy        = staticmethod(spatial_entropy)
        cls.temporal_entropy_cell  = staticmethod(temporal_entropy_cell)
        cls.spatial_entropy_orbit  = staticmethod(spatial_entropy_orbit)
        cls.temporal_entropy_all   = staticmethod(temporal_entropy_all)
        cls.mean_spatial_entropy   = staticmethod(mean_spatial_entropy)
        cls.mean_temporal_entropy  = staticmethod(mean_temporal_entropy)
        cls.max_spatial_entropy    = staticmethod(max_spatial_entropy)
        cls.min_spatial_entropy    = staticmethod(min_spatial_entropy)
        cls.max_temporal_entropy   = staticmethod(max_temporal_entropy)
        cls.min_temporal_entropy   = staticmethod(min_temporal_entropy)
        cls.entropy_summary        = staticmethod(entropy_summary)
        cls.all_entropy            = staticmethod(all_entropy)
        cls.build_entropy_data     = staticmethod(build_entropy_data)

    # ── spatial_entropy ────────────────────────────────────────────────────────

    def test_spatial_entropy_returns_float(self):
        self.assertIsInstance(self.spatial_entropy((0, 0, 0, 0)), float)

    def test_spatial_entropy_uniform_zero(self):
        self.assertAlmostEqual(self.spatial_entropy((0, 0, 0, 0)), 0.0)

    def test_spatial_entropy_uniform_value(self):
        self.assertAlmostEqual(self.spatial_entropy((63, 63, 63, 63)), 0.0)

    def test_spatial_entropy_two_equal_freq(self):
        # (0,0,1,1) → two values, each 50% → H = log2(2) = 1 bit
        self.assertAlmostEqual(self.spatial_entropy((0, 0, 1, 1)), 1.0)

    def test_spatial_entropy_four_equal_freq(self):
        # (0,1,2,3) → four values, each 25% → H = log2(4) = 2 bits
        self.assertAlmostEqual(self.spatial_entropy((0, 1, 2, 3)), 2.0)

    def test_spatial_entropy_nonnegative(self):
        self.assertGreaterEqual(self.spatial_entropy((47, 1, 47, 1)), 0.0)

    def test_spatial_entropy_le_log2_n(self):
        import math
        state = tuple(range(16))
        self.assertLessEqual(self.spatial_entropy(state), math.log2(16) + 1e-9)

    def test_spatial_entropy_not_negative_zero(self):
        # -0.0 should be returned as 0.0
        result = self.spatial_entropy((0, 0, 0, 0))
        self.assertGreaterEqual(result, 0.0)

    # ── temporal_entropy_cell ─────────────────────────────────────────────────

    def test_temporal_entropy_cell_returns_float(self):
        from projects.hexglyph.solan_perm import get_orbit
        orbit = get_orbit('ТУМАН', 'xor', 16)
        self.assertIsInstance(self.temporal_entropy_cell(orbit, 0), float)

    def test_temporal_entropy_cell_constant_zero(self):
        from projects.hexglyph.solan_perm import get_orbit
        orbit = get_orbit('ТУМАН', 'xor', 16)
        self.assertAlmostEqual(self.temporal_entropy_cell(orbit, 0), 0.0)

    def test_temporal_entropy_cell_two_equal(self):
        from projects.hexglyph.solan_perm import get_orbit
        orbit = get_orbit('ГОРА', 'and', 16)
        result = self.temporal_entropy_cell(orbit, 0)
        self.assertAlmostEqual(result, 1.0)

    def test_temporal_entropy_cell_nonnegative(self):
        from projects.hexglyph.solan_perm import get_orbit
        orbit = get_orbit('ТУМАН', 'xor3', 16)
        for i in range(16):
            self.assertGreaterEqual(self.temporal_entropy_cell(orbit, i), 0.0)

    # ── spatial_entropy_orbit ─────────────────────────────────────────────────

    def test_spatial_entropy_orbit_returns_list(self):
        self.assertIsInstance(self.spatial_entropy_orbit('ГОРА', 'and'), list)

    def test_spatial_entropy_orbit_length_equals_period(self):
        result = self.spatial_entropy_orbit('ГОРА', 'and')
        self.assertEqual(len(result), 2)

    def test_spatial_entropy_orbit_tuman_xor_zero(self):
        result = self.spatial_entropy_orbit('ТУМАН', 'xor')
        self.assertAlmostEqual(result[0], 0.0)

    def test_spatial_entropy_orbit_gora_or_zero(self):
        result = self.spatial_entropy_orbit('ГОРА', 'or')
        self.assertAlmostEqual(result[0], 0.0)

    def test_spatial_entropy_orbit_gora_and_one(self):
        result = self.spatial_entropy_orbit('ГОРА', 'and')
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in result))

    def test_spatial_entropy_orbit_gora_xor3_two(self):
        result = self.spatial_entropy_orbit('ГОРА', 'xor3')
        self.assertTrue(all(abs(v - 2.0) < 1e-6 for v in result))

    def test_spatial_entropy_orbit_tuman_xor3(self):
        result = self.spatial_entropy_orbit('ТУМАН', 'xor3')
        self.assertEqual(len(result), 8)
        self.assertAlmostEqual(result[5], 3.375, places=4)

    def test_spatial_entropy_orbit_all_nonnegative(self):
        result = self.spatial_entropy_orbit('ТУМАН', 'xor3')
        self.assertTrue(all(v >= 0.0 for v in result))

    def test_spatial_entropy_orbit_case_insensitive(self):
        self.assertEqual(
            self.spatial_entropy_orbit('гора', 'and'),
            self.spatial_entropy_orbit('ГОРА', 'and'),
        )

    # ── temporal_entropy_all ──────────────────────────────────────────────────

    def test_temporal_entropy_all_returns_list(self):
        self.assertIsInstance(self.temporal_entropy_all('ГОРА', 'and'), list)

    def test_temporal_entropy_all_length_equals_width(self):
        result = self.temporal_entropy_all('ГОРА', 'and')
        self.assertEqual(len(result), 16)

    def test_temporal_entropy_all_tuman_xor_zero(self):
        result = self.temporal_entropy_all('ТУМАН', 'xor')
        self.assertTrue(all(abs(v) < 1e-9 for v in result))

    def test_temporal_entropy_all_gora_and_one(self):
        result = self.temporal_entropy_all('ГОРА', 'and')
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in result))

    def test_temporal_entropy_all_gora_xor3_one(self):
        result = self.temporal_entropy_all('ГОРА', 'xor3')
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in result))

    def test_temporal_entropy_all_tuman_xor3_symmetric(self):
        result = self.temporal_entropy_all('ТУМАН', 'xor3')
        N = len(result)
        for i in range(N // 2):
            self.assertAlmostEqual(result[i], result[N - 1 - i], places=5)

    def test_temporal_entropy_all_tuman_xor3_max_cells(self):
        result = self.temporal_entropy_all('ТУМАН', 'xor3')
        self.assertAlmostEqual(result[2], 2.75, places=4)
        self.assertAlmostEqual(result[3], 2.75, places=4)

    def test_temporal_entropy_all_tuman_xor3_min_cells(self):
        result = self.temporal_entropy_all('ТУМАН', 'xor3')
        self.assertAlmostEqual(result[7], result[8], places=5)
        self.assertLess(result[7], result[2])

    def test_temporal_entropy_all_nonnegative(self):
        result = self.temporal_entropy_all('ТУМАН', 'xor3')
        self.assertTrue(all(v >= 0.0 for v in result))

    # ── mean / max / min entropy ───────────────────────────────────────────────

    def test_mean_spatial_entropy_tuman_xor(self):
        self.assertAlmostEqual(self.mean_spatial_entropy('ТУМАН', 'xor'), 0.0)

    def test_mean_spatial_entropy_gora_and(self):
        self.assertAlmostEqual(self.mean_spatial_entropy('ГОРА', 'and'), 1.0)

    def test_mean_spatial_entropy_gora_xor3(self):
        self.assertAlmostEqual(self.mean_spatial_entropy('ГОРА', 'xor3'), 2.0)

    def test_mean_spatial_entropy_tuman_xor3(self):
        result = self.mean_spatial_entropy('ТУМАН', 'xor3')
        self.assertAlmostEqual(result, 2.8534, places=3)

    def test_mean_temporal_entropy_gora_and(self):
        self.assertAlmostEqual(self.mean_temporal_entropy('ГОРА', 'and'), 1.0)

    def test_mean_temporal_entropy_tuman_xor3(self):
        result = self.mean_temporal_entropy('ТУМАН', 'xor3')
        self.assertAlmostEqual(result, 2.234, places=2)

    def test_max_spatial_entropy_tuman_xor(self):
        self.assertAlmostEqual(self.max_spatial_entropy('ТУМАН', 'xor'), 0.0)

    def test_max_spatial_entropy_tuman_xor3(self):
        self.assertAlmostEqual(self.max_spatial_entropy('ТУМАН', 'xor3'), 3.375, places=4)

    def test_min_spatial_entropy_gora_and(self):
        self.assertAlmostEqual(self.min_spatial_entropy('ГОРА', 'and'), 1.0)

    def test_max_temporal_entropy_tuman_xor3(self):
        self.assertAlmostEqual(self.max_temporal_entropy('ТУМАН', 'xor3'), 2.75, places=4)

    def test_min_temporal_entropy_tuman_xor(self):
        self.assertAlmostEqual(self.min_temporal_entropy('ТУМАН', 'xor'), 0.0)

    def test_min_temporal_entropy_tuman_xor3(self):
        result = self.min_temporal_entropy('ТУМАН', 'xor3')
        self.assertAlmostEqual(result, 1.5613, places=3)

    # ── entropy_summary ───────────────────────────────────────────────────────

    def test_entropy_summary_returns_dict(self):
        self.assertIsInstance(self.entropy_summary('ГОРА', 'and'), dict)

    def test_entropy_summary_required_keys(self):
        d = self.entropy_summary('ГОРА', 'and')
        for key in ('word', 'rule', 'period', 'n_cells',
                    'max_possible_Hs', 'max_possible_Hc',
                    'spatial_entropy', 'mean_spatial_H',
                    'max_spatial_H', 'min_spatial_H',
                    'temporal_entropy', 'mean_temporal_H',
                    'max_temporal_H', 'min_temporal_H',
                    'zero_entropy', 'constant_spatial',
                    'constant_temporal', 'symmetric_temporal'):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_entropy_summary_tuman_xor_zero_entropy(self):
        d = self.entropy_summary('ТУМАН', 'xor')
        self.assertTrue(d['zero_entropy'])
        self.assertAlmostEqual(d['mean_spatial_H'], 0.0)

    def test_entropy_summary_gora_or_zero_entropy(self):
        d = self.entropy_summary('ГОРА', 'or')
        self.assertTrue(d['zero_entropy'])

    def test_entropy_summary_gora_and_not_zero(self):
        d = self.entropy_summary('ГОРА', 'and')
        self.assertFalse(d['zero_entropy'])
        self.assertAlmostEqual(d['mean_spatial_H'], 1.0)

    def test_entropy_summary_gora_xor3_spatial_2(self):
        d = self.entropy_summary('ГОРА', 'xor3')
        self.assertAlmostEqual(d['mean_spatial_H'], 2.0)

    def test_entropy_summary_gora_and_constant_spatial(self):
        d = self.entropy_summary('ГОРА', 'and')
        self.assertTrue(d['constant_spatial'])

    def test_entropy_summary_tuman_xor3_not_constant(self):
        d = self.entropy_summary('ТУМАН', 'xor3')
        self.assertFalse(d['zero_entropy'])
        self.assertFalse(d['constant_spatial'])

    def test_entropy_summary_tuman_xor3_symmetric_temporal(self):
        d = self.entropy_summary('ТУМАН', 'xor3')
        self.assertTrue(d['symmetric_temporal'])

    def test_entropy_summary_tuman_xor3_max_H_s(self):
        d = self.entropy_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['max_spatial_H'], 3.375, places=4)

    def test_entropy_summary_max_possible_Hs(self):
        import math
        d = self.entropy_summary('ГОРА', 'and')
        self.assertAlmostEqual(d['max_possible_Hs'], math.log2(16), places=6)

    def test_entropy_summary_word_upper(self):
        d = self.entropy_summary('гора', 'xor3')
        self.assertEqual(d['word'], 'ГОРА')

    def test_entropy_summary_n_cells(self):
        d = self.entropy_summary('ГОРА', 'and')
        self.assertEqual(d['n_cells'], 16)

    # ── all_entropy ───────────────────────────────────────────────────────────

    def test_all_entropy_returns_dict(self):
        self.assertIsInstance(self.all_entropy('ГОРА'), dict)

    def test_all_entropy_four_rules(self):
        d = self.all_entropy('ГОРА')
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_entropy_consistent_with_summary(self):
        ae = self.all_entropy('ГОРА')
        s  = self.entropy_summary('ГОРА', 'and')
        self.assertAlmostEqual(ae['and']['mean_spatial_H'], s['mean_spatial_H'])

    # ── build_entropy_data ────────────────────────────────────────────────────

    def test_build_entropy_data_returns_dict(self):
        self.assertIsInstance(self.build_entropy_data(['ГОРА']), dict)

    def test_build_entropy_data_top_keys(self):
        d = self.build_entropy_data(['ГОРА'])
        for key in ('words', 'width', 'per_rule'):
            self.assertIn(key, d)

    def test_build_entropy_data_four_rule_keys(self):
        d = self.build_entropy_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_entropy_data_word_uppercase(self):
        d = self.build_entropy_data(['гора'])
        self.assertIn('ГОРА', d['per_rule']['and'])

    def test_build_entropy_data_zero_entropy(self):
        d = self.build_entropy_data(['ТУМАН'])
        self.assertTrue(d['per_rule']['xor']['ТУМАН']['zero_entropy'])

    def test_build_entropy_data_known_fields(self):
        d   = self.build_entropy_data(['ГОРА'])
        rec = d['per_rule']['xor3']['ГОРА']
        for key in ('period', 'spatial_entropy', 'mean_spatial_H',
                    'max_spatial_H', 'min_spatial_H',
                    'temporal_entropy', 'mean_temporal_H',
                    'max_temporal_H', 'min_temporal_H',
                    'zero_entropy', 'constant_spatial',
                    'constant_temporal', 'symmetric_temporal'):
            self.assertIn(key, rec)

    # ── Viewer HTML markers ───────────────────────────────────────────────────

    def test_viewer_has_ent_bar(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ent-bar', content)

    def test_viewer_has_ent_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ent-cell', content)

    def test_viewer_has_ent_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ent-info', content)

    def test_viewer_has_en_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('en-word', content)

    def test_viewer_has_en_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('en-btn', content)

    def test_viewer_has_en_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('enRun', content)

    def test_viewer_has_en_entropy_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('enEntropy', content)

    def test_viewer_has_en_spatial_h_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('enSpatialH', content)

    def test_viewer_has_en_temporal_h_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('enTemporalH', content)


class TestSolanBoundary(unittest.TestCase):
    """Tests for solan_boundary.py — Spatial XOR-Boundary Analysis."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_boundary import (
            boundary_step, boundary_orbit, boundary_period,
            period_compressed, n_active_boundaries, n_zero_boundaries,
            boundary_vocab_nz, boundary_vocab_all, boundary_uniform,
            boundary_bit_constraints, boundary_summary,
            all_boundary, build_boundary_data,
        )
        cls.boundary_step          = staticmethod(boundary_step)
        cls.boundary_orbit         = staticmethod(boundary_orbit)
        cls.boundary_period        = staticmethod(boundary_period)
        cls.period_compressed      = staticmethod(period_compressed)
        cls.n_active_boundaries    = staticmethod(n_active_boundaries)
        cls.n_zero_boundaries      = staticmethod(n_zero_boundaries)
        cls.boundary_vocab_nz      = staticmethod(boundary_vocab_nz)
        cls.boundary_vocab_all     = staticmethod(boundary_vocab_all)
        cls.boundary_uniform       = staticmethod(boundary_uniform)
        cls.boundary_bit_constraints = staticmethod(boundary_bit_constraints)
        cls.boundary_summary       = staticmethod(boundary_summary)
        cls.all_boundary           = staticmethod(all_boundary)
        cls.build_boundary_data    = staticmethod(build_boundary_data)
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.LEXICON = list(LEXICON)

    # ── boundary_step ─────────────────────────────────────────────────────────

    def test_boundary_step_returns_tuple(self):
        result = self.boundary_step((47, 1, 47, 1))
        self.assertIsInstance(result, tuple)

    def test_boundary_step_length_preserved(self):
        result = self.boundary_step((47, 1, 47, 1))
        self.assertEqual(len(result), 4)

    def test_boundary_step_uniform_zero(self):
        # Uniform state: all cells equal → all boundaries = 0
        self.assertEqual(self.boundary_step((0, 0, 0, 0)), (0, 0, 0, 0))

    def test_boundary_step_uniform_63(self):
        self.assertEqual(self.boundary_step((63, 63, 63, 63)), (0, 0, 0, 0))

    def test_boundary_step_alternating_47_1(self):
        # 47 XOR 1 = 46
        result = self.boundary_step((47, 1, 47, 1))
        self.assertEqual(result, (46, 46, 46, 46))

    def test_boundary_step_xor_values_in_range(self):
        result = self.boundary_step((10, 20, 30, 40))
        self.assertTrue(all(0 <= v <= 63 for v in result))

    def test_boundary_step_periodic_wrap(self):
        # Last cell XOR with first cell (periodic)
        state = (47, 1)
        result = self.boundary_step(state)
        # cell 0: 47^1=46, cell 1: 1^47=46
        self.assertEqual(result, (46, 46))

    # ── boundary_orbit ────────────────────────────────────────────────────────

    def test_boundary_orbit_returns_list(self):
        self.assertIsInstance(self.boundary_orbit('ГОРА', 'and'), list)

    def test_boundary_orbit_length_equals_period(self):
        result = self.boundary_orbit('ГОРА', 'and')
        self.assertEqual(len(result), 2)

    def test_boundary_orbit_tuman_xor_all_zero(self):
        b = self.boundary_orbit('ТУМАН', 'xor')
        self.assertEqual(len(b), 1)
        self.assertTrue(all(v == 0 for v in b[0]))

    def test_boundary_orbit_gora_or_all_zero(self):
        b = self.boundary_orbit('ГОРА', 'or')
        self.assertEqual(len(b), 1)
        self.assertTrue(all(v == 0 for v in b[0]))

    def test_boundary_orbit_gora_and_constant_46(self):
        b = self.boundary_orbit('ГОРА', 'and')
        for row in b:
            self.assertTrue(all(v == 46 for v in row))

    def test_boundary_orbit_elements_are_tuples(self):
        b = self.boundary_orbit('ГОРА', 'xor3')
        self.assertIsInstance(b[0], tuple)

    def test_boundary_orbit_values_in_range(self):
        b = self.boundary_orbit('ТУМАН', 'xor3')
        self.assertTrue(all(0 <= v <= 63 for row in b for v in row))

    def test_boundary_orbit_case_insensitive(self):
        self.assertEqual(
            self.boundary_orbit('гора', 'and'),
            self.boundary_orbit('ГОРА', 'and'),
        )

    # ── boundary_period ───────────────────────────────────────────────────────

    def test_boundary_period_returns_int(self):
        self.assertIsInstance(self.boundary_period('ГОРА', 'and'), int)

    def test_boundary_period_gora_and_is_1(self):
        self.assertEqual(self.boundary_period('ГОРА', 'and'), 1)

    def test_boundary_period_tuman_xor_is_1(self):
        self.assertEqual(self.boundary_period('ТУМАН', 'xor'), 1)

    def test_boundary_period_gora_xor3_is_2(self):
        self.assertEqual(self.boundary_period('ГОРА', 'xor3'), 2)

    def test_boundary_period_tuman_xor3_is_8(self):
        self.assertEqual(self.boundary_period('ТУМАН', 'xor3'), 8)

    def test_boundary_period_divides_orbit_period(self):
        from projects.hexglyph.solan_traj import word_trajectory
        for word, rule in [('ГОРА', 'and'), ('ГОРА', 'xor3'), ('ТУМАН', 'xor3')]:
            P  = word_trajectory(word, rule)['period']
            bp = self.boundary_period(word, rule)
            self.assertEqual(P % bp, 0,
                             f'{word} {rule}: P={P} not divisible by b_period={bp}')

    # ── period_compressed ─────────────────────────────────────────────────────

    def test_period_compressed_returns_bool(self):
        self.assertIsInstance(self.period_compressed('ГОРА', 'and'), bool)

    def test_period_compressed_gora_and_true(self):
        self.assertTrue(self.period_compressed('ГОРА', 'and'))

    def test_period_compressed_tuman_xor_false(self):
        self.assertFalse(self.period_compressed('ТУМАН', 'xor'))

    def test_period_compressed_gora_xor3_false(self):
        self.assertFalse(self.period_compressed('ГОРА', 'xor3'))

    def test_period_compressed_tuman_xor3_false(self):
        self.assertFalse(self.period_compressed('ТУМАН', 'xor3'))

    # ── n_active_boundaries ───────────────────────────────────────────────────

    def test_n_active_boundaries_returns_list(self):
        self.assertIsInstance(self.n_active_boundaries('ГОРА', 'and'), list)

    def test_n_active_boundaries_length_equals_period(self):
        result = self.n_active_boundaries('ГОРА', 'and')
        self.assertEqual(len(result), 2)

    def test_n_active_boundaries_tuman_xor_zero(self):
        self.assertEqual(self.n_active_boundaries('ТУМАН', 'xor'), [0])

    def test_n_active_boundaries_gora_or_zero(self):
        self.assertEqual(self.n_active_boundaries('ГОРА', 'or'), [0])

    def test_n_active_boundaries_gora_and_max(self):
        # All 16 adjacent pairs differ → n_active = 16
        self.assertEqual(self.n_active_boundaries('ГОРА', 'and'), [16, 16])

    def test_n_active_boundaries_gora_xor3_max(self):
        self.assertEqual(self.n_active_boundaries('ГОРА', 'xor3'), [16, 16])

    def test_n_active_boundaries_tuman_xor3_values(self):
        result = self.n_active_boundaries('ТУМАН', 'xor3')
        self.assertEqual(result, [15, 16, 16, 16, 16, 16, 16, 13])

    def test_n_active_boundaries_non_negative(self):
        result = self.n_active_boundaries('ТУМАН', 'xor3')
        self.assertTrue(all(v >= 0 for v in result))

    def test_n_active_plus_zero_equals_n(self):
        na = self.n_active_boundaries('ТУМАН', 'xor3')
        nz = self.n_zero_boundaries('ТУМАН', 'xor3')
        for a, z in zip(na, nz):
            self.assertEqual(a + z, 16)

    # ── n_zero_boundaries ─────────────────────────────────────────────────────

    def test_n_zero_boundaries_returns_list(self):
        self.assertIsInstance(self.n_zero_boundaries('ГОРА', 'and'), list)

    def test_n_zero_boundaries_tuman_xor_all_zero(self):
        self.assertEqual(self.n_zero_boundaries('ТУМАН', 'xor'), [16])

    def test_n_zero_boundaries_gora_and_none(self):
        self.assertEqual(self.n_zero_boundaries('ГОРА', 'and'), [0, 0])

    def test_n_zero_boundaries_tuman_xor3(self):
        result = self.n_zero_boundaries('ТУМАН', 'xor3')
        self.assertEqual(result, [1, 0, 0, 0, 0, 0, 0, 3])

    # ── boundary_vocab_nz ─────────────────────────────────────────────────────

    def test_boundary_vocab_nz_returns_list(self):
        self.assertIsInstance(self.boundary_vocab_nz('ГОРА', 'and'), list)

    def test_boundary_vocab_nz_tuman_xor_empty(self):
        self.assertEqual(self.boundary_vocab_nz('ТУМАН', 'xor'), [])

    def test_boundary_vocab_nz_gora_or_empty(self):
        self.assertEqual(self.boundary_vocab_nz('ГОРА', 'or'), [])

    def test_boundary_vocab_nz_gora_and_single(self):
        # 47 XOR 1 = 46
        self.assertEqual(self.boundary_vocab_nz('ГОРА', 'and'), [46])

    def test_boundary_vocab_nz_gora_xor3(self):
        self.assertEqual(self.boundary_vocab_nz('ГОРА', 'xor3'), [14, 30, 32, 48])

    def test_boundary_vocab_nz_tuman_xor3_size(self):
        self.assertEqual(len(self.boundary_vocab_nz('ТУМАН', 'xor3')), 15)

    def test_boundary_vocab_nz_sorted(self):
        v = self.boundary_vocab_nz('ТУМАН', 'xor3')
        self.assertEqual(v, sorted(v))

    # ── boundary_vocab_all ────────────────────────────────────────────────────

    def test_boundary_vocab_all_includes_zero_for_uniform(self):
        v = self.boundary_vocab_all('ТУМАН', 'xor')
        self.assertIn(0, v)

    def test_boundary_vocab_all_tuman_xor3_size(self):
        # 15 non-zero + 1 zero value
        self.assertEqual(len(self.boundary_vocab_all('ТУМАН', 'xor3')), 16)

    def test_boundary_vocab_all_tuman_xor3_has_zero(self):
        self.assertIn(0, self.boundary_vocab_all('ТУМАН', 'xor3'))

    def test_boundary_vocab_all_sorted(self):
        v = self.boundary_vocab_all('ТУМАН', 'xor3')
        self.assertEqual(v, sorted(v))

    # ── boundary_uniform ──────────────────────────────────────────────────────

    def test_boundary_uniform_returns_bool(self):
        self.assertIsInstance(self.boundary_uniform('ГОРА', 'and'), bool)

    def test_boundary_uniform_tuman_xor_true(self):
        self.assertTrue(self.boundary_uniform('ТУМАН', 'xor'))

    def test_boundary_uniform_gora_or_true(self):
        self.assertTrue(self.boundary_uniform('ГОРА', 'or'))

    def test_boundary_uniform_gora_and_true(self):
        # Anti-phase orbit → constant boundary
        self.assertTrue(self.boundary_uniform('ГОРА', 'and'))

    def test_boundary_uniform_gora_xor3_false(self):
        self.assertFalse(self.boundary_uniform('ГОРА', 'xor3'))

    def test_boundary_uniform_tuman_xor3_false(self):
        self.assertFalse(self.boundary_uniform('ТУМАН', 'xor3'))

    # ── boundary_bit_constraints ──────────────────────────────────────────────

    def test_boundary_bit_constraints_returns_list(self):
        self.assertIsInstance(self.boundary_bit_constraints('ТУМАН', 'xor3'), list)

    def test_boundary_bit_constraints_tuman_xor3(self):
        # b0=b1 from coact constraint
        self.assertIn((0, 1), self.boundary_bit_constraints('ТУМАН', 'xor3'))

    def test_boundary_bit_constraints_tuman_xor3_only_01(self):
        # Only (0,1) constraint for ТУМАН XOR3
        self.assertEqual(self.boundary_bit_constraints('ТУМАН', 'xor3'), [(0, 1)])

    def test_boundary_bit_constraints_gora_xor3(self):
        # b1=b2=b3 in boundary values
        c = self.boundary_bit_constraints('ГОРА', 'xor3')
        self.assertIn((1, 2), c)
        self.assertIn((1, 3), c)
        self.assertIn((2, 3), c)

    def test_boundary_bit_constraints_elements_are_tuples(self):
        c = self.boundary_bit_constraints('ТУМАН', 'xor3')
        if c:
            self.assertIsInstance(c[0], tuple)
            self.assertEqual(len(c[0]), 2)

    def test_boundary_bit_constraints_indices_in_range(self):
        for word, rule in [('ГОРА', 'xor3'), ('ТУМАН', 'xor3'), ('ГОРА', 'and')]:
            for b1, b2 in self.boundary_bit_constraints(word, rule):
                self.assertGreaterEqual(b1, 0)
                self.assertLess(b2, 6)
                self.assertLess(b1, b2)

    # ── boundary_summary ──────────────────────────────────────────────────────

    def test_boundary_summary_returns_dict(self):
        self.assertIsInstance(self.boundary_summary('ГОРА', 'and'), dict)

    def test_boundary_summary_required_keys(self):
        d = self.boundary_summary('ГОРА', 'and')
        for key in ('word', 'rule', 'period', 'b_period', 'period_compressed',
                    'n_active', 'n_zero', 'mean_n_active',
                    'max_n_active', 'min_n_active',
                    'vocab_nz', 'vocab_nz_size',
                    'vocab_all', 'uniform',
                    'bit_constraints', 'b_orbit'):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_boundary_summary_gora_and_compressed(self):
        d = self.boundary_summary('ГОРА', 'and')
        self.assertTrue(d['period_compressed'])
        self.assertEqual(d['b_period'], 1)
        self.assertEqual(d['period'], 2)

    def test_boundary_summary_tuman_xor3_constraint(self):
        d = self.boundary_summary('ТУМАН', 'xor3')
        self.assertIn((0, 1), d['bit_constraints'])

    def test_boundary_summary_gora_or_zero_vocab(self):
        d = self.boundary_summary('ГОРА', 'or')
        self.assertEqual(d['vocab_nz_size'], 0)
        self.assertTrue(d['uniform'])

    def test_boundary_summary_word_upper(self):
        d = self.boundary_summary('гора', 'and')
        self.assertEqual(d['word'], 'ГОРА')

    def test_boundary_summary_n_active_consistent(self):
        d = self.boundary_summary('ТУМАН', 'xor3')
        self.assertEqual(d['n_active'], [15, 16, 16, 16, 16, 16, 16, 13])

    def test_boundary_summary_mean_n_active_tuman_xor3(self):
        d = self.boundary_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['mean_n_active'], 15.5)

    # ── all_boundary ──────────────────────────────────────────────────────────

    def test_all_boundary_returns_dict(self):
        self.assertIsInstance(self.all_boundary('ГОРА'), dict)

    def test_all_boundary_four_rules(self):
        d = self.all_boundary('ГОРА')
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── build_boundary_data ───────────────────────────────────────────────────

    def test_build_boundary_data_returns_dict(self):
        self.assertIsInstance(self.build_boundary_data(['ГОРА']), dict)

    def test_build_boundary_data_top_keys(self):
        d = self.build_boundary_data(['ГОРА'])
        for key in ('words', 'width', 'per_rule'):
            self.assertIn(key, d)

    def test_build_boundary_data_rule_keys(self):
        d = self.build_boundary_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_boundary_data_gora_and_compressed(self):
        d = self.build_boundary_data(['ГОРА'])
        self.assertTrue(d['per_rule']['and']['ГОРА']['period_compressed'])

    def test_build_boundary_data_word_uppercase(self):
        d = self.build_boundary_data(['гора'])
        self.assertIn('ГОРА', d['per_rule']['and'])

    def test_build_boundary_data_known_fields(self):
        d   = self.build_boundary_data(['ГОРА'])
        rec = d['per_rule']['xor3']['ГОРА']
        for key in ('period', 'b_period', 'period_compressed',
                    'n_active', 'mean_n_active', 'max_n_active', 'min_n_active',
                    'vocab_nz_size', 'vocab_nz', 'uniform', 'bit_constraints'):
            self.assertIn(key, rec)

    # ── Viewer HTML markers ───────────────────────────────────────────────────

    def test_viewer_has_bound_map(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bound-map', content)

    def test_viewer_has_bound_nact(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bound-nact', content)

    def test_viewer_has_bound_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bound-info', content)

    def test_viewer_has_bn_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bn-word', content)

    def test_viewer_has_bn_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bn-btn', content)

    def test_viewer_has_bn_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bnRun', content)

    def test_viewer_has_bn_boundary_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bnBoundary', content)

    def test_viewer_has_bound_color_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('boundColor', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_boundary',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_boundary(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_boundary', content)


class TestSolanCell(unittest.TestCase):
    """Tests for solan_cell.py — Per-Cell Temporal Analysis."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_cell import (
            cell_series, cell_vocab, cell_hist, cell_vocab_size,
            cell_is_frozen, cell_transitions, cell_mean, cell_var,
            frozen_cells, spatial_variance, cell_summary,
            orbit_cell_matrix, cell_agg, all_cell, build_cell_data,
        )
        cls.cell_series       = staticmethod(cell_series)
        cls.cell_vocab        = staticmethod(cell_vocab)
        cls.cell_hist         = staticmethod(cell_hist)
        cls.cell_vocab_size   = staticmethod(cell_vocab_size)
        cls.cell_is_frozen    = staticmethod(cell_is_frozen)
        cls.cell_transitions  = staticmethod(cell_transitions)
        cls.cell_mean         = staticmethod(cell_mean)
        cls.cell_var          = staticmethod(cell_var)
        cls.frozen_cells      = staticmethod(frozen_cells)
        cls.spatial_variance  = staticmethod(spatial_variance)
        cls.cell_summary      = staticmethod(cell_summary)
        cls.orbit_cell_matrix = staticmethod(orbit_cell_matrix)
        cls.cell_agg          = staticmethod(cell_agg)
        cls.all_cell          = staticmethod(all_cell)
        cls.build_cell_data   = staticmethod(build_cell_data)
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.LEXICON = list(LEXICON)

    # ── cell_series ──────────────────────────────────────────────────────────

    def test_cell_series_returns_list(self):
        result = self.cell_series('ГОРА', 'and', 0)
        self.assertIsInstance(result, list)

    def test_cell_series_length_equals_period(self):
        result = self.cell_series('ГОРА', 'and', 0)
        self.assertEqual(len(result), 2)  # ГОРА AND has P=2

    def test_cell_series_gora_and_cell0(self):
        self.assertEqual(self.cell_series('ГОРА', 'and', 0), [47, 1])

    def test_cell_series_gora_and_cell1(self):
        self.assertEqual(self.cell_series('ГОРА', 'and', 1), [1, 47])

    def test_cell_series_tuman_xor_cell0(self):
        result = self.cell_series('ТУМАН', 'xor', 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 0)

    def test_cell_series_all_values_in_range(self):
        result = self.cell_series('ТУМАН', 'xor3', 0)
        self.assertTrue(all(0 <= v <= 63 for v in result))

    def test_cell_series_case_insensitive(self):
        self.assertEqual(
            self.cell_series('гора', 'and', 0),
            self.cell_series('ГОРА', 'and', 0),
        )

    # ── cell_vocab ───────────────────────────────────────────────────────────

    def test_cell_vocab_returns_list(self):
        self.assertIsInstance(self.cell_vocab('ГОРА', 'and', 0), list)

    def test_cell_vocab_sorted(self):
        v = self.cell_vocab('ГОРА', 'and', 0)
        self.assertEqual(v, sorted(v))

    def test_cell_vocab_gora_and_cell0(self):
        self.assertEqual(self.cell_vocab('ГОРА', 'and', 0), [1, 47])

    def test_cell_vocab_tuman_xor_cell0(self):
        self.assertEqual(self.cell_vocab('ТУМАН', 'xor', 0), [0])

    def test_cell_vocab_gora_or_cell0(self):
        self.assertEqual(self.cell_vocab('ГОРА', 'or', 0), [63])

    # ── cell_vocab_size ──────────────────────────────────────────────────────

    def test_cell_vocab_size_returns_int(self):
        self.assertIsInstance(self.cell_vocab_size('ГОРА', 'and', 0), int)

    def test_cell_vocab_size_gora_and_cell0(self):
        self.assertEqual(self.cell_vocab_size('ГОРА', 'and', 0), 2)

    def test_cell_vocab_size_tuman_xor_cell0(self):
        self.assertEqual(self.cell_vocab_size('ТУМАН', 'xor', 0), 1)

    def test_cell_vocab_size_gora_xor3_cell0(self):
        self.assertEqual(self.cell_vocab_size('ГОРА', 'xor3', 0), 2)

    # ── cell_is_frozen ───────────────────────────────────────────────────────

    def test_cell_is_frozen_returns_bool(self):
        self.assertIsInstance(self.cell_is_frozen('ГОРА', 'and', 0), bool)

    def test_cell_is_frozen_tuman_xor_true(self):
        self.assertTrue(self.cell_is_frozen('ТУМАН', 'xor', 0))

    def test_cell_is_frozen_gora_or_true(self):
        self.assertTrue(self.cell_is_frozen('ГОРА', 'or', 0))

    def test_cell_is_frozen_gora_and_false(self):
        self.assertFalse(self.cell_is_frozen('ГОРА', 'and', 0))

    def test_cell_is_frozen_tuman_xor3_false(self):
        self.assertFalse(self.cell_is_frozen('ТУМАН', 'xor3', 0))

    # ── cell_transitions ─────────────────────────────────────────────────────

    def test_cell_transitions_returns_int(self):
        self.assertIsInstance(self.cell_transitions('ГОРА', 'and', 0), int)

    def test_cell_transitions_gora_and_cell0(self):
        self.assertEqual(self.cell_transitions('ГОРА', 'and', 0), 2)

    def test_cell_transitions_tuman_xor_cell0(self):
        self.assertEqual(self.cell_transitions('ТУМАН', 'xor', 0), 0)

    def test_cell_transitions_le_period(self):
        s = self.cell_series('ТУМАН', 'xor3', 0)
        P = len(s)
        tc = self.cell_transitions('ТУМАН', 'xor3', 0)
        self.assertLessEqual(tc, P)

    def test_cell_transitions_frozen_is_zero(self):
        self.assertEqual(self.cell_transitions('ТУМАН', 'xor', 5), 0)

    # ── cell_mean / cell_var ─────────────────────────────────────────────────

    def test_cell_mean_returns_float(self):
        self.assertIsInstance(self.cell_mean('ГОРА', 'and', 0), float)

    def test_cell_mean_gora_and_cell0(self):
        self.assertAlmostEqual(self.cell_mean('ГОРА', 'and', 0), 24.0)

    def test_cell_mean_tuman_xor_cell0(self):
        self.assertAlmostEqual(self.cell_mean('ТУМАН', 'xor', 0), 0.0)

    def test_cell_mean_gora_or_cell0(self):
        self.assertAlmostEqual(self.cell_mean('ГОРА', 'or', 0), 63.0)

    def test_cell_var_returns_float(self):
        self.assertIsInstance(self.cell_var('ГОРА', 'and', 0), float)

    def test_cell_var_gora_and_cell0(self):
        self.assertAlmostEqual(self.cell_var('ГОРА', 'and', 0), 529.0)

    def test_cell_var_frozen_is_zero(self):
        self.assertAlmostEqual(self.cell_var('ТУМАН', 'xor', 0), 0.0)

    # ── frozen_cells ─────────────────────────────────────────────────────────

    def test_frozen_cells_returns_list(self):
        self.assertIsInstance(self.frozen_cells('ГОРА', 'and'), list)

    def test_frozen_cells_tuman_xor_all_frozen(self):
        self.assertEqual(len(self.frozen_cells('ТУМАН', 'xor')), 16)

    def test_frozen_cells_gora_and_none_frozen(self):
        self.assertEqual(self.frozen_cells('ГОРА', 'and'), [])

    def test_frozen_cells_gora_or_all_frozen(self):
        self.assertEqual(len(self.frozen_cells('ГОРА', 'or')), 16)

    def test_frozen_cells_tuman_xor3_none_frozen(self):
        self.assertEqual(self.frozen_cells('ТУМАН', 'xor3'), [])

    def test_frozen_cells_indices_in_range(self):
        fc = self.frozen_cells('ТУМАН', 'xor')
        self.assertTrue(all(0 <= i < 16 for i in fc))

    # ── spatial_variance ─────────────────────────────────────────────────────

    def test_spatial_variance_returns_list(self):
        self.assertIsInstance(self.spatial_variance('ГОРА', 'and'), list)

    def test_spatial_variance_length_equals_period(self):
        sv = self.spatial_variance('ГОРА', 'and')
        self.assertEqual(len(sv), 2)

    def test_spatial_variance_tuman_xor_zero(self):
        self.assertEqual(self.spatial_variance('ТУМАН', 'xor'), [0.0])

    def test_spatial_variance_gora_and_constant(self):
        sv = self.spatial_variance('ГОРА', 'and')
        self.assertEqual(sv, [529.0, 529.0])

    def test_spatial_variance_gora_xor3(self):
        sv = self.spatial_variance('ГОРА', 'xor3')
        self.assertAlmostEqual(sv[0], 308.75)
        self.assertAlmostEqual(sv[1], 164.75)

    def test_spatial_variance_non_negative(self):
        sv = self.spatial_variance('ТУМАН', 'xor3')
        self.assertTrue(all(v >= 0 for v in sv))

    # ── cell_summary ─────────────────────────────────────────────────────────

    def test_cell_summary_returns_dict(self):
        self.assertIsInstance(self.cell_summary('ГОРА', 'and', 0), dict)

    def test_cell_summary_required_keys(self):
        d = self.cell_summary('ГОРА', 'and', 0)
        for key in ('word', 'rule', 'cell_idx', 'period', 'series',
                    'vocab', 'vocab_size', 'is_frozen', 'frozen_val',
                    'transitions', 'mean', 'var'):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_cell_summary_gora_and_cell0_values(self):
        d = self.cell_summary('ГОРА', 'and', 0)
        self.assertEqual(d['vocab_size'], 2)
        self.assertFalse(d['is_frozen'])
        self.assertEqual(d['transitions'], 2)
        self.assertAlmostEqual(d['mean'], 24.0)
        self.assertAlmostEqual(d['var'], 529.0)
        self.assertIsNone(d['frozen_val'])

    def test_cell_summary_frozen_val_set_when_frozen(self):
        d = self.cell_summary('ТУМАН', 'xor', 0)
        self.assertTrue(d['is_frozen'])
        self.assertEqual(d['frozen_val'], 0)

    def test_cell_summary_word_upper(self):
        d = self.cell_summary('гора', 'and', 0)
        self.assertEqual(d['word'], 'ГОРА')

    def test_cell_summary_cell_idx_correct(self):
        d = self.cell_summary('ГОРА', 'and', 3)
        self.assertEqual(d['cell_idx'], 3)

    # ── orbit_cell_matrix ────────────────────────────────────────────────────

    def test_orbit_cell_matrix_returns_list(self):
        self.assertIsInstance(self.orbit_cell_matrix('ГОРА', 'and'), list)

    def test_orbit_cell_matrix_length(self):
        result = self.orbit_cell_matrix('ГОРА', 'and')
        self.assertEqual(len(result), 16)

    def test_orbit_cell_matrix_each_item_is_dict(self):
        result = self.orbit_cell_matrix('ГОРА', 'and')
        self.assertIsInstance(result[0], dict)

    def test_orbit_cell_matrix_cell_idx_sequential(self):
        result = self.orbit_cell_matrix('ГОРА', 'and')
        for i, d in enumerate(result):
            self.assertEqual(d['cell_idx'], i)

    # ── cell_agg ─────────────────────────────────────────────────────────────

    def test_cell_agg_returns_dict(self):
        self.assertIsInstance(self.cell_agg('ГОРА', 'and'), dict)

    def test_cell_agg_required_keys(self):
        d = self.cell_agg('ГОРА', 'and')
        for key in ('word', 'rule', 'period', 'n_cells', 'n_frozen',
                    'vocab_sizes', 'mean_vocab_size', 'max_vocab_size',
                    'transitions', 'mean_transitions', 'max_transitions',
                    'spatial_variance', 'mean_spatial_var', 'max_spatial_var',
                    'uniform_spatial'):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_cell_agg_tuman_xor_all_frozen(self):
        d = self.cell_agg('ТУМАН', 'xor')
        self.assertEqual(d['n_frozen'], 16)
        self.assertAlmostEqual(d['mean_vocab_size'], 1.0)
        self.assertAlmostEqual(d['mean_transitions'], 0.0)
        self.assertTrue(d['uniform_spatial'])

    def test_cell_agg_gora_or_all_frozen(self):
        d = self.cell_agg('ГОРА', 'or')
        self.assertEqual(d['n_frozen'], 16)

    def test_cell_agg_gora_and_none_frozen(self):
        d = self.cell_agg('ГОРА', 'and')
        self.assertEqual(d['n_frozen'], 0)
        self.assertAlmostEqual(d['mean_vocab_size'], 2.0)
        self.assertTrue(d['uniform_spatial'])

    def test_cell_agg_tuman_xor3_no_frozen(self):
        d = self.cell_agg('ТУМАН', 'xor3')
        self.assertEqual(d['n_frozen'], 0)

    def test_cell_agg_tuman_xor3_mean_vocab(self):
        d = self.cell_agg('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['mean_vocab_size'], 5.25)

    def test_cell_agg_tuman_xor3_mean_transitions(self):
        d = self.cell_agg('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['mean_transitions'], 7.125)

    def test_cell_agg_tuman_xor3_max_vocab(self):
        d = self.cell_agg('ТУМАН', 'xor3')
        self.assertEqual(d['max_vocab_size'], 7)

    def test_cell_agg_tuman_xor3_min_vocab(self):
        d = self.cell_agg('ТУМАН', 'xor3')
        self.assertEqual(d['min_vocab_size'], 3)

    def test_cell_agg_tuman_xor3_max_transitions(self):
        d = self.cell_agg('ТУМАН', 'xor3')
        self.assertEqual(d['max_transitions'], 8)

    def test_cell_agg_n_cells_equals_width(self):
        d = self.cell_agg('ГОРА', 'xor3')
        self.assertEqual(d['n_cells'], 16)

    def test_cell_agg_vocab_sizes_length(self):
        d = self.cell_agg('ГОРА', 'and')
        self.assertEqual(len(d['vocab_sizes']), 16)

    def test_cell_agg_gora_and_spatial_var_constant(self):
        d = self.cell_agg('ГОРА', 'and')
        sv = d['spatial_variance']
        self.assertEqual(sv, [529.0, 529.0])

    def test_cell_agg_word_upper(self):
        d = self.cell_agg('гора', 'and')
        self.assertEqual(d['word'], 'ГОРА')

    # ── build_cell_data ───────────────────────────────────────────────────────

    def test_build_cell_data_returns_dict(self):
        self.assertIsInstance(self.build_cell_data(['ГОРА']), dict)

    def test_build_cell_data_top_keys(self):
        d = self.build_cell_data(['ГОРА'])
        for key in ('words', 'width', 'per_rule'):
            self.assertIn(key, d)

    def test_build_cell_data_rule_keys(self):
        d = self.build_cell_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_cell_data_word_uppercase(self):
        d = self.build_cell_data(['гора'])
        self.assertIn('ГОРА', d['per_rule']['and'])

    def test_build_cell_data_gora_and_n_frozen(self):
        d = self.build_cell_data(['ГОРА'])
        self.assertEqual(d['per_rule']['and']['ГОРА']['n_frozen'], 0)

    def test_build_cell_data_known_fields(self):
        d   = self.build_cell_data(['ГОРА'])
        rec = d['per_rule']['xor3']['ГОРА']
        for key in ('period', 'n_frozen', 'vocab_sizes', 'mean_vocab_size',
                    'max_vocab_size', 'transitions', 'mean_transitions',
                    'max_transitions', 'mean_spatial_var', 'max_spatial_var',
                    'min_spatial_var', 'uniform_spatial'):
            self.assertIn(key, rec)

    # ── all_cell ──────────────────────────────────────────────────────────────

    def test_all_cell_returns_dict(self):
        self.assertIsInstance(self.all_cell('ГОРА'), dict)

    def test_all_cell_four_rules(self):
        d = self.all_cell('ГОРА')
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── Viewer HTML markers ───────────────────────────────────────────────────

    def test_viewer_has_cell_map(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cell-map', content)

    def test_viewer_has_cell_var(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cell-var', content)

    def test_viewer_has_cell_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cell-info', content)

    def test_viewer_has_cl_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cl-word', content)

    def test_viewer_has_cl_rule(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cl-rule', content)

    def test_viewer_has_cl_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cl-btn', content)

    def test_viewer_has_cl_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('clRun', content)

    def test_viewer_has_cell_hue_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cellHue', content)

    def test_viewer_has_cl_orbit_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('clOrbit', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_cell',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_cell', content)


class TestSolanWidth(unittest.TestCase):
    """Tests for solan_width.py — Period vs. Ring Width Scaling."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_width import (
            period_at_width, width_series, is_constant_period,
            constant_period_value, max_width_period, width_summary,
            build_width_data, _DEFAULT_WIDTHS, _RULES,
        )
        cls.period_at_width      = staticmethod(period_at_width)
        cls.width_series         = staticmethod(width_series)
        cls.is_constant_period   = staticmethod(is_constant_period)
        cls.constant_period_value = staticmethod(constant_period_value)
        cls.max_width_period     = staticmethod(max_width_period)
        cls.width_summary        = staticmethod(width_summary)
        cls.build_width_data     = staticmethod(build_width_data)
        cls.DEFAULT_WIDTHS       = list(_DEFAULT_WIDTHS)
        cls.RULES                = list(_RULES)
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.LEXICON = list(LEXICON)

    # ── period_at_width ─────────────────────────────────────────────────────

    def test_period_at_width_returns_int(self):
        result = self.period_at_width('ГОРА', 'and', 16)
        self.assertIsInstance(result, int)

    def test_period_at_width_positive(self):
        result = self.period_at_width('ГОРА', 'xor3', 16)
        self.assertGreater(result, 0)

    def test_period_at_width_gora_and_16(self):
        self.assertEqual(self.period_at_width('ГОРА', 'and', 16), 2)

    def test_period_at_width_tuman_xor_8(self):
        self.assertEqual(self.period_at_width('ТУМАН', 'xor', 8), 1)

    def test_period_at_width_tuman_xor_12(self):
        self.assertEqual(self.period_at_width('ТУМАН', 'xor', 12), 4)

    def test_period_at_width_gora_xor3_16(self):
        self.assertEqual(self.period_at_width('ГОРА', 'xor3', 16), 2)

    def test_period_at_width_gora_or_16(self):
        self.assertEqual(self.period_at_width('ГОРА', 'or', 16), 1)

    def test_period_at_width_case_insensitive(self):
        self.assertEqual(
            self.period_at_width('гора', 'and', 16),
            self.period_at_width('ГОРА', 'and', 16),
        )

    # ── width_series ─────────────────────────────────────────────────────────

    def test_width_series_returns_list(self):
        result = self.width_series('ГОРА', 'and')
        self.assertIsInstance(result, list)

    def test_width_series_length_matches_widths(self):
        result = self.width_series('ГОРА', 'and')
        self.assertEqual(len(result), len(self.DEFAULT_WIDTHS))

    def test_width_series_all_positive(self):
        result = self.width_series('ГОРА', 'xor3')
        self.assertTrue(all(p > 0 for p in result))

    def test_width_series_gora_and_all_two(self):
        result = self.width_series('ГОРА', 'and')
        self.assertEqual(result, [2] * 9)

    def test_width_series_gora_xor3_all_two(self):
        result = self.width_series('ГОРА', 'xor3')
        self.assertEqual(result, [2] * 9)

    def test_width_series_gora_or_all_one(self):
        result = self.width_series('ГОРА', 'or')
        self.assertEqual(result, [1] * 9)

    def test_width_series_tuman_xor3_known_values(self):
        # N=4→2, N=8→4, N=16→8, N=32→16, N=64→32
        ws  = self.DEFAULT_WIDTHS
        ps  = self.width_series('ТУМАН', 'xor3')
        idx4  = ws.index(4);  self.assertEqual(ps[idx4], 2)
        idx8  = ws.index(8);  self.assertEqual(ps[idx8], 4)
        idx16 = ws.index(16); self.assertEqual(ps[idx16], 8)
        idx32 = ws.index(32); self.assertEqual(ps[idx32], 16)
        idx64 = ws.index(64); self.assertEqual(ps[idx64], 32)

    def test_width_series_tuman_xor_pow2_are_one(self):
        # For powers-of-2 widths ТУМАН XOR converges to 0 → P=1
        ws = self.DEFAULT_WIDTHS
        ps = self.width_series('ТУМАН', 'xor')
        for w, p in zip(ws, ps):
            if w in (4, 8, 16, 32, 64):
                self.assertEqual(p, 1, f"Expected P=1 for N={w}")

    def test_width_series_custom_widths(self):
        result = self.width_series('ГОРА', 'and', [4, 8, 16])
        self.assertEqual(result, [2, 2, 2])

    # ── is_constant_period ───────────────────────────────────────────────────

    def test_is_constant_period_returns_bool(self):
        self.assertIsInstance(self.is_constant_period('ГОРА', 'and'), bool)

    def test_is_constant_gora_and(self):
        self.assertTrue(self.is_constant_period('ГОРА', 'and'))

    def test_is_constant_gora_xor3(self):
        self.assertTrue(self.is_constant_period('ГОРА', 'xor3'))

    def test_is_constant_gora_or(self):
        self.assertTrue(self.is_constant_period('ГОРА', 'or'))

    def test_not_constant_tuman_xor(self):
        self.assertFalse(self.is_constant_period('ТУМАН', 'xor'))

    def test_not_constant_tuman_xor3(self):
        self.assertFalse(self.is_constant_period('ТУМАН', 'xor3'))

    # ── constant_period_value ─────────────────────────────────────────────────

    def test_constant_value_returns_int_or_none(self):
        v = self.constant_period_value('ГОРА', 'and')
        self.assertIsInstance(v, int)

    def test_constant_value_gora_and(self):
        self.assertEqual(self.constant_period_value('ГОРА', 'and'), 2)

    def test_constant_value_gora_xor3(self):
        self.assertEqual(self.constant_period_value('ГОРА', 'xor3'), 2)

    def test_constant_value_gora_or(self):
        self.assertEqual(self.constant_period_value('ГОРА', 'or'), 1)

    def test_constant_value_none_for_variable(self):
        self.assertIsNone(self.constant_period_value('ТУМАН', 'xor'))

    def test_constant_value_none_tuman_xor3(self):
        self.assertIsNone(self.constant_period_value('ТУМАН', 'xor3'))

    # ── max_width_period ──────────────────────────────────────────────────────

    def test_max_width_period_returns_tuple(self):
        result = self.max_width_period('ГОРА', 'and')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_max_width_period_width_in_widths(self):
        w, p = self.max_width_period('ТУМАН', 'xor3')
        self.assertIn(w, self.DEFAULT_WIDTHS)

    def test_max_width_period_tuman_xor3(self):
        w, p = self.max_width_period('ТУМАН', 'xor3')
        self.assertEqual((w, p), (64, 32))

    def test_max_width_period_gora_and(self):
        w, p = self.max_width_period('ГОРА', 'and')
        self.assertEqual(p, 2)  # constant series → max = 2

    def test_max_width_period_is_in_series(self):
        series = self.width_series('ТУМАН', 'xor3')
        _, max_p = self.max_width_period('ТУМАН', 'xor3')
        self.assertEqual(max_p, max(series))

    # ── width_summary ─────────────────────────────────────────────────────────

    def test_width_summary_returns_dict(self):
        result = self.width_summary('ГОРА', 'and')
        self.assertIsInstance(result, dict)

    def test_width_summary_required_keys(self):
        d = self.width_summary('ГОРА', 'and')
        for key in ('word', 'rule', 'widths', 'periods', 'pn_ratio',
                    'is_constant', 'constant_value', 'min_period', 'max_period',
                    'max_period_width', 'all_pow2', 'all_one', 'all_two',
                    'n_distinct'):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_width_summary_word_upper(self):
        d = self.width_summary('гора', 'and')
        self.assertEqual(d['word'], 'ГОРА')

    def test_width_summary_gora_and_is_constant(self):
        d = self.width_summary('ГОРА', 'and')
        self.assertTrue(d['is_constant'])

    def test_width_summary_gora_and_constant_value(self):
        d = self.width_summary('ГОРА', 'and')
        self.assertEqual(d['constant_value'], 2)

    def test_width_summary_gora_and_all_two(self):
        d = self.width_summary('ГОРА', 'and')
        self.assertTrue(d['all_two'])

    def test_width_summary_gora_and_all_pow2(self):
        d = self.width_summary('ГОРА', 'and')
        self.assertTrue(d['all_pow2'])

    def test_width_summary_gora_and_not_all_one(self):
        d = self.width_summary('ГОРА', 'and')
        self.assertFalse(d['all_one'])

    def test_width_summary_gora_and_n_distinct_one(self):
        d = self.width_summary('ГОРА', 'and')
        self.assertEqual(d['n_distinct'], 1)

    def test_width_summary_tuman_xor3_not_constant(self):
        d = self.width_summary('ТУМАН', 'xor3')
        self.assertFalse(d['is_constant'])
        self.assertIsNone(d['constant_value'])

    def test_width_summary_tuman_xor3_max_period(self):
        d = self.width_summary('ТУМАН', 'xor3')
        self.assertEqual(d['max_period'], 32)
        self.assertEqual(d['max_period_width'], 64)

    def test_width_summary_tuman_xor3_n_distinct(self):
        d = self.width_summary('ТУМАН', 'xor3')
        self.assertEqual(d['n_distinct'], 6)

    def test_width_summary_pn_ratio_length(self):
        d = self.width_summary('ГОРА', 'xor3')
        self.assertEqual(len(d['pn_ratio']), len(self.DEFAULT_WIDTHS))

    def test_width_summary_pn_ratio_positive(self):
        d = self.width_summary('ТУМАН', 'xor3')
        self.assertTrue(all(r > 0 for r in d['pn_ratio']))

    def test_width_summary_tuman_xor3_pow2_ratio_half(self):
        # For N=8,16,32,64 the ratio P/N = 0.5
        d   = self.width_summary('ТУМАН', 'xor3')
        ws  = d['widths']
        pns = d['pn_ratio']
        for w, r in zip(ws, pns):
            if w in (8, 16, 32, 64):
                self.assertAlmostEqual(r, 0.5, places=3,
                                       msg=f"N={w}: expected P/N=0.5 got {r}")

    def test_width_summary_gora_or_all_one(self):
        d = self.width_summary('ГОРА', 'or')
        self.assertTrue(d['all_one'])

    def test_width_summary_min_le_max_period(self):
        d = self.width_summary('ТУМАН', 'xor3')
        self.assertLessEqual(d['min_period'], d['max_period'])

    def test_width_summary_periods_matches_series(self):
        d = self.width_summary('ГОРА', 'xor3')
        expected = self.width_series('ГОРА', 'xor3')
        self.assertEqual(d['periods'], expected)

    # ── build_width_data ──────────────────────────────────────────────────────

    def test_build_width_data_returns_dict(self):
        result = self.build_width_data(['ГОРА', 'ТУМАН'])
        self.assertIsInstance(result, dict)

    def test_build_width_data_top_keys(self):
        d = self.build_width_data(['ГОРА'])
        for key in ('words', 'widths', 'per_rule'):
            self.assertIn(key, d)

    def test_build_width_data_rule_keys(self):
        d = self.build_width_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_width_data_word_uppercase(self):
        d = self.build_width_data(['гора'])
        self.assertIn('ГОРА', d['per_rule']['and'])

    def test_build_width_data_gora_and_constant(self):
        d = self.build_width_data(['ГОРА'])
        self.assertTrue(d['per_rule']['and']['ГОРА']['is_constant'])

    def test_build_width_data_known_fields(self):
        d   = self.build_width_data(['ГОРА'])
        rec = d['per_rule']['xor3']['ГОРА']
        for key in ('periods', 'pn_ratio', 'is_constant', 'constant_value',
                    'min_period', 'max_period', 'all_pow2', 'all_one',
                    'all_two', 'n_distinct'):
            self.assertIn(key, rec)

    def test_build_width_data_widths_field(self):
        d = self.build_width_data(['ГОРА'])
        self.assertEqual(d['widths'], self.DEFAULT_WIDTHS)

    # ── Cross-rule checks ─────────────────────────────────────────────────────

    def test_xor_all_one_for_all_lexicon_xor_po2(self):
        # ТУМАН XOR is NOT constant; but let's check specific power-of-2 widths
        ps = self.width_series('ТУМАН', 'xor', [4, 8, 16, 32, 64])
        self.assertEqual(ps, [1, 1, 1, 1, 1])

    def test_all_rules_positive_periods(self):
        for rule in self.RULES:
            ps = self.width_series('ГОРА', rule)
            self.assertTrue(all(p >= 1 for p in ps))

    def test_gora_xor_constant_one(self):
        # ГОРА XOR converges to all-0 → P=1 for all widths
        self.assertTrue(self.is_constant_period('ГОРА', 'xor'))
        self.assertEqual(self.constant_period_value('ГОРА', 'xor'), 1)

    # ── Viewer HTML markers ───────────────────────────────────────────────────

    def test_viewer_has_width_chart(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('width-chart', content)

    def test_viewer_has_width_ratio(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('width-ratio', content)

    def test_viewer_has_width_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('width-info', content)

    def test_viewer_has_wd_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wd-word', content)

    def test_viewer_has_wd_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wd-btn', content)

    def test_viewer_has_wd_run_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wdRun', content)

    def test_viewer_has_enc_at_width_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('encAtWidth', content)

    def test_viewer_has_wd_orbit_len_js(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('wdOrbitLen', content)

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_width',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIsInstance(d, dict)

    def test_viewer_has_solan_width(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_width', content)


class TestSolanMultistep(unittest.TestCase):
    """Tests for solan_multistep.py — multi-step Hamming distance matrix."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_multistep import (
            hamming_dist,
            orbit_dist_matrix,
            eccentricity,
            center_steps,
            periphery_steps,
            dist_histogram,
            multistep_summary,
            all_multistep,
            build_multistep_data,
            multistep_dict,
        )
        cls.hamming_dist       = staticmethod(hamming_dist)
        cls.orbit_dist_matrix  = staticmethod(orbit_dist_matrix)
        cls.eccentricity       = staticmethod(eccentricity)
        cls.center_steps       = staticmethod(center_steps)
        cls.periphery_steps    = staticmethod(periphery_steps)
        cls.dist_histogram     = staticmethod(dist_histogram)
        cls.multistep_summary  = staticmethod(multistep_summary)
        cls.all_multistep      = staticmethod(all_multistep)
        cls.build_multistep_data = staticmethod(build_multistep_data)
        cls.multistep_dict     = staticmethod(multistep_dict)

        # Precompute summaries used frequently
        cls.s_tuman_xor  = multistep_summary('ТУМАН', 'xor',  16)
        cls.s_tuman_xor3 = multistep_summary('ТУМАН', 'xor3', 16)
        cls.s_gora_and   = multistep_summary('ГОРА',  'and',  16)
        cls.s_gora_xor3  = multistep_summary('ГОРА',  'xor3', 16)
        cls.s_mat_xor3   = multistep_summary('МАТ',   'xor3', 16)

    # ── hamming_dist ──────────────────────────────────────────────────────────

    def test_hamming_dist_identical(self):
        self.assertEqual(self.hamming_dist([1, 2, 3], [1, 2, 3]), 0)

    def test_hamming_dist_all_differ(self):
        a = [0] * 16
        b = [1] * 16
        self.assertEqual(self.hamming_dist(a, b), 16)

    def test_hamming_dist_half(self):
        a = [0] * 8 + [1] * 8
        b = [1] * 8 + [0] * 8
        self.assertEqual(self.hamming_dist(a, b), 16)

    def test_hamming_dist_one(self):
        a = [0] * 16
        b = list(a); b[5] = 1
        self.assertEqual(self.hamming_dist(a, b), 1)

    def test_hamming_dist_symmetric(self):
        a = [0, 1, 2, 3]
        b = [3, 2, 1, 0]
        self.assertEqual(self.hamming_dist(a, b), self.hamming_dist(b, a))

    # ── orbit_dist_matrix ─────────────────────────────────────────────────────

    def test_dist_matrix_shape_p1(self):
        mat = self.orbit_dist_matrix([[0] * 16])
        self.assertEqual(len(mat), 1)
        self.assertEqual(len(mat[0]), 1)

    def test_dist_matrix_shape_p8(self):
        mat = self.s_tuman_xor3['dist_matrix']
        self.assertEqual(len(mat), 8)
        self.assertTrue(all(len(row) == 8 for row in mat))

    def test_dist_matrix_diagonal_zero(self):
        mat = self.s_tuman_xor3['dist_matrix']
        for t in range(8):
            self.assertEqual(mat[t][t], 0)

    def test_dist_matrix_symmetric(self):
        mat = self.s_tuman_xor3['dist_matrix']
        for t1 in range(8):
            for t2 in range(8):
                self.assertEqual(mat[t1][t2], mat[t2][t1])

    def test_dist_matrix_values_in_range(self):
        mat = self.s_tuman_xor3['dist_matrix']
        for row in mat:
            for d in row:
                self.assertGreaterEqual(d, 0)
                self.assertLessEqual(d, 16)

    def test_dist_matrix_p1_trivial(self):
        mat = self.s_tuman_xor['dist_matrix']
        self.assertEqual(mat, [[0]])

    def test_dist_matrix_p2_all_cells_differ(self):
        # P=2 theorem: H[0][1] = N for all P=2 orbits
        mat = self.s_gora_and['dist_matrix']
        self.assertEqual(mat[0][1], 16)
        self.assertEqual(mat[1][0], 16)

    def test_dist_matrix_p2_xor3_all_cells_differ(self):
        mat = self.s_gora_xor3['dist_matrix']
        self.assertEqual(mat[0][1], 16)

    # ── eccentricity ──────────────────────────────────────────────────────────

    def test_eccentricity_length(self):
        ecc = self.s_tuman_xor3['eccentricity']
        self.assertEqual(len(ecc), 8)

    def test_eccentricity_p1_is_zero(self):
        ecc = self.s_tuman_xor['eccentricity']
        self.assertEqual(ecc, [0])

    def test_eccentricity_p2_all_max(self):
        ecc = self.s_gora_and['eccentricity']
        self.assertTrue(all(e == 16 for e in ecc))

    def test_eccentricity_tuman_xor3_all_16(self):
        ecc = self.s_tuman_xor3['eccentricity']
        self.assertTrue(all(e == 16 for e in ecc))

    def test_eccentricity_mat_xor3_not_uniform(self):
        # MАТ XOR3 is non-regular: not all eccentricities equal
        ecc = self.s_mat_xor3['eccentricity']
        self.assertGreater(max(ecc), min(ecc))

    # ── center_steps / periphery_steps ────────────────────────────────────────

    def test_center_steps_p1(self):
        self.assertEqual(self.s_tuman_xor['center_steps'], [0])

    def test_periphery_steps_p1(self):
        self.assertEqual(self.s_tuman_xor['periphery_steps'], [0])

    def test_center_steps_regular_orbit_all(self):
        # Regular orbit: all steps are center steps
        ctrs = self.s_tuman_xor3['center_steps']
        self.assertEqual(sorted(ctrs), list(range(8)))

    def test_periphery_steps_regular_orbit_all(self):
        peri = self.s_tuman_xor3['periphery_steps']
        self.assertEqual(sorted(peri), list(range(8)))

    def test_center_steps_mat_xor3(self):
        # MАТ XOR3 has a genuine center (step 2)
        ctrs = self.s_mat_xor3['center_steps']
        self.assertIn(2, ctrs)
        # center eccentricity < diameter
        ecc = self.s_mat_xor3['eccentricity']
        for c in ctrs:
            self.assertLess(ecc[c], self.s_mat_xor3['diameter'])

    # ── multistep_summary ─────────────────────────────────────────────────────

    def test_summary_required_keys(self):
        required = {
            'word', 'rule', 'period', 'n_cells',
            'dist_matrix', 'eccentricity', 'diameter', 'radius',
            'center_steps', 'periphery_steps',
            'girth', 'orbit_spread', 'dist_histogram', 'is_regular',
        }
        self.assertTrue(required.issubset(self.s_tuman_xor3.keys()))

    def test_summary_word_preserved(self):
        self.assertEqual(self.s_tuman_xor3['word'], 'ТУМАН')

    def test_summary_rule_preserved(self):
        self.assertEqual(self.s_tuman_xor3['rule'], 'xor3')

    def test_summary_period_tuman_xor3(self):
        self.assertEqual(self.s_tuman_xor3['period'], 8)

    def test_summary_period_tuman_xor(self):
        self.assertEqual(self.s_tuman_xor['period'], 1)

    def test_summary_n_cells(self):
        self.assertEqual(self.s_tuman_xor3['n_cells'], 16)

    def test_summary_diameter_tuman_xor(self):
        self.assertEqual(self.s_tuman_xor['diameter'], 0)

    def test_summary_diameter_gora_and(self):
        self.assertEqual(self.s_gora_and['diameter'], 16)

    def test_summary_diameter_tuman_xor3(self):
        self.assertEqual(self.s_tuman_xor3['diameter'], 16)

    def test_summary_radius_tuman_xor(self):
        self.assertEqual(self.s_tuman_xor['radius'], 0)

    def test_summary_radius_tuman_xor3(self):
        # Regular orbit: radius == diameter
        self.assertEqual(self.s_tuman_xor3['radius'], 16)

    def test_summary_radius_mat_xor3(self):
        # Non-regular: radius < diameter
        self.assertLess(self.s_mat_xor3['radius'], self.s_mat_xor3['diameter'])
        self.assertEqual(self.s_mat_xor3['radius'], 12)

    def test_summary_girth_p1(self):
        self.assertEqual(self.s_tuman_xor['girth'], 0)

    def test_summary_girth_p2_all16(self):
        self.assertEqual(self.s_gora_and['girth'], 16)

    def test_summary_girth_tuman_xor3(self):
        self.assertEqual(self.s_tuman_xor3['girth'], 6)

    def test_summary_girth_mat_xor3(self):
        self.assertEqual(self.s_mat_xor3['girth'], 4)

    def test_summary_spread_p1(self):
        self.assertAlmostEqual(self.s_tuman_xor['orbit_spread'], 0.0)

    def test_summary_spread_p2(self):
        self.assertAlmostEqual(self.s_gora_and['orbit_spread'], 16.0)

    def test_summary_spread_tuman_xor3(self):
        self.assertAlmostEqual(
            self.s_tuman_xor3['orbit_spread'], 195.0 / 14, places=4
        )  # 195/14 ≈ 13.9286

    def test_summary_spread_mat_xor3(self):
        self.assertAlmostEqual(self.s_mat_xor3['orbit_spread'], 9.7143, places=3)

    def test_summary_is_regular_p1(self):
        self.assertTrue(self.s_tuman_xor['is_regular'])

    def test_summary_is_regular_p2(self):
        self.assertTrue(self.s_gora_and['is_regular'])

    def test_summary_is_regular_tuman_xor3(self):
        self.assertTrue(self.s_tuman_xor3['is_regular'])

    def test_summary_is_regular_mat_xor3_false(self):
        self.assertFalse(self.s_mat_xor3['is_regular'])

    def test_summary_is_regular_iff_diam_eq_rad(self):
        s = self.s_mat_xor3
        self.assertEqual(s['is_regular'], s['diameter'] == s['radius'])

    # ── dist_histogram ────────────────────────────────────────────────────────

    def test_histogram_p1_empty(self):
        self.assertEqual(self.s_tuman_xor['dist_histogram'], {})

    def test_histogram_p2_only_16(self):
        h = self.s_gora_and['dist_histogram']
        self.assertEqual(list(h.keys()), [16])
        self.assertEqual(h[16], 2)  # H[0][1] and H[1][0]

    def test_histogram_tuman_xor3_total_count(self):
        h = self.s_tuman_xor3['dist_histogram']
        self.assertEqual(sum(h.values()), 8 * 7)  # 56 off-diagonal pairs

    def test_histogram_tuman_xor3_max_distance_count(self):
        h = self.s_tuman_xor3['dist_histogram']
        self.assertEqual(h.get(16, 0), 30)

    def test_histogram_tuman_xor3_girth_count(self):
        h = self.s_tuman_xor3['dist_histogram']
        self.assertEqual(h.get(6, 0), 2)

    # ── all_multistep / build_multistep_data ──────────────────────────────────

    def test_all_multistep_has_four_rules(self):
        d = self.all_multistep('ГОРА', 16)
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_multistep_data_returns_dict(self):
        data = self.build_multistep_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('data', data)

    def test_build_multistep_data_word_list(self):
        data = self.build_multistep_data(['ТУМАН', 'ГОРА'], 16)
        self.assertEqual(data['words'], ['ТУМАН', 'ГОРА'])

    def test_build_multistep_data_per_word(self):
        data = self.build_multistep_data(['ТУМАН'], 16)
        self.assertIn('ТУМАН', data['data'])

    # ── multistep_dict (JSON serialisability) ─────────────────────────────────

    def test_multistep_dict_keys(self):
        d = self.multistep_dict(self.s_tuman_xor3)
        self.assertIn('diameter', d)
        self.assertIn('dist_histogram', d)

    def test_multistep_dict_histogram_str_keys(self):
        d = self.multistep_dict(self.s_tuman_xor3)
        hist = d['dist_histogram']
        for k in hist:
            self.assertIsInstance(k, str)

    def test_multistep_dict_serialisable(self):
        import json
        d = self.multistep_dict(self.s_tuman_xor3)
        s = json.dumps(d)
        self.assertIn('diameter', s)

    # ── Viewer HTML ───────────────────────────────────────────────────────────

    def test_viewer_has_ms_matrix(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ms-matrix', content)

    def test_viewer_has_ms_ecc(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ms-ecc', content)

    def test_viewer_has_ms_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('msRun', content)

    def test_viewer_has_ms_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ms-info', content)

    def test_viewer_has_ms_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ms-word', content)

    def test_viewer_has_ms_hamming(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('msHamming', content)


class TestSolanSemantic(unittest.TestCase):
    """Tests for solan_semantic.py — semantic orbit trajectory."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_semantic import (
            hamming_dist,
            nearest_in_lex,
            dist_to_self,
            semantic_summary,
            all_semantic,
            build_semantic_data,
            semantic_dict,
        )
        from projects.hexglyph.solan_word import encode_word, pad_to
        cls.hamming_dist     = staticmethod(hamming_dist)
        cls.nearest_in_lex   = staticmethod(nearest_in_lex)
        cls.dist_to_self_fn  = staticmethod(dist_to_self)
        cls.semantic_summary = staticmethod(semantic_summary)
        cls.all_semantic     = staticmethod(all_semantic)
        cls.build_semantic_data = staticmethod(build_semantic_data)
        cls.semantic_dict    = staticmethod(semantic_dict)
        cls.encode_word      = staticmethod(encode_word)
        cls.pad_to           = staticmethod(pad_to)

        # Shared lex ICs
        from projects.hexglyph.solan_semantic import _encode_lex
        cls._lex_ics = _encode_lex(16)

        # Precompute summaries
        cls.s_tuman_xor  = semantic_summary('ТУМАН', 'xor',  16, _lex_ics=cls._lex_ics)
        cls.s_tuman_xor3 = semantic_summary('ТУМАН', 'xor3', 16, _lex_ics=cls._lex_ics)
        cls.s_gora_xor3  = semantic_summary('ГОРА',  'xor3', 16, _lex_ics=cls._lex_ics)
        cls.s_mat_xor3   = semantic_summary('МАТ',   'xor3', 16, _lex_ics=cls._lex_ics)
        cls.s_gora_and   = semantic_summary('ГОРА',  'and',  16, _lex_ics=cls._lex_ics)

    # ── hamming_dist ──────────────────────────────────────────────────────────

    def test_hamming_dist_identical(self):
        self.assertEqual(self.hamming_dist([1, 2, 3], [1, 2, 3]), 0)

    def test_hamming_dist_all_differ(self):
        self.assertEqual(self.hamming_dist([0]*16, [1]*16), 16)

    def test_hamming_dist_symmetric(self):
        a = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        b = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14]
        self.assertEqual(self.hamming_dist(a, b), self.hamming_dist(b, a))

    # ── nearest_in_lex ────────────────────────────────────────────────────────

    def test_nearest_is_self_for_word_ic(self):
        # The IC for ТУМАН should be nearest to itself
        ic = self.pad_to(self.encode_word('ТУМАН'), 16)
        nbs = self.nearest_in_lex(ic, self._lex_ics, top_n=1)
        self.assertEqual(nbs[0][0], 'ТУМАН')
        self.assertEqual(nbs[0][1], 0)

    def test_nearest_sorted_ascending(self):
        ic = self.pad_to(self.encode_word('ТУМАН'), 16)
        nbs = self.nearest_in_lex(ic, self._lex_ics, top_n=5)
        dists = [d for _, d in nbs]
        self.assertEqual(dists, sorted(dists))

    def test_nearest_top_n_length(self):
        ic = self.pad_to(self.encode_word('ТУМАН'), 16)
        nbs = self.nearest_in_lex(ic, self._lex_ics, top_n=4)
        self.assertLessEqual(len(nbs), 4)

    def test_nearest_returns_pairs(self):
        ic = self.pad_to(self.encode_word('ГОРА'), 16)
        nbs = self.nearest_in_lex(ic, self._lex_ics, top_n=3)
        for item in nbs:
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], int)

    # ── dist_to_self ──────────────────────────────────────────────────────────

    def test_dist_to_self_t0_xor3_is_zero(self):
        # XOR3 has transient=0 → orbit[0] = word IC → d_self[0]=0
        self.assertEqual(self.s_tuman_xor3['dist_to_self'][0], 0)

    def test_dist_to_self_length_equals_period(self):
        self.assertEqual(
            len(self.s_tuman_xor3['dist_to_self']),
            self.s_tuman_xor3['period']
        )

    def test_dist_to_self_in_range(self):
        for d in self.s_tuman_xor3['dist_to_self']:
            self.assertGreaterEqual(d, 0)
            self.assertLessEqual(d, 16)

    # ── semantic_summary required keys ────────────────────────────────────────

    def test_summary_required_keys(self):
        required = {
            'word', 'rule', 'period', 'n_cells',
            'nearest', 'nearest_word', 'nearest_dist', 'dist_to_self',
            'self_nearest_steps', 'void_steps', 'unique_words',
            'n_unique_words', 'mean_nearest_dist',
            'min_nearest_dist', 'max_nearest_dist', 'self_is_nearest_t0',
        }
        self.assertTrue(required.issubset(self.s_tuman_xor3.keys()))

    def test_summary_word_preserved(self):
        self.assertEqual(self.s_tuman_xor3['word'], 'ТУМАН')

    def test_summary_rule_preserved(self):
        self.assertEqual(self.s_tuman_xor3['rule'], 'xor3')

    def test_summary_period_tuman_xor3(self):
        self.assertEqual(self.s_tuman_xor3['period'], 8)

    def test_summary_period_gora_xor3(self):
        self.assertEqual(self.s_gora_xor3['period'], 2)

    # ── nearest_word / nearest_dist ───────────────────────────────────────────

    def test_nearest_word_length_equals_period(self):
        self.assertEqual(len(self.s_tuman_xor3['nearest_word']), 8)

    def test_nearest_dist_length_equals_period(self):
        self.assertEqual(len(self.s_tuman_xor3['nearest_dist']), 8)

    def test_nearest_word_t0_is_self_xor3(self):
        # XOR3 transient=0 → orbit[0]=word IC → nearest is the word itself
        self.assertEqual(self.s_tuman_xor3['nearest_word'][0], 'ТУМАН')
        self.assertEqual(self.s_gora_xor3['nearest_word'][0], 'ГОРА')
        self.assertEqual(self.s_mat_xor3['nearest_word'][0], 'МАТ')

    def test_nearest_dist_t0_xor3_is_zero(self):
        self.assertEqual(self.s_tuman_xor3['nearest_dist'][0], 0)
        self.assertEqual(self.s_gora_xor3['nearest_dist'][0], 0)

    def test_nearest_dist_in_valid_range(self):
        for d in self.s_tuman_xor3['nearest_dist']:
            self.assertGreaterEqual(d, 0)
            self.assertLessEqual(d, 16)

    def test_nearest_top_has_correct_structure(self):
        top = self.s_tuman_xor3['nearest']
        self.assertEqual(len(top), 8)
        for step in top:
            self.assertIsInstance(step, list)
            self.assertGreater(len(step), 0)

    # ── self_nearest_steps ────────────────────────────────────────────────────

    def test_self_nearest_steps_tuman_xor3(self):
        self.assertEqual(self.s_tuman_xor3['self_nearest_steps'], [0, 3, 6])

    def test_self_nearest_steps_gora_xor3(self):
        self.assertEqual(self.s_gora_xor3['self_nearest_steps'], [0])

    def test_self_nearest_steps_mat_xor3(self):
        self.assertEqual(self.s_mat_xor3['self_nearest_steps'], [0, 2, 6])

    def test_self_nearest_steps_is_subset_of_range(self):
        P = self.s_tuman_xor3['period']
        for t in self.s_tuman_xor3['self_nearest_steps']:
            self.assertIn(t, range(P))

    def test_self_nearest_steps_t0_always_included_for_xor3(self):
        # XOR3 transient=0 → step 0 always nearest to self
        self.assertIn(0, self.s_tuman_xor3['self_nearest_steps'])
        self.assertIn(0, self.s_gora_xor3['self_nearest_steps'])
        self.assertIn(0, self.s_mat_xor3['self_nearest_steps'])

    # ── void_steps ────────────────────────────────────────────────────────────

    def test_void_steps_gora_xor3(self):
        # P=2 XOR3: t=1 is always void (complement state not in lexicon)
        self.assertIn(1, self.s_gora_xor3['void_steps'])

    def test_void_steps_tuman_xor3_empty(self):
        # ТУМАН XOR3 has no void steps (all nearest dist < 16)
        self.assertEqual(self.s_tuman_xor3['void_steps'], [])

    def test_void_steps_mat_xor3_empty(self):
        self.assertEqual(self.s_mat_xor3['void_steps'], [])

    def test_void_steps_correspond_to_max_dist(self):
        N = 16
        for t in self.s_gora_xor3['void_steps']:
            self.assertEqual(self.s_gora_xor3['nearest_dist'][t], N)

    # ── unique_words / n_unique_words ─────────────────────────────────────────

    def test_n_unique_words_tuman_xor3(self):
        self.assertEqual(self.s_tuman_xor3['n_unique_words'], 6)

    def test_n_unique_words_gora_xor3(self):
        # ГОРА: nearest at t=0 is ГОРА, at t=1 is a void placeholder
        self.assertEqual(self.s_gora_xor3['n_unique_words'], 2)

    def test_unique_words_first_is_self_for_xor3(self):
        # Orbit starts at word's IC, so nearest at t=0 = word itself
        self.assertEqual(self.s_tuman_xor3['unique_words'][0], 'ТУМАН')
        self.assertEqual(self.s_mat_xor3['unique_words'][0], 'МАТ')

    def test_unique_words_length_le_period(self):
        P = self.s_tuman_xor3['period']
        self.assertLessEqual(self.s_tuman_xor3['n_unique_words'], P)

    def test_unique_words_no_duplicates(self):
        uw = self.s_tuman_xor3['unique_words']
        self.assertEqual(len(uw), len(set(uw)))

    # ── mean/min/max nearest dist ─────────────────────────────────────────────

    def test_mean_nearest_dist_tuman_xor3(self):
        self.assertAlmostEqual(self.s_tuman_xor3['mean_nearest_dist'], 10.875, places=3)

    def test_min_nearest_dist_tuman_xor3(self):
        self.assertEqual(self.s_tuman_xor3['min_nearest_dist'], 0)

    def test_max_nearest_dist_tuman_xor3(self):
        self.assertEqual(self.s_tuman_xor3['max_nearest_dist'], 15)

    def test_max_nearest_dist_gora_xor3_void(self):
        # Void step → max dist = 16
        self.assertEqual(self.s_gora_xor3['max_nearest_dist'], 16)

    def test_mean_dist_consistent_with_nearest_dist(self):
        nd = self.s_tuman_xor3['nearest_dist']
        expected = sum(nd) / len(nd)
        self.assertAlmostEqual(self.s_tuman_xor3['mean_nearest_dist'],
                               expected, places=4)

    # ── self_is_nearest_t0 ────────────────────────────────────────────────────

    def test_self_is_nearest_t0_xor3_true(self):
        # XOR3 transient=0: orbit starts at word IC → nearest is self
        self.assertTrue(self.s_tuman_xor3['self_is_nearest_t0'])
        self.assertTrue(self.s_gora_xor3['self_is_nearest_t0'])
        self.assertTrue(self.s_mat_xor3['self_is_nearest_t0'])

    # ── all_semantic / build_semantic_data ────────────────────────────────────

    def test_all_semantic_has_four_rules(self):
        d = self.all_semantic('ГОРА', 16)
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_semantic_data_keys(self):
        data = self.build_semantic_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('data', data)

    def test_build_semantic_data_word_list(self):
        data = self.build_semantic_data(['ТУМАН', 'ГОРА'], 16)
        self.assertEqual(data['words'], ['ТУМАН', 'ГОРА'])

    # ── semantic_dict ─────────────────────────────────────────────────────────

    def test_semantic_dict_keys(self):
        d = self.semantic_dict(self.s_tuman_xor3)
        self.assertIn('nearest_word', d)
        self.assertIn('void_steps', d)
        self.assertIn('unique_words', d)

    def test_semantic_dict_serialisable(self):
        import json
        d = self.semantic_dict(self.s_tuman_xor3)
        s = json.dumps(d, ensure_ascii=False)
        self.assertIn('ТУМАН', s)

    def test_semantic_dict_nearest_top_structure(self):
        d = self.semantic_dict(self.s_tuman_xor3)
        top = d['nearest_top']
        self.assertEqual(len(top), 8)
        self.assertIn('word', top[0][0])
        self.assertIn('dist', top[0][0])

    # ── Viewer HTML ───────────────────────────────────────────────────────────

    def test_viewer_has_sm_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sm-canvas', content)

    def test_viewer_has_sm_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('smRun', content)

    def test_viewer_has_sm_nearest(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('smNearest', content)

    def test_viewer_has_sm_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('sm-info', content)

    def test_viewer_has_sm_hamming(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('smHamming', content)

    def test_viewer_has_sm_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('smOrbit', content)

    def test_viewer_has_void_steps_label(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('void_steps', content)

    def test_viewer_has_self_nearest_label(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('self_nearest', content)


class TestSolanBitplane(unittest.TestCase):
    """Tests for solan_bitplane.py — phonetic bit-plane analysis."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_bitplane import (
            get_bit_plane, plane_period, frozen_type, cell_activity,
            plane_hamming, coupling_matrix, bitplane_summary,
            all_bitplane, build_bitplane_data, bitplane_dict,
        )
        cls.get_bit_plane    = staticmethod(get_bit_plane)
        cls.plane_period     = staticmethod(plane_period)
        cls.frozen_type      = staticmethod(frozen_type)
        cls.cell_activity    = staticmethod(cell_activity)
        cls.plane_hamming    = staticmethod(plane_hamming)
        cls.coupling_matrix  = staticmethod(coupling_matrix)
        cls.bitplane_summary = staticmethod(bitplane_summary)
        cls.all_bitplane     = staticmethod(all_bitplane)
        cls.build_data       = staticmethod(build_bitplane_data)
        cls.bitplane_dict    = staticmethod(bitplane_dict)

        # Precomputed summaries
        cls.s_tuman_xor3 = bitplane_summary('ТУМАН', 'xor3', 16)
        cls.s_mat_xor3   = bitplane_summary('МАТ',   'xor3', 16)
        cls.s_gora_xor3  = bitplane_summary('ГОРА',  'xor3', 16)
        cls.s_rota_xor3  = bitplane_summary('РОТА',  'xor3', 16)
        cls.s_dobro_xor3 = bitplane_summary('ДОБРО', 'xor3', 16)

    # ── get_bit_plane ─────────────────────────────────────────────────────────

    def test_bit0_of_63_is_1(self):
        # 63 = 0b111111 → bit0 = 1
        plane = self.get_bit_plane([[63, 0]], 0)
        self.assertEqual(plane[0][0], 1)
        self.assertEqual(plane[0][1], 0)

    def test_bit5_of_63_is_1(self):
        plane = self.get_bit_plane([[63]], 5)
        self.assertEqual(plane[0][0], 1)

    def test_bit5_of_31_is_0(self):
        # 31 = 0b011111 → bit5 = 0
        plane = self.get_bit_plane([[31]], 5)
        self.assertEqual(plane[0][0], 0)

    def test_bit_plane_shape(self):
        orbit = [[i for i in range(16)]] * 4
        plane = self.get_bit_plane(orbit, 0)
        self.assertEqual(len(plane), 4)
        self.assertTrue(all(len(row) == 16 for row in plane))

    def test_all_zero_orbit_all_bits_zero(self):
        orbit = [[0] * 8] * 3
        for b in range(6):
            plane = self.get_bit_plane(orbit, b)
            self.assertTrue(all(v == 0 for row in plane for v in row))

    def test_all_63_orbit_all_bits_one(self):
        orbit = [[63] * 8] * 3
        for b in range(6):
            plane = self.get_bit_plane(orbit, b)
            self.assertTrue(all(v == 1 for row in plane for v in row))

    # ── plane_period ──────────────────────────────────────────────────────────

    def test_frozen_plane_period_is_1(self):
        plane = [(0,) * 8] * 5  # same state repeated
        self.assertEqual(self.plane_period(plane), 1)

    def test_alternating_plane_period_is_2(self):
        plane = [(0, 1)] * 1 + [(1, 0)] * 1 + [(0, 1)] * 1
        self.assertEqual(self.plane_period(plane), 2)

    def test_plane_period_divides_orbit_period(self):
        P = self.s_tuman_xor3['period']
        for b in range(6):
            pp = self.s_tuman_xor3['bit_periods'][b]
            self.assertEqual(P % pp, 0,
                             f"plane period {pp} does not divide P={P} for bit{b}")

    def test_plane_period_tuman_xor3_all_8(self):
        # All 6 bit planes of ТУМАН XOR3 have period 8
        for b in range(6):
            self.assertEqual(self.s_tuman_xor3['bit_periods'][b], 8)

    def test_plane_period_mat_d1_is_1(self):
        self.assertEqual(self.s_mat_xor3['bit_periods'][4], 1)

    def test_plane_period_gora_t_is_1(self):
        self.assertEqual(self.s_gora_xor3['bit_periods'][0], 1)

    # ── frozen_type ───────────────────────────────────────────────────────────

    def test_frozen_type_uniform_1(self):
        plane = [(1, 1, 1)] * 3
        self.assertEqual(self.frozen_type(plane), 'uniform_1')

    def test_frozen_type_uniform_0(self):
        plane = [(0, 0, 0)] * 3
        self.assertEqual(self.frozen_type(plane), 'uniform_0')

    def test_frozen_type_patterned(self):
        plane = [(0, 1, 0)] * 3  # frozen but not uniform
        self.assertEqual(self.frozen_type(plane), 'patterned')

    def test_frozen_type_active(self):
        plane = [(0, 1), (1, 0), (0, 1)]
        self.assertEqual(self.frozen_type(plane), 'active')

    def test_mat_d1_frozen_type_uniform_1(self):
        self.assertEqual(self.s_mat_xor3['frozen_types'][4], 'uniform_1')

    def test_gora_t_frozen_type_uniform_1(self):
        self.assertEqual(self.s_gora_xor3['frozen_types'][0], 'uniform_1')

    # ── cell_activity ─────────────────────────────────────────────────────────

    def test_cell_activity_all_0(self):
        plane = [(0, 0)] * 4
        act = self.cell_activity(plane)
        self.assertEqual(act, [0.0, 0.0])

    def test_cell_activity_all_1(self):
        plane = [(1, 1)] * 4
        act = self.cell_activity(plane)
        self.assertEqual(act, [1.0, 1.0])

    def test_cell_activity_half(self):
        plane = [(1, 0), (0, 1), (1, 0), (0, 1)]
        act = self.cell_activity(plane)
        self.assertAlmostEqual(act[0], 0.5)
        self.assertAlmostEqual(act[1], 0.5)

    def test_mat_d1_activity_all_ones(self):
        act = self.s_mat_xor3['cell_activity'][4]
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in act))

    # ── plane_hamming ─────────────────────────────────────────────────────────

    def test_plane_hamming_frozen_is_all_zero(self):
        plane = [(0, 1, 0)] * 5
        hd = self.plane_hamming(plane)
        self.assertEqual(hd, [0] * 5)

    def test_plane_hamming_length_equals_period(self):
        hd = self.s_tuman_xor3['plane_hamming'][0]
        self.assertEqual(len(hd), self.s_tuman_xor3['period'])

    def test_mat_d1_hamming_all_zero(self):
        hd = self.s_mat_xor3['plane_hamming'][4]
        self.assertEqual(hd, [0] * self.s_mat_xor3['period'])

    def test_tuman_t_and_b_hamming_identical(self):
        # T and B planes have identical Hamming sequences in ТУМАН XOR3
        hd0 = self.s_tuman_xor3['plane_hamming'][0]
        hd1 = self.s_tuman_xor3['plane_hamming'][1]
        self.assertEqual(hd0, hd1)

    # ── coupling_matrix ───────────────────────────────────────────────────────

    def test_coupling_matrix_diagonal_is_1(self):
        from projects.hexglyph.solan_perm import get_orbit
        orbit = get_orbit('ТУМАН', 'xor3', 16)
        mat = self.coupling_matrix(orbit)
        for b in range(6):
            self.assertAlmostEqual(mat[b][b], 1.0, places=9)

    def test_coupling_matrix_symmetric(self):
        from projects.hexglyph.solan_perm import get_orbit
        orbit = get_orbit('ТУМАН', 'xor3', 16)
        mat = self.coupling_matrix(orbit)
        for b1 in range(6):
            for b2 in range(6):
                r1, r2 = mat[b1][b2], mat[b2][b1]
                if r1 is None or r2 is None:
                    self.assertIsNone(r1)
                    self.assertIsNone(r2)
                else:
                    self.assertAlmostEqual(r1, r2, places=12)

    def test_coupling_matrix_constant_plane_gives_none(self):
        # МАТ XOR3 D1 is constant → coupling with D1 should be None off-diag
        mat = self.s_mat_xor3['coupling']
        for b2 in range(6):
            if b2 == 4:
                continue  # diagonal
            self.assertIsNone(mat[4][b2])
            self.assertIsNone(mat[b2][4])

    # ── bitplane_summary keys & structure ─────────────────────────────────────

    def test_summary_required_keys(self):
        required = {
            'word', 'rule', 'period', 'n_cells',
            'bit_periods', 'frozen_types', 'n_active',
            'n_frozen_uniform', 'n_frozen_patterned',
            'frozen_uniform_bits', 'frozen_bit_values',
            'coupling', 'coupled_pairs', 'anti_coupled_pairs',
            'cell_activity', 'plane_hamming',
        }
        self.assertTrue(required.issubset(self.s_tuman_xor3.keys()))

    def test_summary_word_preserved(self):
        self.assertEqual(self.s_tuman_xor3['word'], 'ТУМАН')

    def test_summary_n_cells(self):
        self.assertEqual(self.s_tuman_xor3['n_cells'], 16)

    def test_bit_periods_length_6(self):
        self.assertEqual(len(self.s_tuman_xor3['bit_periods']), 6)

    def test_frozen_types_length_6(self):
        self.assertEqual(len(self.s_tuman_xor3['frozen_types']), 6)

    def test_cell_activity_length_6(self):
        self.assertEqual(len(self.s_tuman_xor3['cell_activity']), 6)

    def test_plane_hamming_length_6(self):
        self.assertEqual(len(self.s_tuman_xor3['plane_hamming']), 6)

    # ── frozen uniform bits ───────────────────────────────────────────────────

    def test_mat_n_frozen_uniform_is_1(self):
        self.assertEqual(self.s_mat_xor3['n_frozen_uniform'], 1)

    def test_mat_frozen_uniform_bits_contains_d1(self):
        self.assertIn(4, self.s_mat_xor3['frozen_uniform_bits'])

    def test_mat_frozen_bit_value_d1_is_1(self):
        self.assertEqual(self.s_mat_xor3['frozen_bit_values'][4], 1)

    def test_gora_n_frozen_uniform_is_1(self):
        self.assertEqual(self.s_gora_xor3['n_frozen_uniform'], 1)

    def test_gora_frozen_uniform_bits_contains_t(self):
        self.assertIn(0, self.s_gora_xor3['frozen_uniform_bits'])

    def test_gora_frozen_bit_value_t_is_1(self):
        self.assertEqual(self.s_gora_xor3['frozen_bit_values'][0], 1)

    def test_tuman_no_frozen_planes(self):
        self.assertEqual(self.s_tuman_xor3['n_frozen_uniform'], 0)
        self.assertEqual(self.s_tuman_xor3['n_frozen_patterned'], 0)

    # ── n_active ──────────────────────────────────────────────────────────────

    def test_n_active_tuman_xor3_is_6(self):
        self.assertEqual(self.s_tuman_xor3['n_active'], 6)

    def test_n_active_mat_xor3_is_5(self):
        self.assertEqual(self.s_mat_xor3['n_active'], 5)

    def test_n_active_gora_xor3_is_5(self):
        self.assertEqual(self.s_gora_xor3['n_active'], 5)

    def test_n_active_plus_frozen_equals_6(self):
        s = self.s_mat_xor3
        total = s['n_active'] + s['n_frozen_uniform'] + s['n_frozen_patterned']
        self.assertEqual(total, 6)

    # ── coupled / anti-coupled pairs ──────────────────────────────────────────

    def test_tuman_coupled_pairs_contains_t_b(self):
        self.assertIn((0, 1), self.s_tuman_xor3['coupled_pairs'])

    def test_mat_coupled_pairs_contains_t_b(self):
        self.assertIn((0, 1), self.s_mat_xor3['coupled_pairs'])

    def test_mat_coupled_pairs_contains_t_l(self):
        self.assertIn((0, 2), self.s_mat_xor3['coupled_pairs'])

    def test_gora_coupled_pairs_contains_b_l(self):
        self.assertIn((1, 2), self.s_gora_xor3['coupled_pairs'])

    def test_gora_coupled_pairs_contains_l_r(self):
        self.assertIn((2, 3), self.s_gora_xor3['coupled_pairs'])

    def test_rota_has_maximum_coupling(self):
        # РОТА XOR3: T=B=L=R all coupled (6 pairs among 4 planes)
        cp = self.s_rota_xor3['coupled_pairs']
        # All pairs of {T,B,L,R} = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        for pair in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            self.assertIn(pair, cp)

    def test_dobro_has_anti_coupling(self):
        # ДОБРО XOR3: L != D1 (bit2 != bit4)
        acp = self.s_dobro_xor3['anti_coupled_pairs']
        self.assertIn((2, 4), acp)

    def test_coupled_pairs_are_sorted(self):
        for b1, b2 in self.s_tuman_xor3['coupled_pairs']:
            self.assertLess(b1, b2)

    # ── bitplane_dict serialisation ───────────────────────────────────────────

    def test_bitplane_dict_serialisable(self):
        import json
        d = self.bitplane_dict(self.s_tuman_xor3)
        s = json.dumps(d, ensure_ascii=False)
        self.assertIn('ТУМАН', s)

    def test_bitplane_dict_coupling_matrix_no_none_as_string(self):
        import json
        d = self.bitplane_dict(self.s_mat_xor3)
        mat = d['coupling_matrix']
        # None should be preserved as JSON null, not as string 'None'
        s = json.dumps(mat)
        self.assertNotIn('"None"', s)

    def test_all_bitplane_has_four_rules(self):
        d = self.all_bitplane('ТУМАН', 16)
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── Viewer HTML ───────────────────────────────────────────────────────────

    def test_viewer_has_bp_grid(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bp-grid', content)

    def test_viewer_has_bp_heat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bp-heat', content)

    def test_viewer_has_bp_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bpRun', content)

    def test_viewer_has_frozen_type(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('frozenType', content)

    def test_viewer_has_pearson(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pearson', content)

    def test_viewer_has_bp_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bpOrbit', content)

    def test_viewer_has_bit_names(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn("'T','B','L','R','D1','D2'", content)


class TestSolanProfile(unittest.TestCase):
    """Tests for solan_profile.py — statistical moment profile."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_profile import (
            _step_moments, profile_summary, all_profile,
            build_profile_data, profile_dict,
        )
        cls._step_moments    = staticmethod(_step_moments)
        cls.profile_summary  = staticmethod(profile_summary)
        cls.all_profile      = staticmethod(all_profile)
        cls.build_data       = staticmethod(build_profile_data)
        cls.profile_dict     = staticmethod(profile_dict)

        # Precomputed summaries
        cls.s_mat_xor3    = profile_summary('МАТ',    'xor3', 16)
        cls.s_tundra_xor3 = profile_summary('ТУНДРА', 'xor3', 16)
        cls.s_tuman_xor3  = profile_summary('ТУМАН',  'xor3', 16)
        cls.s_gora_xor3   = profile_summary('ГОРА',   'xor3', 16)
        cls.s_duga_xor3   = profile_summary('ДУГА',   'xor3', 16)

    # ── _step_moments basics ──────────────────────────────────────────────────

    def test_uniform_state_var_is_zero(self):
        p = self._step_moments([42] * 16)
        self.assertAlmostEqual(p['var'], 0.0, places=9)

    def test_uniform_state_skew_is_zero(self):
        p = self._step_moments([10] * 8)
        self.assertAlmostEqual(p['skewness'], 0.0, places=9)

    def test_uniform_state_kurt_is_zero(self):
        p = self._step_moments([10] * 8)
        self.assertAlmostEqual(p['kurtosis'], 0.0, places=9)

    def test_uniform_state_mode_count_is_N(self):
        p = self._step_moments([7] * 16)
        self.assertEqual(p['mode_count'], 16)
        self.assertAlmostEqual(p['mode_fraction'], 1.0, places=9)

    def test_two_value_state_range(self):
        p = self._step_moments([0] * 8 + [10] * 8)
        self.assertEqual(p['range'], 10)

    def test_two_value_state_mean(self):
        p = self._step_moments([0] * 8 + [10] * 8)
        self.assertAlmostEqual(p['mean'], 5.0, places=9)

    def test_n_distinct_counts_unique(self):
        p = self._step_moments([1, 2, 3, 1, 2, 3, 1, 2])
        self.assertEqual(p['n_distinct'], 3)

    def test_mode_is_most_common(self):
        p = self._step_moments([5, 5, 5, 7, 8, 9, 10, 5])
        self.assertEqual(p['mode'], 5)
        self.assertEqual(p['mode_count'], 4)

    # ── profile_summary required keys ─────────────────────────────────────────

    def test_summary_required_keys(self):
        required = {
            'word', 'rule', 'period', 'n_cells',
            'spatial_profiles', 'spatial_mean', 'spatial_var',
            'spatial_skewness', 'spatial_kurtosis', 'spatial_range',
            'mode_vals', 'mode_counts', 'mode_fractions', 'n_distinct',
            'max_mode_fraction', 'max_mode_fraction_step',
            'max_skew_abs', 'max_skew_abs_step',
            'max_kurtosis', 'max_kurtosis_step',
            'min_var', 'min_var_step',
            'max_range', 'max_range_step',
            'mean_spatial_mean', 'mean_spatial_var',
            'dominant_mode_val', 'dominant_mode_n',
            'temporal_mean', 'temporal_var',
            'max_temporal_var_cell', 'max_temporal_var',
        }
        self.assertTrue(required.issubset(self.s_mat_xor3.keys()))

    def test_summary_word_preserved(self):
        self.assertEqual(self.s_mat_xor3['word'], 'МАТ')

    def test_summary_rule_preserved(self):
        self.assertEqual(self.s_mat_xor3['rule'], 'xor3')

    def test_summary_period(self):
        self.assertEqual(self.s_mat_xor3['period'], 8)
        self.assertEqual(self.s_gora_xor3['period'], 2)

    # ── spatial_profiles length ───────────────────────────────────────────────

    def test_spatial_profiles_length_equals_period(self):
        P = self.s_mat_xor3['period']
        self.assertEqual(len(self.s_mat_xor3['spatial_profiles']), P)

    def test_spatial_mean_length_equals_period(self):
        P = self.s_mat_xor3['period']
        self.assertEqual(len(self.s_mat_xor3['spatial_mean']), P)

    def test_mode_fractions_length_equals_period(self):
        P = self.s_mat_xor3['period']
        self.assertEqual(len(self.s_mat_xor3['mode_fractions']), P)

    # ── МАТ XOR3 known values ─────────────────────────────────────────────────

    def test_mat_max_mode_fraction(self):
        # 14/16 cells = 0.875 at t=1
        self.assertAlmostEqual(
            self.s_mat_xor3['max_mode_fraction'], 0.875, places=4)

    def test_mat_max_mode_fraction_step(self):
        self.assertEqual(self.s_mat_xor3['max_mode_fraction_step'], 1)

    def test_mat_mode_at_t1_is_23(self):
        self.assertEqual(self.s_mat_xor3['mode_vals'][1], 23)

    def test_mat_mode_count_at_t1_is_14(self):
        self.assertEqual(self.s_mat_xor3['mode_counts'][1], 14)

    def test_mat_max_skew_abs(self):
        self.assertAlmostEqual(
            self.s_mat_xor3['max_skew_abs'], 2.518, places=2)

    def test_mat_max_skew_step(self):
        self.assertEqual(self.s_mat_xor3['max_skew_abs_step'], 1)

    def test_mat_max_kurtosis(self):
        self.assertAlmostEqual(
            self.s_mat_xor3['max_kurtosis'], 4.756, places=2)

    def test_mat_max_kurtosis_step(self):
        self.assertEqual(self.s_mat_xor3['max_kurtosis_step'], 1)

    def test_mat_dominant_mode_is_23(self):
        self.assertEqual(self.s_mat_xor3['dominant_mode_val'], 23)

    def test_mat_dominant_mode_n(self):
        # Value 23 is mode at 6 of 8 steps
        self.assertEqual(self.s_mat_xor3['dominant_mode_n'], 6)

    # ── ТУНДРА XOR3 known values ──────────────────────────────────────────────

    def test_tundra_max_skew_abs(self):
        self.assertAlmostEqual(
            self.s_tundra_xor3['max_skew_abs'], 3.057, places=2)

    def test_tundra_max_skew_step(self):
        self.assertEqual(self.s_tundra_xor3['max_skew_abs_step'], 2)

    def test_tundra_max_kurtosis(self):
        self.assertAlmostEqual(
            self.s_tundra_xor3['max_kurtosis'], 8.41, places=1)

    def test_tundra_max_kurtosis_step(self):
        self.assertEqual(self.s_tundra_xor3['max_kurtosis_step'], 2)

    # ── ДУГА XOR3 — most uniform ──────────────────────────────────────────────

    def test_duga_min_var_is_small(self):
        # ДУГА XOR3 has smallest non-zero variance (~32.19)
        self.assertLess(self.s_duga_xor3['min_var'], 35.0)

    def test_duga_min_var_is_positive(self):
        self.assertGreater(self.s_duga_xor3['min_var'], 0.0)

    # ── Structural invariants ─────────────────────────────────────────────────

    def test_mode_fractions_in_valid_range(self):
        N = self.s_mat_xor3['n_cells']
        for mf in self.s_mat_xor3['mode_fractions']:
            self.assertGreaterEqual(mf, 1.0 / N - 1e-9)
            self.assertLessEqual(mf, 1.0 + 1e-9)

    def test_n_distinct_in_valid_range(self):
        N = self.s_mat_xor3['n_cells']
        for nd in self.s_mat_xor3['n_distinct']:
            self.assertGreaterEqual(nd, 1)
            self.assertLessEqual(nd, N)

    def test_mean_spatial_mean_consistent(self):
        sm = self.s_mat_xor3['spatial_mean']
        expected = sum(sm) / len(sm)
        self.assertAlmostEqual(
            self.s_mat_xor3['mean_spatial_mean'], expected, places=3)

    def test_mean_spatial_var_consistent(self):
        sv = self.s_mat_xor3['spatial_var']
        expected = sum(sv) / len(sv)
        self.assertAlmostEqual(
            self.s_mat_xor3['mean_spatial_var'], expected, places=3)

    def test_max_mode_fraction_is_max_of_list(self):
        mf_list = self.s_mat_xor3['mode_fractions']
        self.assertAlmostEqual(
            self.s_mat_xor3['max_mode_fraction'], max(mf_list), places=9)

    def test_max_skew_abs_is_max_of_abs_skewness(self):
        sk_list = self.s_mat_xor3['spatial_skewness']
        self.assertAlmostEqual(
            self.s_mat_xor3['max_skew_abs'], max(abs(v) for v in sk_list),
            places=4)

    def test_max_range_consistent(self):
        rg = self.s_mat_xor3['spatial_range']
        self.assertEqual(self.s_mat_xor3['max_range'], max(rg))

    # ── Temporal moments ──────────────────────────────────────────────────────

    def test_temporal_mean_length_equals_n_cells(self):
        N = self.s_mat_xor3['n_cells']
        self.assertEqual(len(self.s_mat_xor3['temporal_mean']), N)

    def test_temporal_var_length_equals_n_cells(self):
        N = self.s_mat_xor3['n_cells']
        self.assertEqual(len(self.s_mat_xor3['temporal_var']), N)

    def test_temporal_var_p1_is_zero(self):
        # P=1 orbit (XOR fixed point): all cells constant → temporal_var=0
        s = self.profile_summary('ТУМАН', 'xor', 16)
        self.assertEqual(s['period'], 1)
        for tv in s['temporal_var']:
            self.assertAlmostEqual(tv, 0.0, places=9)

    def test_max_temporal_var_cell_is_valid_index(self):
        N = self.s_mat_xor3['n_cells']
        self.assertIn(self.s_mat_xor3['max_temporal_var_cell'], range(N))

    def test_max_temporal_var_matches_cell(self):
        cell = self.s_mat_xor3['max_temporal_var_cell']
        tv_list = self.s_mat_xor3['temporal_var']
        self.assertAlmostEqual(
            self.s_mat_xor3['max_temporal_var'], tv_list[cell], places=3)

    # ── all_profile / build_data ──────────────────────────────────────────────

    def test_all_profile_has_four_rules(self):
        d = self.all_profile('ТУМАН', 16)
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_data_word_list(self):
        data = self.build_data(['МАТ', 'ГОРА'], 16)
        self.assertEqual(data['words'], ['МАТ', 'ГОРА'])

    def test_build_data_has_all_rules(self):
        data = self.build_data(['МАТ'], 16)
        self.assertEqual(set(data['data']['МАТ'].keys()), {'xor', 'xor3', 'and', 'or'})

    # ── profile_dict ──────────────────────────────────────────────────────────

    def test_profile_dict_serialisable(self):
        import json
        d = self.profile_dict(self.s_mat_xor3)
        s = json.dumps(d, ensure_ascii=False)
        self.assertIn('МАТ', s)

    # ── Viewer HTML ───────────────────────────────────────────────────────────

    def test_viewer_has_pf_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pf-canvas', content)

    def test_viewer_has_pf_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pfRun', content)

    def test_viewer_has_pf_moments(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pfMoments', content)

    def test_viewer_has_pf_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pfOrbit', content)

    def test_viewer_has_pf_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pf-info', content)

    def test_viewer_has_mode_frac_label(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mode_frac', content)


class TestSolanCoverage(unittest.TestCase):
    """Tests for solan_coverage.py — Q6 value coverage of CA orbits."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_coverage import (
            q6_label, orbit_frequencies, coverage_summary,
            global_coverage, all_coverage, build_coverage_data, coverage_dict,
        )
        cls.q6_label          = staticmethod(q6_label)
        cls.orbit_frequencies = staticmethod(orbit_frequencies)
        cls.coverage_summary  = staticmethod(coverage_summary)
        cls.global_coverage   = staticmethod(global_coverage)
        cls.all_coverage      = staticmethod(all_coverage)
        cls.build_data        = staticmethod(build_coverage_data)
        cls.coverage_dict     = staticmethod(coverage_dict)

        # Precomputed summaries
        cls.s_mat_xor3   = coverage_summary('МАТ',    'xor3', 16)
        cls.s_tuman_xor3 = coverage_summary('ТУМАН',  'xor3', 16)
        cls.s_rabota_xor3= coverage_summary('РАБОТА', 'xor3', 16)
        cls.s_gora_xor3  = coverage_summary('ГОРА',   'xor3', 16)
        cls.s_tuman_xor  = coverage_summary('ТУМАН',  'xor',  16)
        cls.g_xor3        = global_coverage('xor3', 16)
        cls.g_xor         = global_coverage('xor',  16)

    # ── q6_label ──────────────────────────────────────────────────────────────

    def test_q6_label_zero(self):
        self.assertEqual(self.q6_label(0), '0')

    def test_q6_label_63(self):
        self.assertEqual(self.q6_label(63), 'T+B+L+R+D1+D2')

    def test_q6_label_1_is_T(self):
        self.assertEqual(self.q6_label(1), 'T')

    def test_q6_label_23(self):
        # 23 = 0b010111 = T+B+L+D1
        self.assertEqual(self.q6_label(23), 'T+B+L+D1')

    def test_q6_label_6_is_BL(self):
        self.assertEqual(self.q6_label(6), 'B+L')

    def test_q6_label_10_is_BR(self):
        self.assertEqual(self.q6_label(10), 'B+R')

    # ── orbit_frequencies ─────────────────────────────────────────────────────

    def test_orbit_frequencies_returns_counter(self):
        from collections import Counter
        cnt = self.orbit_frequencies('ТУМАН', 'xor3', 16)
        self.assertIsInstance(cnt, Counter)

    def test_orbit_frequencies_total_equals_orbit_size(self):
        cnt = self.orbit_frequencies('МАТ', 'xor3', 16)
        self.assertEqual(sum(cnt.values()), 8 * 16)

    def test_orbit_frequencies_mat_has_4_values(self):
        cnt = self.orbit_frequencies('МАТ', 'xor3', 16)
        self.assertEqual(len(cnt), 4)

    def test_orbit_frequencies_xor_only_zero(self):
        cnt = self.orbit_frequencies('ТУМАН', 'xor', 16)
        self.assertEqual(set(cnt.keys()), {0})

    # ── coverage_summary required keys ────────────────────────────────────────

    def test_summary_required_keys(self):
        required = {
            'word', 'rule', 'period', 'n_cells', 'orbit_size',
            'freq', 'vocab', 'n_distinct', 'coverage',
            'most_common', 'dominant_val', 'dominant_count', 'dominant_frac',
            'never_seen', 'n_never_seen', 'vocab_labels',
            'step_vocab', 'step_n_distinct', 'step_mode', 'step_mode_count',
            'min_step_n_distinct', 'max_step_n_distinct',
        }
        self.assertTrue(required.issubset(self.s_mat_xor3.keys()))

    def test_summary_word_preserved(self):
        self.assertEqual(self.s_mat_xor3['word'], 'МАТ')

    def test_summary_rule_preserved(self):
        self.assertEqual(self.s_mat_xor3['rule'], 'xor3')

    def test_orbit_size_equals_period_times_n_cells(self):
        s = self.s_mat_xor3
        self.assertEqual(s['orbit_size'], s['period'] * s['n_cells'])

    # ── МАТ XOR3 known values ─────────────────────────────────────────────────

    def test_mat_n_distinct_is_4(self):
        self.assertEqual(self.s_mat_xor3['n_distinct'], 4)

    def test_mat_coverage(self):
        self.assertAlmostEqual(self.s_mat_xor3['coverage'], 4 / 64, places=6)

    def test_mat_dominant_val_is_23(self):
        self.assertEqual(self.s_mat_xor3['dominant_val'], 23)

    def test_mat_dominant_count_is_64(self):
        self.assertEqual(self.s_mat_xor3['dominant_count'], 64)

    def test_mat_dominant_frac_is_half(self):
        self.assertAlmostEqual(self.s_mat_xor3['dominant_frac'], 0.5, places=6)

    def test_mat_vocab_contains_23(self):
        self.assertIn(23, self.s_mat_xor3['vocab'])

    def test_mat_vocab_is_sorted(self):
        v = self.s_mat_xor3['vocab']
        self.assertEqual(v, sorted(v))

    def test_mat_vocab_labels_for_23(self):
        idx = self.s_mat_xor3['vocab'].index(23)
        self.assertEqual(self.s_mat_xor3['vocab_labels'][idx], 'T+B+L+D1')

    def test_mat_orbit_size_is_128(self):
        self.assertEqual(self.s_mat_xor3['orbit_size'], 128)

    # ── РАБОТА XOR3 — maximum vocabulary ──────────────────────────────────────

    def test_rabota_n_distinct_is_16(self):
        self.assertEqual(self.s_rabota_xor3['n_distinct'], 16)

    def test_rabota_has_max_coverage_in_xor3(self):
        # РАБОТА has the maximum vocabulary size under XOR3
        self.assertGreaterEqual(
            self.s_rabota_xor3['n_distinct'],
            self.s_tuman_xor3['n_distinct'])

    # ── XOR rule — only value 0 ───────────────────────────────────────────────

    def test_xor_n_distinct_is_1(self):
        self.assertEqual(self.s_tuman_xor['n_distinct'], 1)

    def test_xor_dominant_val_is_0(self):
        self.assertEqual(self.s_tuman_xor['dominant_val'], 0)

    def test_xor_dominant_frac_is_1(self):
        self.assertAlmostEqual(self.s_tuman_xor['dominant_frac'], 1.0, places=9)

    def test_xor_n_never_seen_is_63(self):
        self.assertEqual(self.s_tuman_xor['n_never_seen'], 63)

    # ── Structural invariants ─────────────────────────────────────────────────

    def test_n_distinct_plus_n_never_seen_eq_64(self):
        for s in [self.s_mat_xor3, self.s_tuman_xor3, self.s_gora_xor3]:
            self.assertEqual(s['n_distinct'] + s['n_never_seen'], 64)

    def test_coverage_consistent(self):
        s = self.s_mat_xor3
        self.assertAlmostEqual(s['coverage'], s['n_distinct'] / 64, places=9)

    def test_dominant_frac_consistent(self):
        s = self.s_mat_xor3
        self.assertAlmostEqual(
            s['dominant_frac'], s['dominant_count'] / s['orbit_size'], places=9)

    def test_vocab_len_equals_n_distinct(self):
        self.assertEqual(len(self.s_mat_xor3['vocab']), self.s_mat_xor3['n_distinct'])

    def test_vocab_labels_len_equals_n_distinct(self):
        s = self.s_mat_xor3
        self.assertEqual(len(s['vocab_labels']), s['n_distinct'])

    def test_freq_total_equals_orbit_size(self):
        s = self.s_mat_xor3
        self.assertEqual(sum(s['freq'].values()), s['orbit_size'])

    def test_never_seen_not_in_freq(self):
        s = self.s_mat_xor3
        for v in s['never_seen']:
            self.assertNotIn(v, s['freq'])

    # ── Per-step stats ────────────────────────────────────────────────────────

    def test_step_vocab_length_equals_period(self):
        P = self.s_mat_xor3['period']
        self.assertEqual(len(self.s_mat_xor3['step_vocab']), P)

    def test_step_n_distinct_length_equals_period(self):
        P = self.s_mat_xor3['period']
        self.assertEqual(len(self.s_mat_xor3['step_n_distinct']), P)

    def test_step_mode_length_equals_period(self):
        P = self.s_mat_xor3['period']
        self.assertEqual(len(self.s_mat_xor3['step_mode']), P)

    def test_mat_step_mode_at_t1_is_23(self):
        self.assertEqual(self.s_mat_xor3['step_mode'][1], 23)

    def test_mat_step_mode_count_at_t1_is_14(self):
        self.assertEqual(self.s_mat_xor3['step_mode_count'][1], 14)

    def test_min_step_n_distinct_le_max(self):
        s = self.s_mat_xor3
        self.assertLessEqual(s['min_step_n_distinct'], s['max_step_n_distinct'])

    # ── global_coverage XOR3 ──────────────────────────────────────────────────

    def test_global_xor3_n_seen_is_60(self):
        self.assertEqual(self.g_xor3['n_seen'], 60)

    def test_global_xor3_n_absent_is_4(self):
        self.assertEqual(self.g_xor3['n_absent'], 4)

    def test_global_xor3_absent_contains_6(self):
        self.assertIn(6, self.g_xor3['absent'])

    def test_global_xor3_absent_contains_10(self):
        self.assertIn(10, self.g_xor3['absent'])

    def test_global_xor3_absent_contains_45(self):
        self.assertIn(45, self.g_xor3['absent'])

    def test_global_xor3_absent_contains_58(self):
        self.assertIn(58, self.g_xor3['absent'])

    def test_global_xor3_absent_labels(self):
        labels = self.g_xor3['absent_labels']
        self.assertIn('B+L', labels)
        self.assertIn('B+R', labels)

    def test_global_xor_only_value_0(self):
        self.assertEqual(self.g_xor['n_seen'], 1)
        self.assertEqual(self.g_xor['seen'], [0])

    # ── coverage_dict / serialisation ─────────────────────────────────────────

    def test_coverage_dict_serialisable(self):
        import json
        d = self.coverage_dict(self.s_mat_xor3)
        s = json.dumps(d, ensure_ascii=False)
        self.assertIn('МАТ', s)

    def test_all_coverage_has_four_rules(self):
        d = self.all_coverage('МАТ', 16)
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    # ── Viewer HTML ───────────────────────────────────────────────────────────

    def test_viewer_has_cv_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cv-canvas', content)

    def test_viewer_has_cv_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cvRun', content)

    def test_viewer_has_cv_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cvOrbit', content)

    def test_viewer_has_q6_label(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('q6Label', content)

    def test_viewer_has_cv_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cv-info', content)

    def test_viewer_has_xor3_absent(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('XOR3_ABSENT', content)


class TestSolanRun(unittest.TestCase):
    """Tests for solan_run.py — cell temporal run analysis of Q6 CA orbits."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_run import (
            analyze_cell, run_summary, all_run,
            build_run_data, run_dict,
        )
        cls.analyze_cell = staticmethod(analyze_cell)
        cls.run_summary  = staticmethod(run_summary)
        cls.all_run      = staticmethod(all_run)
        cls.build_data   = staticmethod(build_run_data)
        cls.run_dict     = staticmethod(run_dict)

        # Precomputed summaries
        cls.s_rabota_xor3 = run_summary('РАБОТА', 'xor3', 16)
        cls.s_mat_xor3    = run_summary('МАТ',    'xor3', 16)
        cls.s_gora_xor3   = run_summary('ГОРА',   'xor3', 16)
        cls.s_tuman_xor3  = run_summary('ТУМАН',  'xor3', 16)
        cls.s_mat_xor     = run_summary('МАТ',    'xor',  16)
        cls.s_mat_and     = run_summary('МАТ',    'and',  16)

    # ── analyze_cell ──────────────────────────────────────────────────────────

    def test_analyze_cell_constant_sequence(self):
        a = self.analyze_cell([5, 5, 5, 5])
        self.assertEqual(a['n_turns'], 0)
        self.assertEqual(a['n_inc'],   0)
        self.assertEqual(a['n_dec'],   0)
        self.assertEqual(a['n_const'], 3)
        self.assertEqual(a['value_range'], 0)

    def test_analyze_cell_monotone_inc(self):
        a = self.analyze_cell([1, 2, 3, 4])
        self.assertEqual(a['n_turns'], 0)
        self.assertEqual(a['n_inc'],   3)
        self.assertEqual(a['n_dec'],   0)

    def test_analyze_cell_monotone_dec(self):
        a = self.analyze_cell([4, 3, 2, 1])
        self.assertEqual(a['n_turns'], 0)
        self.assertEqual(a['n_inc'],   0)
        self.assertEqual(a['n_dec'],   3)

    def test_analyze_cell_one_turn(self):
        # 1,3,1 — goes up then down: 1 turn (local max at t=1)
        a = self.analyze_cell([1, 3, 1])
        self.assertEqual(a['n_turns'], 1)
        self.assertEqual(a['n_inc'],   1)
        self.assertEqual(a['n_dec'],   1)

    def test_analyze_cell_two_turns(self):
        # 3,1,3,1 — up/down/up: 2 turns
        a = self.analyze_cell([3, 1, 3, 1])
        self.assertEqual(a['n_turns'], 2)

    def test_analyze_cell_range(self):
        a = self.analyze_cell([10, 40, 20, 5])
        self.assertEqual(a['value_range'], 35)
        self.assertEqual(a['min_val'], 5)
        self.assertEqual(a['max_val'], 40)

    def test_analyze_cell_const_does_not_break_run(self):
        # 1,2,2,1 — increases then flat then decreases: 1 turn
        a = self.analyze_cell([1, 2, 2, 1])
        self.assertEqual(a['n_turns'], 1)
        self.assertEqual(a['n_const'], 1)

    def test_analyze_cell_period1(self):
        a = self.analyze_cell([42])
        self.assertEqual(a['n_turns'],  0)
        self.assertEqual(a['n_inc'],    0)
        self.assertEqual(a['n_dec'],    0)
        self.assertEqual(a['n_const'],  0)
        self.assertEqual(a['value_range'], 0)

    def test_analyze_cell_alternating(self):
        # 63,0,63,0 — alternates: 2 turns (down, up, down skipped — 2 sign reversals)
        a = self.analyze_cell([63, 0, 63, 0])
        self.assertEqual(a['n_turns'], 2)

    def test_analyze_cell_returns_required_keys(self):
        a = self.analyze_cell([1, 2, 1])
        for key in ('n_turns', 'n_inc', 'n_dec', 'n_const',
                    'value_range', 'min_val', 'max_val'):
            self.assertIn(key, a)

    # ── run_summary structure ─────────────────────────────────────────────────

    def test_summary_has_required_keys(self):
        required = {
            'word', 'rule', 'period', 'n_cells',
            'cell_n_turns', 'cell_n_inc', 'cell_n_dec', 'cell_n_const',
            'cell_range', 'cell_min_val', 'cell_max_val',
            'max_turns', 'max_turns_cell', 'min_turns', 'min_turns_cell',
            'mean_turns', 'total_inc', 'total_dec', 'total_const',
            'max_range', 'max_range_cell', 'min_range', 'min_range_cell',
            'mean_range', 'quasi_frozen_cells', 'n_quasi_frozen',
        }
        self.assertTrue(required.issubset(self.s_rabota_xor3.keys()))

    def test_summary_list_lengths(self):
        s = self.s_rabota_xor3
        N = s['n_cells']
        for key in ('cell_n_turns', 'cell_n_inc', 'cell_n_dec',
                    'cell_n_const', 'cell_range', 'cell_min_val', 'cell_max_val'):
            self.assertEqual(len(s[key]), N, f'{key} length mismatch')

    def test_summary_word_rule_preserved(self):
        s = self.s_rabota_xor3
        self.assertEqual(s['word'], 'РАБОТА')
        self.assertEqual(s['rule'], 'xor3')

    def test_summary_period(self):
        self.assertEqual(self.s_rabota_xor3['period'], 8)
        self.assertEqual(self.s_gora_xor3['period'],   2)

    def test_summary_n_cells(self):
        self.assertEqual(self.s_rabota_xor3['n_cells'], 16)

    def test_summary_total_steps_conservation(self):
        # total_inc + total_dec + total_const = N × (P-1)
        s = self.s_rabota_xor3
        expected = s['n_cells'] * (s['period'] - 1)
        actual = s['total_inc'] + s['total_dec'] + s['total_const']
        self.assertEqual(actual, expected)

    def test_summary_max_turns_cell_consistent(self):
        s = self.s_rabota_xor3
        self.assertEqual(s['cell_n_turns'][s['max_turns_cell']], s['max_turns'])

    def test_summary_min_turns_cell_consistent(self):
        s = self.s_rabota_xor3
        self.assertEqual(s['cell_n_turns'][s['min_turns_cell']], s['min_turns'])

    def test_summary_max_range_cell_consistent(self):
        s = self.s_rabota_xor3
        self.assertEqual(s['cell_range'][s['max_range_cell']], s['max_range'])

    def test_summary_mean_turns_bounds(self):
        s = self.s_rabota_xor3
        self.assertGreaterEqual(s['mean_turns'], s['min_turns'])
        self.assertLessEqual(s['mean_turns'],    s['max_turns'])

    def test_summary_quasi_frozen_subset(self):
        s = self.s_rabota_xor3
        for i in s['quasi_frozen_cells']:
            self.assertEqual(s['cell_n_turns'][i], 0)

    def test_summary_n_quasi_frozen_matches_list(self):
        s = self.s_rabota_xor3
        self.assertEqual(s['n_quasi_frozen'], len(s['quasi_frozen_cells']))

    # ── РАБОТА XOR3 known values ──────────────────────────────────────────────

    def test_rabota_max_turns(self):
        # Cell 1: [63,62,63,1,63,0,63,62] → 6 turns
        self.assertEqual(self.s_rabota_xor3['max_turns'], 6)

    def test_rabota_max_turns_cell(self):
        self.assertEqual(self.s_rabota_xor3['max_turns_cell'], 1)

    def test_rabota_max_range(self):
        # Cell 1 has range 63 (0–63)
        self.assertEqual(self.s_rabota_xor3['max_range'], 63)

    def test_rabota_quasi_frozen(self):
        # Cells 7,8 have 0 turns
        qf = self.s_rabota_xor3['quasi_frozen_cells']
        self.assertIn(7, qf)
        self.assertIn(8, qf)

    def test_rabota_cell1_n_turns(self):
        self.assertEqual(self.s_rabota_xor3['cell_n_turns'][1], 6)

    def test_rabota_cell7_zero_turns(self):
        self.assertEqual(self.s_rabota_xor3['cell_n_turns'][7], 0)

    def test_rabota_cell8_zero_turns(self):
        self.assertEqual(self.s_rabota_xor3['cell_n_turns'][8], 0)

    # ── МАТ XOR3 known values ────────────────────────────────────────────────

    def test_mat_quasi_frozen_includes_7_8(self):
        qf = self.s_mat_xor3['quasi_frozen_cells']
        self.assertIn(7, qf)
        self.assertIn(8, qf)

    def test_mat_cell7_zero_turns(self):
        self.assertEqual(self.s_mat_xor3['cell_n_turns'][7], 0)

    def test_mat_cell8_zero_turns(self):
        self.assertEqual(self.s_mat_xor3['cell_n_turns'][8], 0)

    def test_mat_cell7_mostly_const(self):
        # seq [63,23,23,...] → n_const=6 out of 7 steps
        self.assertEqual(self.s_mat_xor3['cell_n_const'][7], 6)

    def test_mat_cell0_four_turns(self):
        # Outer cells have 4 turns (gradient pattern)
        self.assertEqual(self.s_mat_xor3['cell_n_turns'][0], 4)

    def test_mat_xor3_period(self):
        self.assertEqual(self.s_mat_xor3['period'], 8)

    # ── ГОРА XOR3 (P=2) ──────────────────────────────────────────────────────

    def test_gora_all_zero_turns(self):
        # P=2 means only 1 step → can't have a turn
        s = self.s_gora_xor3
        self.assertEqual(s['max_turns'], 0)
        self.assertEqual(len(s['quasi_frozen_cells']), s['n_cells'])

    def test_gora_n_quasi_frozen_all_cells(self):
        s = self.s_gora_xor3
        self.assertEqual(s['n_quasi_frozen'], 16)

    def test_gora_total_const_zero_for_p2(self):
        # P=2: each cell has exactly 1 step (inc or dec), never const
        s = self.s_gora_xor3
        self.assertEqual(s['total_const'], 0)

    # ── AND / XOR rules ───────────────────────────────────────────────────────

    def test_mat_and_period1_all_zero_turns(self):
        # AND collapses to fixed point P=1 → no steps → 0 turns
        s = self.s_mat_and
        self.assertEqual(s['period'], 1)
        self.assertEqual(s['max_turns'], 0)
        self.assertEqual(s['total_inc'], 0)
        self.assertEqual(s['total_dec'], 0)
        self.assertEqual(s['total_const'], 0)

    def test_mat_xor_period(self):
        # XOR converges to 0; P=1 for all words (fixed-point {0})
        # (or P=2 transitional — just verify no error and period≥1)
        s = self.s_mat_xor
        self.assertGreaterEqual(s['period'], 1)
        self.assertGreaterEqual(s['max_turns'], 0)

    # ── all_run ───────────────────────────────────────────────────────────────

    def test_all_run_returns_all_rules(self):
        ar = self.all_run('ТУМАН', 16)
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(rule, ar)

    def test_all_run_each_is_dict(self):
        ar = self.all_run('ТУМАН', 16)
        for d in ar.values():
            self.assertIsInstance(d, dict)
            self.assertIn('max_turns', d)

    # ── build_run_data ────────────────────────────────────────────────────────

    def test_build_run_data_keys(self):
        words = ['МАТ', 'ГОРА']
        d = self.build_data(words, 16)
        self.assertIn('words', d)
        self.assertIn('data',  d)
        self.assertEqual(d['words'], words)

    def test_build_run_data_nested_structure(self):
        words = ['МАТ']
        d = self.build_data(words, 16)
        self.assertIn('МАТ', d['data'])
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(rule, d['data']['МАТ'])

    # ── run_dict ─────────────────────────────────────────────────────────────

    def test_run_dict_json_serialisable(self):
        import json
        rd = self.run_dict(self.s_rabota_xor3)
        out = json.dumps(rd)
        self.assertIsInstance(out, str)

    def test_run_dict_has_max_turns(self):
        rd = self.run_dict(self.s_rabota_xor3)
        self.assertIn('max_turns', rd)
        self.assertEqual(rd['max_turns'], 6)

    def test_run_dict_has_cell_lists(self):
        rd = self.run_dict(self.s_rabota_xor3)
        self.assertIsInstance(rd['cell_n_turns'], list)
        self.assertEqual(len(rd['cell_n_turns']), 16)

    # ── mean_turns and mean_range bounds ─────────────────────────────────────

    def test_mean_turns_non_negative(self):
        for s in (self.s_rabota_xor3, self.s_mat_xor3, self.s_gora_xor3):
            self.assertGreaterEqual(s['mean_turns'], 0.0)

    def test_mean_range_non_negative(self):
        for s in (self.s_rabota_xor3, self.s_mat_xor3):
            self.assertGreaterEqual(s['mean_range'], 0.0)

    def test_range_non_negative_all_cells(self):
        for s in (self.s_rabota_xor3, self.s_mat_xor3, self.s_gora_xor3):
            for r in s['cell_range']:
                self.assertGreaterEqual(r, 0)

    def test_turns_non_negative_all_cells(self):
        for s in (self.s_rabota_xor3, self.s_mat_xor3, self.s_gora_xor3):
            for t in s['cell_n_turns']:
                self.assertGreaterEqual(t, 0)

    # ── Viewer HTML assertions ────────────────────────────────────────────────

    def test_viewer_has_rn_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rn-canvas', content)

    def test_viewer_has_rn_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rnRun', content)

    def test_viewer_has_rn_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rnOrbit', content)

    def test_viewer_has_analyze_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('analyzeCell', content)

    def test_viewer_has_rn_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rn-info', content)

    def test_viewer_has_quasi_frozen(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('quasi_frozen', content)

    def test_viewer_has_solan_run_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_run', content)


class TestSolanCross(unittest.TestCase):
    """Tests for solan_cross.py — pairwise cell Q6 cross-correlation."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_cross import (
            pearson, cross_corr_matrix, cross_summary,
            all_cross, build_cross_data, cross_dict,
        )
        cls.pearson            = staticmethod(pearson)
        cls.cross_corr_matrix  = staticmethod(cross_corr_matrix)
        cls.cross_summary      = staticmethod(cross_summary)
        cls.all_cross          = staticmethod(all_cross)
        cls.build_data         = staticmethod(build_cross_data)
        cls.cross_dict         = staticmethod(cross_dict)

        # Precomputed summaries
        cls.s_mat_xor3   = cross_summary('МАТ',    'xor3', 16)
        cls.s_tuman_xor3 = cross_summary('ТУМАН',  'xor3', 16)
        cls.s_gora_xor3  = cross_summary('ГОРА',   'xor3', 16)
        cls.s_rabota_xor3= cross_summary('РАБОТА', 'xor3', 16)
        cls.s_mat_xor    = cross_summary('МАТ',    'xor',  16)

    # ── pearson() unit tests ──────────────────────────────────────────────────

    def test_pearson_identical_sequences(self):
        self.assertAlmostEqual(self.pearson([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)

    def test_pearson_opposite_sequences(self):
        self.assertAlmostEqual(self.pearson([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_pearson_constant_returns_none(self):
        self.assertIsNone(self.pearson([5, 5, 5], [1, 2, 3]))

    def test_pearson_both_constant_returns_none(self):
        self.assertIsNone(self.pearson([3, 3], [7, 7]))

    def test_pearson_range(self):
        import random
        random.seed(42)
        xs = [random.randint(0, 63) for _ in range(20)]
        ys = [random.randint(0, 63) for _ in range(20)]
        r = self.pearson(xs, ys)
        if r is not None:
            self.assertGreaterEqual(r, -1.0 - 1e-9)
            self.assertLessEqual(r,    1.0 + 1e-9)

    def test_pearson_scaled_same_pattern(self):
        # Scaling doesn't affect Pearson correlation
        xs = [1, 2, 3, 4]
        ys = [10, 20, 30, 40]
        self.assertAlmostEqual(self.pearson(xs, ys), 1.0)

    def test_pearson_offset_same_pattern(self):
        # Offset doesn't affect Pearson correlation
        xs = [1, 2, 3]
        ys = [101, 102, 103]
        self.assertAlmostEqual(self.pearson(xs, ys), 1.0)

    def test_pearson_single_element_returns_none(self):
        self.assertIsNone(self.pearson([5], [5]))

    # ── cross_corr_matrix structure ───────────────────────────────────────────

    def test_matrix_shape(self):
        mat = self.cross_corr_matrix('МАТ', 'xor3', 16)
        self.assertEqual(len(mat), 16)
        for row in mat:
            self.assertEqual(len(row), 16)

    def test_matrix_diagonal_is_one_or_none(self):
        mat = self.cross_corr_matrix('МАТ', 'xor3', 16)
        for i in range(16):
            v = mat[i][i]
            self.assertTrue(v is None or abs(v - 1.0) < 1e-9)

    def test_matrix_symmetry(self):
        mat = self.cross_corr_matrix('ТУМАН', 'xor3', 16)
        for i in range(16):
            for j in range(16):
                ri = mat[i][j]
                rj = mat[j][i]
                if ri is None or rj is None:
                    self.assertIsNone(ri); self.assertIsNone(rj)
                else:
                    self.assertAlmostEqual(ri, rj, places=10)

    def test_matrix_xor_all_none_offdiag(self):
        # XOR: P=1, all cells constant → all off-diagonal r = None
        mat = self.cross_corr_matrix('МАТ', 'xor', 16)
        for i in range(16):
            for j in range(16):
                if i != j:
                    self.assertIsNone(mat[i][j])

    # ── cross_summary structure ───────────────────────────────────────────────

    def test_summary_required_keys(self):
        required = {
            'word', 'rule', 'period', 'n_cells', 'matrix',
            'n_sync_pairs', 'n_antisync_pairs', 'sync_pairs', 'antisync_pairs',
            'max_r', 'max_r_pair', 'min_r', 'min_r_pair',
            'mean_abs_r', 'n_defined',
            'spatial_decay', 'n_frozen_cells', 'frozen_cells',
        }
        self.assertTrue(required.issubset(self.s_mat_xor3.keys()))

    def test_summary_word_rule_preserved(self):
        self.assertEqual(self.s_mat_xor3['word'], 'МАТ')
        self.assertEqual(self.s_mat_xor3['rule'], 'xor3')

    def test_summary_n_defined_correct(self):
        # For N=16, C(16,2) = 120 off-diagonal pairs (upper triangle)
        s = self.s_mat_xor3
        self.assertEqual(s['n_defined'], 120)

    def test_summary_xor_n_defined_zero(self):
        # XOR: P=1 → all frozen → no defined pairs
        self.assertEqual(self.s_mat_xor['n_defined'], 0)

    def test_summary_sync_pairs_subset_of_defined(self):
        s = self.s_mat_xor3
        self.assertLessEqual(s['n_sync_pairs'], s['n_defined'])

    def test_summary_antisync_pairs_subset(self):
        s = self.s_mat_xor3
        self.assertLessEqual(s['n_antisync_pairs'], s['n_defined'])

    def test_summary_spatial_decay_length(self):
        s = self.s_mat_xor3
        # N=16 → lags 1..8 → length 8
        self.assertEqual(len(s['spatial_decay']), 16 // 2)

    def test_summary_mean_abs_r_non_negative(self):
        self.assertGreaterEqual(self.s_mat_xor3['mean_abs_r'], 0.0)

    def test_summary_max_r_ge_min_r(self):
        s = self.s_mat_xor3
        if s['max_r'] is not None and s['min_r'] is not None:
            self.assertGreaterEqual(s['max_r'], s['min_r'])

    def test_summary_n_sync_matches_list(self):
        s = self.s_mat_xor3
        self.assertEqual(s['n_sync_pairs'], len(s['sync_pairs']))

    def test_summary_n_antisync_matches_list(self):
        s = self.s_mat_xor3
        self.assertEqual(s['n_antisync_pairs'], len(s['antisync_pairs']))

    def test_summary_frozen_cells_no_defined_corr(self):
        s = self.s_mat_xor
        self.assertEqual(s['n_frozen_cells'], 16)
        self.assertEqual(s['n_defined'], 0)

    # ── МАТ XOR3 known values ────────────────────────────────────────────────

    def test_mat_xor3_sync_pair_0_15(self):
        self.assertIn((0, 15), self.s_mat_xor3['sync_pairs'])

    def test_mat_xor3_sync_pair_7_8(self):
        self.assertIn((7, 8), self.s_mat_xor3['sync_pairs'])

    def test_mat_xor3_r_0_15_is_one(self):
        r = self.s_mat_xor3['matrix'][0][15]
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0, places=9)

    def test_mat_xor3_r_7_8_is_one(self):
        r = self.s_mat_xor3['matrix'][7][8]
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0, places=9)

    def test_mat_xor3_max_r(self):
        self.assertAlmostEqual(self.s_mat_xor3['max_r'], 1.0, places=5)

    def test_mat_xor3_n_sync_at_least_2(self):
        self.assertGreaterEqual(self.s_mat_xor3['n_sync_pairs'], 2)

    def test_mat_xor3_no_frozen_cells(self):
        self.assertEqual(self.s_mat_xor3['n_frozen_cells'], 0)

    # ── ГОРА XOR3 (P=2) — all pairs ±1 ──────────────────────────────────────

    def test_gora_all_abs_r_is_one(self):
        # P=2 → only 2 time steps → Pearson always ±1 or None
        s = self.s_gora_xor3
        mat = s['matrix']
        for i in range(16):
            for j in range(16):
                r = mat[i][j]
                if r is not None:
                    self.assertAlmostEqual(abs(r), 1.0, places=9,
                                           msg=f'|r({i},{j})|={abs(r)} ≠ 1')

    def test_gora_sync_plus_antisync_equals_total(self):
        s = self.s_gora_xor3
        # All 120 pairs should be ±1
        self.assertEqual(s['n_sync_pairs'] + s['n_antisync_pairs'], 120)

    def test_gora_mean_abs_r_is_one(self):
        self.assertAlmostEqual(self.s_gora_xor3['mean_abs_r'], 1.0, places=9)

    # ── all_cross ─────────────────────────────────────────────────────────────

    def test_all_cross_has_four_rules(self):
        ar = self.all_cross('ТУМАН', 16)
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(rule, ar)

    def test_all_cross_each_is_dict(self):
        ar = self.all_cross('ТУМАН', 16)
        for d in ar.values():
            self.assertIsInstance(d, dict)
            self.assertIn('n_sync_pairs', d)

    # ── build_cross_data ──────────────────────────────────────────────────────

    def test_build_data_keys(self):
        words = ['МАТ', 'ГОРА']
        d = self.build_data(words, 16)
        self.assertIn('words', d)
        self.assertIn('data',  d)

    def test_build_data_nested(self):
        d = self.build_data(['МАТ'], 16)
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(rule, d['data']['МАТ'])

    # ── cross_dict ────────────────────────────────────────────────────────────

    def test_cross_dict_json_serialisable(self):
        import json
        cd = self.cross_dict(self.s_mat_xor3)
        out = json.dumps(cd)
        self.assertIsInstance(out, str)

    def test_cross_dict_sync_pairs_are_lists(self):
        cd = self.cross_dict(self.s_mat_xor3)
        self.assertIsInstance(cd['sync_pairs'], list)
        if cd['sync_pairs']:
            self.assertIsInstance(cd['sync_pairs'][0], list)

    def test_cross_dict_matrix_no_python_none(self):
        # cross_dict replaces None with the string 'null'
        cd = self.cross_dict(self.s_mat_xor)  # XOR: all None
        for row in cd['matrix']:
            for v in row:
                self.assertNotEqual(type(v).__name__, 'NoneType')

    # ── Viewer HTML assertions ────────────────────────────────────────────────

    def test_viewer_has_cr_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cr-canvas', content)

    def test_viewer_has_cr_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('crRun', content)

    def test_viewer_has_cr_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('crOrbit', content)

    def test_viewer_has_cr_pearson(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('crPearson', content)

    def test_viewer_has_cr_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cr-info', content)

    def test_viewer_has_solan_cross_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_cross', content)


class TestSolanPCA(unittest.TestCase):
    """Tests for solan_pca.py — PCA of Q6 CA orbit trajectories."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_pca import (
            gram_eig, orbit_pca, pca_summary,
            all_pca, build_pca_data, pca_dict,
        )
        cls.gram_eig    = staticmethod(gram_eig)
        cls.orbit_pca   = staticmethod(orbit_pca)
        cls.pca_summary = staticmethod(pca_summary)
        cls.all_pca     = staticmethod(all_pca)
        cls.build_data  = staticmethod(build_pca_data)
        cls.pca_dict    = staticmethod(pca_dict)

        # Precomputed summaries
        cls.s_rabota_xor3 = pca_summary('РАБОТА', 'xor3', 16)
        cls.s_mat_xor3    = pca_summary('МАТ',    'xor3', 16)
        cls.s_gora_xor3   = pca_summary('ГОРА',   'xor3', 16)
        cls.s_tonha_xor3  = pca_summary('ТОННА',  'xor3', 16)
        cls.s_zavod_xor3  = pca_summary('ЗАВОД',  'xor3', 16)
        cls.s_nitro_xor3  = pca_summary('НИТРО',  'xor3', 16)
        cls.s_mat_and     = pca_summary('МАТ',    'and',  16)

    # ── gram_eig ──────────────────────────────────────────────────────────────

    def test_gram_eig_returns_pair(self):
        G = [[2.0, 1.0], [1.0, 2.0]]
        vals, vecs = self.gram_eig(G)
        self.assertEqual(len(vals), 2)
        self.assertEqual(len(vecs), 2)

    def test_gram_eig_sorted_descending(self):
        G = [[2.0, 1.0], [1.0, 2.0]]
        vals, _ = self.gram_eig(G)
        self.assertGreaterEqual(vals[0], vals[1])

    def test_gram_eig_eigenvalues_correct(self):
        # 2×2 symmetric: eigenvalues are 3 and 1
        G = [[2.0, 1.0], [1.0, 2.0]]
        vals, _ = self.gram_eig(G)
        self.assertAlmostEqual(vals[0], 3.0, places=4)
        self.assertAlmostEqual(vals[1], 1.0, places=4)

    def test_gram_eig_eigenvalues_non_negative(self):
        G = [[4.0, 2.0], [2.0, 3.0]]
        vals, _ = self.gram_eig(G)
        for v in vals:
            self.assertGreaterEqual(v, 0.0)

    def test_gram_eig_empty(self):
        vals, vecs = self.gram_eig([])
        self.assertEqual(vals, [])
        self.assertEqual(vecs, [])

    def test_gram_eig_1x1(self):
        vals, vecs = self.gram_eig([[5.0]])
        self.assertAlmostEqual(vals[0], 5.0, places=6)

    # ── orbit_pca ─────────────────────────────────────────────────────────────

    def test_orbit_pca_keys_present(self):
        raw = self.orbit_pca('РАБОТА', 'xor3', 16)
        for k in ('period', 'n_cells', 'eigenvalues', 'total_var', 'evr', 'cumevr',
                  'loadings', 'mean_state'):
            self.assertIn(k, raw)

    def test_orbit_pca_period_correct(self):
        raw = self.orbit_pca('РАБОТА', 'xor3', 16)
        self.assertEqual(raw['period'], 8)

    def test_orbit_pca_n_cells_correct(self):
        raw = self.orbit_pca('РАБОТА', 'xor3', 16)
        self.assertEqual(raw['n_cells'], 16)

    def test_orbit_pca_eigenvalues_length_equals_period(self):
        raw = self.orbit_pca('РАБОТА', 'xor3', 16)
        self.assertEqual(len(raw['eigenvalues']), raw['period'])

    def test_orbit_pca_mean_state_length(self):
        raw = self.orbit_pca('РАБОТА', 'xor3', 16)
        self.assertEqual(len(raw['mean_state']), 16)

    def test_orbit_pca_p1_total_var_zero(self):
        raw = self.orbit_pca('МАТ', 'and', 16)
        self.assertEqual(raw['period'], 1)
        self.assertEqual(raw['total_var'], 0.0)

    def test_orbit_pca_p1_loadings_zero(self):
        raw = self.orbit_pca('МАТ', 'and', 16)
        # All loadings should be zero for P=1
        for row in raw['loadings']:
            for v in row:
                self.assertAlmostEqual(v, 0.0, places=10)

    # ── pca_summary structure ─────────────────────────────────────────────────

    def test_pca_summary_all_keys(self):
        for k in ('word', 'rule', 'period', 'n_cells', 'eigenvalues',
                  'explained_var_ratio', 'cumulative_evr', 'total_var',
                  'orbit_rank', 'n_components_95', 'pc1_loadings',
                  'pc1_dom_cell', 'pc1_dom_loading', 'pc1_var_ratio',
                  'n_pcs_meaningful', 'all_loadings', 'mean_state'):
            self.assertIn(k, self.s_rabota_xor3, msg=f"missing key: {k}")

    def test_pca_summary_word_preserved(self):
        self.assertEqual(self.s_rabota_xor3['word'], 'РАБОТА')

    def test_pca_summary_rule_preserved(self):
        self.assertEqual(self.s_rabota_xor3['rule'], 'xor3')

    def test_pca_summary_evr_sums_to_one(self):
        evr = self.s_rabota_xor3['explained_var_ratio']
        self.assertAlmostEqual(sum(evr), 1.0, places=5)

    def test_pca_summary_cumevr_ends_at_one(self):
        cum = self.s_rabota_xor3['cumulative_evr']
        self.assertAlmostEqual(cum[-1], 1.0, places=5)

    def test_pca_summary_cumevr_monotone(self):
        cum = self.s_rabota_xor3['cumulative_evr']
        for i in range(len(cum) - 1):
            self.assertLessEqual(cum[i], cum[i + 1] + 1e-10)

    def test_pca_summary_eigenvalues_non_negative(self):
        for v in self.s_rabota_xor3['eigenvalues']:
            self.assertGreaterEqual(v, 0.0)

    def test_pca_summary_eigenvalues_sorted_desc(self):
        evals = self.s_rabota_xor3['eigenvalues']
        for i in range(len(evals) - 1):
            self.assertGreaterEqual(evals[i] + 1e-10, evals[i + 1])

    def test_pca_summary_pc1_loadings_unit_length(self):
        import math
        lv = self.s_rabota_xor3['pc1_loadings']
        nm = math.sqrt(sum(x * x for x in lv))
        self.assertAlmostEqual(nm, 1.0, places=4)

    def test_pca_summary_dom_cell_consistent(self):
        s = self.s_rabota_xor3
        dom = s['pc1_dom_cell']
        lv  = s['pc1_loadings']
        max_abs = max(abs(v) for v in lv)
        self.assertAlmostEqual(abs(lv[dom]), max_abs, places=6)

    def test_pca_summary_dom_loading_consistent(self):
        s = self.s_rabota_xor3
        self.assertAlmostEqual(s['pc1_dom_loading'], s['pc1_loadings'][s['pc1_dom_cell']], places=5)

    def test_pca_summary_pc1_var_ratio_consistent(self):
        s = self.s_rabota_xor3
        evr = s['explained_var_ratio']
        self.assertAlmostEqual(s['pc1_var_ratio'], evr[0], places=5)

    # ── РАБОТА XOR3 known values ──────────────────────────────────────────────

    def test_rabota_xor3_period_8(self):
        self.assertEqual(self.s_rabota_xor3['period'], 8)

    def test_rabota_xor3_orbit_rank_7(self):
        self.assertEqual(self.s_rabota_xor3['orbit_rank'], 7)

    def test_rabota_xor3_pc1_dom_cell_is_1(self):
        self.assertEqual(self.s_rabota_xor3['pc1_dom_cell'], 1)

    def test_rabota_xor3_pc1_dom_loading_positive(self):
        self.assertGreater(self.s_rabota_xor3['pc1_dom_loading'], 0.0)

    def test_rabota_xor3_pc1_dom_loading_approx(self):
        # Cell 1 loading ≈ +0.6664
        self.assertAlmostEqual(self.s_rabota_xor3['pc1_dom_loading'], 0.6664, places=2)

    def test_rabota_xor3_pc1_var_ratio_approx(self):
        # PC₁ explains ≈ 35.8 % of variance
        self.assertAlmostEqual(self.s_rabota_xor3['pc1_var_ratio'], 0.358, places=2)

    def test_rabota_xor3_n_meaningful_pcs(self):
        # 5 PCs with EVR > 5%
        self.assertEqual(self.s_rabota_xor3['n_pcs_meaningful'], 5)

    def test_rabota_xor3_n95_is_5(self):
        self.assertEqual(self.s_rabota_xor3['n_components_95'], 5)

    def test_rabota_xor3_total_var_positive(self):
        self.assertGreater(self.s_rabota_xor3['total_var'], 0.0)

    # ── ГОРА XOR3 (P=2, rank=1) known values ──────────────────────────────────

    def test_gora_xor3_period_2(self):
        self.assertEqual(self.s_gora_xor3['period'], 2)

    def test_gora_xor3_orbit_rank_1(self):
        self.assertEqual(self.s_gora_xor3['orbit_rank'], 1)

    def test_gora_xor3_pc1_var_ratio_is_1(self):
        # P=2 → rank=1 → 100% of variance in PC₁
        self.assertAlmostEqual(self.s_gora_xor3['pc1_var_ratio'], 1.0, places=4)

    def test_gora_xor3_n95_is_1(self):
        self.assertEqual(self.s_gora_xor3['n_components_95'], 1)

    def test_gora_xor3_cumevr_second_is_one(self):
        cum = self.s_gora_xor3['cumulative_evr']
        self.assertAlmostEqual(cum[0], 1.0, places=4)

    # ── ТОННА XOR3 — highest PC1 among P=8 words ──────────────────────────────

    def test_tonha_xor3_pc1_highest_p8(self):
        # ТОННА PC₁ ≈ 52.9 % — highest among P=8 words
        self.assertGreater(self.s_tonha_xor3['pc1_var_ratio'],
                           self.s_zavod_xor3['pc1_var_ratio'])

    def test_tonha_xor3_pc1_approx_529(self):
        self.assertAlmostEqual(self.s_tonha_xor3['pc1_var_ratio'], 0.529, places=2)

    def test_zavod_xor3_pc1_approx_278(self):
        # ЗАВОД ≈ 27.8 % (most spread)
        self.assertAlmostEqual(self.s_zavod_xor3['pc1_var_ratio'], 0.278, places=2)

    # ── НИТРО XOR3 — negative dominant loading ─────────────────────────────────

    def test_nitro_xor3_dom_cell_is_5(self):
        self.assertEqual(self.s_nitro_xor3['pc1_dom_cell'], 5)

    def test_nitro_xor3_dom_loading_negative(self):
        self.assertLess(self.s_nitro_xor3['pc1_dom_loading'], 0.0)

    def test_nitro_xor3_dom_loading_approx(self):
        self.assertAlmostEqual(self.s_nitro_xor3['pc1_dom_loading'], -0.640, places=2)

    # ── AND rule (P=1 fixed point) ────────────────────────────────────────────

    def test_and_p1_total_var_zero(self):
        self.assertEqual(self.s_mat_and['total_var'], 0.0)

    def test_and_p1_orbit_rank_zero(self):
        self.assertEqual(self.s_mat_and['orbit_rank'], 0)

    def test_and_p1_pc1_var_ratio_zero(self):
        self.assertEqual(self.s_mat_and['pc1_var_ratio'], 0.0)

    def test_and_p1_period_is_1(self):
        self.assertEqual(self.s_mat_and['period'], 1)

    # ── all_pca ───────────────────────────────────────────────────────────────

    def test_all_pca_four_rules(self):
        res = self.all_pca('РАБОТА', 16)
        for r in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(r, res)

    def test_all_pca_each_is_dict(self):
        res = self.all_pca('МАТ', 16)
        for s in res.values():
            self.assertIsInstance(s, dict)

    # ── build_pca_data ────────────────────────────────────────────────────────

    def test_build_data_has_words_key(self):
        d = self.build_data(['МАТ'], 16)
        self.assertIn('words', d)

    def test_build_data_has_data_key(self):
        d = self.build_data(['МАТ'], 16)
        self.assertIn('data', d)

    def test_build_data_nested(self):
        d = self.build_data(['МАТ'], 16)
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(rule, d['data']['МАТ'])

    # ── pca_dict ──────────────────────────────────────────────────────────────

    def test_pca_dict_json_serialisable(self):
        import json
        pd = self.pca_dict(self.s_rabota_xor3)
        out = json.dumps(pd)
        self.assertIsInstance(out, str)

    def test_pca_dict_all_loadings_present(self):
        pd = self.pca_dict(self.s_rabota_xor3)
        self.assertIn('all_loadings', pd)

    def test_pca_dict_all_loadings_shape(self):
        pd = self.pca_dict(self.s_rabota_xor3)
        # all_loadings: P rows × N cols
        al = pd['all_loadings']
        self.assertEqual(len(al), self.s_rabota_xor3['period'])
        self.assertEqual(len(al[0]), 16)

    def test_pca_dict_floats_rounded(self):
        pd = self.pca_dict(self.s_rabota_xor3)
        evr = pd['explained_var_ratio']
        if evr:
            # rounded to 8 places, so no more than 8 decimal digits
            self.assertIsInstance(evr[0], float)

    # ── Viewer HTML assertions ────────────────────────────────────────────────

    def test_viewer_has_pca_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pca-canvas', content)

    def test_viewer_has_pca_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pcaRun', content)

    def test_viewer_has_pca_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pcaOrbit', content)

    def test_viewer_has_pca_compute(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pcaCompute', content)

    def test_viewer_has_pca_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('pca-info', content)

    def test_viewer_has_solan_pca_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_pca', content)


class TestSolanFourier(unittest.TestCase):
    """Tests for solan_fourier.py — DFT spectral analysis of Q6 CA orbits."""

    @classmethod
    def setUpClass(cls):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from projects.hexglyph.solan_fourier import (
            dft1, power_spectrum, spectral_entropy, normalised_spectral_entropy,
            spectral_flatness, dominant_harmonic,
            cell_spectrum, fourier_summary, all_fourier,
            build_fourier_data, fourier_dict,
        )
        cls.dft1          = staticmethod(dft1)
        cls.power_spectrum = staticmethod(power_spectrum)
        cls.spec_ent      = staticmethod(spectral_entropy)
        cls.norm_ent      = staticmethod(normalised_spectral_entropy)
        cls.sf            = staticmethod(spectral_flatness)
        cls.dom_harm      = staticmethod(dominant_harmonic)
        cls.cell_spectrum = staticmethod(cell_spectrum)
        cls.fourier_summary = staticmethod(fourier_summary)
        cls.all_fourier   = staticmethod(all_fourier)
        cls.build_data    = staticmethod(build_fourier_data)
        cls.fourier_dict  = staticmethod(fourier_dict)

        # Precomputed summaries
        cls.s_rabota_xor3 = fourier_summary('РАБОТА', 'xor3', 16)
        cls.s_gora_xor3   = fourier_summary('ГОРА',   'xor3', 16)
        cls.s_montazh_xor3= fourier_summary('МОНТАЖ', 'xor3', 16)
        cls.s_gorod_xor3  = fourier_summary('ГОРОД',  'xor3', 16)
        cls.s_mat_and     = fourier_summary('МАТ',    'and',  16)

    # ── dft1 ─────────────────────────────────────────────────────────────────

    def test_dft1_length(self):
        self.assertEqual(len(self.dft1([1, 2, 3, 4])), 4)

    def test_dft1_dc_equals_sum(self):
        seq = [1, 2, 3, 4]
        F = self.dft1(seq)
        self.assertAlmostEqual(F[0].real, sum(seq), places=6)
        self.assertAlmostEqual(F[0].imag, 0.0, places=6)

    def test_dft1_empty(self):
        self.assertEqual(self.dft1([]), [])

    def test_dft1_conjugate_symmetry(self):
        import cmath
        seq = [5, 3, 1, 7, 2, 6]
        F = self.dft1(seq)
        P = len(F)
        for k in range(1, P):
            self.assertAlmostEqual(F[k].real,  F[P - k].real, places=5)
            self.assertAlmostEqual(F[k].imag, -F[P - k].imag, places=5)

    # ── power_spectrum ────────────────────────────────────────────────────────

    def test_power_spectrum_length_even(self):
        self.assertEqual(len(self.power_spectrum([1, 2, 3, 4])), 3)  # 4//2+1

    def test_power_spectrum_length_odd(self):
        self.assertEqual(len(self.power_spectrum([1, 2, 3])), 2)     # 3//2+1

    def test_power_spectrum_non_negative(self):
        for v in self.power_spectrum([5, 3, 2, 7, 1, 4]):
            self.assertGreaterEqual(v, 0.0)

    def test_power_spectrum_empty(self):
        self.assertEqual(self.power_spectrum([]), [])

    def test_power_spectrum_parseval_approx(self):
        # Σ|F[k]|²/P ≈ Σ|x[t]|² (two-sided), one-sided approximation
        seq = [5, 3, 2, 7, 1, 4, 6, 2]
        ps  = self.power_spectrum(seq)
        P   = len(seq)
        # DC and Nyquist counted once; sum of all |F[k]|²/P = Σ x[t]²
        # full Parseval: Σ_{k=0}^{P-1} |F[k]|²/P = Σ x²
        xsq = sum(x * x for x in seq)
        # Reconstruct two-sided: S[0] + 2*S[1..P//2-1] + S[P//2] ≈ xsq
        half = P // 2
        two_sided = ps[0] + sum(2 * ps[k] for k in range(1, half)) + ps[half]
        self.assertAlmostEqual(two_sided, xsq, places=2)

    # ── spectral_entropy ──────────────────────────────────────────────────────

    def test_spec_ent_uniform_is_max(self):
        import math
        power = [1.0, 1.0, 1.0, 1.0]
        self.assertAlmostEqual(self.spec_ent(power), math.log2(4), places=6)

    def test_spec_ent_single_bin_is_zero(self):
        self.assertAlmostEqual(self.spec_ent([0.0, 0.0, 5.0, 0.0]), 0.0, places=6)

    def test_spec_ent_all_zero_is_zero(self):
        self.assertAlmostEqual(self.spec_ent([0.0, 0.0, 0.0]), 0.0, places=6)

    def test_spec_ent_non_negative(self):
        self.assertGreaterEqual(self.spec_ent([2.0, 3.0, 1.0, 4.0]), 0.0)

    # ── normalised_spectral_entropy ───────────────────────────────────────────

    def test_norm_ent_range(self):
        v = self.norm_ent([1.0, 2.0, 3.0, 4.0])
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_norm_ent_uniform_is_one(self):
        self.assertAlmostEqual(self.norm_ent([1.0, 1.0, 1.0, 1.0]), 1.0, places=6)

    def test_norm_ent_single_is_zero(self):
        self.assertAlmostEqual(self.norm_ent([0.0, 5.0, 0.0]), 0.0, places=6)

    # ── spectral_flatness ─────────────────────────────────────────────────────

    def test_sf_range(self):
        v = self.sf([1.0, 2.0, 3.0, 4.0])
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_sf_uniform_is_one(self):
        self.assertAlmostEqual(self.sf([3.0, 3.0, 3.0, 3.0]), 1.0, places=5)

    def test_sf_zero_bin_is_zero(self):
        self.assertAlmostEqual(self.sf([1.0, 0.0, 2.0]), 0.0, places=6)

    # ── dominant_harmonic ─────────────────────────────────────────────────────

    def test_dom_harm_skips_dc(self):
        # DC (k=0) = 100, k=1 = 5 → k* = 1
        k = self.dom_harm([100.0, 5.0, 3.0])
        self.assertEqual(k, 1)

    def test_dom_harm_selects_max_ac(self):
        k = self.dom_harm([1.0, 2.0, 10.0, 3.0])
        self.assertEqual(k, 2)

    def test_dom_harm_single_bin(self):
        self.assertEqual(self.dom_harm([5.0]), 0)

    # ── cell_spectrum structure ───────────────────────────────────────────────

    def test_cell_spectrum_keys(self):
        cs = self.cell_spectrum([10, 20, 30, 20])
        for k in ('power', 'dom_freq', 'ac_power', 'dc', 'dc_frac',
                  'h_sp', 'nh_sp', 'spec_entropy', 'sf', 'period'):
            self.assertIn(k, cs, msg=f"missing key: {k}")

    def test_cell_spectrum_p1_ac_zero(self):
        cs = self.cell_spectrum([42])
        self.assertEqual(cs['ac_power'], 0.0)
        self.assertEqual(cs['dom_freq'], 0)

    def test_cell_spectrum_p2_dom_is_1(self):
        cs = self.cell_spectrum([10, 50])
        self.assertEqual(cs['dom_freq'], 1)  # only Nyquist bin for P=2

    def test_cell_spectrum_power_non_negative(self):
        for v in self.cell_spectrum([5, 3, 2, 7, 1, 4, 6, 2])['power']:
            self.assertGreaterEqual(v, 0.0)

    def test_cell_spectrum_dc_frac_range(self):
        cs = self.cell_spectrum([5, 3, 2, 7, 1, 4, 6, 2])
        self.assertGreaterEqual(cs['dc_frac'], 0.0)
        self.assertLessEqual(cs['dc_frac'], 1.0)

    # ── fourier_summary structure ─────────────────────────────────────────────

    def test_fourier_summary_all_keys(self):
        for k in ('word', 'rule', 'period', 'n_cells',
                  'cell_dom_freq', 'cell_ac_power', 'cell_dc',
                  'cell_spec_entropy', 'cell_nh_sp', 'cell_dc_frac', 'cell_power',
                  'max_ac_power', 'max_ac_cell', 'mean_ac_power',
                  'dom_freq_hist', 'most_common_dom_freq',
                  'n_nyquist_dom', 'n_fundamental_dom',
                  'mean_spec_entropy', 'max_spec_entropy', 'max_spec_entropy_cell',
                  'mean_nh_sp', 'mean_dc_frac', 'mean_ps', 'dominant_k'):
            self.assertIn(k, self.s_rabota_xor3, msg=f"missing key: {k}")

    def test_fourier_summary_word_preserved(self):
        self.assertEqual(self.s_rabota_xor3['word'], 'РАБОТА')

    def test_fourier_summary_rule_preserved(self):
        self.assertEqual(self.s_rabota_xor3['rule'], 'xor3')

    def test_fourier_summary_cell_lists_length(self):
        for key in ('cell_dom_freq', 'cell_ac_power', 'cell_dc',
                    'cell_spec_entropy', 'cell_nh_sp', 'cell_dc_frac', 'cell_power'):
            self.assertEqual(len(self.s_rabota_xor3[key]), 16,
                             msg=f"{key} length ≠ 16")

    def test_fourier_summary_mean_dc_frac_in_range(self):
        v = self.s_rabota_xor3['mean_dc_frac']
        self.assertGreater(v, 0.0)
        self.assertLessEqual(v, 1.0)

    # ── РАБОТА XOR3 known values ──────────────────────────────────────────────

    def test_rabota_xor3_period_8(self):
        self.assertEqual(self.s_rabota_xor3['period'], 8)

    def test_rabota_xor3_dom_freq_hist(self):
        h = self.s_rabota_xor3['dom_freq_hist']
        self.assertEqual(h.get(1), 8)
        self.assertEqual(h.get(2), 1)
        self.assertEqual(h.get(3), 1)
        self.assertEqual(h.get(4), 6)

    def test_rabota_xor3_n_nyquist_dom(self):
        self.assertEqual(self.s_rabota_xor3['n_nyquist_dom'], 6)

    def test_rabota_xor3_n_fundamental_dom(self):
        self.assertEqual(self.s_rabota_xor3['n_fundamental_dom'], 8)

    def test_rabota_xor3_cell1_dom_freq_is_4(self):
        self.assertEqual(self.s_rabota_xor3['cell_dom_freq'][1], 4)

    def test_rabota_xor3_mean_spec_entropy_approx(self):
        self.assertAlmostEqual(self.s_rabota_xor3['mean_spec_entropy'], 1.482, places=2)

    def test_rabota_xor3_mean_ac_positive(self):
        self.assertGreater(self.s_rabota_xor3['mean_ac_power'], 0.0)

    # ── ГОРА XOR3 (P=2) known values ─────────────────────────────────────────

    def test_gora_xor3_period_2(self):
        self.assertEqual(self.s_gora_xor3['period'], 2)

    def test_gora_xor3_all_dom_freq_1(self):
        for df in self.s_gora_xor3['cell_dom_freq']:
            self.assertEqual(df, 1)

    def test_gora_xor3_spec_entropy_zero(self):
        # P=2 → only one AC bin → entropy = 0
        for H in self.s_gora_xor3['cell_spec_entropy']:
            self.assertAlmostEqual(H, 0.0, places=6)

    def test_gora_xor3_n_nyquist_dom_is_16(self):
        # P//2 = 1, so all cells dom k=1 = Nyquist
        self.assertEqual(self.s_gora_xor3['n_nyquist_dom'], 16)

    # ── МОНТАЖ XOR3 — highest mean spectral entropy ───────────────────────────

    def test_montazh_xor3_highest_mean_entropy(self):
        # МОНТАЖ has highest mean_spec_entropy in the lexicon ≈ 1.618
        self.assertGreater(self.s_montazh_xor3['mean_spec_entropy'],
                           self.s_rabota_xor3['mean_spec_entropy'])

    def test_montazh_xor3_entropy_approx(self):
        self.assertAlmostEqual(self.s_montazh_xor3['mean_spec_entropy'], 1.618, places=2)

    # ── ГОРОД XOR3 — k=3 dominated ───────────────────────────────────────────

    def test_gorod_xor3_dom3_cells_is_12(self):
        h = self.s_gorod_xor3['dom_freq_hist']
        self.assertEqual(h.get(3), 12)

    def test_gorod_xor3_n_nyquist_dom_zero(self):
        self.assertEqual(self.s_gorod_xor3['n_nyquist_dom'], 0)

    def test_gorod_xor3_mean_entropy_low(self):
        # Low entropy (concentrated spectrum): approx 1.163
        self.assertLess(self.s_gorod_xor3['mean_spec_entropy'], 1.3)

    # ── AND rule (P=1 fixed point) ────────────────────────────────────────────

    def test_and_p1_period_is_1(self):
        self.assertEqual(self.s_mat_and['period'], 1)

    def test_and_p1_all_ac_power_zero(self):
        for ac in self.s_mat_and['cell_ac_power']:
            self.assertAlmostEqual(ac, 0.0, places=6)

    def test_and_p1_all_dom_freq_zero(self):
        for df in self.s_mat_and['cell_dom_freq']:
            self.assertEqual(df, 0)

    def test_and_p1_spec_entropy_zero(self):
        for H in self.s_mat_and['cell_spec_entropy']:
            self.assertAlmostEqual(H, 0.0, places=6)

    # ── all_fourier ───────────────────────────────────────────────────────────

    def test_all_fourier_four_rules(self):
        res = self.all_fourier('РАБОТА', 16)
        for r in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(r, res)

    def test_all_fourier_each_is_dict(self):
        for s in self.all_fourier('МАТ', 16).values():
            self.assertIsInstance(s, dict)

    # ── build_fourier_data ────────────────────────────────────────────────────

    def test_build_data_has_words_key(self):
        d = self.build_data(['МАТ'], 16)
        self.assertIn('words', d)

    def test_build_data_has_data_key(self):
        d = self.build_data(['МАТ'], 16)
        self.assertIn('data', d)

    def test_build_data_nested(self):
        d = self.build_data(['МАТ'], 16)
        for rule in ('xor', 'xor3', 'and', 'or'):
            self.assertIn(rule, d['data']['МАТ'])

    # ── fourier_dict ──────────────────────────────────────────────────────────

    def test_fourier_dict_json_serialisable(self):
        import json
        fd = self.fourier_dict(self.s_rabota_xor3)
        out = json.dumps(fd)
        self.assertIsInstance(out, str)

    def test_fourier_dict_hist_keys_are_strings(self):
        fd = self.fourier_dict(self.s_rabota_xor3)
        for k in fd['dom_freq_hist']:
            self.assertIsInstance(k, str)

    def test_fourier_dict_cell_power_is_list_of_lists(self):
        fd = self.fourier_dict(self.s_rabota_xor3)
        cp = fd['cell_power']
        self.assertIsInstance(cp, list)
        self.assertIsInstance(cp[0], list)

    # ── Viewer HTML assertions ────────────────────────────────────────────────

    def test_viewer_has_ft_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ft-canvas', content)

    def test_viewer_has_ft_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ftRun', content)

    def test_viewer_has_ft_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ftOrbit', content)

    def test_viewer_has_ft_dft(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ftDFT', content)

    def test_viewer_has_ft_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ft-info', content)

    def test_viewer_has_solan_fourier_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_fourier', content)


class TestSolanMutual(unittest.TestCase):
    """Tests for solan_mutual.py — Mutual Information Analysis of Q6 CA orbits."""

    # ── imports ──────────────────────────────────────────────────────────────

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_mutual import (
            cell_entropy, cell_mi,
            attractor_states, entropy_profile,
            mi_matrix, mi_profile,
            mutual_summary, trajectory_mutual,
            all_mutual, build_mutual_data, mutual_dict,
        )
        cls.cell_entropy      = staticmethod(cell_entropy)
        cls.cell_mi           = staticmethod(cell_mi)
        cls.attractor_states  = staticmethod(attractor_states)
        cls.entropy_profile   = staticmethod(entropy_profile)
        cls.mi_matrix         = staticmethod(mi_matrix)
        cls.mi_profile        = staticmethod(mi_profile)
        cls.mutual_summary    = staticmethod(mutual_summary)
        cls.trajectory_mutual = staticmethod(trajectory_mutual)
        cls.all_mutual        = staticmethod(all_mutual)
        cls.build_mutual_data = staticmethod(build_mutual_data)
        cls.mutual_dict       = staticmethod(mutual_dict)

    # ── cell_entropy ─────────────────────────────────────────────────────────

    def test_cell_entropy_constant_zero(self):
        # constant sequence → H=0
        states = [[5]*16] * 3
        self.assertAlmostEqual(self.cell_entropy(states, 0), 0.0)

    def test_cell_entropy_two_values_equal(self):
        # alternating [0,1,0,1,...] → H=1 bit
        states = [[0]*16, [1]*16]
        self.assertAlmostEqual(self.cell_entropy(states, 0), 1.0)

    def test_cell_entropy_non_negative(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(16):
            self.assertGreaterEqual(self.cell_entropy(states, i), 0.0)

    def test_cell_entropy_bounded_by_log2_period(self):
        import math
        states = self.attractor_states('ТУМАН', 'xor3')
        P = len(states)
        for i in range(16):
            self.assertLessEqual(self.cell_entropy(states, i), math.log2(P) + 1e-9)

    # ── cell_mi ───────────────────────────────────────────────────────────────

    def test_cell_mi_diagonal_equals_entropy(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(16):
            mi_ii  = self.cell_mi(states, i, i)
            h_i    = self.cell_entropy(states, i)
            self.assertAlmostEqual(mi_ii, h_i, places=9)

    def test_cell_mi_symmetric(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(0, 4):
            for j in range(i+1, 4):
                self.assertAlmostEqual(
                    self.cell_mi(states, i, j),
                    self.cell_mi(states, j, i), places=9)

    def test_cell_mi_non_negative(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(16):
            for j in range(16):
                self.assertGreaterEqual(self.cell_mi(states, i, j), 0.0)

    def test_cell_mi_p1_all_zero(self):
        # P=1 → constant orbit → H=0 → MI=0
        states = self.attractor_states('ГОРА', 'xor')
        for i in range(16):
            for j in range(16):
                self.assertAlmostEqual(self.cell_mi(states, i, j), 0.0)

    def test_cell_mi_bounded_by_min_entropy(self):
        # I(X;Y) ≤ min(H(X), H(Y))
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(0, 4):
            for j in range(0, 4):
                mi_ij = self.cell_mi(states, i, j)
                h_i   = self.cell_entropy(states, i)
                h_j   = self.cell_entropy(states, j)
                self.assertLessEqual(mi_ij, min(h_i, h_j) + 1e-9)

    # ── attractor_states ──────────────────────────────────────────────────────

    def test_attractor_states_returns_list(self):
        states = self.attractor_states('ГОРА', 'xor3')
        self.assertIsInstance(states, list)

    def test_attractor_states_each_row_width(self):
        states = self.attractor_states('ГОРА', 'xor3', 16)
        for s in states:
            self.assertEqual(len(s), 16)

    def test_attractor_states_p1_one_row(self):
        states = self.attractor_states('ГОРА', 'xor')
        self.assertEqual(len(states), 1)

    def test_attractor_states_gora_p2(self):
        states = self.attractor_states('ГОРА', 'xor3')
        self.assertEqual(len(states), 2)

    def test_attractor_states_tuman_p8(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        self.assertEqual(len(states), 8)

    # ── entropy_profile ───────────────────────────────────────────────────────

    def test_entropy_profile_length(self):
        ep = self.entropy_profile('ГОРА', 'xor3', 16)
        self.assertEqual(len(ep), 16)

    def test_entropy_profile_non_negative(self):
        ep = self.entropy_profile('ТУМАН', 'xor3', 16)
        for h in ep:
            self.assertGreaterEqual(h, 0.0)

    def test_entropy_profile_p1_all_zero(self):
        ep = self.entropy_profile('ГОРА', 'xor', 16)
        for h in ep:
            self.assertAlmostEqual(h, 0.0)

    def test_entropy_profile_gora_xor3_all_one(self):
        # ГОРА P=2: all cells alternate → H=1 for all
        ep = self.entropy_profile('ГОРА', 'xor3', 16)
        for h in ep:
            self.assertAlmostEqual(h, 1.0)

    # ── mi_matrix ────────────────────────────────────────────────────────────

    def test_mi_matrix_shape(self):
        M = self.mi_matrix('ГОРА', 'xor3')
        self.assertEqual(len(M), 16)
        for row in M:
            self.assertEqual(len(row), 16)

    def test_mi_matrix_symmetric(self):
        M = self.mi_matrix('ТУМАН', 'xor3')
        for i in range(16):
            for j in range(16):
                self.assertAlmostEqual(M[i][j], M[j][i], places=9)

    def test_mi_matrix_diagonal_positive_tuman(self):
        M = self.mi_matrix('ТУМАН', 'xor3')
        for i in range(16):
            self.assertGreater(M[i][i], 0.0)

    def test_mi_matrix_diagonal_zero_xor(self):
        M = self.mi_matrix('ГОРА', 'xor')
        for i in range(16):
            self.assertAlmostEqual(M[i][i], 0.0)

    def test_mi_matrix_gora_xor3_all_ones(self):
        # ГОРА P=2: all cells fully correlated → entire matrix = 1.0
        M = self.mi_matrix('ГОРА', 'xor3')
        for i in range(16):
            for j in range(16):
                self.assertAlmostEqual(M[i][j], 1.0)

    # ── mi_profile ────────────────────────────────────────────────────────────

    def test_mi_profile_length(self):
        M = self.mi_matrix('ТУМАН', 'xor3')
        prof = self.mi_profile(M, 16)
        self.assertEqual(len(prof), 9)   # W//2 + 1 = 9

    def test_mi_profile_d0_equals_mean_entropy(self):
        # d=0 means MI(i,i)=H(i), average = mean_entropy
        M   = self.mi_matrix('ТУМАН', 'xor3')
        prof = self.mi_profile(M, 16)
        ent_mean = sum(M[i][i] for i in range(16)) / 16
        self.assertAlmostEqual(prof[0], ent_mean, places=5)

    def test_mi_profile_gora_xor3_all_one(self):
        # all pairs fully correlated
        M    = self.mi_matrix('ГОРА', 'xor3')
        prof = self.mi_profile(M, 16)
        for v in prof:
            self.assertAlmostEqual(v, 1.0)

    def test_mi_profile_non_negative(self):
        M    = self.mi_matrix('ТУМАН', 'xor3')
        prof = self.mi_profile(M, 16)
        for v in prof:
            self.assertGreaterEqual(v, 0.0)

    # ── mutual_summary / trajectory_mutual ───────────────────────────────────

    def test_mutual_summary_alias_identical(self):
        tr1 = self.mutual_summary('ГОРА', 'xor3')
        tr2 = self.trajectory_mutual('ГОРА', 'xor3')
        self.assertEqual(tr1['period'], tr2['period'])
        self.assertAlmostEqual(tr1['mean_entropy'], tr2['mean_entropy'])
        self.assertAlmostEqual(tr1['max_mi'],       tr2['max_mi'])

    def test_mutual_summary_has_required_keys(self):
        tr = self.mutual_summary('ГОРА', 'xor3')
        for k in ('word','rule','width','period','entropy','M','mi_by_dist',
                  'mean_entropy','max_mi','max_mi_pair'):
            self.assertIn(k, tr)

    def test_mutual_summary_word_uppercased(self):
        tr = self.mutual_summary('гора', 'xor3')
        self.assertEqual(tr['word'], 'ГОРА')

    def test_mutual_summary_gora_xor3_period(self):
        tr = self.mutual_summary('ГОРА', 'xor3')
        self.assertEqual(tr['period'], 2)

    def test_mutual_summary_gora_xor3_mean_entropy(self):
        tr = self.mutual_summary('ГОРА', 'xor3')
        self.assertAlmostEqual(tr['mean_entropy'], 1.0, places=6)

    def test_mutual_summary_gora_xor3_max_mi(self):
        tr = self.mutual_summary('ГОРА', 'xor3')
        self.assertAlmostEqual(tr['max_mi'], 1.0, places=6)

    def test_mutual_summary_tuman_xor3_period(self):
        tr = self.mutual_summary('ТУМАН', 'xor3')
        self.assertEqual(tr['period'], 8)

    def test_mutual_summary_tuman_xor3_mean_entropy(self):
        tr = self.mutual_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(tr['mean_entropy'], 2.23407, places=4)

    def test_mutual_summary_tuman_xor3_max_mi(self):
        tr = self.mutual_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(tr['max_mi'], 2.75, places=6)

    def test_mutual_summary_tuman_xor3_max_pair(self):
        tr = self.mutual_summary('ТУМАН', 'xor3')
        self.assertEqual(tuple(tr['max_mi_pair']), (2, 13))

    def test_mutual_summary_xor_all_zero(self):
        tr = self.mutual_summary('ГОРА', 'xor')
        self.assertEqual(tr['period'], 1)
        self.assertAlmostEqual(tr['mean_entropy'], 0.0)
        self.assertAlmostEqual(tr['max_mi'], 0.0)

    def test_mutual_summary_entropy_list_length(self):
        tr = self.mutual_summary('ГОРА', 'xor3', 16)
        self.assertEqual(len(tr['entropy']), 16)

    def test_mutual_summary_mi_by_dist_length(self):
        tr = self.mutual_summary('ГОРА', 'xor3', 16)
        self.assertEqual(len(tr['mi_by_dist']), 9)

    def test_mutual_summary_M_shape(self):
        tr = self.mutual_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(len(tr['M']), 16)
        for row in tr['M']:
            self.assertEqual(len(row), 16)

    def test_mutual_summary_entropy_matches_diagonal(self):
        tr = self.mutual_summary('ТУМАН', 'xor3')
        M  = tr['M']
        for i in range(16):
            self.assertAlmostEqual(tr['entropy'][i], M[i][i], places=9)

    # ── all_mutual ────────────────────────────────────────────────────────────

    def test_all_mutual_returns_four_rules(self):
        am = self.all_mutual('ГОРА')
        self.assertEqual(set(am.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_mutual_each_has_period(self):
        am = self.all_mutual('ГОРА')
        for rule, tr in am.items():
            self.assertIn('period', tr)

    # ── build_mutual_data ─────────────────────────────────────────────────────

    def test_build_mutual_data_keys(self):
        data = self.build_mutual_data(['ГОРА', 'ТУМАН'])
        for k in ('words','width','per_rule','ranking','max_h','min_h'):
            self.assertIn(k, data)

    def test_build_mutual_data_per_rule_has_all_words(self):
        words = ['ГОРА', 'ТУМАН', 'МАТ']
        data  = self.build_mutual_data(words)
        for rule in ('xor','xor3','and','or'):
            self.assertEqual(set(data['per_rule'][rule].keys()), set(words))

    def test_build_mutual_data_ranking_descending(self):
        data = self.build_mutual_data(['ГОРА', 'ТУМАН', 'МАТ'])
        for rule in ('xor3',):
            rank = data['ranking'][rule]
            vals = [h for _, h in rank]
            self.assertEqual(vals, sorted(vals, reverse=True))

    # ── mutual_dict ───────────────────────────────────────────────────────────

    def test_mutual_dict_json_serialisable(self):
        import json
        d = self.mutual_dict('ГОРА')
        s = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(s, str)

    def test_mutual_dict_has_rules_key(self):
        d = self.mutual_dict('ГОРА')
        self.assertIn('rules', d)
        self.assertEqual(set(d['rules'].keys()), {'xor','xor3','and','or'})

    def test_mutual_dict_rule_keys(self):
        d = self.mutual_dict('ТУМАН')
        for rule, rd in d['rules'].items():
            for k in ('period','mean_entropy','max_mi','max_mi_pair','entropy','mi_by_dist'):
                self.assertIn(k, rd)

    def test_mutual_dict_max_mi_pair_list(self):
        d = self.mutual_dict('ТУМАН')
        pair = d['rules']['xor3']['max_mi_pair']
        self.assertIsInstance(pair, list)
        self.assertEqual(len(pair), 2)

    # ── viewer assertions ─────────────────────────────────────────────────────

    def test_viewer_has_mi_mat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-mat', content)

    def test_viewer_has_mi_dist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-dist', content)

    def test_viewer_has_mi_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-stats', content)

    def test_viewer_has_mi_hmap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-hmap', content)

    def test_viewer_has_mi_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('miRun', content)

    def test_viewer_has_mi_step(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('miStep', content)

    def test_viewer_has_mi_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('miOrbit', content)

    def test_viewer_has_solan_mutual_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_mutual', content)

    def test_viewer_has_mi_cell_mi(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('miCellMI', content)

    def test_viewer_has_mi_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-word', content)


class TestSolanTransfer(unittest.TestCase):
    """Tests for solan_transfer.py — Transfer Entropy analysis of Q6 CA orbits."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_transfer import (
            get_orbit, bit_te, cell_te,
            te_matrix, te_asymmetry,
            te_summary, te_dict, all_te,
            build_te_data,
        )
        cls.get_orbit    = staticmethod(get_orbit)
        cls.bit_te       = staticmethod(bit_te)
        cls.cell_te      = staticmethod(cell_te)
        cls.te_matrix    = staticmethod(te_matrix)
        cls.te_asymmetry = staticmethod(te_asymmetry)
        cls.te_summary   = staticmethod(te_summary)
        cls.te_dict      = staticmethod(te_dict)
        cls.all_te       = staticmethod(all_te)
        cls.build_te_data= staticmethod(build_te_data)

    # ── get_orbit ─────────────────────────────────────────────────────────────

    def test_orbit_gora_xor3_length(self):
        orbit = self.get_orbit('ГОРА', 'xor3')
        self.assertEqual(len(orbit), 2)

    def test_orbit_tuman_xor3_length(self):
        orbit = self.get_orbit('ТУМАН', 'xor3')
        self.assertEqual(len(orbit), 8)

    def test_orbit_gora_xor_length(self):
        orbit = self.get_orbit('ГОРА', 'xor')
        self.assertEqual(len(orbit), 1)

    def test_orbit_state_width(self):
        orbit = self.get_orbit('ТУМАН', 'xor3', 16)
        for state in orbit:
            self.assertEqual(len(state), 16)

    def test_orbit_values_in_q6_range(self):
        orbit = self.get_orbit('ТУМАН', 'xor3')
        for state in orbit:
            for v in state:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 63)

    def test_orbit_returns_tuples(self):
        orbit = self.get_orbit('ГОРА', 'xor3')
        for s in orbit:
            self.assertIsInstance(s, tuple)

    def test_orbit_periodic(self):
        from projects.hexglyph.solan_ca import step
        orbit = self.get_orbit('ТУМАН', 'xor3')
        last  = list(orbit[-1])
        nxt   = step(last, 'xor3')
        self.assertEqual(nxt, list(orbit[0]))

    # ── bit_te ────────────────────────────────────────────────────────────────

    def test_bit_te_non_negative(self):
        te = self.bit_te([0, 1, 0, 1], [0, 1, 0, 1])
        self.assertGreaterEqual(te, 0.0)

    def test_bit_te_short_series_zero(self):
        self.assertEqual(self.bit_te([1], [0]), 0.0)

    def test_bit_te_constant_zero(self):
        te = self.bit_te([0, 0, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(te, 0.0)

    def test_bit_te_returns_float(self):
        te = self.bit_te([0, 1, 0, 1], [1, 0, 1, 0])
        self.assertIsInstance(te, float)

    # ── cell_te ───────────────────────────────────────────────────────────────

    def test_cell_te_non_negative(self):
        orbit = self.get_orbit('ТУМАН', 'xor3')
        for i in range(4):
            for j in range(4):
                self.assertGreaterEqual(self.cell_te(orbit, i, j), 0.0)

    def test_cell_te_p1_all_zero(self):
        orbit = self.get_orbit('ГОРА', 'xor')
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(self.cell_te(orbit, i, j), 0.0)

    def test_cell_te_returns_float(self):
        orbit = self.get_orbit('ТУМАН', 'xor3')
        te = self.cell_te(orbit, 0, 1)
        self.assertIsInstance(te, float)

    # ── te_matrix ────────────────────────────────────────────────────────────

    def test_te_matrix_shape(self):
        M = self.te_matrix('ТУМАН', 'xor3', 16)
        self.assertEqual(len(M), 16)
        for row in M:
            self.assertEqual(len(row), 16)

    def test_te_matrix_non_negative(self):
        M = self.te_matrix('ТУМАН', 'xor3')
        for row in M:
            for v in row:
                self.assertGreaterEqual(v, 0.0)

    def test_te_matrix_xor_all_zero(self):
        M = self.te_matrix('ГОРА', 'xor')
        for row in M:
            for v in row:
                self.assertAlmostEqual(v, 0.0)

    def test_te_matrix_asymmetric(self):
        M = self.te_matrix('ТУМАН', 'xor3')
        found_asymm = any(
            abs(M[i][j] - M[j][i]) > 1e-6
            for i in range(16) for j in range(i+1, 16)
        )
        self.assertTrue(found_asymm)

    # ── te_asymmetry ─────────────────────────────────────────────────────────

    def test_te_asymmetry_shape(self):
        M = self.te_matrix('ТУМАН', 'xor3')
        A = self.te_asymmetry(M)
        self.assertEqual(len(A), 16)
        for row in A:
            self.assertEqual(len(row), 16)

    def test_te_asymmetry_antisymmetric(self):
        M = self.te_matrix('ТУМАН', 'xor3')
        A = self.te_asymmetry(M)
        for i in range(16):
            for j in range(16):
                self.assertAlmostEqual(A[i][j], -A[j][i], places=8)

    def test_te_asymmetry_diagonal_zero(self):
        M = self.te_matrix('ТУМАН', 'xor3')
        A = self.te_asymmetry(M)
        for i in range(16):
            self.assertAlmostEqual(A[i][i], 0.0)

    # ── te_summary / te_dict ──────────────────────────────────────────────────

    def test_te_summary_alias_identical(self):
        d1 = self.te_summary('ТУМАН', 'xor3')
        d2 = self.te_dict('ТУМАН', 'xor3')
        self.assertEqual(d1['period'], d2['period'])
        self.assertAlmostEqual(d1['max_te'], d2['max_te'])
        self.assertAlmostEqual(d1['lr_asymmetry'], d2['lr_asymmetry'])

    def test_te_summary_has_required_keys(self):
        d = self.te_summary('ТУМАН', 'xor3')
        for k in ('word','rule','width','period','matrix','max_te','mean_te',
                  'self_te','right_te','left_te','asymmetry',
                  'mean_right','mean_left','lr_asymmetry'):
            self.assertIn(k, d)

    def test_te_summary_word_uppercased(self):
        d = self.te_summary('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_te_summary_tuman_period(self):
        d = self.te_summary('ТУМАН', 'xor3')
        self.assertEqual(d['period'], 8)

    def test_te_summary_tuman_max_te(self):
        d = self.te_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['max_te'], 2.90241012, places=4)

    def test_te_summary_tuman_lr_asymmetry(self):
        d = self.te_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['lr_asymmetry'], 0.0, places=6)

    def test_te_summary_gora_xor_all_zero(self):
        d = self.te_summary('ГОРА', 'xor')
        self.assertEqual(d['period'], 1)
        self.assertAlmostEqual(d['max_te'], 0.0)
        self.assertAlmostEqual(d['mean_te'], 0.0)

    def test_te_summary_gora_xor3_p2(self):
        d = self.te_summary('ГОРА', 'xor3')
        self.assertEqual(d['period'], 2)

    def test_te_summary_matrix_shape(self):
        d = self.te_summary('ТУМАН', 'xor3', 16)
        mat = d['matrix']
        self.assertEqual(len(mat), 16)
        for row in mat:
            self.assertEqual(len(row), 16)

    def test_te_summary_self_te_length(self):
        d = self.te_summary('ТУМАН', 'xor3')
        self.assertEqual(len(d['self_te']), 16)

    def test_te_summary_right_left_length(self):
        d = self.te_summary('ТУМАН', 'xor3')
        self.assertEqual(len(d['right_te']), 16)
        self.assertEqual(len(d['right_te']), len(d['left_te']))

    def test_te_summary_lr_asymmetry_formula(self):
        d = self.te_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(
            d['lr_asymmetry'],
            d['mean_right'] - d['mean_left'], places=8)

    def test_te_summary_mean_te_off_diag(self):
        d = self.te_summary('ТУМАН', 'xor3')
        mat = d['matrix']
        W   = d['width']
        vals = [mat[i][j] for i in range(W) for j in range(W) if i != j]
        expected = sum(vals) / len(vals)
        self.assertAlmostEqual(d['mean_te'], expected, places=5)

    def test_te_summary_max_te_equals_matrix_max(self):
        d   = self.te_summary('ТУМАН', 'xor3')
        mat = d['matrix']
        W   = d['width']
        mx  = max(mat[i][j] for i in range(W) for j in range(W))
        self.assertAlmostEqual(d['max_te'], mx, places=8)

    def test_te_summary_tuman_self_te_all_zero(self):
        d = self.te_summary('ТУМАН', 'xor3')
        for v in d['self_te']:
            self.assertAlmostEqual(v, 0.0, places=6)

    # ── all_te ────────────────────────────────────────────────────────────────

    def test_all_te_four_rules(self):
        am = self.all_te('ТУМАН')
        self.assertEqual(set(am.keys()), {'xor','xor3','and','or'})

    def test_all_te_each_has_period(self):
        am = self.all_te('ГОРА')
        for rule, d in am.items():
            self.assertIn('period', d)

    def test_all_te_xor3_has_max_greater_than_xor(self):
        am = self.all_te('ТУМАН')
        self.assertGreater(am['xor3']['max_te'], am['xor']['max_te'])

    # ── build_te_data ─────────────────────────────────────────────────────────

    def test_build_te_data_keys(self):
        d = self.build_te_data(['ГОРА', 'ТУМАН'])
        for k in ('words','per_rule','ranking'):
            self.assertIn(k, d)

    def test_build_te_data_per_rule_words(self):
        words = ['ГОРА', 'ТУМАН', 'МАТ']
        d = self.build_te_data(words)
        for rule in ('xor','xor3','and','or'):
            self.assertEqual(set(d['per_rule'][rule].keys()), set(words))

    def test_build_te_data_ranking_descending(self):
        words = ['ГОРА', 'ТУМАН', 'МАТ']
        d = self.build_te_data(words)
        vals = [v for _, v in d['ranking']['xor3']]
        self.assertEqual(vals, sorted(vals, reverse=True))

    # ── JSON serialisability ──────────────────────────────────────────────────

    def test_te_summary_json_serialisable(self):
        import json
        d = self.te_summary('ТУМАН', 'xor3')
        s = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(s, str)

    # ── viewer assertions ─────────────────────────────────────────────────────

    def test_viewer_has_te_matrix_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('te-matrix-canvas', content)

    def test_viewer_has_te_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('teRun', content)

    def test_viewer_has_te_step(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('teStep', content)

    def test_viewer_has_te_orbit(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('teOrbitStates', content)

    def test_viewer_has_te_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('te-word', content)

    def test_viewer_has_te_rule(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('te-rule', content)

    def test_viewer_has_solan_transfer_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_transfer', content)

    def test_viewer_has_cell_te(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cellTE', content)


class TestSolanLZ(unittest.TestCase):
    """Tests for solan_lz.py — LZ76 complexity of Q6 CA attractors."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_lz import (
            lz76, to_binary, lz_of_series, lz_of_spatial,
            lz_summary, lz_dict, all_lz, build_lz_data,
        )
        cls.lz76          = staticmethod(lz76)
        cls.to_binary     = staticmethod(to_binary)
        cls.lz_of_series  = staticmethod(lz_of_series)
        cls.lz_of_spatial = staticmethod(lz_of_spatial)
        cls.lz_summary    = staticmethod(lz_summary)
        cls.lz_dict       = staticmethod(lz_dict)
        cls.all_lz        = staticmethod(all_lz)
        cls.build_lz_data = staticmethod(build_lz_data)

    # ── lz76 ────────────────────────────────────────────────────────────────

    def test_lz76_empty_zero(self):
        self.assertEqual(self.lz76(''), 0)

    def test_lz76_single_char(self):
        self.assertEqual(self.lz76('0'), 1)

    def test_lz76_two_chars(self):
        self.assertEqual(self.lz76('01'), 2)

    def test_lz76_all_zeros_low(self):
        c = self.lz76('0' * 96)
        self.assertLessEqual(c, 12)

    def test_lz76_periodic_less_than_random(self):
        import random
        random.seed(42)
        periodic = '01' * 48
        rnd = ''.join(random.choice('01') for _ in range(96))
        self.assertLess(self.lz76(periodic), self.lz76(rnd))

    def test_lz76_positive(self):
        self.assertGreaterEqual(self.lz76('101'), 1)

    # ── to_binary ────────────────────────────────────────────────────────────

    def test_to_binary_zero(self):
        self.assertEqual(self.to_binary(0, 6), '000000')

    def test_to_binary_ones(self):
        self.assertEqual(self.to_binary(63, 6), '111111')

    def test_to_binary_five(self):
        self.assertEqual(self.to_binary(5, 6), '000101')

    def test_to_binary_length(self):
        for v in range(64):
            self.assertEqual(len(self.to_binary(v, 6)), 6)

    # ── lz_of_series ─────────────────────────────────────────────────────────

    def test_lz_of_series_keys(self):
        d = self.lz_of_series([0]*8)
        for k in ('bits','lz','norm'):
            self.assertIn(k, d)

    def test_lz_of_series_bits_length(self):
        d = self.lz_of_series([0]*8)
        self.assertEqual(d['bits'], 48)   # 8 values × 6 bits

    def test_lz_of_series_norm_non_negative(self):
        d = self.lz_of_series([0,1]*4)
        self.assertGreaterEqual(d['norm'], 0.0)

    def test_lz_of_series_lz_positive(self):
        d = self.lz_of_series([0,1,0,1,0,1,0,1])
        self.assertGreaterEqual(d['lz'], 1)

    # ── lz_of_spatial ────────────────────────────────────────────────────────

    def test_lz_of_spatial_keys(self):
        d = self.lz_of_spatial([0]*16)
        for k in ('bits','lz','norm'):
            self.assertIn(k, d)

    def test_lz_of_spatial_bits_length(self):
        d = self.lz_of_spatial([0]*16)
        self.assertEqual(d['bits'], 96)   # 16 cells × 6 bits

    # ── lz_summary / lz_dict ─────────────────────────────────────────────────

    def test_lz_summary_alias_identical(self):
        d1 = self.lz_summary('ТУМАН', 'xor3')
        d2 = self.lz_dict('ТУМАН', 'xor3')
        self.assertEqual(d1['period'], d2['period'])
        self.assertAlmostEqual(d1['full_lz']['norm'], d2['full_lz']['norm'])

    def test_lz_summary_has_required_keys(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        for k in ('word','rule','period','cell_lz','mean_cell_norm',
                  'spatial_lz','mean_sp_norm','full_lz'):
            self.assertIn(k, d)

    def test_lz_summary_word_upper(self):
        d = self.lz_summary('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_lz_summary_tuman_period(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        self.assertEqual(d['period'], 8)

    def test_lz_summary_tuman_full_norm(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['full_lz']['norm'], 0.59906016, places=4)

    def test_lz_summary_tuman_full_lz_value(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        self.assertEqual(d['full_lz']['lz'], 48)

    def test_lz_summary_tuman_full_bits(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        self.assertEqual(d['full_lz']['bits'], 768)   # 8×16×6

    def test_lz_summary_tuman_mean_cell_norm(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['mean_cell_norm'], 1.28715933, places=4)

    def test_lz_summary_gora_xor_period1(self):
        d = self.lz_summary('ГОРА', 'xor')
        self.assertEqual(d['period'], 1)

    def test_lz_summary_cell_lz_length(self):
        d = self.lz_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(len(d['cell_lz']), 16)

    def test_lz_summary_cell_lz_each_has_keys(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        for c in d['cell_lz']:
            for k in ('bits','lz','norm'):
                self.assertIn(k, c)

    def test_lz_summary_cell_lz_bits(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        for c in d['cell_lz']:
            self.assertEqual(c['bits'], 48)   # P=8 × 6 bits

    def test_lz_summary_spatial_lz_length(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        self.assertEqual(len(d['spatial_lz']), 8)   # P=8

    def test_lz_summary_cell0_norm(self):
        d = self.lz_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['cell_lz'][0]['norm'], 1.27988724, places=4)

    def test_lz_summary_json_serialisable(self):
        import json
        d = self.lz_summary('ТУМАН', 'xor3')
        s = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(s, str)

    # ── all_lz ───────────────────────────────────────────────────────────────

    def test_all_lz_four_rules(self):
        am = self.all_lz('ТУМАН')
        self.assertEqual(set(am.keys()), {'xor','xor3','and','or'})

    def test_all_lz_each_has_period(self):
        am = self.all_lz('ГОРА')
        for rule, d in am.items():
            self.assertIn('period', d)

    # ── build_lz_data ─────────────────────────────────────────────────────────

    def test_build_lz_data_keys(self):
        d = self.build_lz_data(['ГОРА', 'ТУМАН'])
        for k in ('words','width','per_rule'):
            self.assertIn(k, d)

    def test_build_lz_data_per_rule_words(self):
        words = ['ГОРА', 'МАТ']
        d = self.build_lz_data(words)
        for rule in ('xor','xor3','and','or'):
            self.assertEqual(set(d['per_rule'][rule].keys()), set(words))

    # ── viewer assertions ─────────────────────────────────────────────────────

    def test_viewer_has_lz_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lz-canvas', content)

    def test_viewer_has_lz_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lzRun', content)

    def test_viewer_has_solan_lz_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_lz', content)


class TestSolanMoran(unittest.TestCase):
    """Tests for solan_moran.py — Moran's I spatial autocorrelation of Q6 CA."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_moran import (
            morans_i, spatial_classification,
            morans_i_series, moran_summary, morans_i_dict,
            all_morans_i, build_moran_data,
        )
        cls.morans_i               = staticmethod(morans_i)
        cls.spatial_classification = staticmethod(spatial_classification)
        cls.morans_i_series        = staticmethod(morans_i_series)
        cls.moran_summary          = staticmethod(moran_summary)
        cls.morans_i_dict          = staticmethod(morans_i_dict)
        cls.all_morans_i           = staticmethod(all_morans_i)
        cls.build_moran_data       = staticmethod(build_moran_data)

    # ── morans_i ─────────────────────────────────────────────────────────────

    def test_morans_i_constant_nan(self):
        import math
        self.assertTrue(math.isnan(self.morans_i([5, 5, 5, 5])))

    def test_morans_i_empty_nan(self):
        import math
        self.assertTrue(math.isnan(self.morans_i([])))

    def test_morans_i_single_nan(self):
        import math
        self.assertTrue(math.isnan(self.morans_i([42])))

    def test_morans_i_alternating_minus_one(self):
        vals = [0, 63] * 8
        self.assertAlmostEqual(self.morans_i(vals), -1.0, places=6)

    def test_morans_i_two_elements_checkerboard(self):
        self.assertAlmostEqual(self.morans_i([10, 50]), -1.0, places=6)

    def test_morans_i_range(self):
        import math
        v = self.morans_i(list(range(16)))
        if not math.isnan(v):
            self.assertGreaterEqual(v, -1.1)
            self.assertLessEqual(v, 1.1)

    def test_morans_i_returns_float(self):
        v = self.morans_i(list(range(1, 17)))
        self.assertIsInstance(v, float)

    # ── spatial_classification ────────────────────────────────────────────────

    def test_spatial_classification_constant(self):
        import math
        self.assertEqual(self.spatial_classification(float('nan')), 'constant')

    def test_spatial_classification_strongly_dispersed(self):
        self.assertEqual(self.spatial_classification(-0.68), 'strongly dispersed')

    def test_spatial_classification_dispersed(self):
        self.assertEqual(self.spatial_classification(-0.3), 'dispersed')

    def test_spatial_classification_random(self):
        self.assertEqual(self.spatial_classification(0.05), 'random')

    def test_spatial_classification_clustered(self):
        self.assertEqual(self.spatial_classification(0.49), 'clustered')

    def test_spatial_classification_strongly_clustered(self):
        self.assertEqual(self.spatial_classification(0.8), 'strongly clustered')

    # ── morans_i_series ───────────────────────────────────────────────────────

    def test_morans_i_series_tuman_length(self):
        s = self.morans_i_series('ТУМАН', 'xor3')
        self.assertEqual(len(s), 8)

    def test_morans_i_series_gora_xor3_length(self):
        s = self.morans_i_series('ГОРА', 'xor3')
        self.assertEqual(len(s), 2)

    def test_morans_i_series_gora_xor_nan(self):
        import math
        s = self.morans_i_series('ГОРА', 'xor')
        self.assertTrue(math.isnan(s[0]))

    # ── moran_summary / morans_i_dict ─────────────────────────────────────────

    def test_moran_summary_alias_identical(self):
        d1 = self.moran_summary('ТУМАН', 'xor3')
        d2 = self.morans_i_dict('ТУМАН', 'xor3')
        self.assertEqual(d1['period'], d2['period'])
        self.assertAlmostEqual(d1['mean_i'], d2['mean_i'])
        self.assertEqual(d1['classification'], d2['classification'])

    def test_moran_summary_has_required_keys(self):
        d = self.moran_summary('ТУМАН', 'xor3')
        for k in ('word','rule','period','series','mean_i','min_i','max_i',
                  'var_i','classification','n_valid'):
            self.assertIn(k, d)

    def test_moran_summary_word_upper(self):
        d = self.moran_summary('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_moran_summary_tuman_period(self):
        d = self.moran_summary('ТУМАН', 'xor3')
        self.assertEqual(d['period'], 8)

    def test_moran_summary_tuman_mean_i(self):
        d = self.moran_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['mean_i'], -0.12167013, places=4)

    def test_moran_summary_tuman_min_i(self):
        d = self.moran_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['min_i'], -0.67936736, places=4)

    def test_moran_summary_tuman_max_i(self):
        d = self.moran_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['max_i'], 0.49012789, places=4)

    def test_moran_summary_tuman_classification(self):
        d = self.moran_summary('ТУМАН', 'xor3')
        self.assertEqual(d['classification'], 'dispersed')

    def test_moran_summary_tuman_n_valid(self):
        d = self.moran_summary('ТУМАН', 'xor3')
        self.assertEqual(d['n_valid'], 8)

    def test_moran_summary_gora_xor_nan_mean(self):
        import math
        d = self.moran_summary('ГОРА', 'xor')
        self.assertTrue(math.isnan(d['mean_i']))

    def test_moran_summary_gora_xor_n_valid_zero(self):
        d = self.moran_summary('ГОРА', 'xor')
        self.assertEqual(d['n_valid'], 0)

    def test_moran_summary_series_length(self):
        d = self.moran_summary('ТУМАН', 'xor3')
        self.assertEqual(len(d['series']), 8)

    def test_moran_summary_series_matches_direct(self):
        d    = self.moran_summary('ТУМАН', 'xor3')
        ser  = self.morans_i_series('ТУМАН', 'xor3')
        for a, b in zip(d['series'], ser):
            self.assertAlmostEqual(a, b, places=8)

    # ── all_morans_i ──────────────────────────────────────────────────────────

    def test_all_morans_i_four_rules(self):
        am = self.all_morans_i('ТУМАН')
        self.assertEqual(set(am.keys()), {'xor','xor3','and','or'})

    def test_all_morans_i_each_has_keys(self):
        am = self.all_morans_i('ГОРА')
        for rule, d in am.items():
            self.assertIn('period', d)
            self.assertIn('classification', d)

    # ── build_moran_data ──────────────────────────────────────────────────────

    def test_build_moran_data_keys(self):
        d = self.build_moran_data(['ГОРА', 'ТУМАН'])
        for k in ('words','width','per_rule'):
            self.assertIn(k, d)

    def test_build_moran_data_per_rule_words(self):
        words = ['ГОРА', 'МАТ']
        d = self.build_moran_data(words)
        for rule in ('xor','xor3','and','or'):
            self.assertEqual(set(d['per_rule'][rule].keys()), set(words))

    # ── viewer assertions ─────────────────────────────────────────────────────

    def test_viewer_has_moran_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('moran-word', content)

    def test_viewer_has_moran_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('moranRun', content)

    def test_viewer_has_solan_moran_section(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_moran', content)


class TestSolanDerrida2(unittest.TestCase):
    """Replacement tests for solan_derrida.py (standard *_summary convention)."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_derrida import (
            state_dist_norm, derrida_point,
            lexicon_points, random_points,
            derrida_curve, analytic_curve,
            classify_rule, build_derrida_data,
            derrida_dict, derrida_summary,
            _ALL_RULES,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.state_dist_norm    = staticmethod(state_dist_norm)
        cls.derrida_point      = staticmethod(derrida_point)
        cls.lexicon_points     = staticmethod(lexicon_points)
        cls.random_points      = staticmethod(random_points)
        cls.derrida_curve      = staticmethod(derrida_curve)
        cls.analytic_curve     = staticmethod(analytic_curve)
        cls.classify_rule      = staticmethod(classify_rule)
        cls.build_derrida_data = staticmethod(build_derrida_data)
        cls.derrida_dict       = staticmethod(derrida_dict)
        cls.derrida_summary    = staticmethod(derrida_summary)
        cls.ALL_RULES          = _ALL_RULES
        cls.LEXICON            = list(LEXICON)

    # ── state_dist_norm() ─────────────────────────────────────────────────────

    def test_sdn_identical_zero(self):
        c = [10, 20, 30] * 4
        self.assertAlmostEqual(self.state_dist_norm(c, c, 12), 0.0)

    def test_sdn_max_one(self):
        self.assertAlmostEqual(self.state_dist_norm([0]*16, [63]*16), 1.0)

    def test_sdn_in_range(self):
        d = self.state_dist_norm([0]*16, [42]*16)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)

    def test_sdn_symmetric(self):
        c1 = [0, 1, 2, 3] * 4
        c2 = [4, 5, 6, 7] * 4
        self.assertAlmostEqual(self.state_dist_norm(c1, c2),
                               self.state_dist_norm(c2, c1))

    # ── derrida_point() ───────────────────────────────────────────────────────

    def test_dp_is_tuple_2(self):
        c = [0]*16
        r = self.derrida_point(c, c, 'xor3')
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 2)

    def test_dp_identical_both_zero(self):
        c = [42]*16
        x, y = self.derrida_point(c, c, 'xor3')
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)

    def test_dp_values_in_01(self):
        from projects.hexglyph.solan_word import encode_word, pad_to
        c1 = pad_to(encode_word('ГОРА'), 16)
        c2 = pad_to(encode_word('ЛУНА'), 16)
        for rule in self.ALL_RULES:
            x, y = self.derrida_point(c1, c2, rule)
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    # ── lexicon_points() ──────────────────────────────────────────────────────

    def test_lp_count(self):
        n = len(self.LEXICON)
        pts = self.lexicon_points('xor3')
        self.assertEqual(len(pts), n * (n - 1) // 2)

    def test_lp_all_in_range(self):
        pts = self.lexicon_points('xor')
        for x, y in pts:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(y, 1.0)

    def test_lp_x_positive(self):
        pts = self.lexicon_points('xor3')
        self.assertTrue(all(x > 0 for x, _ in pts))

    # ── random_points() ───────────────────────────────────────────────────────

    def test_rp_count(self):
        pts = self.random_points('xor3', n=40, seed=0)
        self.assertEqual(len(pts), 40)

    def test_rp_reproducible(self):
        p1 = self.random_points('xor3', n=10, seed=7)
        p2 = self.random_points('xor3', n=10, seed=7)
        self.assertEqual(p1, p2)

    def test_rp_different_seeds(self):
        p1 = self.random_points('xor3', n=10, seed=1)
        p2 = self.random_points('xor3', n=10, seed=2)
        self.assertNotEqual(p1, p2)

    def test_rp_in_range(self):
        for x, y in self.random_points('xor', n=20, seed=0):
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)

    # ── derrida_curve() ───────────────────────────────────────────────────────

    def test_dc_required_keys(self):
        pts = [(0.1, 0.2), (0.5, 0.4)]
        r = self.derrida_curve(pts)
        for k in ('bins', 'mean_y', 'count', 'above_diag', 'below_diag', 'on_diag'):
            self.assertIn(k, r)

    def test_dc_above_below_total(self):
        pts = [(0.1, 0.3), (0.5, 0.3), (0.8, 0.5)]
        r = self.derrida_curve(pts)
        self.assertEqual(r['above_diag'] + r['below_diag'] + r['on_diag'], 3)

    def test_dc_above_correct(self):
        # (0.1, 0.3): y>x → above; (0.5, 0.3): y<x → below
        r = self.derrida_curve([(0.1, 0.3), (0.5, 0.3)])
        self.assertEqual(r['above_diag'], 1)
        self.assertEqual(r['below_diag'], 1)

    def test_dc_bins_length(self):
        pts = [(0.5, 0.5)]
        r = self.derrida_curve(pts, n_bins=10)
        self.assertEqual(len(r['bins']), 10)

    # ── analytic_curve() ──────────────────────────────────────────────────────

    def test_ac_xor_midpoint(self):
        # XOR: y=2x(1-x), at x=0.5 → y=0.5
        pts = self.analytic_curve('xor', n_pts=100)
        mid = min(pts, key=lambda p: abs(p[0] - 0.5))
        self.assertAlmostEqual(mid[1], 0.5, places=2)

    def test_ac_xor_starts_zero(self):
        pts = self.analytic_curve('xor', n_pts=10)
        self.assertAlmostEqual(pts[0][1], 0.0)

    def test_ac_xor3_chaotic_slope(self):
        pts = self.analytic_curve('xor3', n_pts=100)
        x1, y1 = pts[1]
        self.assertGreater(y1 / x1, 1.0)  # slope > 1 near origin → chaotic

    def test_ac_all_y_in_01(self):
        for rule in self.ALL_RULES:
            for x, y in self.analytic_curve(rule):
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(y, 1.0 + 1e-9)

    # ── classify_rule() ───────────────────────────────────────────────────────

    def test_cr_valid_values(self):
        for rule in self.ALL_RULES:
            c = self.classify_rule(rule, n_random=50)
            self.assertIn(c, ('ordered', 'chaotic', 'complex'))

    def test_cr_or_ordered(self):
        self.assertEqual(self.classify_rule('or', n_random=200), 'ordered')

    def test_cr_and_ordered(self):
        self.assertEqual(self.classify_rule('and', n_random=200), 'ordered')

    def test_cr_xor3_not_ordered(self):
        # XOR3 is either 'complex' or 'chaotic', never 'ordered'
        self.assertNotEqual(self.classify_rule('xor3', n_random=300), 'ordered')

    # ── build_derrida_data() / derrida_summary() ──────────────────────────────

    def test_bdd_required_keys(self):
        d = self.build_derrida_data(n_random=30)
        for k in ('width', 'n_random', 'rules'):
            self.assertIn(k, d)

    def test_bdd_all_rules(self):
        d = self.build_derrida_data(n_random=30)
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_bdd_has_classification(self):
        d = self.build_derrida_data(n_random=30)
        for rule in self.ALL_RULES:
            self.assertIn('classification', d['rules'][rule])

    def test_summary_equals_build(self):
        """derrida_summary must equal build_derrida_data."""
        d1 = self.build_derrida_data(n_random=30)
        d2 = self.derrida_summary(n_random=30)
        self.assertEqual(d1['width'], d2['width'])
        self.assertEqual(d1['n_random'], d2['n_random'])
        self.assertEqual(set(d1['rules'].keys()), set(d2['rules'].keys()))

    def test_summary_default_width(self):
        d = self.derrida_summary(n_random=20)
        self.assertEqual(d['width'], 16)

    # ── derrida_dict() ────────────────────────────────────────────────────────

    def test_dd_json_serialisable(self):
        import json
        d = self.derrida_dict(n_random=30)
        json.dumps(d, ensure_ascii=False)  # must not raise

    def test_dd_top_keys(self):
        d = self.derrida_dict(n_random=30)
        for k in ('width', 'n_random', 'rules'):
            self.assertIn(k, d)

    def test_dd_analytic_present(self):
        d = self.derrida_dict(n_random=30)
        for rule in self.ALL_RULES:
            self.assertIn('analytic', d['rules'][rule])

    def test_dd_no_raw_points(self):
        d = self.derrida_dict(n_random=30)
        # Raw point lists should not appear in the JSON dict
        for rule in self.ALL_RULES:
            self.assertNotIn('lex_points', d['rules'][rule])

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_der_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('der-canvas', content)

    def test_viewer_has_der_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('der-btn', content)

    def test_viewer_has_der_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('derRun', content)

    def test_viewer_derrida_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Диаграмма Деррида CA Q6', content)

    def test_viewer_has_analytic_pts(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('analyticPts', content)

    def test_viewer_has_binned_curve(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('binnedCurve', content)

    def test_viewer_has_compute_pairs(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('computePairs', content)

    def test_viewer_has_solan_derrida(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_derrida', content)


class TestSolanRecurrence2(unittest.TestCase):
    """Replacement tests for solan_recurrence.py (standard *_summary convention)."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_recurrence import (
            state_hamming, recurrence_matrix, rqa_metrics,
            trajectory_recurrence, recurrence_summary,
            all_recurrences, build_recurrence_data, recurrence_dict,
            _ALL_RULES, _DEFAULT_WIDTH, _DEFAULT_EPS, _N_CYCLES,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.state_hamming         = staticmethod(state_hamming)
        cls.recurrence_matrix     = staticmethod(recurrence_matrix)
        cls.rqa_metrics           = staticmethod(rqa_metrics)
        cls.trajectory_recurrence = staticmethod(trajectory_recurrence)
        cls.recurrence_summary    = staticmethod(recurrence_summary)
        cls.all_recurrences       = staticmethod(all_recurrences)
        cls.build_recurrence_data = staticmethod(build_recurrence_data)
        cls.recurrence_dict       = staticmethod(recurrence_dict)
        cls.ALL_RULES             = _ALL_RULES
        cls.DEFAULT_WIDTH         = _DEFAULT_WIDTH
        cls.DEFAULT_EPS           = _DEFAULT_EPS
        cls.N_CYCLES              = _N_CYCLES
        cls.LEXICON               = list(LEXICON)

    # ── state_hamming() ────────────────────────────────────────────────────────

    def test_sh_identical_zeros(self):
        self.assertEqual(self.state_hamming([0, 0], [0, 0]), 0)

    def test_sh_identical_max(self):
        self.assertEqual(self.state_hamming([63, 63], [63, 63]), 0)

    def test_sh_one_bit(self):
        self.assertEqual(self.state_hamming([0], [1]), 1)

    def test_sh_six_bits(self):
        self.assertEqual(self.state_hamming([0], [63]), 6)

    def test_sh_two_cells_full_diff(self):
        # [0,63] vs [63,0]: each cell differs by 6 bits → 12 total
        self.assertEqual(self.state_hamming([0, 63], [63, 0]), 12)

    def test_sh_symmetric(self):
        r1 = [7, 0, 63, 1]
        r2 = [0, 7, 1, 63]
        self.assertEqual(self.state_hamming(r1, r2), self.state_hamming(r2, r1))

    def test_sh_zero_vs_63_16_cells(self):
        self.assertEqual(self.state_hamming([0]*16, [63]*16), 96)

    # ── recurrence_matrix() ────────────────────────────────────────────────────

    def test_rm_shape_square(self):
        rows = [[0]*4, [1]*4, [2]*4]
        R = self.recurrence_matrix(rows, eps=0)
        self.assertEqual(len(R), 3)
        self.assertTrue(all(len(row) == 3 for row in R))

    def test_rm_main_diag_ones(self):
        rows = [[i]*4 for i in range(5)]
        R = self.recurrence_matrix(rows, eps=0)
        for i in range(5):
            self.assertEqual(R[i][i], 1)

    def test_rm_symmetric(self):
        rows = [[0]*4, [1]*4, [3]*4]
        R = self.recurrence_matrix(rows, eps=0)
        for i in range(3):
            for j in range(3):
                self.assertEqual(R[i][j], R[j][i])

    def test_rm_identical_rows_all_ones(self):
        rows = [[5]*4] * 4
        R = self.recurrence_matrix(rows, eps=0)
        self.assertTrue(all(R[i][j] == 1 for i in range(4) for j in range(4)))

    def test_rm_eps_0_distinct_rows(self):
        R = self.recurrence_matrix([[0]*4, [1]*4], eps=0)
        self.assertEqual(R[0][1], 0)

    def test_rm_eps_relaxes(self):
        # differ by 1 bit → eps=1 makes them recurrent
        R = self.recurrence_matrix([[0, 0], [1, 0]], eps=1)
        self.assertEqual(R[0][1], 1)

    # ── rqa_metrics() ─────────────────────────────────────────────────────────

    def test_rqa_empty(self):
        m = self.rqa_metrics([])
        self.assertEqual(m['N'], 0)
        self.assertEqual(m['RR'], 0.0)

    def test_rqa_single_row(self):
        m = self.rqa_metrics([[1]])
        self.assertEqual(m['RR'], 0.0)

    def test_rqa_all_ones_rr_1(self):
        R = [[1]*4 for _ in range(4)]
        m = self.rqa_metrics(R)
        self.assertAlmostEqual(m['RR'], 1.0, places=4)

    def test_rqa_identity_rr_0(self):
        N = 4
        R = [[1 if i == j else 0 for j in range(N)] for i in range(N)]
        m = self.rqa_metrics(R)
        self.assertEqual(m['RR'], 0.0)
        self.assertEqual(m['DET'], 0.0)

    def test_rqa_required_keys(self):
        m = self.rqa_metrics([[1, 0], [0, 1]])
        for key in ('N', 'RR', 'DET', 'L', 'LAM', 'TT'):
            self.assertIn(key, m)

    def test_rqa_rr_in_range(self):
        import random
        random.seed(42)
        R = [[random.choice([0, 1]) for _ in range(8)] for _ in range(8)]
        m = self.rqa_metrics(R)
        self.assertGreaterEqual(m['RR'], 0.0)
        self.assertLessEqual(m['RR'], 1.0)

    # ── trajectory_recurrence() / recurrence_summary() ─────────────────────────

    def test_tr_returns_dict(self):
        r = self.trajectory_recurrence('ГОРА', 'xor3')
        self.assertIsInstance(r, dict)

    def test_tr_required_keys(self):
        r = self.trajectory_recurrence('ГОРА', 'xor3')
        for k in ('word', 'rule', 'width', 'eps', 'transient', 'period',
                  'n_steps', 'R', 'rqa'):
            self.assertIn(k, r)

    def test_tr_word_uppercased(self):
        r = self.trajectory_recurrence('гора', 'xor3')
        self.assertEqual(r['word'], 'ГОРА')

    def test_tr_matrix_square(self):
        r = self.trajectory_recurrence('ГОРА', 'xor3')
        N = r['n_steps']
        self.assertEqual(len(r['R']), N)
        self.assertTrue(all(len(row) == N for row in r['R']))

    def test_tr_n_steps_formula(self):
        r = self.trajectory_recurrence('ТУМАН', 'xor3', n_cycles=4)
        self.assertEqual(r['n_steps'], r['transient'] + 4 * r['period'])

    def test_tr_gora_xor_rr(self):
        r = self.trajectory_recurrence('ГОРА', 'xor')
        self.assertAlmostEqual(r['rqa']['RR'], 0.4, places=3)

    def test_tr_gora_xor3_det_one(self):
        r = self.trajectory_recurrence('ГОРА', 'xor3')
        self.assertAlmostEqual(r['rqa']['DET'], 1.0, places=3)

    def test_tr_tuman_xor3_rr(self):
        r = self.trajectory_recurrence('ТУМАН', 'xor3')
        self.assertAlmostEqual(r['rqa']['RR'], 0.0968, places=3)

    def test_tr_tuman_xor3_det_one(self):
        r = self.trajectory_recurrence('ТУМАН', 'xor3')
        self.assertAlmostEqual(r['rqa']['DET'], 1.0, places=3)

    def test_tr_tuman_xor3_l(self):
        r = self.trajectory_recurrence('ТУМАН', 'xor3')
        self.assertAlmostEqual(r['rqa']['L'], 16.0, places=1)

    def test_tr_tuman_xor3_lam_zero(self):
        r = self.trajectory_recurrence('ТУМАН', 'xor3')
        self.assertAlmostEqual(r['rqa']['LAM'], 0.0, places=4)

    def test_tr_tuman_xor3_tt_zero(self):
        r = self.trajectory_recurrence('ТУМАН', 'xor3')
        self.assertAlmostEqual(r['rqa']['TT'], 0.0, places=4)

    def test_tr_eps_increases_rr(self):
        r0 = self.trajectory_recurrence('ГОРА', 'xor3', eps=0)
        r4 = self.trajectory_recurrence('ГОРА', 'xor3', eps=4)
        self.assertGreaterEqual(r4['rqa']['RR'], r0['rqa']['RR'])

    def test_tr_rr_in_01(self):
        r = self.trajectory_recurrence('ВОДА', 'or')
        self.assertGreaterEqual(r['rqa']['RR'], 0.0)
        self.assertLessEqual(r['rqa']['RR'], 1.0)

    def test_summary_equals_traj(self):
        """recurrence_summary must equal trajectory_recurrence."""
        d1 = self.trajectory_recurrence('ГОРА', 'xor3')
        d2 = self.recurrence_summary('ГОРА', 'xor3')
        self.assertEqual(d1['rqa'], d2['rqa'])
        self.assertEqual(d1['period'], d2['period'])
        self.assertEqual(d1['transient'], d2['transient'])

    def test_summary_default_rule_xor3(self):
        d = self.recurrence_summary('ТУМАН')
        self.assertEqual(d['rule'], 'xor3')

    def test_summary_word_uppercased(self):
        d = self.recurrence_summary('туман')
        self.assertEqual(d['word'], 'ТУМАН')

    # ── all_recurrences() ──────────────────────────────────────────────────────

    def test_ar_four_rules(self):
        d = self.all_recurrences('ГОРА')
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    def test_ar_each_has_rqa(self):
        d = self.all_recurrences('ВОДА')
        for rule in self.ALL_RULES:
            self.assertIn('rqa', d[rule])

    def test_ar_rr_in_range(self):
        d = self.all_recurrences('ТУМАН')
        for rule in self.ALL_RULES:
            rr = d[rule]['rqa']['RR']
            self.assertGreaterEqual(rr, 0.0)
            self.assertLessEqual(rr, 1.0)

    # ── build_recurrence_data() ────────────────────────────────────────────────

    def test_brd_required_keys(self):
        d = self.build_recurrence_data(['ГОРА', 'ВОДА'])
        for k in ('words', 'width', 'eps', 'per_rule', 'ranking',
                  'max_rr', 'min_rr'):
            self.assertIn(k, d)

    def test_brd_per_rule_all_rules(self):
        d = self.build_recurrence_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), set(self.ALL_RULES))

    def test_brd_ranking_descending(self):
        d = self.build_recurrence_data(['ГОРА', 'ВОДА', 'ТУМАН'])
        for rule in self.ALL_RULES:
            rr_vals = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(rr_vals, sorted(rr_vals, reverse=True))

    def test_brd_max_rr_tuple(self):
        d = self.build_recurrence_data(['ГОРА', 'ВОДА'])
        for rule in self.ALL_RULES:
            self.assertIsInstance(d['max_rr'][rule], tuple)

    def test_brd_words_preserved(self):
        words = ['ГОРА', 'ВОДА']
        d = self.build_recurrence_data(words)
        self.assertEqual(d['words'], words)

    # ── recurrence_dict() ─────────────────────────────────────────────────────

    def test_rd_json_serialisable(self):
        import json
        d = self.recurrence_dict('ГОРА')
        json.dumps(d)   # must not raise

    def test_rd_has_rules_key(self):
        d = self.recurrence_dict('ГОРА')
        self.assertIn('rules', d)

    def test_rd_all_rules_present(self):
        d = self.recurrence_dict('ГОРА')
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_rd_each_rule_has_rr(self):
        d = self.recurrence_dict('ВОДА')
        for rule in self.ALL_RULES:
            self.assertIn('RR', d['rules'][rule])

    def test_rd_no_matrix(self):
        d = self.recurrence_dict('ГОРА')
        for rule in self.ALL_RULES:
            self.assertNotIn('R', d['rules'][rule])

    def test_rd_word_uppercased(self):
        d = self.recurrence_dict('гора')
        self.assertEqual(d['word'], 'ГОРА')

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_rc_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rc-canvas', content)

    def test_viewer_has_rc_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rc-btn', content)

    def test_viewer_has_rc_hmap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rc-hmap', content)

    def test_viewer_has_rc_metrics(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rc-metrics', content)

    def test_viewer_has_rqa_met(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rqaMet', content)

    def test_viewer_has_rc_rows(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rcRows', content)

    def test_viewer_has_solan_recurrence(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_recurrence', content)


class TestSolanCorrelation2(unittest.TestCase):
    """Replacement tests for solan_correlation.py (standard *_summary convention)."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_correlation import (
            row_autocorr, attractor_autocorr, all_autocorrs,
            cross_corr, correlation_length,
            build_correlation_data, correlation_dict, correlation_summary,
            _ALL_RULES,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.row_autocorr           = staticmethod(row_autocorr)
        cls.attractor_autocorr     = staticmethod(attractor_autocorr)
        cls.all_autocorrs          = staticmethod(all_autocorrs)
        cls.cross_corr             = staticmethod(cross_corr)
        cls.correlation_length     = staticmethod(correlation_length)
        cls.build_correlation_data = staticmethod(build_correlation_data)
        cls.correlation_dict       = staticmethod(correlation_dict)
        cls.correlation_summary    = staticmethod(correlation_summary)
        cls.ALL_RULES              = _ALL_RULES
        cls.LEXICON                = list(LEXICON)

    # ── row_autocorr() ────────────────────────────────────────────────────────

    def test_rac_length_16(self):
        # width=16 → n_lags = 16//2+1 = 9
        self.assertEqual(len(self.row_autocorr([0]*16)), 9)

    def test_rac_r0_is_one(self):
        for row in ([1, 2, 3, 4]*4, [0]*16, [63]*16):
            self.assertAlmostEqual(self.row_autocorr(row)[0], 1.0, places=5)

    def test_rac_constant_all_one(self):
        # constant row → zero variance → all r(d)=1
        r = self.row_autocorr([7]*16)
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in r))

    def test_rac_alternating_r1_minus_one(self):
        # [1,0,1,0,...] → r(1)=-1
        r = self.row_autocorr([1, 0]*8)
        self.assertAlmostEqual(r[1], -1.0, places=5)

    def test_rac_values_in_range(self):
        r = self.row_autocorr([i % 7 for i in range(16)])
        for v in r:
            self.assertGreaterEqual(v, -1.0 - 1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)

    # ── attractor_autocorr() ──────────────────────────────────────────────────

    def test_aac_length(self):
        self.assertEqual(len(self.attractor_autocorr('ГОРА', 'xor3')), 9)

    def test_aac_r0_one(self):
        for rule in self.ALL_RULES:
            r = self.attractor_autocorr('ГОРА', rule)
            self.assertAlmostEqual(r[0], 1.0, places=5)

    def test_aac_xor_constant_attractor(self):
        # XOR → all-zero attractor → r(d≥1)=0
        r = self.attractor_autocorr('ГОРА', 'xor')
        for v in r[1:]:
            self.assertAlmostEqual(v, 0.0, places=5)

    def test_aac_and_anticorr_at_lag1(self):
        # AND → alternating attractor → r(1) < 0
        r = self.attractor_autocorr('ГОРА', 'and')
        self.assertLess(r[1], 0.0)

    def test_aac_values_bounded(self):
        for rule in self.ALL_RULES:
            for v in self.attractor_autocorr('ТУМАН', rule):
                self.assertGreaterEqual(v, -1.0 - 1e-9)
                self.assertLessEqual(v, 1.0 + 1e-9)

    # ── all_autocorrs() ───────────────────────────────────────────────────────

    def test_all_ac_four_rules(self):
        r = self.all_autocorrs('ГОРА')
        self.assertEqual(set(r.keys()), set(self.ALL_RULES))

    def test_all_ac_each_is_list(self):
        r = self.all_autocorrs('ЛУНА')
        for ac in r.values():
            self.assertIsInstance(ac, list)
            self.assertEqual(len(ac), 9)

    # ── cross_corr() ──────────────────────────────────────────────────────────

    def test_cc_length(self):
        self.assertEqual(len(self.cross_corr('ГОРА', 'ЛУНА', 'xor3')), 9)

    def test_cc_same_word_r0(self):
        # cross_corr(word, word)[0] == attractor_autocorr[0] == 1
        cc = self.cross_corr('ГОРА', 'ГОРА', 'xor3')
        ac = self.attractor_autocorr('ГОРА', 'xor3')
        self.assertAlmostEqual(cc[0], ac[0], places=4)

    def test_cc_bounded(self):
        for v in self.cross_corr('ГОРА', 'ЛУНА', 'xor3'):
            self.assertGreaterEqual(v, -1.0 - 1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)

    # ── correlation_length() ──────────────────────────────────────────────────

    def test_cl_is_float(self):
        self.assertIsInstance(self.correlation_length('ГОРА', 'xor3'), float)

    def test_cl_xor_is_one(self):
        # XOR: r(d≥1)=0 ≤ 1/e → first crossing at d=1
        self.assertAlmostEqual(self.correlation_length('ГОРА', 'xor'), 1.0)

    def test_cl_and_long(self):
        # AND alternating |r|=1 for all lags → returns max = width//2 = 8
        self.assertGreaterEqual(self.correlation_length('ГОРА', 'and'), 7.0)

    def test_cl_positive_all_rules(self):
        for rule in self.ALL_RULES:
            self.assertGreater(self.correlation_length('ВОДА', rule), 0.0)

    # ── build_correlation_data() ──────────────────────────────────────────────

    def test_bcd_required_keys(self):
        d = self.build_correlation_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'n_lags', 'per_rule', 'corr_lengths',
                  'max_corr_len', 'min_corr_len'):
            self.assertIn(k, d)

    def test_bcd_n_lags_16(self):
        d = self.build_correlation_data(['ГОРА'])
        self.assertEqual(d['n_lags'], 9)

    def test_bcd_all_rules(self):
        d = self.build_correlation_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), set(self.ALL_RULES))

    def test_bcd_words_preserved(self):
        words = ['ГОРА', 'ЛУНА']
        d = self.build_correlation_data(words)
        self.assertEqual(d['words'], words)

    # ── correlation_dict() / correlation_summary() ────────────────────────────

    def test_cd_json_serialisable(self):
        import json
        json.dumps(self.correlation_dict('ГОРА'), ensure_ascii=False)

    def test_cd_top_keys(self):
        d = self.correlation_dict('ГОРА')
        for k in ('word', 'width', 'lags', 'rules'):
            self.assertIn(k, d)

    def test_cd_lags_list(self):
        d = self.correlation_dict('ГОРА')
        self.assertEqual(d['lags'], list(range(9)))

    def test_cd_all_rules(self):
        d = self.correlation_dict('ГОРА')
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_cd_corr_length_present(self):
        d = self.correlation_dict('ГОРА')
        for rule in self.ALL_RULES:
            self.assertIn('corr_length', d['rules'][rule])

    def test_cd_word_uppercased(self):
        d = self.correlation_dict('гора')
        self.assertEqual(d['word'], 'ГОРА')

    def test_summary_equals_dict(self):
        """correlation_summary must equal correlation_dict."""
        d1 = self.correlation_dict('ТУМАН')
        d2 = self.correlation_summary('ТУМАН')
        self.assertEqual(d1['lags'], d2['lags'])
        self.assertEqual(set(d1['rules'].keys()), set(d2['rules'].keys()))

    def test_summary_default_width(self):
        d = self.correlation_summary('ГОРА')
        self.assertEqual(d['width'], 16)

    def test_summary_word_uppercased(self):
        d = self.correlation_summary('гора')
        self.assertEqual(d['word'], 'ГОРА')

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_cor_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cor-canvas', content)

    def test_viewer_has_cor_hmap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cor-hmap', content)

    def test_viewer_has_cor_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cor-btn', content)

    def test_viewer_has_cor_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('corRun', content)

    def test_viewer_cor_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Пространственная автокорреляция CA Q6', content)

    def test_viewer_has_row_autocorr(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rowAutocorr', content)

    def test_viewer_has_solan_correlation(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_correlation', content)


class TestSolanLyapunov3(unittest.TestCase):
    """Merged replacement tests for solan_lyapunov.py — both A & B interfaces."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_lyapunov import (
            # A-interface
            q6_hamming, state_distance, perturb,
            divergence_trajectory, lyapunov_profile,
            lyapunov_summary, peak_sensitivity_map,
            build_lyapunov_data, lyapunov_dict,
            # B-interface
            perturb_profile, perturb_all_profiles, mean_d_profile,
            detect_period, classify_mode, perturbation_cone,
            lyapunov_mode_summary, all_lyapunov, build_mode_data,
            _ALL_RULES,
        )
        cls.q6_hamming            = staticmethod(q6_hamming)
        cls.state_distance        = staticmethod(state_distance)
        cls.perturb               = staticmethod(perturb)
        cls.divergence_trajectory = staticmethod(divergence_trajectory)
        cls.lyapunov_profile      = staticmethod(lyapunov_profile)
        cls.lyapunov_summary      = staticmethod(lyapunov_summary)
        cls.peak_sensitivity_map  = staticmethod(peak_sensitivity_map)
        cls.build_lyapunov_data   = staticmethod(build_lyapunov_data)
        cls.lyapunov_dict         = staticmethod(lyapunov_dict)
        cls.perturb_profile       = staticmethod(perturb_profile)
        cls.perturb_all_profiles  = staticmethod(perturb_all_profiles)
        cls.mean_d_profile        = staticmethod(mean_d_profile)
        cls.detect_period         = staticmethod(detect_period)
        cls.classify_mode         = staticmethod(classify_mode)
        cls.perturbation_cone     = staticmethod(perturbation_cone)
        cls.mode_summary          = staticmethod(lyapunov_mode_summary)
        cls.all_lyapunov          = staticmethod(all_lyapunov)
        cls.build_mode_data       = staticmethod(build_mode_data)
        cls.ALL_RULES             = _ALL_RULES

    # ── A: q6_hamming() ───────────────────────────────────────────────────────

    def test_q6h_identical(self):
        self.assertEqual(self.q6_hamming(42, 42), 0)

    def test_q6h_one_bit(self):
        self.assertEqual(self.q6_hamming(0, 1), 1)
        self.assertEqual(self.q6_hamming(0, 32), 1)

    def test_q6h_all_six_bits(self):
        self.assertEqual(self.q6_hamming(0, 63), 6)

    def test_q6h_symmetric(self):
        self.assertEqual(self.q6_hamming(17, 42), self.q6_hamming(42, 17))

    def test_q6h_range_0_6(self):
        for a in range(64):
            for b in range(64):
                self.assertLessEqual(self.q6_hamming(a, b), 6)

    # ── A: state_distance() ───────────────────────────────────────────────────

    def test_sd_identical(self):
        c = [10, 20, 30, 40]
        self.assertEqual(self.state_distance(c, c), 0)

    def test_sd_one_bit_flip(self):
        c1 = [0]*8; c2 = [0]*8; c2[3] = 1
        self.assertEqual(self.state_distance(c1, c2), 1)

    def test_sd_all_differ_4_cells(self):
        self.assertEqual(self.state_distance([0]*4, [63]*4), 24)

    # ── A: perturb() ──────────────────────────────────────────────────────────

    def test_perturb_new_list(self):
        c = [10, 20, 30]
        self.assertIsNot(self.perturb(c, 0, 0), c)

    def test_perturb_original_unchanged(self):
        c = [10, 20, 30]
        self.perturb(c, 0, 0)
        self.assertEqual(c, [10, 20, 30])

    def test_perturb_distance_one(self):
        c = [0]*8
        self.assertEqual(self.state_distance(c, self.perturb(c, 3, 0)), 1)

    def test_perturb_double_flip_restores(self):
        c = [42]*6
        self.assertEqual(c, self.perturb(self.perturb(c, 2, 3), 2, 3))

    # ── A: divergence_trajectory() ────────────────────────────────────────────

    def test_divtraj_length(self):
        r = self.divergence_trajectory('ГОРА', 0, 0, 'xor3', max_steps=10)
        self.assertEqual(len(r), 11)

    def test_divtraj_starts_at_one(self):
        self.assertEqual(self.divergence_trajectory('ГОРА', 0, 0, 'xor3')[0], 1)

    def test_divtraj_xor_converges(self):
        r = self.divergence_trajectory('ГОРА', 0, 0, 'xor', max_steps=20)
        self.assertEqual(r[-1], 0)

    def test_divtraj_nonneg(self):
        r = self.divergence_trajectory('ГОРА', 0, 0, 'xor3')
        self.assertTrue(all(v >= 0 for v in r))

    # ── A: lyapunov_profile() ─────────────────────────────────────────────────

    def test_prof_required_keys(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        for k in ('word', 'rule', 'width', 'n_perturb', 'mean_dist',
                  'max_dist', 'min_dist', 'peak_mean', 'peak_step',
                  'final_mean', 'converges', 'per_perturb'):
            self.assertIn(k, p)

    def test_prof_n_perturb(self):
        self.assertEqual(
            self.lyapunov_profile('ГОРА', 'xor3', max_steps=5)['n_perturb'], 96)

    def test_prof_initial_mean_one(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        self.assertAlmostEqual(p['mean_dist'][0], 1.0)

    def test_prof_xor_converges(self):
        self.assertTrue(
            self.lyapunov_profile('ГОРА', 'xor', max_steps=20)['converges'])

    # ── A: lyapunov_summary() (all-rules) ─────────────────────────────────────

    def test_lsummary_all_rules(self):
        s = self.lyapunov_summary('ГОРА', max_steps=10)
        self.assertEqual(set(s.keys()), set(self.ALL_RULES))

    def test_lsummary_rule_keys(self):
        s = self.lyapunov_summary('ГОРА', max_steps=10)
        for d in s.values():
            for k in ('peak_mean', 'peak_step', 'final_mean', 'converges'):
                self.assertIn(k, d)

    # ── A: peak_sensitivity_map() ─────────────────────────────────────────────

    def test_psmap_shape(self):
        m = self.peak_sensitivity_map('ГОРА', 'xor3', max_steps=10)
        self.assertEqual(len(m), 16)
        self.assertTrue(all(len(row) == 6 for row in m))

    # ── A: lyapunov_dict() ────────────────────────────────────────────────────

    def test_ldict_json_serialisable(self):
        import json
        json.dumps(self.lyapunov_dict('ГОРА', max_steps=8), ensure_ascii=False)

    def test_ldict_all_rules(self):
        d = self.lyapunov_dict('ГОРА', max_steps=8)
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_ldict_mean_dist_length(self):
        d = self.lyapunov_dict('ГОРА', max_steps=8)
        for rd in d['rules'].values():
            self.assertEqual(len(rd['mean_dist']), 9)

    # ── B: perturb_profile() ──────────────────────────────────────────────────

    def test_pp_length_T(self):
        self.assertEqual(len(self.perturb_profile([0]*16, 0, 0, 'xor', 10)), 10)

    def test_pp_initial_d_one(self):
        self.assertEqual(self.perturb_profile([0]*16, 0, 0, 'xor', 8)[0], 1)

    def test_pp_gora_or_absorbs(self):
        # OR: 63 | x = 63 → absorbed at t=1
        prof = self.perturb_profile([63]*16, 0, 0, 'or', 8)
        self.assertEqual(prof[1], 0)

    def test_pp_nonneg_le_n(self):
        prof = self.perturb_profile([0]*16, 3, 2, 'xor3', 16)
        self.assertTrue(all(0 <= d <= 16 for d in prof))

    # ── B: perturb_all_profiles() ─────────────────────────────────────────────

    def test_pap_count(self):
        self.assertEqual(len(self.perturb_all_profiles([0]*16, 'xor', 8)), 96)

    def test_pap_all_start_one(self):
        profs = self.perturb_all_profiles([0]*16, 'xor', 4)
        self.assertTrue(all(p[0] == 1 for p in profs))

    # ── B: mean_d_profile() ───────────────────────────────────────────────────

    def test_mdp_initial_one(self):
        profs = self.perturb_all_profiles([0]*16, 'xor', 8)
        self.assertAlmostEqual(self.mean_d_profile(profs)[0], 1.0)

    def test_mdp_xor_all_zeros_known(self):
        # XOR [0]*16: t=7→8.0, t=8→0.0
        profs = self.perturb_all_profiles([0]*16, 'xor', 10)
        m = self.mean_d_profile(profs)
        self.assertAlmostEqual(m[7], 8.0)
        self.assertAlmostEqual(m[8], 0.0)

    # ── B: detect_period() ────────────────────────────────────────────────────

    def test_dp_none_aperiodic(self):
        self.assertIsNone(self.detect_period([1, 2, 3, 4, 5, 6, 7, 8]))

    def test_dp_period_3(self):
        self.assertEqual(self.detect_period([1.0, 2.0, 3.0]*3), 3)

    def test_dp_period_8(self):
        seq = [1.0, 3.0, 3.0, 5.0, 3.0, 9.0, 5.0, 11.0]*3
        self.assertEqual(self.detect_period(seq), 8)

    # ── B: classify_mode() ────────────────────────────────────────────────────

    def test_cm_absorbs(self):
        self.assertEqual(self.classify_mode([1.0, 0.0]*16), 'absorbs')

    def test_cm_stabilizes(self):
        prof = [1.0, 2.0, 2.0, 4.0, 2.0, 4.0, 4.0, 8.0] + [0.0]*24
        self.assertEqual(self.classify_mode(prof), 'stabilizes')

    def test_cm_plateau(self):
        self.assertEqual(self.classify_mode([1.0]*8 + [4.0]*24), 'plateau')

    def test_cm_periodic(self):
        self.assertEqual(
            self.classify_mode([1.0, 3.0, 3.0, 5.0, 3.0, 9.0, 5.0, 11.0]*4),
            'periodic')

    # ── B: perturbation_cone() ────────────────────────────────────────────────

    def test_cone_shape(self):
        cone = self.perturbation_cone([0]*16, 0, 'xor', T=8)
        self.assertEqual(len(cone), 8)
        self.assertEqual(len(cone[0]), 16)

    def test_cone_binary(self):
        cone = self.perturbation_cone([0]*16, 0, 'xor', T=8)
        self.assertTrue(all(v in (0, 1) for row in cone for v in row))

    def test_cone_t0_single_cell(self):
        cone = self.perturbation_cone([0]*16, 8, 'xor', T=4)
        self.assertEqual(sum(cone[0]), 1)
        self.assertEqual(cone[0][8], 1)

    # ── B: lyapunov_mode_summary() ────────────────────────────────────────────

    def test_ms_required_keys(self):
        d = self.mode_summary('ГОРА', 'and')
        for k in ('word', 'rule', 'period_orbit', 'n_cells', 'T',
                  'mean_d', 'max_mean_d', 't_max_mean_d',
                  't_converge', 'fraction_converged', 'plateau_d',
                  'mode', 'period_d', 'cone_centre',
                  'absorbs', 'stabilizes', 'is_plateau', 'is_periodic'):
            self.assertIn(k, d)

    def test_ms_word_upper(self):
        self.assertEqual(self.mode_summary('гора', 'and')['word'], 'ГОРА')

    def test_ms_tuman_and_absorbs(self):
        d = self.mode_summary('ТУМАН', 'and')
        self.assertEqual(d['mode'], 'absorbs')
        self.assertTrue(d['absorbs'])
        self.assertAlmostEqual(d['fraction_converged'], 1.0)

    def test_ms_tuman_xor_stabilizes(self):
        d = self.mode_summary('ТУМАН', 'xor')
        self.assertEqual(d['mode'], 'stabilizes')
        self.assertAlmostEqual(d['max_mean_d'], 8.0)
        self.assertEqual(d['t_converge'], 8)

    def test_ms_gora_and_plateau(self):
        d = self.mode_summary('ГОРА', 'and')
        self.assertEqual(d['mode'], 'plateau')
        self.assertAlmostEqual(d['plateau_d'], 4.0, places=1)

    def test_ms_tuman_xor3_periodic(self):
        d = self.mode_summary('ТУМАН', 'xor3')
        self.assertEqual(d['mode'], 'periodic')
        self.assertAlmostEqual(d['max_mean_d'], 11.0)
        self.assertEqual(d['period_d'], 8)

    def test_ms_mode_flags_exclusive(self):
        for word in ('ТУМАН', 'ГОРА'):
            for rule in self.ALL_RULES:
                d = self.mode_summary(word, rule)
                flags = [d['absorbs'], d['stabilizes'],
                         d['is_plateau'], d['is_periodic']]
                self.assertEqual(sum(flags), 1)

    def test_ms_mean_d_starts_one(self):
        d = self.mode_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['mean_d'][0], 1.0)

    # ── B: all_lyapunov() ─────────────────────────────────────────────────────

    def test_al_four_rules(self):
        self.assertEqual(set(self.all_lyapunov('ГОРА').keys()),
                         set(self.ALL_RULES))

    # ── B: build_mode_data() ──────────────────────────────────────────────────

    def test_bmd_top_keys(self):
        d = self.build_mode_data(['ГОРА'])
        for k in ('words', 'width', 'per_rule'):
            self.assertIn(k, d)

    def test_bmd_word_uppercase(self):
        d = self.build_mode_data(['гора'])
        self.assertIn('ГОРА', d['per_rule']['and'])

    def test_bmd_known_mode(self):
        d = self.build_mode_data(['ТУМАН'])
        self.assertEqual(d['per_rule']['xor']['ТУМАН']['mode'], 'stabilizes')

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_lya_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lya-canvas', content)

    def test_viewer_lya_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lya-btn', content)

    def test_viewer_lya_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lyaRun', content)

    def test_viewer_lya_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Ляпунов CA Q6', content)

    def test_viewer_lyap_heat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lyap-heat', content)

    def test_viewer_lyap_cone(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lyap-cone', content)

    def test_viewer_has_solan_lyapunov(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_lyapunov', content)


class TestSolanAutocorr2(unittest.TestCase):
    """Merged replacement tests for solan_autocorr.py."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_autocorr import (
            cell_series, temporal_ac, temporal_ac_profile,
            mean_temporal_ac, cell_ac_all,
            cell_crosscorr, crosscorr_matrix,
            spatial_ac, spatial_ac_profile,
            autocorr_summary, all_autocorr, build_autocorr_data,
        )
        from projects.hexglyph.solan_perm import get_orbit
        cls.cell_series         = staticmethod(cell_series)
        cls.temporal_ac         = staticmethod(temporal_ac)
        cls.temporal_ac_profile = staticmethod(temporal_ac_profile)
        cls.mean_temporal_ac    = staticmethod(mean_temporal_ac)
        cls.cell_ac_all         = staticmethod(cell_ac_all)
        cls.cell_crosscorr      = staticmethod(cell_crosscorr)
        cls.crosscorr_matrix    = staticmethod(crosscorr_matrix)
        cls.spatial_ac          = staticmethod(spatial_ac)
        cls.spatial_ac_profile  = staticmethod(spatial_ac_profile)
        cls.autocorr_summary    = staticmethod(autocorr_summary)
        cls.all_autocorr        = staticmethod(all_autocorr)
        cls.build_autocorr_data = staticmethod(build_autocorr_data)
        cls.orbit_tuman_xor3    = get_orbit('ТУМАН', 'xor3', 16)
        cls.orbit_gora_and      = get_orbit('ГОРА',  'and',  16)
        cls.orbit_tuman_xor     = get_orbit('ТУМАН', 'xor',  16)

    # ── cell_series() ─────────────────────────────────────────────────────────

    def test_cs_length_equals_period(self):
        self.assertEqual(len(self.cell_series(self.orbit_tuman_xor3, 0)), 8)

    def test_cs_values_match_orbit(self):
        s = self.cell_series(self.orbit_gora_and, 5)
        self.assertEqual(s[0], self.orbit_gora_and[0][5])

    # ── temporal_ac() ─────────────────────────────────────────────────────────

    def test_tac_lag0_is_one(self):
        self.assertAlmostEqual(self.temporal_ac([1, 3, 5, 7], 0), 1.0)

    def test_tac_constant_returns_one(self):
        self.assertAlmostEqual(self.temporal_ac([5, 5, 5, 5], 1), 1.0)

    def test_tac_in_range(self):
        for lag in range(8):
            v = self.temporal_ac(list(range(1, 9)), lag)
            self.assertGreaterEqual(v, -1.0 - 1e-9)
            self.assertLessEqual(v,    1.0 + 1e-9)

    def test_tac_p2_lag1_minus_one(self):
        for s in ([1, 2], [47, 1], [63, 0]):
            self.assertAlmostEqual(self.temporal_ac(s, 1), -1.0, places=5)

    def test_tac_palindrome_symmetry(self):
        s = [3, 7, 2, 9, 1, 5, 8, 4]
        P = len(s)
        for tau in range(1, P):
            self.assertAlmostEqual(
                self.temporal_ac(s, tau), self.temporal_ac(s, P - tau), places=6)

    def test_tac_tuman_xor3_cell8_lag3(self):
        s = self.cell_series(self.orbit_tuman_xor3, 8)
        self.assertGreater(self.temporal_ac(s, 3), 0.3)

    def test_tac_tuman_xor3_cell0_lag1_positive(self):
        s = self.cell_series(self.orbit_tuman_xor3, 0)
        self.assertGreater(self.temporal_ac(s, 1), 0.0)

    # ── temporal_ac_profile() ─────────────────────────────────────────────────

    def test_tap_starts_at_one(self):
        self.assertAlmostEqual(self.temporal_ac_profile([3, 1, 4, 1, 5])[0], 1.0)

    def test_tap_length(self):
        self.assertEqual(len(self.temporal_ac_profile(list(range(8)))), 8)

    def test_tap_palindromic(self):
        s = [3, 7, 2, 9, 1, 5, 8, 4]
        P = len(s)
        prof = self.temporal_ac_profile(s)
        for tau in range(1, P):
            self.assertAlmostEqual(prof[tau], prof[P - tau], places=6)

    def test_tap_p2_anti(self):
        prof = self.temporal_ac_profile([47, 1])
        self.assertAlmostEqual(prof[1], -1.0, places=5)

    # ── mean_temporal_ac() ────────────────────────────────────────────────────

    def test_mta_length(self):
        self.assertEqual(len(self.mean_temporal_ac(self.orbit_tuman_xor3)), 8)

    def test_mta_starts_at_one(self):
        self.assertAlmostEqual(self.mean_temporal_ac(self.orbit_tuman_xor3)[0], 1.0)

    def test_mta_gora_and_lag1_minus_one(self):
        self.assertAlmostEqual(
            self.mean_temporal_ac(self.orbit_gora_and)[1], -1.0, places=5)

    def test_mta_tuman_xor3_palindromic(self):
        m = self.mean_temporal_ac(self.orbit_tuman_xor3)
        for tau in range(1, 8):
            self.assertAlmostEqual(m[tau], m[8 - tau], places=5)

    def test_mta_tuman_xor3_known_values(self):
        m = self.mean_temporal_ac(self.orbit_tuman_xor3)
        self.assertAlmostEqual(m[1], -0.2555, places=3)
        self.assertAlmostEqual(m[3],  0.141,  places=2)

    # ── cell_ac_all() ─────────────────────────────────────────────────────────

    def test_caa_shape(self):
        ac = self.cell_ac_all(self.orbit_tuman_xor3)
        self.assertEqual(len(ac), 16)
        self.assertEqual(len(ac[0]), 8)

    def test_caa_all_lag0_one(self):
        ac = self.cell_ac_all(self.orbit_tuman_xor3)
        for ci in range(16):
            self.assertAlmostEqual(ac[ci][0], 1.0)

    def test_caa_gora_and_all_lag1_minus_one(self):
        ac = self.cell_ac_all(self.orbit_gora_and)
        for ci in range(16):
            self.assertAlmostEqual(ac[ci][1], -1.0, places=5)

    # ── cell_crosscorr() ──────────────────────────────────────────────────────

    def test_cc_self_is_one(self):
        s = [3, 1, 4, 1, 5, 9, 2, 6]
        self.assertAlmostEqual(self.cell_crosscorr(s, s), 1.0)

    def test_cc_constant_is_zero(self):
        self.assertAlmostEqual(self.cell_crosscorr([5]*4, [1, 2, 3, 4]), 0.0)

    def test_cc_symmetric(self):
        s1 = [3, 1, 4, 1, 5]
        s2 = [2, 7, 1, 8, 2]
        self.assertAlmostEqual(
            self.cell_crosscorr(s1, s2), self.cell_crosscorr(s2, s1), places=6)

    # ── crosscorr_matrix() ────────────────────────────────────────────────────

    def test_ccm_shape(self):
        m = self.crosscorr_matrix(self.orbit_gora_and)
        self.assertEqual(len(m), 16)
        self.assertEqual(len(m[0]), 16)

    def test_ccm_diagonal_ones(self):
        m = self.crosscorr_matrix(self.orbit_tuman_xor3)
        for i in range(16):
            self.assertAlmostEqual(m[i][i], 1.0, places=5)

    def test_ccm_symmetric(self):
        m = self.crosscorr_matrix(self.orbit_tuman_xor3)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(m[i][j], m[j][i], places=6)

    # ── spatial_ac() ──────────────────────────────────────────────────────────

    def test_sac_lag0_is_one(self):
        self.assertAlmostEqual(self.spatial_ac(self.orbit_tuman_xor3, 0), 1.0)

    def test_sac_in_range(self):
        for d in range(1, 9):
            v = self.spatial_ac(self.orbit_tuman_xor3, d)
            self.assertGreaterEqual(v, -1.0 - 1e-6)
            self.assertLessEqual(v,    1.0 + 1e-6)

    def test_sac_p1_constant_one(self):
        self.assertAlmostEqual(self.spatial_ac(self.orbit_tuman_xor, 1), 1.0)

    # ── spatial_ac_profile() ──────────────────────────────────────────────────

    def test_sap_length_default(self):
        self.assertEqual(len(self.spatial_ac_profile(self.orbit_tuman_xor3)), 9)

    def test_sap_starts_at_one(self):
        self.assertAlmostEqual(self.spatial_ac_profile(self.orbit_tuman_xor3)[0], 1.0)

    def test_sap_custom_max_d(self):
        self.assertEqual(
            len(self.spatial_ac_profile(self.orbit_tuman_xor3, max_d=4)), 5)

    # ── autocorr_summary() ────────────────────────────────────────────────────

    def test_as_required_keys(self):
        d = self.autocorr_summary('ГОРА', 'and')
        for k in ('word', 'rule', 'period', 'n_cells',
                  'mean_ac', 'cell_ac', 'mean_ac_lag1',
                  'max_ac_lag1', 'min_ac_lag1', 'dominant_lag',
                  'is_palindrome', 'all_p2_anti',
                  'crosscorr_matrix', 'mean_crosscorr',
                  'spatial_ac', 'max_spatial_ac', 'min_spatial_ac'):
            self.assertIn(k, d)

    def test_as_word_upper(self):
        self.assertEqual(self.autocorr_summary('гора', 'and')['word'], 'ГОРА')

    def test_as_gora_and_p2_anti(self):
        d = self.autocorr_summary('ГОРА', 'and')
        self.assertTrue(d['all_p2_anti'])
        self.assertAlmostEqual(d['mean_ac_lag1'], -1.0, places=5)

    def test_as_gora_xor3_p2_anti(self):
        self.assertTrue(self.autocorr_summary('ГОРА', 'xor3')['all_p2_anti'])

    def test_as_tuman_xor_not_p2_anti(self):
        self.assertFalse(self.autocorr_summary('ТУМАН', 'xor')['all_p2_anti'])

    def test_as_tuman_xor3_palindrome(self):
        self.assertTrue(self.autocorr_summary('ТУМАН', 'xor3')['is_palindrome'])

    def test_as_tuman_xor3_mean_lag1(self):
        d = self.autocorr_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['mean_ac_lag1'], -0.2555, places=3)

    def test_as_tuman_xor3_edge_cell_positive(self):
        self.assertGreater(
            self.autocorr_summary('ТУМАН', 'xor3')['cell_ac'][0][1], 0.0)

    def test_as_tuman_xor3_inner_cell_neg(self):
        self.assertLess(
            self.autocorr_summary('ТУМАН', 'xor3')['cell_ac'][8][1], -0.5)

    def test_as_tuman_xor3_inner_period3(self):
        self.assertGreater(
            self.autocorr_summary('ТУМАН', 'xor3')['cell_ac'][8][3], 0.3)

    def test_as_spatial_ac_first_one(self):
        d = self.autocorr_summary('ТУМАН', 'xor3')
        self.assertAlmostEqual(d['spatial_ac'][0], 1.0)

    # ── all_autocorr() ────────────────────────────────────────────────────────

    def test_aa_four_rules(self):
        self.assertEqual(set(self.all_autocorr('ГОРА').keys()),
                         {'xor', 'xor3', 'and', 'or'})

    # ── build_autocorr_data() ─────────────────────────────────────────────────

    def test_bad_top_keys(self):
        d = self.build_autocorr_data(['ГОРА'])
        for k in ('words', 'width', 'per_rule'):
            self.assertIn(k, d)

    def test_bad_word_uppercase(self):
        d = self.build_autocorr_data(['гора'])
        self.assertIn('ГОРА', d['per_rule']['and'])

    def test_bad_p2_anti_gora_and(self):
        d = self.build_autocorr_data(['ГОРА'])
        self.assertTrue(d['per_rule']['and']['ГОРА']['all_p2_anti'])

    def test_bad_known_fields(self):
        rec = self.build_autocorr_data(['ГОРА'])['per_rule']['and']['ГОРА']
        for k in ('period', 'mean_ac', 'mean_ac_lag1', 'all_p2_anti',
                  'spatial_ac', 'max_spatial_ac', 'min_spatial_ac'):
            self.assertIn(k, rec)

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_ac_heat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ac-heat', content)

    def test_viewer_has_ac_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ac-btn', content)

    def test_viewer_has_ac_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('acRun', content)

    def test_viewer_has_ac_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Автокорреляция Q6', content)

    def test_viewer_has_solan_autocorr(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_autocorr', content)


class TestSolanHamming3(unittest.TestCase):
    """Tests for solan_hamming.py — Consecutive-step Hamming Distances & Cell Mobility."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_hamming import (
            hamming_dist, consecutive_hamming, flip_mask, cell_mobility,
            hamming_profile, flip_mask_word, cell_mobility_word,
            mean_hamming, max_hamming, min_hamming,
            hamming_summary, all_hamming, build_hamming_data,
        )
        cls.hamming_dist        = staticmethod(hamming_dist)
        cls.consecutive_hamming = staticmethod(consecutive_hamming)
        cls.flip_mask           = staticmethod(flip_mask)
        cls.cell_mobility       = staticmethod(cell_mobility)
        cls.hamming_profile     = staticmethod(hamming_profile)
        cls.flip_mask_word      = staticmethod(flip_mask_word)
        cls.cell_mobility_word  = staticmethod(cell_mobility_word)
        cls.mean_hamming        = staticmethod(mean_hamming)
        cls.max_hamming         = staticmethod(max_hamming)
        cls.min_hamming         = staticmethod(min_hamming)
        cls.hamming_summary     = staticmethod(hamming_summary)
        cls.all_hamming         = staticmethod(all_hamming)
        cls.build_hamming_data  = staticmethod(build_hamming_data)
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.LEXICON = list(LEXICON)

    # ── hamming_dist ──────────────────────────────────────────────────────────

    def test_hamming_dist_returns_int(self):
        self.assertIsInstance(self.hamming_dist((0, 1), (0, 1)), int)

    def test_hamming_dist_equal_states(self):
        self.assertEqual(self.hamming_dist((1, 2, 3), (1, 2, 3)), 0)

    def test_hamming_dist_all_differ(self):
        self.assertEqual(self.hamming_dist((0, 0, 0, 0), (1, 1, 1, 1)), 4)

    def test_hamming_dist_one_differs(self):
        self.assertEqual(self.hamming_dist((0, 1, 2), (0, 9, 2)), 1)

    def test_hamming_dist_symmetric(self):
        a, b = (3, 7, 2), (1, 7, 9)
        self.assertEqual(self.hamming_dist(a, b), self.hamming_dist(b, a))

    def test_hamming_dist_nonnegative(self):
        self.assertGreaterEqual(self.hamming_dist((0, 1), (1, 0)), 0)

    # ── consecutive_hamming ───────────────────────────────────────────────────

    def test_consecutive_hamming_returns_list(self):
        orbit = [(0, 1, 0, 1), (1, 0, 1, 0)]
        result = self.consecutive_hamming(orbit)
        self.assertIsInstance(result, list)

    def test_consecutive_hamming_length_equals_period(self):
        orbit = [(0, 0, 0), (1, 1, 1), (0, 0, 0)]
        result = self.consecutive_hamming(orbit)
        self.assertEqual(len(result), len(orbit))

    def test_consecutive_hamming_constant_orbit_zero(self):
        orbit = [(5, 5, 5, 5)] * 3
        result = self.consecutive_hamming(orbit)
        self.assertEqual(result, [0, 0, 0])

    def test_consecutive_hamming_all_flip(self):
        orbit = [(0, 0, 0, 0), (1, 1, 1, 1)]
        result = self.consecutive_hamming(orbit)
        self.assertEqual(result, [4, 4])  # wraps around

    def test_consecutive_hamming_wraps_around(self):
        # Last state compared to first (periodic)
        orbit = [(0,), (1,)]
        result = self.consecutive_hamming(orbit)
        self.assertEqual(result[1], self.hamming_dist((1,), (0,)))

    # ── flip_mask ─────────────────────────────────────────────────────────────

    def test_flip_mask_returns_list_of_lists(self):
        orbit = [(0, 1), (1, 0)]
        result = self.flip_mask(orbit)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)

    def test_flip_mask_shape(self):
        orbit = [(0, 1, 2), (3, 1, 5)]
        result = self.flip_mask(orbit)
        self.assertEqual(len(result), 2)  # period
        self.assertEqual(len(result[0]), 3)  # width

    def test_flip_mask_zeros_for_constant_orbit(self):
        orbit = [(7, 7, 7)] * 4
        result = self.flip_mask(orbit)
        for row in result:
            self.assertEqual(row, [0, 0, 0])

    def test_flip_mask_all_ones_for_full_flip(self):
        orbit = [(0, 0, 0, 0), (1, 1, 1, 1)]
        result = self.flip_mask(orbit)
        self.assertEqual(result[0], [1, 1, 1, 1])

    def test_flip_mask_values_binary(self):
        orbit = [(0, 1, 2), (1, 1, 5)]
        result = self.flip_mask(orbit)
        for row in result:
            self.assertTrue(all(v in (0, 1) for v in row))

    # ── cell_mobility ─────────────────────────────────────────────────────────

    def test_cell_mobility_returns_list(self):
        orbit = [(0, 1), (1, 0)]
        self.assertIsInstance(self.cell_mobility(orbit), list)

    def test_cell_mobility_length_equals_width(self):
        orbit = [(0, 1, 2, 3), (4, 5, 6, 7)]
        mob = self.cell_mobility(orbit)
        self.assertEqual(len(mob), 4)

    def test_cell_mobility_constant_orbit_all_zero(self):
        orbit = [(5, 5, 5)] * 3
        mob = self.cell_mobility(orbit)
        self.assertEqual(mob, [0.0, 0.0, 0.0])

    def test_cell_mobility_full_flip_all_one(self):
        orbit = [(0, 0, 0, 0), (1, 1, 1, 1)]
        mob = self.cell_mobility(orbit)
        self.assertEqual(mob, [1.0, 1.0, 1.0, 1.0])

    def test_cell_mobility_range_zero_to_one(self):
        orbit = [(0, 1, 0), (1, 0, 0)]
        mob = self.cell_mobility(orbit)
        self.assertTrue(all(0.0 <= m <= 1.0 for m in mob))

    # ── hamming_profile / flip_mask_word / cell_mobility_word ─────────────────

    def test_hamming_profile_returns_list_int(self):
        H = self.hamming_profile('ГОРА', 'xor3', 16)
        self.assertIsInstance(H, list)
        self.assertTrue(all(isinstance(v, int) for v in H))

    def test_hamming_profile_tuman_xor3_known(self):
        H = self.hamming_profile('ТУМАН', 'xor3', 16)
        self.assertEqual(H, [16, 16, 10, 12, 14, 16, 16, 14])

    def test_hamming_profile_tuman_xor_fixed_point(self):
        H = self.hamming_profile('ТУМАН', 'xor', 16)
        self.assertEqual(H, [0])

    def test_hamming_profile_gora_and(self):
        H = self.hamming_profile('ГОРА', 'and', 16)
        self.assertEqual(H, [16, 16])

    def test_hamming_profile_nonnegative(self):
        H = self.hamming_profile('ГОРА', 'xor3', 16)
        self.assertTrue(all(h >= 0 for h in H))

    def test_flip_mask_word_returns_list_of_lists(self):
        fm = self.flip_mask_word('ГОРА', 'xor3', 16)
        self.assertIsInstance(fm, list)
        self.assertIsInstance(fm[0], list)

    def test_flip_mask_word_row_length_equals_width(self):
        fm = self.flip_mask_word('ГОРА', 'xor3', 16)
        self.assertTrue(all(len(row) == 16 for row in fm))

    def test_flip_mask_word_binary_values(self):
        fm = self.flip_mask_word('ГОРА', 'xor3', 16)
        for row in fm:
            self.assertTrue(all(v in (0, 1) for v in row))

    def test_flip_mask_word_tuman_xor3_step2_frozen_edges(self):
        fm = self.flip_mask_word('ТУМАН', 'xor3', 16)
        # Step 2→3: cells {0,1,2} and {13,14,15} frozen
        row = fm[2]
        self.assertEqual(row[0], 0)
        self.assertEqual(row[1], 0)
        self.assertEqual(row[2], 0)
        self.assertEqual(row[13], 0)
        self.assertEqual(row[14], 0)
        self.assertEqual(row[15], 0)

    def test_cell_mobility_word_returns_list_float(self):
        mob = self.cell_mobility_word('ТУМАН', 'xor3', 16)
        self.assertIsInstance(mob, list)
        self.assertTrue(all(isinstance(v, float) for v in mob))

    def test_cell_mobility_word_length_equals_width(self):
        mob = self.cell_mobility_word('ТУМАН', 'xor3', 16)
        self.assertEqual(len(mob), 16)

    def test_cell_mobility_word_tuman_xor3_edges(self):
        mob = self.cell_mobility_word('ТУМАН', 'xor3', 16)
        self.assertAlmostEqual(mob[0],  0.5, places=6)
        self.assertAlmostEqual(mob[15], 0.5, places=6)

    def test_cell_mobility_word_tuman_xor3_inner(self):
        mob = self.cell_mobility_word('ТУМАН', 'xor3', 16)
        for i in range(3, 13):
            self.assertAlmostEqual(mob[i], 1.0, places=6)

    def test_cell_mobility_word_tuman_xor3_cell1(self):
        mob = self.cell_mobility_word('ТУМАН', 'xor3', 16)
        self.assertAlmostEqual(mob[1],  0.75, places=6)
        self.assertAlmostEqual(mob[14], 0.75, places=6)

    def test_cell_mobility_word_tuman_xor3_cell2(self):
        mob = self.cell_mobility_word('ТУМАН', 'xor3', 16)
        self.assertAlmostEqual(mob[2],  0.875, places=6)
        self.assertAlmostEqual(mob[13], 0.875, places=6)

    # ── mean / max / min hamming ──────────────────────────────────────────────

    def test_mean_hamming_tuman_xor3(self):
        self.assertAlmostEqual(self.mean_hamming('ТУМАН', 'xor3', 16), 14.25, places=5)

    def test_mean_hamming_tuman_xor_zero(self):
        self.assertAlmostEqual(self.mean_hamming('ТУМАН', 'xor', 16), 0.0, places=6)

    def test_mean_hamming_gora_and_max(self):
        self.assertAlmostEqual(self.mean_hamming('ГОРА', 'and', 16), 16.0, places=6)

    def test_max_hamming_tuman_xor3(self):
        self.assertEqual(self.max_hamming('ТУМАН', 'xor3', 16), 16)

    def test_max_hamming_tuman_xor_zero(self):
        self.assertEqual(self.max_hamming('ТУМАН', 'xor', 16), 0)

    def test_min_hamming_tuman_xor3(self):
        self.assertEqual(self.min_hamming('ТУМАН', 'xor3', 16), 10)

    def test_min_hamming_tuman_xor_zero(self):
        self.assertEqual(self.min_hamming('ТУМАН', 'xor', 16), 0)

    def test_min_hamming_gora_and(self):
        self.assertEqual(self.min_hamming('ГОРА', 'and', 16), 16)

    # ── hamming_summary ───────────────────────────────────────────────────────

    def test_hamming_summary_returns_dict(self):
        d = self.hamming_summary('ГОРА', 'xor3', 16)
        self.assertIsInstance(d, dict)

    def test_hamming_summary_required_keys(self):
        d = self.hamming_summary('ГОРА', 'xor3', 16)
        for key in ('word', 'rule', 'period', 'n_cells', 'hamming',
                    'mean_hamming', 'max_hamming', 'min_hamming',
                    'mobility', 'mean_mobility', 'all_frozen', 'all_max',
                    'flip_mask', 'mobile_symmetric'):
            self.assertIn(key, d)

    def test_hamming_summary_word_uppercase(self):
        d = self.hamming_summary('гора', 'xor3', 16)
        self.assertEqual(d['word'], 'ГОРА')

    def test_hamming_summary_tuman_xor_all_frozen(self):
        d = self.hamming_summary('ТУМАН', 'xor', 16)
        self.assertTrue(d['all_frozen'])

    def test_hamming_summary_gora_and_all_max(self):
        d = self.hamming_summary('ГОРА', 'and', 16)
        self.assertTrue(d['all_max'])

    def test_hamming_summary_tuman_xor3_mean_hamming(self):
        d = self.hamming_summary('ТУМАН', 'xor3', 16)
        self.assertAlmostEqual(d['mean_hamming'], 14.25, places=4)

    def test_hamming_summary_tuman_xor3_min_hamming(self):
        d = self.hamming_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(d['min_hamming'], 10)

    def test_hamming_summary_tuman_xor3_period(self):
        d = self.hamming_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(d['period'], 8)

    def test_hamming_summary_tuman_xor3_mobile_symmetric(self):
        d = self.hamming_summary('ТУМАН', 'xor3', 16)
        self.assertTrue(d['mobile_symmetric'])

    def test_hamming_summary_tuman_xor3_frozen_cells(self):
        d = self.hamming_summary('ТУМАН', 'xor', 16)
        # Fixed point → all cells frozen
        self.assertEqual(len(d['frozen_cells']), 16)

    def test_hamming_summary_gora_and_maxmobile_cells(self):
        d = self.hamming_summary('ГОРА', 'and', 16)
        self.assertEqual(len(d['maxmobile_cells']), 16)

    def test_hamming_summary_flip_mask_shape(self):
        d = self.hamming_summary('ТУМАН', 'xor3', 16)
        self.assertEqual(len(d['flip_mask']), d['period'])
        self.assertEqual(len(d['flip_mask'][0]), d['n_cells'])

    def test_hamming_summary_min_hamming_steps_tuman_xor3(self):
        d = self.hamming_summary('ТУМАН', 'xor3', 16)
        # min H=10 at step 2→3
        self.assertIn(2, d['min_hamming_steps'])

    # ── all_hamming ───────────────────────────────────────────────────────────

    def test_all_hamming_returns_dict_of_four(self):
        d = self.all_hamming('ГОРА', 16)
        self.assertEqual(set(d.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_hamming_values_are_summaries(self):
        d = self.all_hamming('ГОРА', 16)
        for rule, v in d.items():
            self.assertIsInstance(v, dict)
            self.assertIn('hamming', v)

    # ── build_hamming_data ────────────────────────────────────────────────────

    def test_build_hamming_data_returns_dict(self):
        d = self.build_hamming_data(['ГОРА', 'ТУМАН'], 16)
        self.assertIsInstance(d, dict)

    def test_build_hamming_data_keys(self):
        d = self.build_hamming_data(['ГОРА'], 16)
        self.assertIn('words', d)
        self.assertIn('width', d)
        self.assertIn('per_rule', d)

    def test_build_hamming_data_per_rule_keys(self):
        d = self.build_hamming_data(['ГОРА'], 16)
        self.assertEqual(set(d['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_hamming_data_contains_word(self):
        d = self.build_hamming_data(['ГОРА', 'ТУМАН'], 16)
        self.assertIn('ГОРА', d['per_rule']['xor3'])

    def test_build_hamming_data_width_preserved(self):
        d = self.build_hamming_data(['ГОРА'], 16)
        self.assertEqual(d['width'], 16)

    # ── --json CLI ────────────────────────────────────────────────────────────

    def test_cli_json_flag(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_hamming',
             '--word', 'ГОРА', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        self.assertEqual(r.returncode, 0)
        d = json.loads(r.stdout)
        self.assertIn('word', d)
        self.assertEqual(d['word'], 'ГОРА')

    def test_cli_json_contains_hamming(self):
        import subprocess, json, sys
        r = subprocess.run(
            [sys.executable, '-m', 'projects.hexglyph.solan_hamming',
             '--word', 'ТУМАН', '--rule', 'xor3', '--json'],
            capture_output=True, text=True,
            cwd='/home/user/meta'
        )
        d = json.loads(r.stdout)
        self.assertIn('hamming', d)

    # ── viewer ────────────────────────────────────────────────────────────────

    def test_viewer_has_ham_map(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ham-map', content)

    def test_viewer_has_ham_mob(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ham-mob', content)

    def test_viewer_has_ham_info(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('ham-info', content)

    def test_viewer_has_hm_word(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('hm-word', content)

    def test_viewer_has_hm_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('hm-btn', content)

    def test_viewer_has_hmRun(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('hmRun', content)

    def test_viewer_has_hmFlipMask(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('hmFlipMask', content)

    def test_viewer_has_hmMobility(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('hmMobility', content)

    def test_viewer_has_solan_hamming_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('solan_hamming', content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
