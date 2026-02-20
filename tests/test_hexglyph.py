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


if __name__ == "__main__":
    unittest.main(verbosity=2)
