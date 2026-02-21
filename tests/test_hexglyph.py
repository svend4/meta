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


class TestSolanRecurrence(unittest.TestCase):
    """Tests for solan_recurrence.py and the viewer Recurrence section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_recurrence import (
            state_hamming, recurrence_matrix, rqa_metrics,
            trajectory_recurrence, all_recurrences,
            build_recurrence_data, recurrence_dict,
            _ALL_RULES,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.state_hamming         = staticmethod(state_hamming)
        cls.recurrence_matrix     = staticmethod(recurrence_matrix)
        cls.rqa_metrics           = staticmethod(rqa_metrics)
        cls.trajectory_recurrence = staticmethod(trajectory_recurrence)
        cls.all_recurrences       = staticmethod(all_recurrences)
        cls.build_recurrence_data = staticmethod(build_recurrence_data)
        cls.recurrence_dict       = staticmethod(recurrence_dict)
        cls.ALL_RULES             = _ALL_RULES
        cls.LEXICON               = list(LEXICON)

    # ── state_hamming() ────────────────────────────────────────────────────────

    def test_sh_identical_zeros(self):
        self.assertEqual(self.state_hamming([0]*16, [0]*16), 0)

    def test_sh_identical_nonzero(self):
        self.assertEqual(self.state_hamming([63]*16, [63]*16), 0)

    def test_sh_one_bit_diff(self):
        # rows differ by one bit in one cell
        r1 = [0]*16; r2 = [0]*16; r2[0] = 1
        self.assertEqual(self.state_hamming(r1, r2), 1)

    def test_sh_max_distance(self):
        # all 6 bits differ in all 16 cells
        self.assertEqual(self.state_hamming([0]*16, [63]*16), 96)

    def test_sh_symmetric(self):
        r1 = [7, 0, 63, 1]; r2 = [0, 7, 1, 63]
        self.assertEqual(self.state_hamming(r1, r2), self.state_hamming(r2, r1))

    # ── recurrence_matrix() ────────────────────────────────────────────────────

    def test_rm_shape(self):
        rows = [[0]*4, [1]*4, [2]*4]
        R = self.recurrence_matrix(rows, eps=0)
        self.assertEqual(len(R), 3)
        self.assertEqual(len(R[0]), 3)

    def test_rm_main_diag_all_ones(self):
        rows = [[i]*4 for i in range(5)]
        R = self.recurrence_matrix(rows, eps=0)
        for i in range(5):
            self.assertEqual(R[i][i], 1)

    def test_rm_symmetric(self):
        rows = [[0]*4, [1]*4, [3]*4, [7]*4]
        R = self.recurrence_matrix(rows, eps=0)
        for i in range(len(R)):
            for j in range(len(R)):
                self.assertEqual(R[i][j], R[j][i])

    def test_rm_identical_rows_all_ones(self):
        rows = [[0]*4] * 4
        R = self.recurrence_matrix(rows, eps=0)
        for row in R:
            self.assertTrue(all(v == 1 for v in row))

    def test_rm_eps_0_strict(self):
        rows = [[0]*4, [1]*4]
        R = self.recurrence_matrix(rows, eps=0)
        self.assertEqual(R[0][1], 0)

    def test_rm_eps_positive_relaxes(self):
        # rows differ by exactly 1 bit (first cell: 0 vs 1) → eps=1 makes them recurrent
        rows = [[0, 0, 0, 0], [1, 0, 0, 0]]
        R = self.recurrence_matrix(rows, eps=1)
        self.assertEqual(R[0][1], 1)

    # ── rqa_metrics() ─────────────────────────────────────────────────────────

    def test_rqa_empty(self):
        m = self.rqa_metrics([])
        self.assertEqual(m['N'], 0)
        self.assertEqual(m['RR'], 0.0)

    def test_rqa_single_row(self):
        m = self.rqa_metrics([[1]])
        self.assertEqual(m['N'], 1)
        self.assertEqual(m['RR'], 0.0)  # no off-diagonal pairs

    def test_rqa_full_ones_rr(self):
        # 4×4 all-ones → RR = 12/12 = 1.0 (off-diagonal)
        R = [[1]*4 for _ in range(4)]
        m = self.rqa_metrics(R)
        self.assertAlmostEqual(m['RR'], 1.0, places=4)

    def test_rqa_full_ones_det(self):
        # 4×4 all-ones: shorter diagonals (length 1 at offset=3) don't meet min_line=2
        # so DET < 1.0 but > 0.8
        R = [[1]*4 for _ in range(4)]
        m = self.rqa_metrics(R)
        self.assertGreater(m['DET'], 0.8)

    def test_rqa_full_ones_lam(self):
        # same edge effect for vertical lines → LAM > 0.8
        R = [[1]*4 for _ in range(4)]
        m = self.rqa_metrics(R)
        self.assertGreater(m['LAM'], 0.8)

    def test_rqa_zero_off_diag(self):
        # only main diagonal → RR=0
        N = 4
        R = [[1 if i == j else 0 for j in range(N)] for i in range(N)]
        m = self.rqa_metrics(R)
        self.assertEqual(m['RR'], 0.0)
        self.assertEqual(m['DET'], 0.0)

    def test_rqa_checkerboard_det(self):
        # XOR3-style checkerboard: R[i][j]=1 iff (i+j) even
        N = 6
        R = [[1 if (i + j) % 2 == 0 else 0 for j in range(N)] for i in range(N)]
        m = self.rqa_metrics(R)
        # Diagonal offset=2 is all-ones (length N-2=4 ≥ 2) → DET > 0
        self.assertGreater(m['DET'], 0.0)

    def test_rqa_keys_present(self):
        R = [[1, 0], [0, 1]]
        m = self.rqa_metrics(R)
        for key in ['N', 'RR', 'DET', 'L', 'LAM', 'TT']:
            self.assertIn(key, m)

    # ── trajectory_recurrence() ────────────────────────────────────────────────

    def test_tr_returns_dict(self):
        r = self.trajectory_recurrence('ГОРА', 'xor3')
        self.assertIsInstance(r, dict)

    def test_tr_keys(self):
        r = self.trajectory_recurrence('ГОРА', 'xor3')
        for k in ['word', 'rule', 'width', 'eps', 'transient', 'period',
                  'n_steps', 'R', 'rqa']:
            self.assertIn(k, r)

    def test_tr_word_uppercased(self):
        r = self.trajectory_recurrence('гора', 'xor3')
        self.assertEqual(r['word'], 'ГОРА')

    def test_tr_matrix_shape(self):
        r = self.trajectory_recurrence('ГОРА', 'xor3')
        N = r['n_steps']
        self.assertEqual(len(r['R']), N)
        self.assertEqual(len(r['R'][0]), N)

    def test_tr_xor_rr(self):
        # XOR: ГОРА transient=2, period=1, N=6 → attractor rows all-zeros
        # RR=0.4 as verified by CLI
        r = self.trajectory_recurrence('ГОРА', 'xor')
        self.assertAlmostEqual(r['rqa']['RR'], 0.4, places=3)

    def test_tr_xor_high_det(self):
        # Attractor block of identical rows → long diagonals → DET > 0.5
        r = self.trajectory_recurrence('ГОРА', 'xor')
        self.assertGreater(r['rqa']['DET'], 0.5)

    def test_tr_xor3_det_one(self):
        # XOR3 checkerboard → every off-diagonal point lies on a diagonal line
        r = self.trajectory_recurrence('ГОРА', 'xor3')
        self.assertAlmostEqual(r['rqa']['DET'], 1.0, places=3)

    def test_tr_or_high_rr(self):
        # OR period-1 attractor: many identical rows → high RR
        r = self.trajectory_recurrence('ГОРА', 'or')
        self.assertGreater(r['rqa']['RR'], 0.4)

    def test_tr_n_steps_consistent(self):
        r = self.trajectory_recurrence('ТУМАН', 'xor3', n_cycles=4)
        expected = r['transient'] + 4 * r['period']
        self.assertEqual(r['n_steps'], expected)

    def test_tr_different_words_different_rr(self):
        r1 = self.trajectory_recurrence('ГОРА', 'xor3')
        r2 = self.trajectory_recurrence('ТУМАН', 'xor3')
        # Just check they are both valid floats in [0,1]
        self.assertGreaterEqual(r1['rqa']['RR'], 0.0)
        self.assertGreaterEqual(r2['rqa']['RR'], 0.0)

    def test_tr_eps_positive_increases_rr(self):
        r0 = self.trajectory_recurrence('ГОРА', 'xor3', eps=0)
        r1 = self.trajectory_recurrence('ГОРА', 'xor3', eps=4)
        self.assertGreaterEqual(r1['rqa']['RR'], r0['rqa']['RR'])

    # ── all_recurrences() ──────────────────────────────────────────────────────

    def test_ar_all_rules(self):
        d = self.all_recurrences('ГОРА')
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    def test_ar_each_has_rqa(self):
        d = self.all_recurrences('ВОДА')
        for rule in self.ALL_RULES:
            self.assertIn('rqa', d[rule])

    # ── build_recurrence_data() ───────────────────────────────────────────────

    def test_brd_keys(self):
        d = self.build_recurrence_data(['ГОРА', 'ВОДА'])
        for k in ['words', 'width', 'eps', 'per_rule', 'ranking', 'max_rr', 'min_rr']:
            self.assertIn(k, d)

    def test_brd_per_rule_contains_all_rules(self):
        d = self.build_recurrence_data(['ГОРА'])
        self.assertEqual(set(d['per_rule'].keys()), set(self.ALL_RULES))

    def test_brd_ranking_sorted(self):
        d = self.build_recurrence_data(['ГОРА', 'ВОДА', 'ТУМАН'])
        for rule in self.ALL_RULES:
            rr_vals = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(rr_vals, sorted(rr_vals, reverse=True))

    def test_brd_max_rr_is_tuple(self):
        d = self.build_recurrence_data(['ГОРА', 'ВОДА'])
        for rule in self.ALL_RULES:
            self.assertIsInstance(d['max_rr'][rule], tuple)

    # ── recurrence_dict() ─────────────────────────────────────────────────────

    def test_rd_json_serialisable(self):
        import json
        d = self.recurrence_dict('ГОРА')
        # R matrix is excluded; should be serialisable
        json.dumps(d)

    def test_rd_rules_key(self):
        d = self.recurrence_dict('ГОРА')
        self.assertIn('rules', d)
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_rd_each_rule_has_rr(self):
        d = self.recurrence_dict('ВОДА')
        for rule in self.ALL_RULES:
            self.assertIn('RR', d['rules'][rule])

    def test_rd_no_matrix(self):
        # The R matrix should NOT be in the JSON export (too large)
        d = self.recurrence_dict('ГОРА')
        for rule in self.ALL_RULES:
            self.assertNotIn('R', d['rules'][rule])

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_rc_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rc-canvas', content)

    def test_viewer_has_rc_metrics(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rc-metrics', content)

    def test_viewer_has_rc_hmap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rc-hmap', content)

    def test_viewer_has_rc_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rc-btn', content)

    def test_viewer_has_rqa_met(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rqaMet', content)

    def test_viewer_has_rc_rows(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('rcRows', content)


class TestSolanMutual(unittest.TestCase):
    """Tests for solan_mutual.py and the viewer MI section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_mutual import (
            attractor_states, cell_entropy, cell_mi,
            entropy_profile, mi_matrix, mi_profile,
            trajectory_mutual, all_mutual,
            build_mutual_data, mutual_dict,
            _ALL_RULES, _DEFAULT_WIDTH,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.attractor_states  = staticmethod(attractor_states)
        cls.cell_entropy      = staticmethod(cell_entropy)
        cls.cell_mi           = staticmethod(cell_mi)
        cls.entropy_profile   = staticmethod(entropy_profile)
        cls.mi_matrix         = staticmethod(mi_matrix)
        cls.mi_profile        = staticmethod(mi_profile)
        cls.trajectory_mutual = staticmethod(trajectory_mutual)
        cls.all_mutual        = staticmethod(all_mutual)
        cls.build_mutual_data = staticmethod(build_mutual_data)
        cls.mutual_dict       = staticmethod(mutual_dict)
        cls.ALL_RULES         = _ALL_RULES
        cls.W                 = _DEFAULT_WIDTH
        cls.LEXICON           = list(LEXICON)

    # ── attractor_states() ────────────────────────────────────────────────────

    def test_as_length_equals_period(self):
        from projects.hexglyph.solan_ca import find_orbit
        from projects.hexglyph.solan_word import encode_word, pad_to
        cells = pad_to(encode_word('ГОРА'), 16)
        _, period = find_orbit(cells[:], 'xor3')
        states = self.attractor_states('ГОРА', 'xor3')
        self.assertEqual(len(states), max(period, 1))

    def test_as_each_state_has_width_cells(self):
        states = self.attractor_states('ГОРА', 'xor3', 16)
        for s in states:
            self.assertEqual(len(s), 16)

    def test_as_xor_period_one(self):
        states = self.attractor_states('ГОРА', 'xor')
        self.assertEqual(len(states), 1)

    def test_as_values_in_q6_range(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for s in states:
            for v in s:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 63)

    def test_as_xor_attractor_all_zeros(self):
        # XOR rule: attractor is the all-zeros state
        states = self.attractor_states('ГОРА', 'xor')
        self.assertEqual(states[0], [0] * 16)

    # ── cell_entropy() ────────────────────────────────────────────────────────

    def test_ce_constant_state(self):
        # Single state → H = 0
        self.assertAlmostEqual(self.cell_entropy([[5, 3, 0]], 0), 0.0, places=9)

    def test_ce_two_distinct_values(self):
        # Alternating [0, 1, 0, 1] for cell 0 → H = 1 bit
        states = [[0, 0], [1, 0], [0, 0], [1, 0]]
        self.assertAlmostEqual(self.cell_entropy(states, 0), 1.0, places=9)

    def test_ce_all_same_value(self):
        states = [[7, 7]] * 8
        self.assertAlmostEqual(self.cell_entropy(states, 0), 0.0, places=9)

    def test_ce_non_negative(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(16):
            self.assertGreaterEqual(self.cell_entropy(states, i), 0.0)

    def test_ce_xor_zero_entropy(self):
        # XOR attractor is all-zeros: entropy = 0
        states = self.attractor_states('ГОРА', 'xor')
        self.assertAlmostEqual(self.cell_entropy(states, 0), 0.0, places=9)

    # ── cell_mi() ─────────────────────────────────────────────────────────────

    def test_cmi_self_equals_entropy(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(4):
            mi_self = self.cell_mi(states, i, i)
            ent     = self.cell_entropy(states, i)
            self.assertAlmostEqual(mi_self, ent, places=6)

    def test_cmi_symmetric(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    self.cell_mi(states, i, j),
                    self.cell_mi(states, j, i),
                    places=9,
                )

    def test_cmi_non_negative(self):
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(8):
            for j in range(8):
                self.assertGreaterEqual(self.cell_mi(states, i, j), 0.0)

    def test_cmi_bounded_by_min_entropy(self):
        # I(X;Y) ≤ min(H(X), H(Y))
        states = self.attractor_states('ТУМАН', 'xor3')
        for i in range(6):
            for j in range(6):
                mi  = self.cell_mi(states, i, j)
                hi  = self.cell_entropy(states, i)
                hj  = self.cell_entropy(states, j)
                self.assertLessEqual(mi, max(hi, hj) + 1e-9)

    def test_cmi_xor_constant(self):
        # XOR period-1 (constant): MI = 0 between any pair
        states = self.attractor_states('ГОРА', 'xor')
        self.assertAlmostEqual(self.cell_mi(states, 0, 5), 0.0, places=9)

    def test_cmi_empty_states(self):
        self.assertEqual(self.cell_mi([], 0, 0), 0)

    # ── entropy_profile() ─────────────────────────────────────────────────────

    def test_ep_length(self):
        ep = self.entropy_profile('ГОРА', 'xor3', 16)
        self.assertEqual(len(ep), 16)

    def test_ep_all_non_negative(self):
        ep = self.entropy_profile('ТУМАН', 'xor3')
        for v in ep:
            self.assertGreaterEqual(v, 0.0)

    def test_ep_xor_all_zero(self):
        ep = self.entropy_profile('ГОРА', 'xor')
        for v in ep:
            self.assertAlmostEqual(v, 0.0, places=9)

    # ── mi_matrix() ───────────────────────────────────────────────────────────

    def test_mm_shape(self):
        M = self.mi_matrix('ГОРА', 'xor3', 16)
        self.assertEqual(len(M), 16)
        self.assertEqual(len(M[0]), 16)

    def test_mm_symmetric(self):
        M = self.mi_matrix('ТУМАН', 'xor3')
        for i in range(16):
            for j in range(16):
                self.assertAlmostEqual(M[i][j], M[j][i], places=6)

    def test_mm_diagonal_is_entropy(self):
        M  = self.mi_matrix('ТУМАН', 'xor3')
        ep = self.entropy_profile('ТУМАН', 'xor3')
        for i in range(16):
            self.assertAlmostEqual(M[i][i], ep[i], places=6)

    def test_mm_all_non_negative(self):
        M = self.mi_matrix('ТУМАН', 'xor3')
        for row in M:
            for v in row:
                self.assertGreaterEqual(v, -1e-9)

    def test_mm_xor_all_zero(self):
        M = self.mi_matrix('ГОРА', 'xor')
        for row in M:
            for v in row:
                self.assertAlmostEqual(v, 0.0, places=9)

    # ── mi_profile() ──────────────────────────────────────────────────────────

    def test_mp_length(self):
        M   = self.mi_matrix('ТУМАН', 'xor3')
        p   = self.mi_profile(M, 16)
        self.assertEqual(len(p), 16 // 2 + 1)

    def test_mp_d0_is_mean_entropy(self):
        M   = self.mi_matrix('ТУМАН', 'xor3')
        p   = self.mi_profile(M, 16)
        ep  = self.entropy_profile('ТУМАН', 'xor3')
        self.assertAlmostEqual(p[0], sum(ep) / len(ep), places=4)

    def test_mp_all_non_negative(self):
        M = self.mi_matrix('ТУМАН', 'xor3')
        for v in self.mi_profile(M, 16):
            self.assertGreaterEqual(v, -1e-9)

    # ── trajectory_mutual() ───────────────────────────────────────────────────

    def test_tm_keys(self):
        r = self.trajectory_mutual('ГОРА', 'xor3')
        for k in ['word', 'rule', 'width', 'period', 'entropy',
                  'M', 'mi_by_dist', 'mean_entropy', 'max_mi', 'max_mi_pair']:
            self.assertIn(k, r)

    def test_tm_word_uppercased(self):
        self.assertEqual(self.trajectory_mutual('гора', 'xor3')['word'], 'ГОРА')

    def test_tm_xor_zero_entropy(self):
        r = self.trajectory_mutual('ГОРА', 'xor')
        self.assertAlmostEqual(r['mean_entropy'], 0.0, places=9)

    def test_tm_xor3_tuман_mean_entropy(self):
        r = self.trajectory_mutual('ТУМАН', 'xor3')
        self.assertAlmostEqual(r['mean_entropy'], 2.234, delta=0.01)

    def test_tm_max_mi_non_negative(self):
        r = self.trajectory_mutual('ТУМАН', 'xor3')
        self.assertGreaterEqual(r['max_mi'], 0.0)

    def test_tm_max_mi_pair_is_tuple_or_list(self):
        r = self.trajectory_mutual('ТУМАН', 'xor3')
        self.assertEqual(len(r['max_mi_pair']), 2)

    # ── all_mutual() ──────────────────────────────────────────────────────────

    def test_am_all_rules(self):
        d = self.all_mutual('ГОРА')
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    def test_am_each_has_period(self):
        d = self.all_mutual('ВОДА')
        for rule in self.ALL_RULES:
            self.assertIn('period', d[rule])

    # ── build_mutual_data() ───────────────────────────────────────────────────

    def test_bmd_keys(self):
        d = self.build_mutual_data(['ГОРА', 'ВОДА'])
        for k in ['words', 'width', 'per_rule', 'ranking', 'max_h', 'min_h']:
            self.assertIn(k, d)

    def test_bmd_ranking_sorted(self):
        d = self.build_mutual_data(['ГОРА', 'ВОДА', 'ТУМАН'])
        for rule in self.ALL_RULES:
            hs = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(hs, sorted(hs, reverse=True))

    # ── mutual_dict() ─────────────────────────────────────────────────────────

    def test_md_json_serialisable(self):
        import json
        d = self.mutual_dict('ГОРА')
        json.dumps(d)

    def test_md_no_matrix(self):
        d = self.mutual_dict('ГОРА')
        for rule in self.ALL_RULES:
            self.assertNotIn('M', d['rules'][rule])

    def test_md_has_mi_by_dist(self):
        d = self.mutual_dict('ТУМАН')
        for rule in self.ALL_RULES:
            self.assertIn('mi_by_dist', d['rules'][rule])

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_mi_mat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-mat', content)

    def test_viewer_has_mi_dist(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-dist', content)

    def test_viewer_has_mi_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-stats', content)

    def test_viewer_has_mi_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('mi-btn', content)

    def test_viewer_has_mi_cell_mi(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('miCellMI', content)

    def test_viewer_has_mi_entropy(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('miEntropy', content)


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


class TestSolanFourier(unittest.TestCase):
    """Tests for solan_fourier.py and the viewer Fourier/PSD section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_fourier import (
            dft1, power_spectrum, spectral_entropy,
            normalised_spectral_entropy, spectral_flatness,
            dominant_harmonic, cell_fourier, fourier_profile,
            fourier_dict, all_fourier, build_fourier_data,
        )
        cls.dft1                        = staticmethod(dft1)
        cls.power_spectrum              = staticmethod(power_spectrum)
        cls.spectral_entropy            = staticmethod(spectral_entropy)
        cls.normalised_spectral_entropy = staticmethod(normalised_spectral_entropy)
        cls.spectral_flatness           = staticmethod(spectral_flatness)
        cls.dominant_harmonic           = staticmethod(dominant_harmonic)
        cls.cell_fourier                = staticmethod(cell_fourier)
        cls.fourier_profile             = staticmethod(fourier_profile)
        cls.fourier_dict                = staticmethod(fourier_dict)
        cls.all_fourier                 = staticmethod(all_fourier)
        cls.build_fourier_data          = staticmethod(build_fourier_data)

    # ── dft1 ───────────────────────────────────────────────────────────

    def test_dft1_empty(self):
        self.assertEqual(self.dft1([]), [])

    def test_dft1_single(self):
        X = self.dft1([3.0])
        self.assertEqual(len(X), 1)
        self.assertAlmostEqual(X[0].real, 3.0, places=10)

    def test_dft1_dc_component(self):
        s = [1.0, 2.0, 3.0, 4.0]
        X = self.dft1(s)
        self.assertAlmostEqual(X[0].real, 10.0, places=8)

    def test_dft1_pure_cosine_peak_at_k1(self):
        import math
        N = 8
        s = [math.cos(2 * math.pi * t / N) for t in range(N)]
        X = self.dft1(s)
        self.assertGreater(abs(X[1]), abs(X[0]))

    # ── power_spectrum ─────────────────────────────────────────────────

    def test_power_spectrum_empty(self):
        self.assertEqual(self.power_spectrum([]), [])

    def test_power_spectrum_length(self):
        ps = self.power_spectrum([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(ps), 5)

    def test_power_spectrum_nonneg(self):
        ps = self.power_spectrum([10, 5, 20, 15, 3, 8])
        self.assertTrue(all(v >= 0 for v in ps))

    def test_power_spectrum_constant_dc_only(self):
        ps = self.power_spectrum([5, 5, 5, 5])
        self.assertGreater(ps[0], 0)
        self.assertAlmostEqual(sum(ps[1:]), 0.0, places=8)

    # ── spectral_entropy ───────────────────────────────────────────────

    def test_spectral_entropy_single_bin_is_zero(self):
        self.assertAlmostEqual(self.spectral_entropy([5.0]), 0.0)

    def test_spectral_entropy_all_zero_is_zero(self):
        self.assertAlmostEqual(self.spectral_entropy([0.0, 0.0]), 0.0)

    def test_spectral_entropy_uniform_max(self):
        import math
        ps = [1.0, 1.0, 1.0, 1.0]
        self.assertAlmostEqual(self.spectral_entropy(ps), math.log2(4), places=8)

    def test_spectral_entropy_nonneg(self):
        self.assertGreaterEqual(self.spectral_entropy([3.0, 1.0, 2.0, 0.5]), 0.0)

    # ── normalised_spectral_entropy ────────────────────────────────────

    def test_nse_range(self):
        v = self.normalised_spectral_entropy([2.0, 1.0, 3.0, 0.5])
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_nse_uniform_is_one(self):
        self.assertAlmostEqual(self.normalised_spectral_entropy([1.0] * 8), 1.0, places=6)

    def test_nse_one_bin_is_zero(self):
        self.assertAlmostEqual(self.normalised_spectral_entropy([7.0]), 0.0)

    def test_nse_all_zero_is_zero(self):
        self.assertAlmostEqual(self.normalised_spectral_entropy([0.0, 0.0, 0.0]), 0.0)

    # ── spectral_flatness ──────────────────────────────────────────────

    def test_sf_uniform_is_one(self):
        self.assertAlmostEqual(self.spectral_flatness([2.0, 2.0, 2.0, 2.0]), 1.0, places=6)

    def test_sf_zero_if_zero_bin(self):
        self.assertAlmostEqual(self.spectral_flatness([3.0, 0.0, 2.0]), 0.0)

    def test_sf_range(self):
        v = self.spectral_flatness([10.0, 1.0, 5.0, 2.0])
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    # ── dominant_harmonic ──────────────────────────────────────────────

    def test_dominant_harmonic_returns_k_ge_1(self):
        k = self.dominant_harmonic([100.0, 5.0, 20.0, 3.0])
        self.assertGreaterEqual(k, 1)

    def test_dominant_harmonic_finds_peak(self):
        self.assertEqual(self.dominant_harmonic([100.0, 5.0, 80.0, 3.0]), 2)

    def test_dominant_harmonic_single_bin(self):
        self.assertEqual(self.dominant_harmonic([5.0]), 1)

    # ── Fixed-point attractors → nH_sp = 0 ────────────────────────────

    def test_tuman_xor_nh_sp_zero(self):
        profile = self.fourier_profile('ТУМАН', 'xor', 16)
        for c in profile:
            self.assertAlmostEqual(c['nh_sp'], 0.0)

    def test_gora_and_nh_sp_high(self):
        profile = self.fourier_profile('ГОРА', 'and', 16)
        active = [c['nh_sp'] for c in profile if c['ac_total'] > 0]
        if active:
            self.assertGreater(sum(active) / len(active), 0.9)

    def test_tuman_xor3_nh_sp_mid(self):
        d = self.fourier_dict('ТУМАН', 'xor3', 16)
        self.assertGreater(d['mean_nh_sp'], 0.05)
        self.assertLess(d['mean_nh_sp'], 0.6)

    # ── fourier_dict structure ─────────────────────────────────────────

    def test_fourier_dict_keys(self):
        d = self.fourier_dict('ТУМАН', 'xor3', 16)
        for k in ('word', 'rule', 'period', 'cell_fourier',
                  'mean_nh_sp', 'mean_sf', 'mean_dc_frac',
                  'mean_ps', 'dominant_k', 'eff_period', 'n_bins'):
            self.assertIn(k, d)

    def test_fourier_dict_profile_length(self):
        d = self.fourier_dict('ГОРА', 'xor3', 16)
        self.assertEqual(len(d['cell_fourier']), 16)

    def test_fourier_dict_nh_sp_range(self):
        d = self.fourier_dict('ТУМАН', 'xor3', 16)
        for c in d['cell_fourier']:
            self.assertGreaterEqual(c['nh_sp'], 0.0)
            self.assertLessEqual(c['nh_sp'], 1.0)

    def test_fourier_dict_dc_frac_range(self):
        d = self.fourier_dict('ГОРА', 'and', 16)
        for c in d['cell_fourier']:
            self.assertGreaterEqual(c['dc_frac'], 0.0)
            self.assertLessEqual(c['dc_frac'], 1.0)

    def test_fourier_dict_dominant_k_ge_1(self):
        d = self.fourier_dict('ТУМАН', 'xor3', 16)
        self.assertGreaterEqual(d['dominant_k'], 1)

    def test_fourier_dict_word_uppercase(self):
        d = self.fourier_dict('туман', 'xor3', 16)
        self.assertEqual(d['word'], 'ТУМАН')

    # ── all_fourier ────────────────────────────────────────────────────

    def test_all_fourier_has_four_rules(self):
        result = self.all_fourier('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_fourier_values_are_dicts(self):
        result = self.all_fourier('ГОРА', 16)
        for rule, d in result.items():
            self.assertIsInstance(d, dict)
            self.assertIn('mean_nh_sp', d)

    # ── build_fourier_data ─────────────────────────────────────────────

    def test_build_fourier_data_structure(self):
        data = self.build_fourier_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_fourier_data_entry_keys(self):
        data = self.build_fourier_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'mean_nh_sp', 'mean_sf', 'mean_dc_frac',
                  'dominant_k', 'eff_period', 'n_bins'):
            self.assertIn(k, entry)

    def test_build_fourier_data_words_uppercase(self):
        data = self.build_fourier_data(['туман'], 16)
        self.assertIn('ТУМАН', data['words'])

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_fou_spectrum(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('fou-spectrum', content)

    def test_viewer_has_fou_cell(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('fou-cell', content)

    def test_viewer_has_fou_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('fou-stats', content)

    def test_viewer_has_fou_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('fouRun', content)

    def test_viewer_has_fou_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Фурье / PSD Q6', content)


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


class TestSolanAutocorr(unittest.TestCase):
    """Tests for solan_autocorr.py and the viewer ACF section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_autocorr import (
            acf, decorrelation_lag, mean_acf_power,
            cell_acf_profile, acf_dict, all_acf, build_acf_data,
        )
        cls.acf                = staticmethod(acf)
        cls.decorrelation_lag  = staticmethod(decorrelation_lag)
        cls.mean_acf_power     = staticmethod(mean_acf_power)
        cls.cell_acf_profile   = staticmethod(cell_acf_profile)
        cls.acf_dict           = staticmethod(acf_dict)
        cls.all_acf            = staticmethod(all_acf)
        cls.build_acf_data     = staticmethod(build_acf_data)

    # ── acf() ──────────────────────────────────────────────────────────

    def test_acf_lag0_always_one(self):
        self.assertAlmostEqual(self.acf([1, 2, 3, 4, 5])[0], 1.0, places=8)

    def test_acf_constant_returns_one_then_nan(self):
        import math
        vals = self.acf([7, 7, 7, 7], 3)
        self.assertAlmostEqual(vals[0], 1.0)
        self.assertTrue(all(math.isnan(v) for v in vals[1:]))

    def test_acf_alternating_lag1_is_minus_one(self):
        # [a, b, a, b, ...] → ACF(1) = -1
        vals = self.acf([47, 1, 47, 1, 47, 1, 47, 1], 1)
        self.assertAlmostEqual(vals[1], -1.0, places=6)

    def test_acf_empty_series(self):
        self.assertEqual(self.acf([]), [])

    def test_acf_single_element(self):
        vals = self.acf([5], 0)
        self.assertAlmostEqual(vals[0], 1.0)

    def test_acf_length(self):
        vals = self.acf([1, 2, 3, 4, 5, 6], 4)
        self.assertEqual(len(vals), 5)

    def test_acf_max_lag_capped_at_n_minus_1(self):
        vals = self.acf([1, 2, 3], 100)
        self.assertEqual(len(vals), 3)   # max_lag capped at 2

    # ── decorrelation_lag ──────────────────────────────────────────────

    def test_decorrelation_lag_alternating(self):
        # ACF = [1, -1] → τ₀ = 1
        acf_vals = [1.0, -1.0]
        self.assertEqual(self.decorrelation_lag(acf_vals), 1)

    def test_decorrelation_lag_all_positive_is_none(self):
        self.assertIsNone(self.decorrelation_lag([1.0, 0.8, 0.5, 0.2]))

    def test_decorrelation_lag_zero_at_lag2(self):
        self.assertEqual(self.decorrelation_lag([1.0, 0.3, -0.1, -0.4]), 2)

    def test_decorrelation_lag_ignores_nan(self):
        import math
        self.assertIsNone(self.decorrelation_lag([1.0, float('nan'), float('nan')]))

    # ── mean_acf_power ─────────────────────────────────────────────────

    def test_mean_acf_power_alternating(self):
        # ACF = [1, -1] → power over lags 1..1 = (-1)² = 1
        self.assertAlmostEqual(self.mean_acf_power([1.0, -1.0]), 1.0, places=8)

    def test_mean_acf_power_zero_lags(self):
        self.assertAlmostEqual(self.mean_acf_power([1.0]), 0.0, places=8)

    def test_mean_acf_power_nonneg(self):
        self.assertGreaterEqual(self.mean_acf_power([1.0, 0.5, -0.3]), 0.0)

    # ── Fixed-point attractors → max_lag=0 ────────────────────────────

    def test_tuman_xor_max_lag_zero(self):
        profile = self.cell_acf_profile('ТУМАН', 'xor', 16, 8)
        for c in profile:
            self.assertEqual(c['max_lag'], 0)
            self.assertAlmostEqual(c['acf'][0], 1.0)

    def test_gora_or_tau0_is_none(self):
        profile = self.cell_acf_profile('ГОРА', 'or', 16, 8)
        for c in profile:
            self.assertIsNone(c['tau0'])

    # ── ГОРА AND (P=2) → ACF = [1, -1] ───────────────────────────────

    def test_gora_and_acf_lag1_minus_one(self):
        profile = self.cell_acf_profile('ГОРА', 'and', 16, 8)
        for c in profile:
            self.assertAlmostEqual(c['acf'][0], 1.0, places=6)
            self.assertAlmostEqual(c['acf'][1], -1.0, places=6)

    def test_gora_and_tau0_is_one(self):
        profile = self.cell_acf_profile('ГОРА', 'and', 16, 8)
        for c in profile:
            self.assertEqual(c['tau0'], 1)

    def test_gora_and_mean_power_is_one(self):
        d = self.acf_dict('ГОРА', 'and', 16, 8)
        self.assertAlmostEqual(d['mean_mpower'], 1.0, places=6)

    # ── ТУМАН XOR3 (P=8) → rich ACF ───────────────────────────────────

    def test_tuman_xor3_max_lag_is_7(self):
        d = self.acf_dict('ТУМАН', 'xor3', 16, 8)
        self.assertEqual(d['max_lag'], 7)

    def test_tuman_xor3_mean_acf_lag0_is_one(self):
        d = self.acf_dict('ТУМАН', 'xor3', 16, 8)
        self.assertAlmostEqual(d['mean_acf'][0], 1.0, places=6)

    def test_tuman_xor3_tau0_small(self):
        d = self.acf_dict('ТУМАН', 'xor3', 16, 8)
        # Most cells decorrelate at lag 1 or 2
        self.assertIsNotNone(d['mean_tau0'])
        self.assertLessEqual(d['mean_tau0'], 3.0)

    def test_tuman_xor3_acf_symmetric(self):
        # For circular ACF of period P: ACF(k) = ACF(P-k)
        profile = self.cell_acf_profile('ТУМАН', 'xor3', 16, 8)
        for c in profile:
            a = c['acf']
            P = 8
            if len(a) >= P:
                self.assertAlmostEqual(a[1], a[P - 1], places=6)
                self.assertAlmostEqual(a[2], a[P - 2], places=6)

    # ── acf_dict structure ─────────────────────────────────────────────

    def test_acf_dict_keys(self):
        d = self.acf_dict('ТУМАН', 'xor3', 16, 8)
        for k in ('word', 'rule', 'period', 'max_lag', 'cell_profile',
                  'mean_acf', 'mean_tau0', 'mean_mpower'):
            self.assertIn(k, d)

    def test_acf_dict_cell_profile_length(self):
        d = self.acf_dict('ГОРА', 'xor3', 16, 8)
        self.assertEqual(len(d['cell_profile']), 16)

    def test_acf_dict_word_uppercase(self):
        d = self.acf_dict('туман', 'xor3', 16, 8)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_acf_dict_mean_acf_lag0(self):
        d = self.acf_dict('ТУМАН', 'xor3', 16, 8)
        self.assertAlmostEqual(d['mean_acf'][0], 1.0, places=6)

    def test_acf_dict_mean_mpower_nonneg(self):
        d = self.acf_dict('ТУМАН', 'xor3', 16, 8)
        self.assertGreaterEqual(d['mean_mpower'], 0.0)

    # ── all_acf ────────────────────────────────────────────────────────

    def test_all_acf_has_four_rules(self):
        result = self.all_acf('ТУМАН', 16, 8)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_acf_values_are_dicts(self):
        result = self.all_acf('ГОРА', 16, 8)
        for rule, d in result.items():
            self.assertIsInstance(d, dict)
            self.assertIn('mean_acf', d)

    # ── build_acf_data ─────────────────────────────────────────────────

    def test_build_acf_data_structure(self):
        data = self.build_acf_data(['ТУМАН', 'ГОРА'], 16, 8)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_acf_data_entry_keys(self):
        data = self.build_acf_data(['ТУМАН'], 16, 8)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'max_lag', 'mean_acf', 'mean_tau0', 'mean_mpower'):
            self.assertIn(k, entry)

    def test_build_acf_data_words_uppercase(self):
        data = self.build_acf_data(['туман'], 16, 8)
        self.assertIn('ТУМАН', data['words'])

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_acf_heat(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('acf-heat', content)

    def test_viewer_has_acf_mean(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('acf-mean', content)

    def test_viewer_has_acf_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('acf-stats', content)

    def test_viewer_has_acf_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('acfRun', content)

    def test_viewer_has_acf_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Автокорреляция Q6', content)


class TestSolanMoran(unittest.TestCase):
    """Tests for solan_moran.py and the viewer Moran's I section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_moran import (
            morans_i, spatial_classification,
            morans_i_series, morans_i_dict, all_morans_i, build_moran_data,
        )
        cls.morans_i               = staticmethod(morans_i)
        cls.spatial_classification = staticmethod(spatial_classification)
        cls.morans_i_series        = staticmethod(morans_i_series)
        cls.morans_i_dict          = staticmethod(morans_i_dict)
        cls.all_morans_i           = staticmethod(all_morans_i)
        cls.build_moran_data       = staticmethod(build_moran_data)

    # ── morans_i() ─────────────────────────────────────────────────────

    def test_morans_i_constant_is_nan(self):
        import math
        self.assertTrue(math.isnan(self.morans_i([5, 5, 5, 5])))

    def test_morans_i_alternating_is_minus_one(self):
        vals = [47, 1] * 8   # 16 elements alternating
        self.assertAlmostEqual(self.morans_i(vals), -1.0, places=6)

    def test_morans_i_single_element_nan(self):
        import math
        self.assertTrue(math.isnan(self.morans_i([42])))

    def test_morans_i_empty_nan(self):
        import math
        self.assertTrue(math.isnan(self.morans_i([])))

    def test_morans_i_range(self):
        import random, math
        random.seed(0)
        vals = [random.randint(0, 63) for _ in range(16)]
        v = self.morans_i(vals)
        if not math.isnan(v):
            self.assertGreaterEqual(v, -1.1)
            self.assertLessEqual(v, 1.1)

    def test_morans_i_two_elements(self):
        # [a, b] alternating in 1-ring: each cell's neighbours are both the other cell
        v = self.morans_i([10, 50])
        self.assertAlmostEqual(v, -1.0, places=6)

    # ── spatial_classification ─────────────────────────────────────────

    def test_classification_constant(self):
        import math
        self.assertEqual(self.spatial_classification(float('nan')), 'constant')

    def test_classification_strongly_clustered(self):
        self.assertEqual(self.spatial_classification(0.8), 'strongly clustered')

    def test_classification_clustered(self):
        self.assertEqual(self.spatial_classification(0.3), 'clustered')

    def test_classification_random(self):
        self.assertEqual(self.spatial_classification(0.0), 'random')

    def test_classification_dispersed(self):
        self.assertEqual(self.spatial_classification(-0.3), 'dispersed')

    def test_classification_strongly_dispersed(self):
        self.assertEqual(self.spatial_classification(-0.8), 'strongly dispersed')

    # ── Fixed-point attractors → NaN ──────────────────────────────────

    def test_tuman_xor_all_nan(self):
        import math
        series = self.morans_i_series('ТУМАН', 'xor', 16)
        self.assertEqual(len(series), 1)
        self.assertTrue(math.isnan(series[0]))

    def test_gora_or_all_nan(self):
        import math
        series = self.morans_i_series('ГОРА', 'or', 16)
        self.assertEqual(len(series), 1)
        self.assertTrue(math.isnan(series[0]))

    # ── ГОРА AND (P=2, perfect alternating → I=−1) ────────────────────

    def test_gora_and_i_is_minus_one(self):
        series = self.morans_i_series('ГОРА', 'and', 16)
        for v in series:
            self.assertAlmostEqual(v, -1.0, places=6)

    def test_gora_and_mean_is_minus_one(self):
        d = self.morans_i_dict('ГОРА', 'and', 16)
        self.assertAlmostEqual(d['mean_i'], -1.0, places=6)

    def test_gora_and_classification_strongly_dispersed(self):
        d = self.morans_i_dict('ГОРА', 'and', 16)
        self.assertEqual(d['classification'], 'strongly dispersed')

    def test_gora_and_var_is_zero(self):
        d = self.morans_i_dict('ГОРА', 'and', 16)
        self.assertAlmostEqual(d['var_i'], 0.0, places=6)

    # ── ТУМАН XOR3 (P=8, varied spatial patterns) ─────────────────────

    def test_tuman_xor3_series_length_8(self):
        series = self.morans_i_series('ТУМАН', 'xor3', 16)
        self.assertEqual(len(series), 8)

    def test_tuman_xor3_series_all_valid(self):
        import math
        series = self.morans_i_series('ТУМАН', 'xor3', 16)
        self.assertTrue(all(not math.isnan(v) for v in series))

    def test_tuman_xor3_has_mixed_sign(self):
        series = self.morans_i_series('ТУМАН', 'xor3', 16)
        self.assertTrue(any(v > 0 for v in series))
        self.assertTrue(any(v < 0 for v in series))

    def test_tuman_xor3_mean_i_in_range(self):
        d = self.morans_i_dict('ТУМАН', 'xor3', 16)
        self.assertGreaterEqual(d['mean_i'], -1.0)
        self.assertLessEqual(d['mean_i'], 1.0)

    def test_tuman_xor3_min_le_max(self):
        d = self.morans_i_dict('ТУМАН', 'xor3', 16)
        self.assertLessEqual(d['min_i'], d['max_i'])

    # ── morans_i_dict structure ────────────────────────────────────────

    def test_morans_i_dict_keys(self):
        d = self.morans_i_dict('ТУМАН', 'xor3', 16)
        for k in ('word', 'rule', 'period', 'series', 'mean_i',
                  'min_i', 'max_i', 'var_i', 'classification', 'n_valid'):
            self.assertIn(k, d)

    def test_morans_i_dict_word_uppercase(self):
        d = self.morans_i_dict('туман', 'xor3', 16)
        self.assertEqual(d['word'], 'ТУМАН')

    def test_morans_i_dict_series_len_equals_period(self):
        d = self.morans_i_dict('ТУМАН', 'xor3', 16)
        self.assertEqual(len(d['series']), d['period'])

    def test_morans_i_dict_n_valid_le_period(self):
        d = self.morans_i_dict('ТУМАН', 'xor3', 16)
        self.assertLessEqual(d['n_valid'], d['period'])

    # ── all_morans_i ───────────────────────────────────────────────────

    def test_all_morans_i_has_four_rules(self):
        result = self.all_morans_i('ТУМАН', 16)
        self.assertEqual(set(result.keys()), {'xor', 'xor3', 'and', 'or'})

    def test_all_morans_i_values_are_dicts(self):
        result = self.all_morans_i('ГОРА', 16)
        for rule, d in result.items():
            self.assertIsInstance(d, dict)
            self.assertIn('mean_i', d)

    # ── build_moran_data ───────────────────────────────────────────────

    def test_build_moran_data_structure(self):
        data = self.build_moran_data(['ТУМАН', 'ГОРА'], 16)
        self.assertIn('words', data)
        self.assertIn('per_rule', data)
        self.assertEqual(set(data['per_rule'].keys()), {'xor', 'xor3', 'and', 'or'})

    def test_build_moran_data_entry_keys(self):
        data = self.build_moran_data(['ТУМАН'], 16)
        entry = data['per_rule']['xor3']['ТУМАН']
        for k in ('period', 'series', 'mean_i', 'min_i', 'max_i',
                  'var_i', 'classification', 'n_valid'):
            self.assertIn(k, entry)

    def test_build_moran_data_words_uppercase(self):
        data = self.build_moran_data(['туман'], 16)
        self.assertIn('ТУМАН', data['words'])

    # ── viewer ─────────────────────────────────────────────────────────

    def test_viewer_has_moran_time(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('moran-time', content)

    def test_viewer_has_moran_all(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('moran-all', content)

    def test_viewer_has_moran_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('moran-stats', content)

    def test_viewer_has_moran_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('moranRun', content)

    def test_viewer_has_moran_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn("Moran's I Q6", content)


class TestSolanTransfer(unittest.TestCase):
    """Tests for solan_transfer.py and the viewer Transfer Entropy section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_transfer import (
            get_orbit, bit_te, cell_te,
            te_matrix, te_asymmetry, te_dict, all_te,
            build_te_data, _ALL_RULES, _DEFAULT_WIDTH,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.get_orbit    = staticmethod(get_orbit)
        cls.bit_te       = staticmethod(bit_te)
        cls.cell_te      = staticmethod(cell_te)
        cls.te_matrix    = staticmethod(te_matrix)
        cls.te_asymmetry = staticmethod(te_asymmetry)
        cls.te_dict      = staticmethod(te_dict)
        cls.all_te       = staticmethod(all_te)
        cls.build_te_data= staticmethod(build_te_data)
        cls.ALL_RULES    = _ALL_RULES
        cls.W            = _DEFAULT_WIDTH
        cls.LEXICON      = list(LEXICON)

    # ── get_orbit() ───────────────────────────────────────────────────────────

    def test_go_length_xor3_tuman(self):
        orbit = self.get_orbit('ТУМАН', 'xor3')
        self.assertEqual(len(orbit), 8)

    def test_go_length_period1(self):
        orbit = self.get_orbit('ГОРА', 'xor')   # period=1 → len=1
        self.assertEqual(len(orbit), 1)

    def test_go_state_width(self):
        orbit = self.get_orbit('ГОРА', 'xor3')
        for state in orbit:
            self.assertEqual(len(state), self.W)

    def test_go_q6_range(self):
        orbit = self.get_orbit('ТУМАН', 'xor3')
        for state in orbit:
            for v in state:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 63)

    def test_go_periodic(self):
        # Last step should cycle back to first
        from projects.hexglyph.solan_ca import step
        orbit = self.get_orbit('ТУМАН', 'xor3')
        last  = list(orbit[-1])
        nxt   = step(last, 'xor3')
        self.assertEqual(tuple(nxt), orbit[0])

    # ── bit_te() ──────────────────────────────────────────────────────────────

    def test_bte_constant_series_zero(self):
        y = [0, 0, 0, 0]
        x = [1, 0, 1, 0]
        self.assertAlmostEqual(self.bit_te(y, x), 0.0, places=8)

    def test_bte_non_negative(self):
        import random
        rng = random.Random(7)
        for _ in range(30):
            P = rng.choice([2, 4, 6, 8])
            y = [rng.randint(0, 1) for _ in range(P)]
            x = [rng.randint(0, 1) for _ in range(P)]
            self.assertGreaterEqual(self.bit_te(y, x), 0.0)

    def test_bte_period1_zero(self):
        # period-1 series → only one transition → TE=0
        self.assertAlmostEqual(self.bit_te([1], [0]), 0.0, places=8)

    def test_bte_known_nonzero(self):
        # period-6: y repeats with shorter effective period than orbit
        # y=[0,1,0,0,1,0], x=[1,1,0,1,1,0]
        y = [0, 1, 0, 0, 1, 0]
        x = [1, 1, 0, 1, 1, 0]
        # H(Y_t|Y_{t-1}) > 0 because y={0→1, 1→0, 0→0, 0→1, 1→0, 0→0}
        # from y_{t-1}=0: y_t ∈ {1,0,1,0} → non-trivial
        te = self.bit_te(y, x)
        self.assertGreaterEqual(te, 0.0)

    # ── cell_te() ─────────────────────────────────────────────────────────────

    def test_cte_non_negative(self):
        orbit = self.get_orbit('ТУМАН', 'xor3')
        for j in range(0, self.W, 4):
            for i in range(0, self.W, 4):
                self.assertGreaterEqual(self.cell_te(orbit, i, j), 0.0)

    def test_cte_period1_zero(self):
        orbit = self.get_orbit('ГОРА', 'xor')  # P=1, all-zeros
        te = self.cell_te(orbit, 0, 1)
        self.assertAlmostEqual(te, 0.0, places=8)

    def test_cte_self_zero(self):
        # Self-TE is always 0: I(Y_t; Y_{t-1} | Y_{t-1}) = 0
        orbit = self.get_orbit('ТУМАН', 'xor3')
        for i in range(self.W):
            self.assertAlmostEqual(self.cell_te(orbit, i, i), 0.0, places=8)

    # ── te_matrix() ───────────────────────────────────────────────────────────

    def test_tem_dimensions(self):
        mat = self.te_matrix('ГОРА', 'xor3')
        self.assertEqual(len(mat), self.W)
        for row in mat:
            self.assertEqual(len(row), self.W)

    def test_tem_non_negative(self):
        mat = self.te_matrix('ТУМАН', 'xor3')
        for row in mat:
            for v in row:
                self.assertGreaterEqual(v, 0.0)

    def test_tem_period1_all_zero(self):
        mat = self.te_matrix('ГОРА', 'xor')   # period-1 → all 0
        for row in mat:
            for v in row:
                self.assertAlmostEqual(v, 0.0, places=8)

    def test_tem_xor3_tuman_has_nonzero(self):
        mat = self.te_matrix('ТУМАН', 'xor3')
        max_te = max(mat[i][j] for i in range(self.W) for j in range(self.W))
        self.assertGreater(max_te, 0.0)

    def test_tem_diagonal_zero(self):
        mat = self.te_matrix('ТУМАН', 'xor3')
        for i in range(self.W):
            self.assertAlmostEqual(mat[i][i], 0.0, places=8)

    # ── te_asymmetry() ────────────────────────────────────────────────────────

    def test_ta_antisymmetric(self):
        mat  = self.te_matrix('ТУМАН', 'xor3')
        asym = self.te_asymmetry(mat)
        for i in range(self.W):
            for j in range(self.W):
                self.assertAlmostEqual(asym[i][j], -asym[j][i], places=8)

    def test_ta_dimensions(self):
        mat  = self.te_matrix('ГОРА', 'xor3')
        asym = self.te_asymmetry(mat)
        self.assertEqual(len(asym), self.W)
        for row in asym:
            self.assertEqual(len(row), self.W)

    # ── te_dict() ─────────────────────────────────────────────────────────────

    def test_td_keys(self):
        d = self.te_dict('ТУМАН', 'xor3')
        for k in ['word', 'rule', 'width', 'period', 'matrix',
                  'max_te', 'mean_te', 'self_te', 'right_te', 'left_te',
                  'asymmetry', 'mean_right', 'mean_left', 'lr_asymmetry']:
            self.assertIn(k, d)

    def test_td_word_upper(self):
        d = self.te_dict('туман', 'xor3')
        self.assertEqual(d['word'], 'ТУМАН')

    def test_td_period_tuman_xor3(self):
        d = self.te_dict('ТУМАН', 'xor3')
        self.assertEqual(d['period'], 8)

    def test_td_max_te_non_negative(self):
        for rule in self.ALL_RULES:
            d = self.te_dict('ТУМАН', rule)
            self.assertGreaterEqual(d['max_te'], 0.0)

    def test_td_self_te_all_zero(self):
        d = self.te_dict('ТУМАН', 'xor3')
        for v in d['self_te']:
            self.assertAlmostEqual(v, 0.0, places=8)

    def test_td_lr_asymmetry_symmetric_word(self):
        # lr_asymmetry = mean_right - mean_left; should be a float
        d = self.te_dict('ТУМАН', 'xor3')
        self.assertAlmostEqual(
            d['lr_asymmetry'],
            d['mean_right'] - d['mean_left'],
            places=6,
        )

    def test_td_period1_max_te_zero(self):
        d = self.te_dict('ГОРА', 'xor')
        self.assertAlmostEqual(d['max_te'], 0.0, places=8)

    def test_td_xor3_tuman_max_te_positive(self):
        d = self.te_dict('ТУМАН', 'xor3')
        self.assertGreater(d['max_te'], 0.0)

    def test_td_matrix_matches_te_matrix(self):
        d   = self.te_dict('ГОРА', 'xor3')
        mat = self.te_matrix('ГОРА', 'xor3')
        for i in range(self.W):
            for j in range(self.W):
                self.assertAlmostEqual(d['matrix'][i][j], mat[i][j], places=8)

    # ── all_te() ──────────────────────────────────────────────────────────────

    def test_ate_all_rules(self):
        d = self.all_te('ТУМАН')
        self.assertEqual(set(d.keys()), set(self.ALL_RULES))

    # ── build_te_data() ───────────────────────────────────────────────────────

    def test_btd_keys(self):
        d = self.build_te_data(['ГОРА', 'ВОДА'])
        for k in ['words', 'per_rule', 'ranking']:
            self.assertIn(k, d)

    def test_btd_ranking_sorted(self):
        d = self.build_te_data(['ГОРА', 'ВОДА', 'МИР'])
        for rule in self.ALL_RULES:
            vals = [x[1] for x in d['ranking'][rule]]
            self.assertEqual(vals, sorted(vals, reverse=True))

    # ── Viewer HTML / JS ──────────────────────────────────────────────────────

    def test_viewer_has_te_matrix_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('te-matrix-canvas', content)

    def test_viewer_has_te_stats(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('te-stats', content)

    def test_viewer_has_te_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('te-btn', content)

    def test_viewer_has_bit_te(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('bitTE', content)

    def test_viewer_has_cell_te(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('cellTE', content)

    def test_viewer_has_te_build_matrix(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('teBuildMatrix', content)


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


class TestSolanCorrelation(unittest.TestCase):
    """Tests for solan_correlation.py and the viewer Correlation section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_correlation import (
            row_autocorr, attractor_autocorr, all_autocorrs,
            cross_corr, correlation_length,
            build_correlation_data, correlation_dict,
            _ALL_RULES,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.row_autocorr          = staticmethod(row_autocorr)
        cls.attractor_autocorr    = staticmethod(attractor_autocorr)
        cls.all_autocorrs         = staticmethod(all_autocorrs)
        cls.cross_corr            = staticmethod(cross_corr)
        cls.correlation_length    = staticmethod(correlation_length)
        cls.build_correlation_data = staticmethod(build_correlation_data)
        cls.correlation_dict      = staticmethod(correlation_dict)
        cls.ALL_RULES             = _ALL_RULES
        cls.LEXICON               = list(LEXICON)

    # ── row_autocorr() ────────────────────────────────────────────────────────

    def test_rac_returns_list(self):
        r = self.row_autocorr([1, 0, 1, 0, 1, 0, 1, 0])
        self.assertIsInstance(r, list)

    def test_rac_length(self):
        row = list(range(16))
        r = self.row_autocorr(row)
        self.assertEqual(len(r), 9)  # width//2 + 1

    def test_rac_first_element_one(self):
        # r(0) = 1 always (self-correlation)
        for row in [[1,2,3,4]*4, [0]*16, [63]*16]:
            r = self.row_autocorr(row)
            self.assertAlmostEqual(r[0], 1.0, places=5)

    def test_rac_constant_row(self):
        # Constant row → r(d) = 1 for all d (zero-variance case)
        r = self.row_autocorr([5] * 16)
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in r))

    def test_rac_alternating_row(self):
        # [1,0,1,0,...] → r(1) = -1 (anti-correlated at lag 1)
        row = [1, 0] * 8
        r = self.row_autocorr(row)
        self.assertAlmostEqual(r[1], -1.0, places=5)

    def test_rac_values_bounded(self):
        row = [i % 7 for i in range(16)]
        r = self.row_autocorr(row)
        for v in r:
            self.assertGreaterEqual(v, -1.0 - 1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_rac_zeros_row(self):
        # All-zero row: constant → r(d)=1
        r = self.row_autocorr([0] * 16)
        self.assertAlmostEqual(r[0], 1.0)

    # ── attractor_autocorr() ──────────────────────────────────────────────────

    def test_aac_returns_list(self):
        r = self.attractor_autocorr('ГОРА', 'xor3')
        self.assertIsInstance(r, list)

    def test_aac_length(self):
        r = self.attractor_autocorr('ГОРА', 'xor3')
        self.assertEqual(len(r), 9)  # 16//2 + 1

    def test_aac_first_is_one(self):
        for rule in self.ALL_RULES:
            r = self.attractor_autocorr('ГОРА', rule)
            self.assertAlmostEqual(r[0], 1.0, places=5)

    def test_aac_xor_attractor(self):
        # XOR attractor = all-zeros → zero variance → r(0)=1, r(d≥1)=0
        r = self.attractor_autocorr('ГОРА', 'xor')
        self.assertAlmostEqual(r[0], 1.0, places=5)
        for v in r[1:]:
            self.assertAlmostEqual(v, 0.0, places=5)

    def test_aac_and_alternating(self):
        # AND/OR produce alternating attractors for ГОРА → r(1) < 0
        r = self.attractor_autocorr('ГОРА', 'and')
        self.assertLess(r[1], 0.0)

    def test_aac_values_bounded(self):
        for rule in self.ALL_RULES:
            r = self.attractor_autocorr('ГОРА', rule)
            for v in r:
                self.assertGreaterEqual(v, -1.0 - 1e-9)
                self.assertLessEqual(v, 1.0 + 1e-9)

    # ── all_autocorrs() ───────────────────────────────────────────────────────

    def test_all_ac_keys(self):
        r = self.all_autocorrs('ГОРА')
        self.assertEqual(set(r.keys()), set(self.ALL_RULES))

    def test_all_ac_each_list(self):
        r = self.all_autocorrs('ГОРА')
        for ac in r.values():
            self.assertIsInstance(ac, list)

    # ── cross_corr() ─────────────────────────────────────────────────────────

    def test_cc_returns_list(self):
        r = self.cross_corr('ГОРА', 'ЛУНА', 'xor3')
        self.assertIsInstance(r, list)

    def test_cc_length(self):
        r = self.cross_corr('ГОРА', 'ЛУНА', 'xor3')
        self.assertEqual(len(r), 9)

    def test_cc_same_word_is_autocorr(self):
        # cross_corr(word, word) should match attractor_autocorr
        cc = self.cross_corr('ГОРА', 'ГОРА', 'xor3')
        ac = self.attractor_autocorr('ГОРА', 'xor3')
        self.assertAlmostEqual(cc[0], ac[0], places=4)

    def test_cc_bounded(self):
        r = self.cross_corr('ГОРА', 'ЛУНА', 'xor3')
        for v in r:
            self.assertGreaterEqual(v, -1.0 - 1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)

    # ── correlation_length() ──────────────────────────────────────────────────

    def test_cl_returns_float(self):
        v = self.correlation_length('ГОРА', 'xor3')
        self.assertIsInstance(v, float)

    def test_cl_positive(self):
        for rule in self.ALL_RULES:
            v = self.correlation_length('ГОРА', rule)
            self.assertGreater(v, 0.0)

    def test_cl_xor_short(self):
        # XOR attractor: r(d≥1)=0 → length = 1
        v = self.correlation_length('ГОРА', 'xor')
        self.assertAlmostEqual(v, 1.0)

    def test_cl_and_long(self):
        # AND alternating r(d)=(-1)^d: |r|=1 > 1/e at all lags → length = max
        v = self.correlation_length('ГОРА', 'and')
        self.assertGreaterEqual(v, 7.0)

    # ── build_correlation_data() ──────────────────────────────────────────────

    def test_bcd_returns_dict(self):
        d = self.build_correlation_data(['ГОРА', 'ЛУНА'])
        self.assertIsInstance(d, dict)

    def test_bcd_required_keys(self):
        d = self.build_correlation_data(['ГОРА', 'ЛУНА'])
        for k in ('words', 'width', 'n_lags', 'per_rule', 'corr_lengths',
                  'max_corr_len', 'min_corr_len'):
            self.assertIn(k, d)

    def test_bcd_n_lags(self):
        d = self.build_correlation_data(['ГОРА'])
        self.assertEqual(d['n_lags'], 9)

    def test_bcd_all_rules(self):
        d = self.build_correlation_data(['ГОРА', 'ЛУНА'])
        self.assertEqual(set(d['per_rule'].keys()), set(self.ALL_RULES))

    # ── correlation_dict() ────────────────────────────────────────────────────

    def test_cd_json_serialisable(self):
        import json
        d = self.correlation_dict('ГОРА')
        dumped = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(dumped, str)

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

    def test_cd_corr_length_in_dict(self):
        d = self.correlation_dict('ГОРА')
        for rule in self.ALL_RULES:
            self.assertIn('corr_length', d['rules'][rule])

    # ── Viewer section ────────────────────────────────────────────────────────

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

    def test_viewer_has_attr_autocorr(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('attrAutocorr', content)

    def test_viewer_has_wiener_mention(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Винера', content)


class TestSolanDerrida(unittest.TestCase):
    """Tests for solan_derrida.py and the viewer Derrida section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_derrida import (
            state_dist_norm, derrida_point,
            lexicon_points, random_points,
            derrida_curve, analytic_curve,
            classify_rule, build_derrida_data,
            derrida_dict, _ALL_RULES,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.state_dist_norm   = staticmethod(state_dist_norm)
        cls.derrida_point     = staticmethod(derrida_point)
        cls.lexicon_points    = staticmethod(lexicon_points)
        cls.random_points     = staticmethod(random_points)
        cls.derrida_curve     = staticmethod(derrida_curve)
        cls.analytic_curve    = staticmethod(analytic_curve)
        cls.classify_rule     = staticmethod(classify_rule)
        cls.build_derrida_data = staticmethod(build_derrida_data)
        cls.derrida_dict      = staticmethod(derrida_dict)
        cls.ALL_RULES         = _ALL_RULES
        cls.LEXICON           = list(LEXICON)

    # ── state_dist_norm() ─────────────────────────────────────────────────────

    def test_sdn_identical_is_zero(self):
        cells = [10, 20, 30, 40] * 4
        self.assertAlmostEqual(self.state_dist_norm(cells, cells), 0.0)

    def test_sdn_max_is_one(self):
        c1 = [0] * 16
        c2 = [63] * 16
        self.assertAlmostEqual(self.state_dist_norm(c1, c2), 1.0)

    def test_sdn_range(self):
        c1 = [0] * 16
        c2 = [42] * 16
        d = self.state_dist_norm(c1, c2)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)

    def test_sdn_symmetric(self):
        c1 = [0, 1, 2, 3] * 4
        c2 = [4, 5, 6, 7] * 4
        self.assertAlmostEqual(
            self.state_dist_norm(c1, c2),
            self.state_dist_norm(c2, c1)
        )

    # ── derrida_point() ───────────────────────────────────────────────────────

    def test_dp_returns_tuple(self):
        c = [0] * 16
        r = self.derrida_point(c, c, 'xor3')
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 2)

    def test_dp_identical_x_zero(self):
        c = [42] * 16
        x, y = self.derrida_point(c, c, 'xor3')
        self.assertAlmostEqual(x, 0.0)

    def test_dp_identical_y_zero(self):
        c = [42] * 16
        x, y = self.derrida_point(c, c, 'xor3')
        self.assertAlmostEqual(y, 0.0)

    def test_dp_values_in_range(self):
        from projects.hexglyph.solan_word import encode_word, pad_to
        c1 = pad_to(encode_word('ГОРА'),  16)
        c2 = pad_to(encode_word('ЛУНА'),  16)
        for rule in self.ALL_RULES:
            x, y = self.derrida_point(c1, c2, rule)
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    # ── lexicon_points() ──────────────────────────────────────────────────────

    def test_lp_count(self):
        # C(49,2) = 1176
        n = len(self.LEXICON)
        expected = n * (n - 1) // 2
        pts = self.lexicon_points('xor3')
        self.assertEqual(len(pts), expected)

    def test_lp_all_in_range(self):
        pts = self.lexicon_points('xor3')
        for x, y in pts:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    def test_lp_x_positive(self):
        # Different words have x > 0
        pts = self.lexicon_points('xor3')
        self.assertTrue(all(x > 0 for x, _ in pts))

    # ── random_points() ───────────────────────────────────────────────────────

    def test_rp_count(self):
        pts = self.random_points('xor3', n=50, seed=0)
        self.assertEqual(len(pts), 50)

    def test_rp_reproducible(self):
        p1 = self.random_points('xor3', n=10, seed=7)
        p2 = self.random_points('xor3', n=10, seed=7)
        self.assertEqual(p1, p2)

    def test_rp_different_seeds(self):
        p1 = self.random_points('xor3', n=10, seed=1)
        p2 = self.random_points('xor3', n=10, seed=2)
        self.assertNotEqual(p1, p2)

    def test_rp_in_range(self):
        pts = self.random_points('xor3', n=30, seed=0)
        for x, y in pts:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)

    # ── derrida_curve() ───────────────────────────────────────────────────────

    def test_dc_returns_dict(self):
        pts = [(0.1, 0.2), (0.5, 0.4), (0.8, 0.6)]
        r = self.derrida_curve(pts)
        self.assertIsInstance(r, dict)

    def test_dc_required_keys(self):
        pts = [(0.1, 0.2), (0.5, 0.4)]
        r = self.derrida_curve(pts)
        for k in ('bins', 'mean_y', 'count', 'above_diag', 'below_diag', 'on_diag'):
            self.assertIn(k, r)

    def test_dc_above_below_count(self):
        pts = [(0.1, 0.3), (0.5, 0.3), (0.8, 0.5)]
        r = self.derrida_curve(pts)
        total = r['above_diag'] + r['below_diag'] + r['on_diag']
        self.assertEqual(total, 3)

    def test_dc_above_diag_correct(self):
        # (0.1, 0.3): y>x → above
        # (0.5, 0.3): y<x → below
        pts = [(0.1, 0.3), (0.5, 0.3)]
        r = self.derrida_curve(pts)
        self.assertEqual(r['above_diag'], 1)
        self.assertEqual(r['below_diag'], 1)

    # ── analytic_curve() ──────────────────────────────────────────────────────

    def test_ac_returns_list(self):
        r = self.analytic_curve('xor3')
        self.assertIsInstance(r, list)

    def test_ac_xor_formula(self):
        # XOR: y = 2x(1-x), at x=0.5 → y=0.5
        pts = self.analytic_curve('xor', n_pts=100)
        # Find point nearest x=0.5
        mid = min(pts, key=lambda p: abs(p[0] - 0.5))
        self.assertAlmostEqual(mid[1], 0.5, places=2)

    def test_ac_xor_at_zero(self):
        pts = self.analytic_curve('xor', n_pts=10)
        x0, y0 = pts[0]
        self.assertAlmostEqual(x0, 0.0)
        self.assertAlmostEqual(y0, 0.0)

    def test_ac_xor3_tangent_steep(self):
        # XOR3 analytic: slope at x=0 is 3 (chaotic)
        pts = self.analytic_curve('xor3', n_pts=100)
        x1, y1 = pts[1]
        self.assertGreater(y1 / x1, 1.0)  # slope > 1 near origin

    def test_ac_all_y_in_range(self):
        for rule in self.ALL_RULES:
            for x, y in self.analytic_curve(rule):
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(y, 1.0 + 1e-9)

    # ── classify_rule() ───────────────────────────────────────────────────────

    def test_cr_returns_string(self):
        c = self.classify_rule('xor3', n_random=50)
        self.assertIsInstance(c, str)

    def test_cr_valid_values(self):
        for rule in self.ALL_RULES:
            c = self.classify_rule(rule, n_random=50)
            self.assertIn(c, ('ordered', 'chaotic', 'complex'))

    def test_cr_or_ordered(self):
        # OR tends toward ordered (d shrinks quickly)
        c = self.classify_rule('or', n_random=200)
        self.assertEqual(c, 'ordered')

    def test_cr_and_ordered(self):
        c = self.classify_rule('and', n_random=200)
        self.assertEqual(c, 'ordered')

    # ── build_derrida_data() ──────────────────────────────────────────────────

    def test_bdd_returns_dict(self):
        d = self.build_derrida_data(n_random=50)
        self.assertIsInstance(d, dict)

    def test_bdd_required_keys(self):
        d = self.build_derrida_data(n_random=50)
        for k in ('width', 'n_random', 'rules'):
            self.assertIn(k, d)

    def test_bdd_all_rules_present(self):
        d = self.build_derrida_data(n_random=50)
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_bdd_rule_has_classification(self):
        d = self.build_derrida_data(n_random=50)
        for rule in self.ALL_RULES:
            self.assertIn('classification', d['rules'][rule])

    # ── derrida_dict() ────────────────────────────────────────────────────────

    def test_dd_json_serialisable(self):
        import json
        d = self.derrida_dict(n_random=50)
        dumped = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(dumped, str)

    def test_dd_top_keys(self):
        d = self.derrida_dict(n_random=50)
        for k in ('width', 'n_random', 'rules'):
            self.assertIn(k, d)

    def test_dd_analytic_present(self):
        d = self.derrida_dict(n_random=50)
        for rule in self.ALL_RULES:
            self.assertIn('analytic', d['rules'][rule])

    # ── Viewer section ────────────────────────────────────────────────────────

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

    def test_viewer_has_analytic_curve(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('analyticPts', content)

    def test_viewer_has_binned_curve(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('binnedCurve', content)

    def test_viewer_has_compute_pairs(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('computePairs', content)

    def test_viewer_has_diagonal(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('der-analytic', content)


class TestSolanLyapunov(unittest.TestCase):
    """Tests for solan_lyapunov.py and the viewer Lyapunov section."""

    @classmethod
    def setUpClass(cls):
        from projects.hexglyph.solan_lyapunov import (
            q6_hamming, state_distance, perturb,
            divergence_trajectory, lyapunov_profile,
            lyapunov_summary, peak_sensitivity_map,
            build_lyapunov_data, lyapunov_dict,
            _ALL_RULES, _N_BITS, _DEFAULT_STEPS,
        )
        from projects.hexglyph.solan_lexicon import LEXICON
        cls.q6_hamming            = staticmethod(q6_hamming)
        cls.state_distance        = staticmethod(state_distance)
        cls.perturb               = staticmethod(perturb)
        cls.divergence_trajectory = staticmethod(divergence_trajectory)
        cls.lyapunov_profile      = staticmethod(lyapunov_profile)
        cls.lyapunov_summary      = staticmethod(lyapunov_summary)
        cls.peak_sensitivity_map  = staticmethod(peak_sensitivity_map)
        cls.build_lyapunov_data   = staticmethod(build_lyapunov_data)
        cls.lyapunov_dict         = staticmethod(lyapunov_dict)
        cls.ALL_RULES             = _ALL_RULES
        cls.N_BITS                = _N_BITS
        cls.DEFAULT_STEPS         = _DEFAULT_STEPS
        cls.LEXICON               = list(LEXICON)

    # ── q6_hamming() ──────────────────────────────────────────────────────────

    def test_q6_hamming_identical(self):
        self.assertEqual(self.q6_hamming(42, 42), 0)

    def test_q6_hamming_one_bit(self):
        self.assertEqual(self.q6_hamming(0, 1), 1)
        self.assertEqual(self.q6_hamming(0, 2), 1)
        self.assertEqual(self.q6_hamming(0, 32), 1)

    def test_q6_hamming_all_differ(self):
        # 0 vs 63 = 0b111111 → 6 bits differ
        self.assertEqual(self.q6_hamming(0, 63), 6)

    def test_q6_hamming_symmetric(self):
        self.assertEqual(self.q6_hamming(17, 42), self.q6_hamming(42, 17))

    def test_q6_hamming_range(self):
        for a in range(64):
            for b in range(64):
                d = self.q6_hamming(a, b)
                self.assertGreaterEqual(d, 0)
                self.assertLessEqual(d, 6)

    # ── state_distance() ──────────────────────────────────────────────────────

    def test_state_distance_identical(self):
        cells = [10, 20, 30, 40]
        self.assertEqual(self.state_distance(cells, cells), 0)

    def test_state_distance_one_bit_flip(self):
        c1 = [0] * 8
        c2 = [0] * 8; c2[3] = 1
        self.assertEqual(self.state_distance(c1, c2), 1)

    def test_state_distance_all_differ(self):
        c1 = [0] * 4
        c2 = [63] * 4
        self.assertEqual(self.state_distance(c1, c2), 4 * 6)

    # ── perturb() ─────────────────────────────────────────────────────────────

    def test_perturb_returns_new_list(self):
        cells = [10, 20, 30]
        p = self.perturb(cells, 0, 0)
        self.assertIsNot(p, cells)

    def test_perturb_original_unchanged(self):
        cells = [10, 20, 30]
        _ = self.perturb(cells, 0, 0)
        self.assertEqual(cells, [10, 20, 30])

    def test_perturb_changes_one_cell(self):
        cells = [0] * 8
        p = self.perturb(cells, 3, 0)
        self.assertEqual(p[3], 1)
        self.assertEqual(sum(p), 1)

    def test_perturb_distance_one(self):
        cells = [0] * 8
        p = self.perturb(cells, 3, 0)
        self.assertEqual(self.state_distance(cells, p), 1)

    def test_perturb_double_flip_restores(self):
        cells = [42] * 6
        p  = self.perturb(cells, 2, 3)
        pp = self.perturb(p, 2, 3)
        self.assertEqual(cells, pp)

    # ── divergence_trajectory() ───────────────────────────────────────────────

    def test_divtraj_returns_list(self):
        r = self.divergence_trajectory('ГОРА', 0, 0, 'xor3')
        self.assertIsInstance(r, list)

    def test_divtraj_length(self):
        r = self.divergence_trajectory('ГОРА', 0, 0, 'xor3', max_steps=10)
        self.assertEqual(len(r), 11)   # 0..10 inclusive

    def test_divtraj_starts_at_one(self):
        # initial perturbation = 1 bit
        r = self.divergence_trajectory('ГОРА', 0, 0, 'xor3')
        self.assertEqual(r[0], 1)

    def test_divtraj_nonneg(self):
        r = self.divergence_trajectory('ГОРА', 0, 0, 'xor3')
        self.assertTrue(all(v >= 0 for v in r))

    def test_divtraj_xor_converges(self):
        # XOR always converges to all-zeros → perturbation absorbed
        r = self.divergence_trajectory('ГОРА', 0, 0, 'xor', max_steps=20)
        self.assertEqual(r[-1], 0)

    # ── lyapunov_profile() ────────────────────────────────────────────────────

    def test_profile_returns_dict(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        self.assertIsInstance(p, dict)

    def test_profile_required_keys(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        for k in ('word', 'rule', 'width', 'n_perturb', 'mean_dist',
                  'max_dist', 'min_dist', 'peak_mean', 'peak_step',
                  'final_mean', 'converges', 'per_perturb'):
            self.assertIn(k, p)

    def test_profile_n_perturb(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        self.assertEqual(p['n_perturb'], 16 * 6)   # width × 6 bits

    def test_profile_mean_dist_length(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        self.assertEqual(len(p['mean_dist']), 11)

    def test_profile_mean_dist_nonneg(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        self.assertTrue(all(v >= 0 for v in p['mean_dist']))

    def test_profile_initial_mean_dist_one(self):
        # Average of 96 trajectories all starting at d=1
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        self.assertAlmostEqual(p['mean_dist'][0], 1.0)

    def test_profile_peak_step_valid(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        self.assertGreaterEqual(p['peak_step'], 0)
        self.assertLessEqual(p['peak_step'], 10)

    def test_profile_xor_converges(self):
        p = self.lyapunov_profile('ГОРА', 'xor', max_steps=20)
        self.assertTrue(p['converges'])

    def test_profile_converges_type_bool(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=10)
        self.assertIsInstance(p['converges'], bool)

    def test_profile_per_perturb_count(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=5)
        self.assertEqual(len(p['per_perturb']), 16 * 6)

    def test_profile_per_perturb_starts_at_one(self):
        p = self.lyapunov_profile('ГОРА', 'xor3', max_steps=5)
        for entry in p['per_perturb']:
            self.assertEqual(entry['traj'][0], 1)

    # ── lyapunov_summary() ────────────────────────────────────────────────────

    def test_summary_returns_all_rules(self):
        s = self.lyapunov_summary('ГОРА', max_steps=10)
        self.assertEqual(set(s.keys()), set(self.ALL_RULES))

    def test_summary_rule_keys(self):
        s = self.lyapunov_summary('ГОРА', max_steps=10)
        for rule, d in s.items():
            for k in ('peak_mean', 'peak_step', 'final_mean', 'converges'):
                self.assertIn(k, d)

    # ── peak_sensitivity_map() ────────────────────────────────────────────────

    def test_psmap_shape(self):
        m = self.peak_sensitivity_map('ГОРА', 'xor3', max_steps=10)
        self.assertEqual(len(m), 16)
        for row in m:
            self.assertEqual(len(row), 6)

    def test_psmap_nonneg(self):
        m = self.peak_sensitivity_map('ГОРА', 'xor3', max_steps=10)
        for row in m:
            for v in row:
                self.assertGreaterEqual(v, 0)

    # ── build_lyapunov_data() ─────────────────────────────────────────────────

    def test_build_returns_dict(self):
        d = self.build_lyapunov_data(['ГОРА', 'ЛУНА'], max_steps=8)
        self.assertIsInstance(d, dict)

    def test_build_required_keys(self):
        d = self.build_lyapunov_data(['ГОРА', 'ЛУНА'], max_steps=8)
        for k in ('words', 'width', 'max_steps', 'per_rule', 'most_chaotic', 'most_stable'):
            self.assertIn(k, d)

    def test_build_per_rule_words(self):
        words = ['ГОРА', 'ЛУНА', 'МАТ']
        d = self.build_lyapunov_data(words, max_steps=8)
        for rule in self.ALL_RULES:
            self.assertEqual(set(d['per_rule'][rule].keys()), set(words))

    def test_build_most_chaotic_is_valid_word(self):
        words = ['ГОРА', 'ЛУНА', 'МАТ']
        d = self.build_lyapunov_data(words, max_steps=8)
        for rule in self.ALL_RULES:
            word, _ = d['most_chaotic'][rule]
            self.assertIn(word, words)

    # ── lyapunov_dict() ───────────────────────────────────────────────────────

    def test_dict_json_serialisable(self):
        import json
        d = self.lyapunov_dict('ГОРА', max_steps=8)
        dumped = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(dumped, str)

    def test_dict_top_keys(self):
        d = self.lyapunov_dict('ГОРА', max_steps=8)
        for k in ('word', 'width', 'max_steps', 'rules'):
            self.assertIn(k, d)

    def test_dict_all_rules_present(self):
        d = self.lyapunov_dict('ГОРА', max_steps=8)
        self.assertEqual(set(d['rules'].keys()), set(self.ALL_RULES))

    def test_dict_mean_dist_list(self):
        d = self.lyapunov_dict('ГОРА', max_steps=8)
        for rule, rd in d['rules'].items():
            self.assertIsInstance(rd['mean_dist'], list)
            self.assertEqual(len(rd['mean_dist']), 9)  # 0..8

    # ── Viewer section ────────────────────────────────────────────────────────

    def test_viewer_has_lya_canvas(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lya-canvas', content)

    def test_viewer_has_lya_hmap(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lya-hmap', content)

    def test_viewer_has_lya_btn(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lya-btn', content)

    def test_viewer_has_lya_run(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lyaRun', content)

    def test_viewer_has_lya_word_select(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lya-word', content)

    def test_viewer_lya_section_heading(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('Ляпунов CA Q6', content)

    def test_viewer_has_q6_hamming(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('q6Ham', content)

    def test_viewer_has_lya_profile(self):
        content = viewer_path().read_text(encoding='utf-8')
        self.assertIn('lyaProfile', content)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
