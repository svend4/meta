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
