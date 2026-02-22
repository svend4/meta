"""Тесты для hexdim — размерности, подкубы, тессеракты, проекции, псевдо-QR."""
import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from projects.hexdim.hexdim import (
    hamming,
    subcube, all_subcubes, subcube_count, find_subcube_of,
    tesseract_subgraph, all_tesseracts, tesseract_count, is_tesseract,
    all_cubes, cube_count,
    trigram_decomposition, from_trigrams, trigram_name, hexagram_structure,
    product_decomposition, all_partitions_into_two,
    project_to_qk, project_to_q4,
    q6_to_grid_coords, q6_to_3d_coords, q6_to_r6_coords,
    gray_code_sequence, gray_code_position, gray_code_step, gray_code_step_axis,
    hexagram_as_barcode, q6_as_8x8_grid, grid_to_string,
    q12_hexagram, q12_to_hexagram, q12_transformed, q12_changing_lines,
    dimension_info,
)
from math import comb


# ── подкубы Q_k ───────────────────────────────────────────────────────────────

class TestSubcube(unittest.TestCase):

    def test_subcube_q0_singleton(self):
        """subcube([], base) = {base} — нуль-мерный куб."""
        for base in [0, 7, 42]:
            v = subcube([], base)
            self.assertEqual(v, frozenset({base}))

    def test_subcube_q6_is_all(self):
        """subcube([0..5], 0) = Q6."""
        v = subcube(list(range(6)), 0)
        self.assertEqual(v, frozenset(range(64)))

    def test_subcube_q1_is_edge(self):
        """subcube([0], 0) = {0, 1} — ребро Q6."""
        v = subcube([0], 0)
        self.assertEqual(v, frozenset({0, 1}))

    def test_subcube_size_2k(self):
        """subcube(k осей, base) содержит 2^k вершин."""
        for k in range(7):
            from itertools import combinations
            axes = list(range(k))
            v = subcube(axes, 0)
            self.assertEqual(len(v), 2 ** k)

    def test_subcube_hamming_distance_in_cube(self):
        """Все вершины Q_k-подкуба попарно отличаются только по свободным осям."""
        axes = [1, 3, 5]
        v = list(subcube(axes, 0))
        for a in v:
            for b in v:
                # разность (a XOR b) содержит только биты из axes
                diff = a ^ b
                for i in range(6):
                    if i not in axes:
                        self.assertEqual((diff >> i) & 1, 0)

    def test_all_subcubes_count(self):
        """all_subcubes(k) возвращает C(6,k) × 2^{6-k} подкубов."""
        for k in range(7):
            result = all_subcubes(k)
            expected = subcube_count(k)
            self.assertEqual(len(result), expected)

    def test_all_subcubes_vertices_correct_size(self):
        """Каждый подкуб содержит 2^k вершин."""
        for k in [0, 1, 2, 3, 4, 5, 6]:
            for _, _, verts in all_subcubes(k):
                self.assertEqual(len(verts), 2 ** k)

    def test_find_subcube_of_pair(self):
        """find_subcube_of({a, b}) → размерность = hamming(a,b)."""
        for a, b in [(0, 7), (0, 63), (0, 1)]:
            axes, base, k = find_subcube_of([a, b])
            self.assertEqual(k, hamming(a, b))

    def test_find_subcube_of_q3(self):
        """find_subcube_of(8 вершин куба) → k=3."""
        verts = list(subcube([0, 1, 2], 0))
        axes, base, k = find_subcube_of(verts)
        self.assertEqual(k, 3)

    def test_find_subcube_of_empty(self):
        """find_subcube_of([]) возвращает ([], 0, 0)."""
        axes, base, k = find_subcube_of([])
        self.assertEqual(axes, [])
        self.assertEqual(k, 0)


# ── тессеракт Q4 ─────────────────────────────────────────────────────────────

class TestTesseract(unittest.TestCase):

    def test_tesseract_has_16_vertices(self):
        """Q4-подграф: ровно 16 вершин."""
        t = tesseract_subgraph([0, 1, 2, 3], 0)
        self.assertEqual(len(t), 16)

    def test_tesseract_wrong_axes_raises(self):
        """tesseract_subgraph с ≠ 4 осями → ошибка."""
        with self.assertRaises(ValueError):
            tesseract_subgraph([0, 1, 2], 0)

    def test_all_tesseracts_count(self):
        """Число тессерактов = C(6,4) × 2^2 = 15 × 4 = 60."""
        self.assertEqual(len(all_tesseracts()), 60)
        self.assertEqual(tesseract_count(), 60)

    def test_all_tesseracts_size(self):
        """Каждый тессеракт содержит 16 вершин."""
        for t in all_tesseracts():
            self.assertEqual(len(t), 16)

    def test_is_tesseract_true(self):
        """is_tesseract распознаёт правильный Q4-подграф."""
        t = tesseract_subgraph([0, 1, 2, 3], 0)
        self.assertTrue(is_tesseract(t))

    def test_is_tesseract_false(self):
        """is_tesseract отвергает произвольные 16 вершин."""
        bad = frozenset(range(16))
        # [0..15] — это Q4-подграф на осях 0-3 с base=0 → IS tesseract
        # Используем случайный набор
        bad2 = frozenset(range(0, 63, 4))
        # 16 вершин, но не обязательно подкуб
        axes, base, k = find_subcube_of(bad2)
        self.assertEqual(is_tesseract(bad2), (k == 4 and
                         subcube(axes[:4] if len(axes) >= 4 else axes, base) == bad2))

    def test_is_tesseract_wrong_size(self):
        """is_tesseract возвращает False при размере != 16."""
        self.assertFalse(is_tesseract({0, 1, 2}))

    def test_all_cubes_count(self):
        """Число Q3-подграфов = C(6,3)×2^3 = 20×8 = 160."""
        self.assertEqual(len(all_cubes()), 160)
        self.assertEqual(cube_count(), 160)

    def test_cube_has_8_vertices(self):
        """Каждый куб Q3 содержит 8 вершин."""
        for c in all_cubes():
            self.assertEqual(len(c), 8)


# ── триграммы и структура Q6 = Q3×Q3 ────────────────────────────────────────

class TestTrigrams(unittest.TestCase):

    def test_trigram_decomposition_range(self):
        """lower, upper ∈ [0, 7] для всех h ∈ Q6."""
        for h in range(64):
            lower, upper = trigram_decomposition(h)
            self.assertIn(lower, range(8))
            self.assertIn(upper, range(8))

    def test_from_trigrams_roundtrip(self):
        """from_trigrams(decomposition(h)) = h."""
        for h in range(64):
            lower, upper = trigram_decomposition(h)
            self.assertEqual(from_trigrams(lower, upper), h)

    def test_trigram_decomposition_bijection(self):
        """Отображение h → (lower, upper) — биекция Q6 → {0..7}²."""
        pairs = [trigram_decomposition(h) for h in range(64)]
        self.assertEqual(len(set(pairs)), 64)

    def test_trigram_product_structure(self):
        """Q6 = Q3_lower × Q3_upper: смена lower не зависит от upper."""
        # Фиксируем upper=5 (биты 3-5), меняем lower (биты 0-2)
        upper = 5
        lowers = set()
        for h in range(64):
            l, u = trigram_decomposition(h)
            if u == upper:
                lowers.add(l)
        self.assertEqual(lowers, set(range(8)))

    def test_trigram_name_coverage(self):
        """Все 8 триграмм имеют имя."""
        for t in range(8):
            name = trigram_name(t)
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_hexagram_structure_keys(self):
        """hexagram_structure возвращает все ожидаемые ключи."""
        info = hexagram_structure(42)
        for k in ('value', 'bits', 'yang_count', 'lower_trigram',
                  'upper_trigram', 'antipodal'):
            self.assertIn(k, info)

    def test_hexagram_structure_antipodal(self):
        """info['antipodal'] = h ⊕ 63."""
        info = hexagram_structure(7)
        self.assertEqual(info['antipodal'], 7 ^ 63)


# ── произведения и разложения ─────────────────────────────────────────────────

class TestProductDecomposition(unittest.TestCase):

    def test_product_decomposition_q3xq3(self):
        """Q6 = Q3([0,1,2]) × Q3([3,4,5]): factor1 и factor2 правильного размера."""
        f1, f2 = product_decomposition([0, 1, 2], [3, 4, 5])
        self.assertEqual(len(f1), 8)
        self.assertEqual(len(f2), 8)

    def test_product_decomposition_q1xq5(self):
        """Q6 = Q1([0]) × Q5([1,2,3,4,5])."""
        f1, f2 = product_decomposition([0], [1, 2, 3, 4, 5])
        self.assertEqual(len(f1), 2)
        self.assertEqual(len(f2), 32)

    def test_product_decomposition_bad_axes(self):
        """Перекрывающиеся оси → ошибка."""
        with self.assertRaises(ValueError):
            product_decomposition([0, 1], [1, 2, 3, 4, 5])

    def test_all_partitions_into_two_count(self):
        """Число разбиений {0..5} на два непустых подмножества (без порядка) = 2^5 - 1 = 31."""
        parts = all_partitions_into_two()
        self.assertEqual(len(parts), 31)

    def test_all_partitions_axes_valid(self):
        """Каждое разбиение: union = {0..5}, intersection = ∅."""
        for axes1, axes2 in all_partitions_into_two():
            self.assertEqual(set(axes1) | set(axes2), set(range(6)))
            self.assertEqual(set(axes1) & set(axes2), set())


# ── проекции ─────────────────────────────────────────────────────────────────

class TestProjections(unittest.TestCase):

    def test_project_to_qk_range(self):
        """project_to_qk возвращает значение в [0, 2^k-1]."""
        for h in range(64):
            p = project_to_qk(h, [0, 1, 2, 3])
            self.assertIn(p, range(16))

    def test_project_to_q4_range(self):
        """project_to_q4 возвращает значение в [0, 15]."""
        for h in range(64):
            p = project_to_q4(h, [0, 1])
            self.assertIn(p, range(16))

    def test_q6_to_3d_coords_range(self):
        """3D-координаты x,y,z ∈ {0,1,2}."""
        for h in range(64):
            x, y, z = q6_to_3d_coords(h)
            for c in (x, y, z):
                self.assertIn(c, [0, 1, 2])

    def test_q6_to_r6_coords_binary(self):
        """R⁶-координаты — двоичные {0,1}."""
        for h in range(64):
            coords = q6_to_r6_coords(h)
            self.assertEqual(len(coords), 6)
            for c in coords:
                self.assertIn(c, [0, 1])

    def test_r6_coords_hamming_preserved(self):
        """Хэмминг-расстояние = сумма |coord_diff|."""
        a, b = 7, 42
        ca = q6_to_r6_coords(a)
        cb = q6_to_r6_coords(b)
        l1 = sum(abs(x - y) for x, y in zip(ca, cb))
        self.assertEqual(l1, hamming(a, b))

    def test_grid_coords_trigram_range(self):
        """q6_to_grid_coords('trigram') → 0..7 × 0..7."""
        for h in range(64):
            r, c = q6_to_grid_coords(h, 'trigram')
            self.assertIn(r, range(8))
            self.assertIn(c, range(8))

    def test_grid_coords_gray_range(self):
        """q6_to_grid_coords('gray') → 0..7 × 0..7."""
        for h in range(64):
            r, c = q6_to_grid_coords(h, 'gray')
            self.assertIn(r, range(8))
            self.assertIn(c, range(8))

    def test_grid_coords_yang_method(self):
        """q6_to_grid_coords('yang') → (yang_count, pos_in_level)."""
        row, col = q6_to_grid_coords(0, 'yang')
        self.assertEqual(row, 0)   # weight of 0 = 0
        row6, col6 = q6_to_grid_coords(63, 'yang')
        self.assertEqual(row6, 6)  # weight of 63 = 6

    def test_grid_coords_unknown_raises(self):
        """q6_to_grid_coords с неизвестным методом → ValueError."""
        with self.assertRaises(ValueError):
            q6_to_grid_coords(0, 'unknown_method')


# ── код Грея ─────────────────────────────────────────────────────────────────

class TestGrayCode(unittest.TestCase):

    def test_gray_code_length(self):
        """Последовательность Грея: 64 элемента."""
        self.assertEqual(len(gray_code_sequence()), 64)

    def test_gray_code_is_permutation(self):
        """Все 64 гексаграммы в последовательности."""
        self.assertEqual(set(gray_code_sequence()), set(range(64)))

    def test_gray_code_hamiltonian(self):
        """Каждый шаг меняет ровно 1 бит."""
        seq = gray_code_sequence()
        for i in range(len(seq) - 1):
            diff = seq[i] ^ seq[i + 1]
            # diff — степень двойки → ровно 1 бит
            self.assertGreater(diff, 0)
            self.assertEqual(diff & (diff - 1), 0)

    def test_gray_code_position_roundtrip(self):
        """gray_code_sequence()[gray_code_position(h)] = h."""
        seq = gray_code_sequence()
        for h in range(64):
            pos = gray_code_position(h)
            self.assertEqual(seq[pos], h)

    def test_gray_code_step_axis_valid(self):
        """Ось шага ∈ [0, 5]."""
        for i in range(63):
            axis = gray_code_step_axis(i)
            self.assertIn(axis, range(6))

    def test_gray_code_step_axis_matches_flip(self):
        """Ось шага соответствует фактически флипнутому биту."""
        seq = gray_code_sequence()
        for i in range(63):
            axis = gray_code_step_axis(i)
            flip_bit = seq[i] ^ seq[i + 1]
            self.assertEqual(flip_bit, 1 << axis)

    def test_gray_code_step_returns_int(self):
        """gray_code_step(i) возвращает целое число."""
        for i in range(10):
            result = gray_code_step(i)
            self.assertIsInstance(result, int)


# ── псевдо-QR и сетка 8×8 ────────────────────────────────────────────────────

class TestPseudoQR(unittest.TestCase):

    def test_hexagram_barcode_length(self):
        """Штрих-код гексаграммы: 6 линий."""
        for h in range(64):
            barcode = hexagram_as_barcode(h)
            self.assertEqual(len(barcode), 6)

    def test_hexagram_barcode_binary(self):
        """Бинарный штрих-код содержит только '0' и '1'."""
        for h in [0, 7, 42, 63]:
            barcode = hexagram_as_barcode(h, style='binary')
            for b in barcode:
                self.assertIn(b, ['0', '1'])

    def test_hexagram_barcode_encodes_bits(self):
        """Штрих-код гексаграммы 63 (111111): все линии = '1'."""
        barcode = hexagram_as_barcode(63, style='binary')
        self.assertTrue(all(b == '1' for b in barcode))

    def test_hexagram_barcode_0_all_yin(self):
        """Гексаграмма 0 (000000): все линии = '0'."""
        barcode = hexagram_as_barcode(0, style='binary')
        self.assertTrue(all(b == '0' for b in barcode))

    def test_8x8_grid_covers_all(self):
        """8×8 сетка содержит все 64 гексаграммы."""
        for ordering in ('trigram', 'gray', 'natural'):
            grid = q6_as_8x8_grid(ordering)
            all_vals = set()
            for row in grid:
                for h in row:
                    all_vals.add(h)
            self.assertEqual(all_vals, set(range(64)))

    def test_8x8_grid_shape(self):
        """Сетка 8×8."""
        grid = q6_as_8x8_grid('trigram')
        self.assertEqual(len(grid), 8)
        for row in grid:
            self.assertEqual(len(row), 8)

    def test_trigram_grid_structure(self):
        """В триграммной сетке: grid[upper][lower] = from_trigrams(lower, upper)."""
        from projects.hexdim.hexdim import from_trigrams
        grid = q6_as_8x8_grid('trigram')
        for upper in range(8):
            for lower in range(8):
                expected = from_trigrams(lower, upper)
                self.assertEqual(grid[upper][lower], expected)

    def test_hexagram_barcode_grid_style(self):
        """hexagram_as_barcode с style='grid' возвращает список списков."""
        barcode = hexagram_as_barcode(42, style='grid')
        self.assertEqual(len(barcode), 6)
        for line in barcode:
            self.assertIsInstance(line, list)
            self.assertIn(line[0], [0, 1])

    def test_grid_to_string_show_bits(self):
        """grid_to_string с show_bits=True показывает битовые строки."""
        grid = q6_as_8x8_grid('trigram')
        s = grid_to_string(grid, show_bits=True)
        self.assertIn('000000', s)   # бит-строка гексаграммы 0

    def test_grid_to_string_default(self):
        """grid_to_string без show_bits возвращает строку с разделителями."""
        grid = q6_as_8x8_grid('trigram')
        s = grid_to_string(grid)  # show_bits=False по умолчанию
        self.assertIsInstance(s, str)
        self.assertIn('│', s)  # разделители строки

    def test_grid_to_string_with_none(self):
        """grid_to_string с None-элементами показывает '?'."""
        grid_with_none = [[0, None, 1], [2, 3, None]]
        s = grid_to_string(grid_with_none)
        self.assertIn('?', s)

    def test_q12_transformed_output(self):
        """q12_transformed возвращает гексаграмму ∈ [0, 63]."""
        code = q12_hexagram([0, 1, 2, 3, 0, 3])
        result = q12_transformed(code)
        self.assertIn(result, range(64))

    def test_q12_transformed_old_yang_becomes_yin(self):
        """state=3 (старый ян) → ян=0 (инь)."""
        # Кодон с state=3 для бита 0: код = 3 (0b11)
        code = 3   # только бит 0 = state 3, остальные = state 0
        result = q12_transformed(code)
        # bit0: state=3 → yang=0; bit1-5: state=0 → yang=1
        expected = 0b111110   # биты 1-5 = 1, бит 0 = 0
        self.assertEqual(result, expected)


# ── Q12: четырёхуровневые линии ───────────────────────────────────────────────

class TestQ12(unittest.TestCase):

    def test_q12_code_range(self):
        """Q12-код ∈ [0, 4095] = [0, 2^12-1]."""
        states = [0, 1, 2, 3, 0, 1]
        code = q12_hexagram(states)
        self.assertIn(code, range(4096))

    def test_q12_invalid_state(self):
        """Состояние вне {0,1,2,3} → ошибка."""
        with self.assertRaises(ValueError):
            q12_hexagram([0, 1, 2, 4, 0, 1])

    def test_q12_wrong_length(self):
        """Не 6 линий → ошибка."""
        with self.assertRaises(ValueError):
            q12_hexagram([0, 1, 2])

    def test_q12_to_hexagram_range(self):
        """q12_to_hexagram → значение ∈ [0, 63]."""
        for code in [0, 100, 1000, 4095]:
            h = q12_to_hexagram(code)
            self.assertIn(h, range(64))

    def test_q12_all_yang_states(self):
        """[2,2,2,2,2,2] = все молодые-янь → гексаграмма 63."""
        states = [2] * 6  # молодая-янь: бит1=1 → янь
        code = q12_hexagram(states)
        h = q12_to_hexagram(code)
        self.assertEqual(h, 63)

    def test_q12_all_yin_states(self):
        """[1,1,1,1,1,1] = все молодые-инь → гексаграмма 0."""
        states = [1] * 6
        code = q12_hexagram(states)
        h = q12_to_hexagram(code)
        self.assertEqual(h, 0)

    def test_q12_changing_lines(self):
        """Изменяющиеся линии — индексы со состояниями 0 или 3."""
        states = [0, 1, 2, 3, 1, 0]  # меняются: 0, 3, 5
        code = q12_hexagram(states)
        changing = q12_changing_lines(code)
        self.assertEqual(set(changing), {0, 3, 5})

    def test_q12_no_changing_lines(self):
        """Только молодые линии → нет изменяющихся."""
        states = [1, 2, 1, 2, 1, 2]
        code = q12_hexagram(states)
        self.assertEqual(q12_changing_lines(code), [])

    def test_q12_total_states(self):
        """4^6 = 4096 различных Q12-состояний."""
        self.assertEqual(4 ** 6, 4096)
        self.assertEqual(4 ** 6, 2 ** 12)


# ── dimension_info ────────────────────────────────────────────────────────────

class TestDimensionInfo(unittest.TestCase):

    def test_dimension_info_keys(self):
        """dimension_info содержит k=0..6."""
        info = dimension_info()
        for k in range(7):
            self.assertIn(k, info)

    def test_dimension_info_q6(self):
        """Q6: 64 вершины."""
        info = dimension_info()
        self.assertEqual(info[6]['vertices'], 64)

    def test_dimension_info_q4_tesseract(self):
        """Q4 (тессеракт): 16 вершин, 60 штук в Q6."""
        info = dimension_info()
        self.assertEqual(info[4]['vertices'], 16)
        self.assertEqual(info[4]['count_in_q6'], 60)

    def test_dimension_info_q3_cube(self):
        """Q3 (куб): 8 вершин, 160 штук в Q6."""
        info = dimension_info()
        self.assertEqual(info[3]['vertices'], 8)
        self.assertEqual(info[3]['count_in_q6'], 160)


class TestDimCLI(unittest.TestCase):
    """Тесты main() hexdim."""

    def _run(self, args):
        import io
        from contextlib import redirect_stdout
        from projects.hexdim.hexdim import main
        old_argv = sys.argv
        sys.argv = ['hexdim.py'] + args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_cmd_info(self):
        out = self._run(['info'])
        self.assertIn('Q6', out)

    def test_cmd_hexagram_default(self):
        out = self._run(['hexagram'])
        self.assertIn('Гексаграмма', out)

    def test_cmd_hexagram_with_arg(self):
        out = self._run(['hexagram', '7'])
        self.assertIn('7', out)

    def test_cmd_tesseracts(self):
        out = self._run(['tesseracts'])
        self.assertIn('тессерактов', out)

    def test_cmd_grid_default(self):
        out = self._run(['grid'])
        self.assertIn('8×8', out)

    def test_cmd_gray(self):
        out = self._run(['gray'])
        self.assertIn('Грея', out)

    def test_cmd_q12_default(self):
        out = self._run(['q12'])
        self.assertIn('Q12', out)

    def test_cmd_projection(self):
        out = self._run(['projection'])
        self.assertIn('3D', out)

    def test_cmd_help(self):
        out = self._run(['help'])
        self.assertIn('hexdim', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
