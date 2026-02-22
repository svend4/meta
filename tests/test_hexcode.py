"""Тесты hexcode — двоичные линейные коды в Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexcode.hexcode import (
    BinaryCode, repetition_code, parity_check_code, shortened_hamming_code,
    hexcode_312, full_space_code, even_weight_code,
    singleton_bound, hamming_bound, plotkin_bound, feasible,
    dual_repetition_code, find_codes, min_covering_code,
    _gf2_dot, _int_to_bits, _bits_to_int, _row_reduce_gf2,
)
from libs.hexcore.hexcore import hamming, SIZE


class TestGF2Utils(unittest.TestCase):
    def test_gf2_dot(self):
        self.assertEqual(_gf2_dot([1, 1, 0], [1, 0, 1]), 1)
        self.assertEqual(_gf2_dot([1, 1], [1, 1]), 0)

    def test_int_to_bits_roundtrip(self):
        for v in range(64):
            bits = _int_to_bits(v, 6)
            self.assertEqual(_bits_to_int(bits), v)
            self.assertEqual(len(bits), 6)

    def test_row_reduce_identity(self):
        I = [[int(i == j) for j in range(4)] for i in range(4)]
        reduced, pivots = _row_reduce_gf2(I)
        self.assertEqual(pivots, [0, 1, 2, 3])

    def test_row_reduce_rank_deficient(self):
        # Две одинаковые строки → ранг 1
        M = [[1, 0, 1], [1, 0, 1]]
        _, pivots = _row_reduce_gf2(M)
        self.assertEqual(len(pivots), 1)


class TestBinaryCodeBasic(unittest.TestCase):
    def test_creates_code(self):
        code = BinaryCode([[1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1]])
        self.assertIsInstance(code, BinaryCode)

    def test_dimension(self):
        code = BinaryCode([[1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1]])
        self.assertEqual(code.k, 2)

    def test_length(self):
        code = BinaryCode([[1, 0, 0, 1, 1, 0]])
        self.assertEqual(code.n, 6)

    def test_codewords_contains_zero(self):
        """Нулевое слово всегда в коде."""
        code = BinaryCode([[1, 1, 1, 1, 1, 1]])
        self.assertIn(0, code.codewords())

    def test_codewords_count(self):
        """Код размерности k содержит 2^k слов."""
        code = BinaryCode([[1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1]])
        self.assertEqual(len(code.codewords()), 4)

    def test_is_codeword(self):
        code = repetition_code()
        self.assertTrue(code.is_codeword(0))
        self.assertTrue(code.is_codeword(63))
        self.assertFalse(code.is_codeword(1))

    def test_linearity(self):
        """XOR двух кодовых слов — тоже кодовое слово."""
        code = hexcode_312()
        cws = code.codewords()
        for a in cws:
            for b in cws:
                self.assertIn(a ^ b, cws)

    def test_empty_generator_raises(self):
        with self.assertRaises(ValueError):
            BinaryCode([])

    def test_wrong_row_length_raises(self):
        with self.assertRaises(ValueError):
            BinaryCode([[1, 0, 0, 1]])   # длина 4, не 6


class TestRepetitionCode(unittest.TestCase):
    def test_parameters(self):
        code = repetition_code()
        self.assertEqual(code.k, 1)
        self.assertEqual(code.min_distance(), 6)

    def test_codewords(self):
        code = repetition_code()
        self.assertEqual(set(code.codewords()), {0, 63})

    def test_mds(self):
        """[6,1,6]: d = n-k+1 = 6. MDS."""
        self.assertTrue(repetition_code().is_mds())

    def test_covering_radius(self):
        """Покрывающий радиус = 3 (ровно половина пути до антипода)."""
        self.assertEqual(repetition_code().covering_radius(), 3)

    def test_not_perfect(self):
        """[6,1,6] не совершенный: 2*Σ C(6,i) ≠ 64."""
        self.assertFalse(repetition_code().is_perfect())


class TestParityCheckCode(unittest.TestCase):
    def test_parameters(self):
        code = parity_check_code()
        self.assertEqual(code.k, 5)
        self.assertEqual(code.min_distance(), 2)

    def test_codewords_even_weight(self):
        """Все кодовые слова имеют чётный вес."""
        code = parity_check_code()
        for cw in code.codewords():
            self.assertEqual(bin(cw).count('1') % 2, 0)

    def test_size(self):
        self.assertEqual(len(parity_check_code().codewords()), 32)

    def test_mds(self):
        """[6,5,2]: d = n-k+1 = 2. MDS."""
        self.assertTrue(parity_check_code().is_mds())


class TestShortenedHamming(unittest.TestCase):
    def test_dimension(self):
        code = shortened_hamming_code()
        self.assertEqual(code.k, 3)

    def test_min_distance(self):
        code = shortened_hamming_code()
        self.assertEqual(code.min_distance(), 3)

    def test_size(self):
        code = shortened_hamming_code()
        self.assertEqual(len(code.codewords()), 8)


class TestHexCode312(unittest.TestCase):
    def test_parameters(self):
        code = hexcode_312()
        self.assertEqual(code.k, 3)
        self.assertEqual(code.min_distance(), 3)

    def test_codewords_count(self):
        self.assertEqual(len(hexcode_312().codewords()), 8)

    def test_error_correction_capability(self):
        """t = (3-1)//2 = 1: исправляет 1 ошибку."""
        code = hexcode_312()
        d = code.min_distance()
        self.assertEqual((d - 1) // 2, 1)


class TestFullSpaceCode(unittest.TestCase):
    def test_parameters(self):
        code = full_space_code()
        self.assertEqual(code.k, 6)
        self.assertEqual(code.min_distance(), 1)

    def test_all_codewords(self):
        """Полный код содержит все 64 гексаграммы."""
        code = full_space_code()
        self.assertEqual(set(code.codewords()), set(range(SIZE)))


class TestEncodeDecode(unittest.TestCase):
    def test_encode_zero_message(self):
        """Нулевое сообщение → нулевое кодовое слово."""
        code = hexcode_312()
        cw = code.encode([0, 0, 0])
        self.assertEqual(cw, 0)

    def test_encode_is_codeword(self):
        """Результат кодирования — кодовое слово."""
        code = hexcode_312()
        for msg in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]:
            cw = code.encode(msg)
            self.assertIn(cw, code.codewords())

    def test_encode_wrong_length_raises(self):
        code = hexcode_312()
        with self.assertRaises(ValueError):
            code.encode([1, 0])

    def test_decode_codeword_unchanged(self):
        """Декодирование кодового слова без ошибок → то же слово."""
        code = hexcode_312()
        for cw in code.codewords():
            result = code.decode(cw)
            self.assertEqual(result, cw)

    def test_decode_single_error(self):
        """Исправление одной ошибки для кода с d=3."""
        code = hexcode_312()
        for cw in code.codewords():
            for bit in range(6):
                received = cw ^ (1 << bit)
                decoded = code.decode(received)
                self.assertEqual(decoded, cw,
                    f"Ошибка декодирования: cw={cw:06b}, bit={bit}, decoded={decoded}")

    def test_decode_repetition_single_error(self):
        """Repetition code исправляет ошибки весом ≤ 2."""
        code = repetition_code()
        # 0-слово = 000000, ошибка в 1 бите
        decoded = code.decode(1)   # 000001 → должно дать 0
        self.assertEqual(decoded, 0)

    def test_nearest_codeword(self):
        """nearest_codeword всегда возвращает кодовое слово."""
        code = hexcode_312()
        for h in range(SIZE):
            nc = code.nearest_codeword(h)
            self.assertIn(nc, code.codewords())


class TestParityCheckMatrix(unittest.TestCase):
    def test_orthogonality(self):
        """H·c = 0 для всех кодовых слов c."""
        for factory in [repetition_code, parity_check_code, hexcode_312]:
            code = factory()
            H = code.parity_check_matrix()
            if not H:
                continue
            for cw in code.codewords():
                cw_bits = _int_to_bits(cw, 6)
                for h_row in H:
                    dot = _gf2_dot(h_row, cw_bits)
                    self.assertEqual(dot, 0, f"Ортогональность нарушена: {factory.__name__}")

    def test_rows_count(self):
        """Матрица H имеет n-k строк."""
        code = hexcode_312()
        H = code.parity_check_matrix()
        self.assertEqual(len(H), code.n - code.k)


class TestCosets(unittest.TestCase):
    def test_cosets_partition(self):
        """Смежные классы разбивают (Z₂)⁶ без пересечений."""
        code = hexcode_312()
        cosets = code.cosets()
        all_pts = set()
        for coset in cosets:
            s = set(coset)
            self.assertEqual(s & all_pts, set(), "Пересечение смежных классов")
            all_pts |= s
        self.assertEqual(all_pts, set(range(SIZE)))

    def test_cosets_count(self):
        """Число смежных классов = 2^(n-k)."""
        code = hexcode_312()
        self.assertEqual(len(code.cosets()), SIZE // len(code.codewords()))


class TestDualCode(unittest.TestCase):
    def test_dual_rep_is_parity(self):
        """Двойственный к [6,1,6] = [6,5,2] (код чётности)."""
        rep = repetition_code()
        dual = BinaryCode.dual(rep)
        self.assertEqual(dual.k, 5)
        self.assertEqual(dual.min_distance(), 2)

    def test_double_dual(self):
        """Двойственный к двойственному = исходный код (по кодовым словам)."""
        code = hexcode_312()
        dual = BinaryCode.dual(code)
        dual_dual = BinaryCode.dual(dual)
        self.assertEqual(frozenset(code.codewords()), frozenset(dual_dual.codewords()))


class TestBounds(unittest.TestCase):
    def test_singleton_feasible(self):
        """d ≤ n-k+1 для всех реальных кодов."""
        for code in [repetition_code(), parity_check_code(), hexcode_312()]:
            self.assertTrue(singleton_bound(code.k, code.min_distance()))

    def test_hamming_bound_perfect(self):
        """Совершенный код достигает границы Хэмминга с равенством."""
        # Нет совершенного [6,k,d] кода с n=6, но проверим что граница Хэмминга работает
        # Repetition [6,1,6]: t=2, sphere_vol = C(6,0)+C(6,1)+C(6,2) = 1+6+15=22
        # 2^1 * 22 = 44 ≤ 64 = 2^6 ✓
        self.assertTrue(hamming_bound(1, 6))

    def test_plotkin_inapplicable(self):
        """Граница Плоткина неприменима если 2d ≤ n."""
        # d=3, n=6: 2*3=6=n → граница применима
        # d=2, n=6: 2*2=4 < n=6 → неприменима (всегда True)
        self.assertTrue(plotkin_bound(5, 2))   # 2*2 ≤ 6 → True (не применима)

    def test_feasible_known_codes(self):
        """Параметры реальных кодов должны быть допустимы."""
        self.assertTrue(feasible(1, 6))   # [6,1,6] repetition
        self.assertTrue(feasible(5, 2))   # [6,5,2] parity
        self.assertTrue(feasible(3, 3))   # [6,3,3] hexcode

    def test_infeasible_params(self):
        """[6,6,6] — весь код не может иметь расстояние 6."""
        self.assertFalse(feasible(6, 6))


class TestWeightDistribution(unittest.TestCase):
    def test_rep_code_distribution(self):
        """Repetition code: веса {0: 1, 6: 1}."""
        d = repetition_code().weight_distribution()
        self.assertEqual(d[0], 1)
        self.assertEqual(d[6], 1)
        self.assertEqual(sum(d.values()), 2)

    def test_parity_distribution_symmetric(self):
        """Код чётности: только чётные веса."""
        d = parity_check_code().weight_distribution()
        for w in d:
            self.assertEqual(w % 2, 0)

    def test_total_count(self):
        """Сумма весов = число кодовых слов."""
        code = hexcode_312()
        d = code.weight_distribution()
        self.assertEqual(sum(d.values()), len(code.codewords()))


class TestDualRepetitionCode(unittest.TestCase):
    """Тесты dual_repetition_code — [6,5,2]-код."""

    def test_returns_binary_code(self):
        code = dual_repetition_code()
        self.assertIsInstance(code, BinaryCode)

    def test_n_equals_6(self):
        code = dual_repetition_code()
        self.assertEqual(code.n, 6)

    def test_k_equals_5(self):
        code = dual_repetition_code()
        self.assertEqual(code.k, 5)

    def test_min_distance_2(self):
        code = dual_repetition_code()
        self.assertEqual(code.min_distance(), 2)

    def test_codewords_count(self):
        """[6,5,2] имеет 2^5 = 32 кодовых слова."""
        code = dual_repetition_code()
        self.assertEqual(len(code.codewords()), 32)


class TestFindCodes(unittest.TestCase):
    """Тесты find_codes — поиск линейных кодов."""

    def test_returns_list(self):
        codes = find_codes(d_min=1, k=1)
        self.assertIsInstance(codes, list)

    def test_k1_d1_finds_63_codes(self):
        """При k=1, d_min=1 находит все 63 ненулевых однострочных кода."""
        codes = find_codes(d_min=1, k=1)
        self.assertEqual(len(codes), 63)

    def test_k1_d6_finds_one_code(self):
        """Код d=6 при k=1 — только repetition code ([6,1,6])."""
        codes = find_codes(d_min=6, k=1)
        self.assertEqual(len(codes), 1)
        self.assertEqual(codes[0].min_distance(), 6)

    def test_all_found_meet_d_min(self):
        """Все найденные коды имеют min_distance ≥ d_min."""
        codes = find_codes(d_min=3, k=1)
        for c in codes:
            self.assertGreaterEqual(c.min_distance(), 3)


class TestMinCoveringCode(unittest.TestCase):
    """Тесты min_covering_code — минимальный линейный код с покрывающим радиусом."""

    def test_returns_binary_code(self):
        code = min_covering_code(radius=3)
        self.assertIsInstance(code, BinaryCode)

    def test_n_equals_6(self):
        code = min_covering_code(radius=3)
        self.assertEqual(code.n, 6)

    def test_k_is_positive(self):
        code = min_covering_code(radius=3)
        self.assertGreater(code.k, 0)

    def test_covering_radius_satisfied(self):
        """Покрывающий радиус ≤ заданному radius=3."""
        from projects.hexgeom.hexgeom import hamming_ball
        code = min_covering_code(radius=3)
        cws = code.codewords()
        # Каждая из 64 вершин Q6 на расстоянии ≤ 3 от какого-то кодового слова
        covered = set()
        for cw in cws:
            for h in hamming_ball(cw, 3):
                covered.add(h)
        self.assertEqual(len(covered), 64)


class TestBinaryCodeDisplay(unittest.TestCase):
    """Тесты метода display() для BinaryCode."""

    def test_display_returns_string(self):
        from projects.hexcode.hexcode import even_weight_code
        code = even_weight_code()
        result = code.display()
        self.assertIsInstance(result, str)

    def test_display_contains_rate(self):
        from projects.hexcode.hexcode import even_weight_code
        code = even_weight_code()
        result = code.display()
        self.assertIn('R =', result)

    def test_display_contains_codewords(self):
        from projects.hexcode.hexcode import even_weight_code
        code = even_weight_code()
        result = code.display()
        # should contain bit strings of codewords
        self.assertIn('000000', result)


class TestPlotkinBoundHighD(unittest.TestCase):
    """Тесты plotkin_bound с 2*d > n (фактическое применение границы)."""

    def test_plotkin_bound_high_d_true(self):
        """[6,1,6] repetition code: 2*6=12>6, 2^1=2 ≤ 12/(12-6)=2 → True."""
        self.assertTrue(plotkin_bound(1, 6))

    def test_plotkin_bound_high_d_false(self):
        """[6,3,4] — если бы k=3, d=4: 2^3=8 > 2*4/(2*4-6) = 8/2 = 4 → False."""
        self.assertFalse(plotkin_bound(3, 4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
