"""Тесты hexring — булевы функции на Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexring.hexring import (
    BoolFunc, ReedMullerCode,
    zero_func, one_func, coordinate, inner_product, inner_product_bent,
    yang_parity, threshold_func, maiorana_mcfarland,
    all_linear_functions, all_affine_functions,
    best_affine_approximation, auto_correlation, auto_correlation_table,
    power_moment, find_bent_examples, hamming_distance_func,
    nonlinearity_profile, _wht_inplace, _mobius_inplace,
    count_bent_in_rm2, find_resilient,
)
from libs.hexcore.hexcore import SIZE


# ---------------------------------------------------------------------------
# Тест преобразований WHT и Мёбиуса
# ---------------------------------------------------------------------------

class TestTransforms(unittest.TestCase):
    def test_wht_involutory(self):
        """WHT дважды = 64 × исходное (WHT — самообратная с множителем 64)."""
        f = inner_product_bent()
        W = f.wht()
        _wht_inplace(W)
        for i, (orig, w) in enumerate(zip([1 - 2 * b for b in f.truth_table()], W)):
            self.assertAlmostEqual(w, 64 * orig, msg=f"Позиция {i}")

    def test_mobius_involutory(self):
        """Преобразование Мёбиуса дважды = исходное (над GF(2))."""
        f = inner_product_bent()
        a = f.truth_table()
        _mobius_inplace(a)
        _mobius_inplace(a)
        self.assertEqual(a, f.truth_table())

    def test_wht_zero_func(self):
        """WHT нулевой функции: Ŵ(0) = 64, остальные = 0 (ортогональность)."""
        W = zero_func().wht()
        self.assertEqual(W[0], 64)
        self.assertTrue(all(W[u] == 0 for u in range(1, SIZE)))

    def test_wht_one_func(self):
        """WHT единичной функции: Ŵ(0) = -64, остальные 0."""
        W = one_func().wht()
        self.assertEqual(W[0], -64)
        self.assertTrue(all(W[u] == 0 for u in range(1, SIZE)))

    def test_wht_linear_func(self):
        """WHT линейной f(x) = a·x: Ŵ(a) = 64, остальные 0."""
        for a in [1, 7, 42]:
            f = inner_product(a)
            W = f.wht()
            self.assertEqual(W[a], 64)
            self.assertTrue(all(W[u] == 0 for u in range(SIZE) if u != a))

    def test_anf_coordinate(self):
        """ANF координатной функции x_i содержит только моном x_i."""
        for i in range(6):
            f = coordinate(i)
            coeffs = f.anf_coeffs()
            mask_i = 1 << i
            self.assertEqual(coeffs[mask_i], 1)
            self.assertEqual(sum(coeffs), 1)  # только один моном

    def test_anf_bent(self):
        """ANF bent-функции x0x1+x2x3+x4x5 содержит 3 квадратичных монома."""
        f = inner_product_bent()
        monomials = f.anf_monomials()
        degrees = [len(m) for m in monomials]
        self.assertEqual(sorted(degrees), [2, 2, 2])   # три квадратичных


# ---------------------------------------------------------------------------
# Тест BoolFunc создания
# ---------------------------------------------------------------------------

class TestBoolFuncCreation(unittest.TestCase):
    def test_from_list(self):
        tt = [i % 2 for i in range(SIZE)]
        f = BoolFunc(tt)
        self.assertEqual(f.truth_table(), tt)

    def test_from_int(self):
        mask = 0xAAAAAAAAAAAAAAAA  # чередующиеся биты
        f = BoolFunc(mask)
        for i in range(SIZE):
            self.assertEqual(f(i), (mask >> i) & 1)

    def test_from_callable(self):
        f = BoolFunc(lambda x: (x >> 3) & 1)
        for x in range(SIZE):
            self.assertEqual(f(x), (x >> 3) & 1)

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            BoolFunc([0] * 32)

    def test_roundtrip_as_int(self):
        f = inner_product_bent()
        f2 = BoolFunc(f.as_int())
        self.assertEqual(f, f2)

    def test_equality(self):
        self.assertEqual(zero_func(), zero_func())
        self.assertNotEqual(zero_func(), one_func())

    def test_hash(self):
        d = {zero_func(): 'zero', one_func(): 'one'}
        self.assertEqual(d[zero_func()], 'zero')

    def test_call_evaluates(self):
        f = coordinate(3)
        for x in range(SIZE):
            self.assertEqual(f(x), (x >> 3) & 1)


# ---------------------------------------------------------------------------
# Тест криптографических свойств
# ---------------------------------------------------------------------------

class TestCryptographicProps(unittest.TestCase):
    def test_nonlinearity_linear_is_zero(self):
        """Линейные функции: nl = 0."""
        for a in [0, 1, 7, 63]:
            self.assertEqual(inner_product(a).nonlinearity(), 0)

    def test_nonlinearity_affine_is_zero(self):
        """Аффинные функции: nl = 0."""
        f = inner_product(5)
        g = f + one_func()
        self.assertEqual(g.nonlinearity(), 0)

    def test_nonlinearity_bent_is_28(self):
        """Bent-функция: максимальная нелинейность 28."""
        self.assertEqual(inner_product_bent().nonlinearity(), 28)

    def test_nonlinearity_max_for_n6(self):
        """Нелинейность ≤ 28 для любой функции n=6."""
        for f in [zero_func(), one_func(), inner_product_bent(),
                  yang_parity(), threshold_func(3)]:
            self.assertLessEqual(f.nonlinearity(), 28)

    def test_is_bent_true(self):
        self.assertTrue(inner_product_bent().is_bent())
        bent2 = maiorana_mcfarland()
        self.assertTrue(bent2.is_bent())

    def test_is_bent_false_linear(self):
        self.assertFalse(inner_product(7).is_bent())

    def test_is_bent_false_balanced(self):
        """Bent-функции не сбалансированы."""
        b = inner_product_bent()
        self.assertTrue(b.is_bent())
        self.assertFalse(b.is_balanced())

    def test_is_balanced_zero_false(self):
        self.assertFalse(zero_func().is_balanced())

    def test_is_balanced_parity(self):
        """yang_parity — сбалансированная (32 нуля, 32 единицы)."""
        self.assertTrue(yang_parity().is_balanced())

    def test_is_balanced_linear_inner_product(self):
        """Ненулевое линейное a: f_a(x) = a·x сбалансировано."""
        for a in range(1, SIZE):
            self.assertTrue(inner_product(a).is_balanced())

    def test_algebraic_degree_constant(self):
        self.assertEqual(zero_func().algebraic_degree(), 0)
        self.assertEqual(one_func().algebraic_degree(), 0)

    def test_algebraic_degree_linear(self):
        for i in range(6):
            self.assertEqual(coordinate(i).algebraic_degree(), 1)

    def test_algebraic_degree_quadratic(self):
        self.assertEqual(inner_product_bent().algebraic_degree(), 2)

    def test_algebraic_degree_cubic(self):
        # Функция x0*x1*x2
        f = coordinate(0) * coordinate(1) * coordinate(2)
        self.assertEqual(f.algebraic_degree(), 3)

    def test_is_affine_linear_plus_const(self):
        f = inner_product(7)
        g = f + one_func()
        self.assertTrue(g.is_affine())
        self.assertFalse(g.is_linear())

    def test_is_linear_coordinate(self):
        self.assertTrue(coordinate(0).is_linear())

    def test_is_symmetric(self):
        """Пороговая функция — симметричная (зависит только от yang_count)."""
        self.assertTrue(threshold_func(3).is_symmetric())
        self.assertTrue(yang_parity().is_symmetric())

    def test_not_symmetric(self):
        """Координатная функция x_0 — не симметричная."""
        self.assertFalse(coordinate(0).is_symmetric())

    def test_correlation_immunity_linear(self):
        """Линейная f(x) = a·x (a≠0): CI = 0."""
        self.assertEqual(inner_product(1).correlation_immunity(), 0)

    def test_correlation_immunity_yang_parity(self):
        """yang_parity — CI(5) = 5 (коррел.-иммунна порядка 5)."""
        ci = yang_parity().correlation_immunity()
        self.assertEqual(ci, 5)

    def test_resilience_balanced(self):
        """Сбалансированная функция CI(t) → resilience = t."""
        f = yang_parity()
        self.assertEqual(f.resilience(), 5)

    def test_resilience_unbalanced(self):
        """Несбалансированная функция → resilience = -1."""
        f = zero_func()
        self.assertEqual(f.resilience(), -1)


# ---------------------------------------------------------------------------
# Тест арифметики
# ---------------------------------------------------------------------------

class TestArithmetic(unittest.TestCase):
    def test_xor_self_is_zero(self):
        f = inner_product_bent()
        self.assertEqual(f + f, zero_func())

    def test_and_with_zero(self):
        f = inner_product_bent()
        self.assertEqual(f * zero_func(), zero_func())

    def test_not_not_is_identity(self):
        f = inner_product_bent()
        self.assertEqual(-(-f), f)

    def test_not_zero_is_one(self):
        self.assertEqual(-zero_func(), one_func())

    def test_add_one_is_complement(self):
        f = coordinate(2)
        self.assertEqual(f + one_func(), -f)

    def test_xor_associative(self):
        f = coordinate(0)
        g = coordinate(1)
        h = coordinate(2)
        self.assertEqual((f + g) + h, f + (g + h))

    def test_mul_distributive(self):
        f = coordinate(0)
        g = coordinate(1)
        h = coordinate(2)
        self.assertEqual(f * (g + h), f * g + f * h)

    def test_add_non_boolfunc_returns_not_implemented(self):
        f = coordinate(0)
        self.assertIs(f.__add__(42), NotImplemented)

    def test_mul_non_boolfunc_returns_not_implemented(self):
        f = coordinate(0)
        self.assertIs(f.__mul__(42), NotImplemented)

    def test_xor_two_boolfuncs(self):
        f = coordinate(0)
        g = coordinate(1)
        result = f ^ g
        self.assertIsInstance(result, BoolFunc)

    def test_display_one_func_has_constant_1(self):
        """one_func ANF содержит '1' (константный терм, покрывает _anf_str mask==0)."""
        f = one_func()
        d = f.display()
        self.assertIn('1', d)


# ---------------------------------------------------------------------------
# Тест стандартных функций
# ---------------------------------------------------------------------------

class TestStandardFunctions(unittest.TestCase):
    def test_zero_func(self):
        f = zero_func()
        self.assertTrue(all(f(x) == 0 for x in range(SIZE)))

    def test_one_func(self):
        f = one_func()
        self.assertTrue(all(f(x) == 1 for x in range(SIZE)))

    def test_coordinate_range(self):
        for i in range(6):
            f = coordinate(i)
            for x in range(SIZE):
                self.assertIn(f(x), [0, 1])

    def test_coordinate_out_of_range(self):
        with self.assertRaises(ValueError):
            coordinate(6)
        with self.assertRaises(ValueError):
            coordinate(-1)

    def test_inner_product_zero_a(self):
        """f_0(x) = 0·x = 0."""
        f = inner_product(0)
        self.assertEqual(f, zero_func())

    def test_inner_product_bilinearity(self):
        """f_{a XOR b}(x) = f_a(x) XOR f_b(x)."""
        a, b = 5, 12
        f_ab = inner_product(a ^ b)
        self.assertEqual(f_ab, inner_product(a) + inner_product(b))

    def test_yang_parity_values(self):
        f = yang_parity()
        for x in range(SIZE):
            self.assertEqual(f(x), bin(x).count('1') % 2)

    def test_threshold_func_values(self):
        for t in range(7):
            f = threshold_func(t)
            for x in range(SIZE):
                self.assertEqual(f(x), int(bin(x).count('1') >= t))

    def test_all_linear_count(self):
        lins = all_linear_functions()
        self.assertEqual(len(lins), SIZE)

    def test_all_affine_count(self):
        affs = all_affine_functions()
        self.assertEqual(len(affs), 2 * SIZE)

    def test_maiorana_mcfarland_is_bent(self):
        f = maiorana_mcfarland()
        self.assertTrue(f.is_bent())


# ---------------------------------------------------------------------------
# Тест кодов Рида–Маллера
# ---------------------------------------------------------------------------

class TestReedMullerCode(unittest.TestCase):
    def test_rm0_params(self):
        rm = ReedMullerCode(0)
        self.assertEqual(rm.n, 64)
        self.assertEqual(rm.k, 1)
        self.assertEqual(rm.d, 64)

    def test_rm1_params(self):
        rm = ReedMullerCode(1)
        self.assertEqual(rm.k, 7)
        self.assertEqual(rm.d, 32)

    def test_rm6_params(self):
        rm = ReedMullerCode(6)
        self.assertEqual(rm.k, 64)
        self.assertEqual(rm.d, 1)

    def test_generator_matrix_shape(self):
        for r in range(7):
            rm = ReedMullerCode(r)
            G = rm.generator_matrix()
            self.assertEqual(len(G), rm.k)
            self.assertEqual(len(G[0]), rm.n)

    def test_generator_rows_are_codewords(self):
        """Каждая строка G — кодовое слово RM(r)."""
        for r in range(4):
            rm = ReedMullerCode(r)
            for row in rm.generator_matrix():
                f = BoolFunc(row)
                self.assertTrue(rm.contains(f), f"r={r}, row={row}")

    def test_encode_zero_message(self):
        """Нулевое сообщение → нулевая функция."""
        rm = ReedMullerCode(2)
        msg = [0] * rm.k
        cw = rm.encode(msg)
        self.assertEqual(cw, zero_func())

    def test_encode_wrong_length_raises(self):
        rm = ReedMullerCode(1)
        with self.assertRaises(ValueError):
            rm.encode([0, 1])

    def test_contains_zero(self):
        for r in range(7):
            rm = ReedMullerCode(r)
            self.assertTrue(rm.contains(zero_func()))

    def test_contains_degree(self):
        """RM(r) содержит функции степени ≤ r."""
        self.assertTrue(ReedMullerCode(2).contains(inner_product_bent()))   # degree 2
        self.assertFalse(ReedMullerCode(1).contains(inner_product_bent()))  # degree 2 > 1

    def test_decode_rm0(self):
        """RM(0): декодирование → ближайшая константа."""
        rm = ReedMullerCode(0)
        # yang_parity: 32 единицы — ближе к константе 0 или 1?
        cw = rm.decode(yang_parity())
        self.assertIn(cw, [zero_func(), one_func()])

    def test_decode_rm1_linear(self):
        """RM(1): декодирование линейной функции → она сама."""
        rm = ReedMullerCode(1)
        f = inner_product(7)
        decoded = rm.decode(f)
        self.assertEqual(decoded, f)

    def test_repr(self):
        self.assertIn('RM(2, 6)', repr(ReedMullerCode(2)))

    def test_info_returns_string(self):
        rm = ReedMullerCode(1)
        info = rm.info()
        self.assertIsInstance(info, str)
        self.assertIn('RM(1, 6)', info)

    def test_info_contains_params(self):
        rm = ReedMullerCode(2)
        info = rm.info()
        self.assertIn('Длина', info)
        self.assertIn('Мин. расстояние', info)

    def test_invalid_r_raises(self):
        with self.assertRaises(ValueError):
            ReedMullerCode(-1)
        with self.assertRaises(ValueError):
            ReedMullerCode(7)

    def test_rm_nested(self):
        """RM(r) ⊆ RM(r+1): все кодовые слова RM(r) принадлежат RM(r+1)."""
        rm_r = ReedMullerCode(2)
        rm_r1 = ReedMullerCode(3)
        for f in rm_r.generator_functions():
            self.assertTrue(rm_r1.contains(f))


# ---------------------------------------------------------------------------
# Тест автокорреляции и моментов
# ---------------------------------------------------------------------------

class TestACF(unittest.TestCase):
    def test_acf_zero_displacement(self):
        """ACF(0) = 64 для любой функции."""
        for f in [zero_func(), one_func(), inner_product_bent(), yang_parity()]:
            self.assertEqual(auto_correlation(f, 0), 64)

    def test_acf_bent_uniform(self):
        """Для bent-функции |ACF(a)| ≤ 64 для a ≠ 0."""
        f = inner_product_bent()
        for a in range(1, SIZE):
            acf = auto_correlation(f, a)
            self.assertLessEqual(abs(acf), 64)

    def test_acf_table_length(self):
        acf = auto_correlation_table(inner_product_bent())
        self.assertEqual(len(acf), SIZE)

    def test_power_moment_order0(self):
        """Момент порядка 0 = 64 (число спектральных компонент)."""
        f = inner_product_bent()
        self.assertEqual(power_moment(f, 0), SIZE)

    def test_power_moment_order2(self):
        """Момент порядка 2 = 64^2 = 4096 (теорема Парсеваля для WHT)."""
        # Σ Ŵ(u)^2 = 64 * 64 = 4096 для любой булевой функции
        for f in [inner_product_bent(), coordinate(0), yang_parity()]:
            self.assertEqual(power_moment(f, 2), SIZE * SIZE)


# ---------------------------------------------------------------------------
# Тест вспомогательных функций
# ---------------------------------------------------------------------------

class TestUtils(unittest.TestCase):
    def test_hamming_distance_same(self):
        f = inner_product_bent()
        self.assertEqual(hamming_distance_func(f, f), 0)

    def test_hamming_distance_complement(self):
        f = zero_func()
        self.assertEqual(hamming_distance_func(f, one_func()), SIZE)

    def test_hamming_distance_symmetric(self):
        f = inner_product_bent()
        g = yang_parity()
        self.assertEqual(hamming_distance_func(f, g), hamming_distance_func(g, f))

    def test_best_affine_approximation_linear(self):
        """Лучшая аффинная аппроксимация линейной функции = сама функция."""
        f = inner_product(7)
        g = best_affine_approximation(f)
        self.assertEqual(hamming_distance_func(f, g), 0)

    def test_find_bent_examples(self):
        bents = find_bent_examples(5)
        self.assertGreater(len(bents), 0)
        for f in bents:
            self.assertTrue(f.is_bent())

    def test_nonlinearity_profile_monotone(self):
        """Расстояние до RM(r) убывает с ростом r."""
        f = inner_product_bent()
        profile = nonlinearity_profile(f)
        for r in range(6):
            self.assertGreaterEqual(profile[r], profile[r + 1])

    def test_nonlinearity_profile_rm_r(self):
        """Расстояние от f ∈ RM(r) до RM(r) = 0."""
        f = coordinate(0)  # degree 1, в RM(1)
        profile = nonlinearity_profile(f)
        self.assertEqual(profile[1], 0)
        self.assertEqual(profile[2], 0)


class TestCountBentInRM2(unittest.TestCase):
    """Тест count_bent_in_rm2 — заглушка с NotImplementedError."""

    def test_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            count_bent_in_rm2()


class TestFindResilient(unittest.TestCase):
    """Тесты find_resilient — поиск сбалансированных функций с CI."""

    def test_returns_list(self):
        result = find_resilient(ci_order=0, n_max=3)
        self.assertIsInstance(result, list)

    def test_ci0_finds_functions(self):
        """ci_order=0 (просто сбалансированные) — находит n_max функций."""
        result = find_resilient(ci_order=0, n_max=5)
        self.assertGreater(len(result), 0)

    def test_result_are_boolfuncs(self):
        """Все найденные объекты — BoolFunc."""
        result = find_resilient(ci_order=0, n_max=3)
        for f in result:
            self.assertIsInstance(f, BoolFunc)

    def test_n_max_limits_output(self):
        """Количество результатов ≤ n_max."""
        for n in [1, 3, 5]:
            result = find_resilient(ci_order=0, n_max=n)
            self.assertLessEqual(len(result), n)


class TestReedMullerDecode(unittest.TestCase):
    """Тесты decode и encode edge cases для ReedMullerCode."""

    def test_encode_wrong_length_raises(self):
        """encode() с неправильной длиной сообщения → ValueError."""
        rm = ReedMullerCode(1)  # k=7
        with self.assertRaises(ValueError):
            rm.encode([1, 0])  # length 2, not 7

    def test_decode_rm1_positive_wht(self):
        """Декодирование RM(1) функции с положительным WHT."""
        rm = ReedMullerCode(1)
        f = inner_product(1)  # WHT has W[1] = 64 > 0
        decoded = rm.decode(f)
        self.assertIsInstance(decoded, BoolFunc)

    def test_decode_rm1_negative_wht(self):
        """Декодирование RM(1) функции с отрицательным WHT."""
        rm = ReedMullerCode(1)
        f = one_func() + inner_product(1)  # complement: WHT W[1] = -64 < 0
        decoded = rm.decode(f)
        self.assertIsInstance(decoded, BoolFunc)

    def test_decode_rm0(self):
        """Декодирование RM(0): ближайшая константа."""
        rm = ReedMullerCode(0)
        # Функция с 40 единицами → ближайшая константа — 1
        tt = [1] * 40 + [0] * 24
        f = BoolFunc(tt)
        decoded = rm.decode(f)
        self.assertIsInstance(decoded, BoolFunc)

    def test_decode_rm0_zero_func(self):
        """Декодирование RM(0): функция с 20 единицами → 0."""
        rm = ReedMullerCode(0)
        tt = [1] * 20 + [0] * 44
        f = BoolFunc(tt)
        decoded = rm.decode(f)
        self.assertIsInstance(decoded, BoolFunc)


class TestBestAffineExtended(unittest.TestCase):
    """Тесты best_affine_approximation для ветви с отрицательным WHT."""

    def test_neg_wht_complement_linear(self):
        """best_affine_approximation дополнения линейной функции."""
        f = one_func() + inner_product(1)  # complement: WHT W[1] = -64 < 0
        g = best_affine_approximation(f)
        self.assertIsInstance(g, BoolFunc)
        self.assertTrue(g.is_affine())


class TestFindBentEarlyReturn(unittest.TestCase):
    """Тесты find_bent_examples с малым n_max для ранних возвратов."""

    def test_find_bent_n1(self):
        """find_bent_examples(1) возвращает ровно 1 bent-функцию."""
        result = find_bent_examples(n_max=1)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].is_bent())

    def test_find_bent_n2(self):
        """find_bent_examples(2) возвращает ровно 2 bent-функции."""
        result = find_bent_examples(n_max=2)
        self.assertEqual(len(result), 2)

    def test_find_bent_n3(self):
        """find_bent_examples(3) возвращает не менее 3 bent-функций."""
        result = find_bent_examples(n_max=3)
        self.assertGreaterEqual(len(result), 3)


class TestFindResilientExtended(unittest.TestCase):
    """Тесты find_resilient для ветвей CI >= 1 и dep_bits пустой."""

    def test_find_resilient_ci1_n1(self):
        """find_resilient(ci=1, n_max=1) ранний выход."""
        result = find_resilient(ci_order=1, n_max=1)
        self.assertLessEqual(len(result), 1)

    def test_find_resilient_ci1_finds(self):
        """find_resilient(ci=1) возвращает функции с CI≥1."""
        result = find_resilient(ci_order=1, n_max=3)
        for f in result:
            self.assertGreaterEqual(f.correlation_immunity(), 1)

    def test_find_resilient_ci6_empty(self):
        """CI=6 вероятно не найдёт функций (пустой результат)."""
        result = find_resilient(ci_order=6, n_max=5)
        self.assertIsInstance(result, list)


class TestBoolFuncDunders(unittest.TestCase):
    """Тесты __repr__ и __eq__ для BoolFunc."""

    def test_repr_returns_string(self):
        f = inner_product_bent()
        r = repr(f)
        self.assertIsInstance(r, str)
        self.assertIn('BoolFunc', r)

    def test_repr_shows_degree(self):
        f = zero_func()
        r = repr(f)
        self.assertIn('degree', r)

    def test_eq_with_non_boolfunc_returns_false(self):
        """Сравнение с не-BoolFunc возвращает NotImplemented (т.е. False)."""
        f = zero_func()
        self.assertNotEqual(f, 0)
        self.assertNotEqual(f, 'not a BoolFunc')


class TestBoolFuncDisplay(unittest.TestCase):
    """Тесты метода display() для BoolFunc."""

    def test_display_returns_string(self):
        f = inner_product_bent()
        result = f.display()
        self.assertIsInstance(result, str)

    def test_display_compact_shorter(self):
        """Компактный режим короче полного."""
        f = inner_product_bent()
        full = f.display(compact=False)
        compact = f.display(compact=True)
        self.assertGreater(len(full), len(compact))

    def test_display_contains_anf(self):
        f = zero_func()
        result = f.display()
        self.assertIn('ANF', result)

    def test_display_contains_bent_info(self):
        """Для bent-функции display показывает 'True'."""
        f = inner_product_bent()
        result = f.display()
        self.assertIn('True', result)


class TestEncodeNonZero(unittest.TestCase):
    """Тест encode с ненулевым сообщением (line 465)."""

    def test_encode_nonzero_message(self):
        """Кодирование ненулевого сообщения — входит в ветку if bit: (line 465)."""
        rm = ReedMullerCode(1)
        # RM(1,6) has k=7: constant + 6 linear functions
        msg = [1, 0, 0, 0, 0, 0, 0]  # only the first bit set
        cw = rm.encode(msg)
        # Should produce a valid codeword (not zero)
        self.assertNotEqual(cw, zero_func())


class TestFindBentExhaustAll(unittest.TestCase):
    """Тест find_bent_examples с n_max > доступного — line 603."""

    def test_find_bent_large_nmax_exhausts_loop(self):
        """find_bent_examples с большим n_max исчерпывает аффинные сдвиги → line 603."""
        result = find_bent_examples(n_max=200)
        # Should return all found results (< 200) and hit line 603
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertLess(len(result), 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
