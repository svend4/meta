"""Тесты для hexgf — поле Галуа GF(2^6)."""
import unittest
from math import gcd

from projects.hexgf import (
    ORDER, SIZE, PRIMITIVE, POLY,
    gf_add, gf_sub, gf_mul, gf_pow, gf_inv, gf_div,
    gf_exp, gf_log, gf_mul_via_log, build_exp_log_tables,
    gf_trace, gf_norm, trace_bilinear,
    element_order, is_primitive, primitive_elements, count_primitive,
    cyclotomic_coset_of_exp, cyclotomic_coset_of, all_cyclotomic_cosets,
    minimal_polynomial, poly_eval_gf,
    subfield_elements,
    build_zech_log_table,
    additive_character, additive_character_b,
    frobenius, frobenius_orbit,
    character_sum,
    bch_zeros, bch_generator_degree,
)
from projects.hexgf.hexgf import all_elements, nonzero_elements, is_subfield, build_zech_log_table


# ─────────────────────────────────────────────────────────────────────────────
class TestGFConstants(unittest.TestCase):

    def test_order(self):
        """ORDER = 2^6 − 1 = 63."""
        self.assertEqual(ORDER, 63)

    def test_size(self):
        """SIZE = 2^6 = 64."""
        self.assertEqual(SIZE, 64)

    def test_primitive_is_2(self):
        """Примитивный элемент = 2 = x."""
        self.assertEqual(PRIMITIVE, 2)

    def test_poly(self):
        """POLY = 0x43 = 67 = x^6+x+1."""
        self.assertEqual(POLY, 67)


# ─────────────────────────────────────────────────────────────────────────────
class TestGFAddition(unittest.TestCase):

    def test_add_commutativity(self):
        """a + b = b + a."""
        for a in range(0, 64, 7):
            for b in range(0, 64, 7):
                self.assertEqual(gf_add(a, b), gf_add(b, a))

    def test_add_zero(self):
        """a + 0 = a."""
        for a in range(64):
            self.assertEqual(gf_add(a, 0), a)

    def test_add_self_zero(self):
        """a + a = 0 (характеристика 2)."""
        for a in range(64):
            self.assertEqual(gf_add(a, a), 0)

    def test_add_associativity(self):
        """(a+b)+c = a+(b+c)."""
        self.assertEqual(gf_add(gf_add(7, 13), 42),
                         gf_add(7, gf_add(13, 42)))

    def test_sub_equals_add(self):
        """В GF(2): a − b = a + b."""
        for a in [3, 17, 42, 63]:
            for b in [5, 20, 31]:
                self.assertEqual(gf_sub(a, b), gf_add(a, b))


# ─────────────────────────────────────────────────────────────────────────────
class TestGFMultiplication(unittest.TestCase):

    def test_mul_by_one(self):
        """1 · a = a."""
        for a in range(64):
            self.assertEqual(gf_mul(1, a), a)
            self.assertEqual(gf_mul(a, 1), a)

    def test_mul_by_zero(self):
        """0 · a = 0."""
        for a in range(64):
            self.assertEqual(gf_mul(0, a), 0)

    def test_mul_commutativity(self):
        """a · b = b · a."""
        for a in range(0, 64, 8):
            for b in range(0, 64, 8):
                self.assertEqual(gf_mul(a, b), gf_mul(b, a))

    def test_mul_associativity(self):
        """(a·b)·c = a·(b·c)."""
        self.assertEqual(gf_mul(gf_mul(7, 3), 5),
                         gf_mul(7, gf_mul(3, 5)))

    def test_distributivity(self):
        """a·(b+c) = a·b + a·c."""
        for a in [3, 7, 42]:
            for b, c in [(5, 11), (13, 17)]:
                lhs = gf_mul(a, gf_add(b, c))
                rhs = gf_add(gf_mul(a, b), gf_mul(a, c))
                self.assertEqual(lhs, rhs)

    def test_mul_result_in_gf(self):
        """Результат умножения ∈ [0..63]."""
        for a in range(0, 64, 4):
            for b in range(0, 64, 4):
                self.assertIn(gf_mul(a, b), range(64))

    def test_mul_specific(self):
        """Проверка: 2 * 32 = 3 (x · x^5 = x^6 = x+1)."""
        self.assertEqual(gf_mul(2, 32), 3)   # x^6 = x+1 = 3

    def test_mul_via_log_matches(self):
        """gf_mul_via_log совпадает с gf_mul."""
        for a in range(1, 64, 7):
            for b in range(1, 64, 7):
                self.assertEqual(gf_mul_via_log(a, b), gf_mul(a, b))


# ─────────────────────────────────────────────────────────────────────────────
class TestGFInverse(unittest.TestCase):

    def test_inv_times_self(self):
        """a · a^{−1} = 1 для всех a ≠ 0."""
        for a in range(1, 64):
            self.assertEqual(gf_mul(a, gf_inv(a)), 1)

    def test_inv_one(self):
        """1^{−1} = 1."""
        self.assertEqual(gf_inv(1), 1)

    def test_inv_zero_raises(self):
        """gf_inv(0) бросает исключение."""
        with self.assertRaises((ZeroDivisionError, ValueError)):
            gf_inv(0)

    def test_div(self):
        """a / b · b = a."""
        for a in [3, 7, 42]:
            for b in [1, 2, 5, 31]:
                self.assertEqual(gf_mul(gf_div(a, b), b), a)


# ─────────────────────────────────────────────────────────────────────────────
class TestGFPower(unittest.TestCase):

    def test_pow_zero(self):
        """a^0 = 1 для всех a ≠ 0."""
        for a in range(1, 64):
            self.assertEqual(gf_pow(a, 0), 1)

    def test_pow_one(self):
        """a^1 = a."""
        for a in range(64):
            self.assertEqual(gf_pow(a, 1), a)

    def test_pow_ORDER(self):
        """a^63 = 1 для всех a ≠ 0 (теорема Ферма)."""
        for a in range(1, 64):
            self.assertEqual(gf_pow(a, ORDER), 1)

    def test_pow_consistency(self):
        """a^{m+n} = a^m · a^n."""
        a = 7
        for m in [3, 10, 31]:
            for n in [2, 5, 20]:
                self.assertEqual(gf_pow(a, m + n), gf_mul(gf_pow(a, m), gf_pow(a, n)))


# ─────────────────────────────────────────────────────────────────────────────
class TestGFTables(unittest.TestCase):

    def test_exp_log_inverse(self):
        """gf_exp(gf_log(a)) = a для a ≠ 0."""
        for a in range(1, 64):
            self.assertEqual(gf_exp(gf_log(a)), a)

    def test_log_exp_inverse(self):
        """gf_log(gf_exp(k)) = k mod 63."""
        for k in range(63):
            self.assertEqual(gf_log(gf_exp(k)), k)

    def test_exp_table_distinct(self):
        """g^0,...,g^62 — 63 различных элементов."""
        vals = {gf_exp(k) for k in range(ORDER)}
        self.assertEqual(len(vals), ORDER)

    def test_exp_table_nonzero(self):
        """g^k ≠ 0 для всех k."""
        for k in range(ORDER):
            self.assertNotEqual(gf_exp(k), 0)

    def test_primitive_order(self):
        """g = PRIMITIVE имеет порядок 63."""
        self.assertEqual(element_order(PRIMITIVE), ORDER)


# ─────────────────────────────────────────────────────────────────────────────
class TestGFTrace(unittest.TestCase):

    def test_trace_in_gf2(self):
        """Tr(a) ∈ {0, 1} для всех a."""
        for a in range(64):
            self.assertIn(gf_trace(a), [0, 1])

    def test_trace_zero(self):
        """Tr(0) = 0."""
        self.assertEqual(gf_trace(0), 0)

    def test_trace_one(self):
        """Tr(1) = ?. В GF(2^6)/GF(2): Tr(1) = 6 mod 2 = 0."""
        self.assertEqual(gf_trace(1), 0)  # Tr(1) = sum of 1 six times = 6 mod 2 = 0

    def test_trace_linearity(self):
        """Tr(a+b) = Tr(a) + Tr(b) (линейность над GF(2))."""
        for a in range(0, 64, 8):
            for b in range(0, 64, 8):
                self.assertEqual(gf_trace(gf_add(a, b)),
                                 (gf_trace(a) + gf_trace(b)) % 2)

    def test_trace_balanced(self):
        """Tr принимает значения 0 и 1 по 32 раза каждое."""
        t0 = sum(1 for a in range(64) if gf_trace(a) == 0)
        t1 = sum(1 for a in range(64) if gf_trace(a) == 1)
        self.assertEqual(t0, 32)
        self.assertEqual(t1, 32)

    def test_trace_frobenius(self):
        """Tr(a^2) = Tr(a) (стабильность относительно Фробениуса)."""
        for a in range(0, 64, 5):
            self.assertEqual(gf_trace(gf_mul(a, a)), gf_trace(a))


# ─────────────────────────────────────────────────────────────────────────────
class TestGFNorm(unittest.TestCase):

    def test_norm_zero(self):
        """N(0) = 0."""
        self.assertEqual(gf_norm(0), 0)

    def test_norm_nonzero(self):
        """N(a) = 1 для всех a ≠ 0."""
        for a in range(1, 64):
            self.assertEqual(gf_norm(a), 1)


# ─────────────────────────────────────────────────────────────────────────────
class TestGFPrimitivity(unittest.TestCase):

    def test_count_primitive(self):
        """Число примитивных элементов = φ(63) = 36."""
        self.assertEqual(count_primitive(), 36)
        self.assertEqual(len(primitive_elements()), 36)

    def test_primitive_2(self):
        """PRIMITIVE = 2 является примитивным."""
        self.assertTrue(is_primitive(PRIMITIVE))

    def test_not_primitive_zero(self):
        """0 не является примитивным."""
        self.assertFalse(is_primitive(0))

    def test_not_primitive_one(self):
        """1 не является примитивным (ord = 1 ≠ 63)."""
        self.assertFalse(is_primitive(1))

    def test_element_order_divides_63(self):
        """Порядок любого a ≠ 0 делит 63."""
        for a in range(1, 64):
            self.assertEqual(ORDER % element_order(a), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestCyclotomicCosets(unittest.TestCase):

    def test_coset0_is_one(self):
        """Класс показателя 0: {0} (элемент g^0 = 1)."""
        coset = cyclotomic_coset_of_exp(0)
        self.assertEqual(coset, frozenset([0]))

    def test_coset_of_element(self):
        """Циклотомический класс элемента замкнут относительно Фробениуса."""
        for a in [2, 7, 11, 42]:
            coset = cyclotomic_coset_of(a)
            for c in coset:
                self.assertIn(frobenius(c), coset)

    def test_all_cosets_partition(self):
        """Все классы разбивают {0,...,62} (+ дополнительный для показателя 0)."""
        cosets = all_cyclotomic_cosets()
        all_indices = set()
        for c in cosets:
            self.assertEqual(len(all_indices & c), 0, "Классы пересекаются!")
            all_indices |= c
        self.assertEqual(all_indices, set(range(ORDER)))

    def test_coset_sizes_divide_6(self):
        """Размер каждого класса делит 6."""
        cosets = all_cyclotomic_cosets()
        for c in cosets:
            self.assertEqual(6 % len(c), 0)

    def test_coset1_size_6(self):
        """Класс показателя 1: {1,2,4,8,16,32} (размер 6)."""
        coset = cyclotomic_coset_of_exp(1)
        self.assertEqual(len(coset), 6)
        self.assertIn(1, coset)


# ─────────────────────────────────────────────────────────────────────────────
class TestMinimalPolynomials(unittest.TestCase):

    def test_minpoly_zero(self):
        """Мин. многочлен 0 = x = [0, 1]."""
        poly = minimal_polynomial(0)
        self.assertEqual(poly, [0, 1])

    def test_minpoly_root(self):
        """a является корнем своего минимального многочлена: m_a(a) = 0."""
        for a in [1, 2, 3, 7, 42, 63]:
            poly = minimal_polynomial(a)
            self.assertEqual(poly_eval_gf(poly, a), 0,
                             msg=f"m_{a}({a}) ≠ 0")

    def test_minpoly_coeffs_in_gf2(self):
        """Все коэффициенты минимального многочлена ∈ {0, 1}."""
        for a in [2, 7, 11, 42]:
            poly = minimal_polynomial(a)
            for c in poly:
                self.assertIn(c, [0, 1])

    def test_minpoly_monic(self):
        """Минимальный многочлен является унитарным (старший коэффициент = 1)."""
        for a in [2, 3, 5, 42]:
            poly = minimal_polynomial(a)
            self.assertEqual(poly[-1], 1)

    def test_minpoly_degree_divides_6(self):
        """Степень мин. многочлена делит 6."""
        for a in range(1, 10):
            poly = minimal_polynomial(a)
            self.assertEqual(6 % (len(poly) - 1), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestSubfields(unittest.TestCase):

    def test_gf2_subfield(self):
        """GF(2^1) = {0, 1}."""
        sf = subfield_elements(1)
        self.assertEqual(sf, frozenset([0, 1]))

    def test_gf4_subfield(self):
        """|GF(2^2)| = 4."""
        sf = subfield_elements(2)
        self.assertEqual(len(sf), 4)
        # 0 и 1 всегда входят
        self.assertIn(0, sf)
        self.assertIn(1, sf)

    def test_gf8_subfield(self):
        """|GF(2^3)| = 8."""
        sf = subfield_elements(3)
        self.assertEqual(len(sf), 8)

    def test_gf64_subfield(self):
        """GF(2^6) = все 64 элемента."""
        sf = subfield_elements(6)
        self.assertEqual(sf, frozenset(range(64)))

    def test_subfield_closed_add(self):
        """GF(2^3) замкнуто относительно сложения."""
        sf = subfield_elements(3)
        for a in sf:
            for b in sf:
                self.assertIn(gf_add(a, b), sf)

    def test_subfield_closed_mul(self):
        """GF(2^2) замкнуто относительно умножения."""
        sf = subfield_elements(2)
        for a in sf:
            for b in sf:
                self.assertIn(gf_mul(a, b), sf)

    def test_invalid_subfield(self):
        """GF(2^4) не является подполем GF(2^6) (4 не делит 6)."""
        with self.assertRaises(ValueError):
            subfield_elements(4)


# ─────────────────────────────────────────────────────────────────────────────
class TestAdditiveCharacters(unittest.TestCase):

    def test_character_pm1(self):
        """ψ(a) ∈ {+1, −1}."""
        for a in range(64):
            self.assertIn(additive_character(a), [1, -1])

    def test_character_zero(self):
        """ψ(0) = (−1)^{Tr(0)} = +1."""
        self.assertEqual(additive_character(0), 1)

    def test_character_homomorphism(self):
        """ψ(a+b) = ψ(a)·ψ(b) (гомоморфизм)."""
        for a in range(0, 64, 8):
            for b in range(0, 64, 8):
                self.assertEqual(additive_character(gf_add(a, b)),
                                 additive_character(a) * additive_character(b))

    def test_character_b_orthogonality(self):
        """Σ_a ψ_b(a) = 64·δ_{b,0} (ортогональность характеров)."""
        for b in range(64):
            s = sum(additive_character_b(b, a) for a in range(64))
            expected = 64 if b == 0 else 0
            self.assertEqual(s, expected)

    def test_trace_bilinear_symmetric(self):
        """Tr(a·b) = Tr(b·a) (симметричность)."""
        for a in range(0, 64, 8):
            for b in range(0, 64, 8):
                self.assertEqual(trace_bilinear(a, b), trace_bilinear(b, a))

    def test_trace_bilinear_nondegenerate(self):
        """Билинейная форма Трейса невырождена: для каждого a≠0 существует b с Tr(ab)=1."""
        for a in range(1, 64):
            has_one = any(trace_bilinear(a, b) == 1 for b in range(64))
            self.assertTrue(has_one, f"Tr(a·b)=1 не достигается для a={a}")


class TestElementSets(unittest.TestCase):
    """Тесты all_elements, nonzero_elements."""

    def test_all_elements_count(self):
        self.assertEqual(len(all_elements()), 64)

    def test_all_elements_range(self):
        self.assertEqual(sorted(all_elements()), list(range(64)))

    def test_nonzero_elements_count(self):
        self.assertEqual(len(nonzero_elements()), 63)

    def test_nonzero_elements_no_zero(self):
        self.assertNotIn(0, nonzero_elements())

    def test_all_minus_zero_equals_nonzero(self):
        self.assertEqual(set(all_elements()) - {0}, set(nonzero_elements()))


class TestCharacterSum(unittest.TestCase):
    """Тесты character_sum — суммы характеров над подмножеством."""

    def test_character_sum_b0_equals_size(self):
        """character_sum(0, S) = |S| (ψ_0 = 1 везде)."""
        subset = list(range(10))
        self.assertEqual(character_sum(0, subset), len(subset))

    def test_character_sum_full_field_b_nonzero(self):
        """Σ_{a∈GF(2^6)} ψ_b(a) = 0 для b ≠ 0."""
        for b in [1, 2, 7, 63]:
            result = character_sum(b, all_elements())
            self.assertEqual(result, 0)

    def test_character_sum_empty(self):
        self.assertEqual(character_sum(1, []), 0)


class TestBCH(unittest.TestCase):
    """Тесты bch_zeros, bch_generator_degree."""

    def test_bch_zeros_d1_empty(self):
        """d=1: нет нулей (BCH с проектным расстоянием 1 = trivial код)."""
        self.assertEqual(bch_zeros(1), [])

    def test_bch_zeros_d2_one_element(self):
        """d=2: один нуль {g^1}."""
        z = bch_zeros(2)
        self.assertEqual(len(z), 1)

    def test_bch_zeros_d_n_minus_one(self):
        """bch_zeros(d) содержит d-1 элементов."""
        for d in range(1, 5):
            self.assertEqual(len(bch_zeros(d)), d - 1)

    def test_bch_generator_degree_d1_is_0(self):
        """d=1: нет нулей → степень 0."""
        self.assertEqual(bch_generator_degree(1), 0)

    def test_bch_generator_degree_positive_for_d2(self):
        """d=2: есть нули → степень > 0."""
        self.assertGreater(bch_generator_degree(2), 0)

    def test_bch_generator_degree_nondecreasing(self):
        """Степень не убывает с ростом d."""
        degs = [bch_generator_degree(d) for d in range(1, 6)]
        for i in range(len(degs) - 1):
            self.assertLessEqual(degs[i], degs[i + 1])


class TestIsSubfield(unittest.TestCase):
    """Тесты is_subfield."""

    def test_gf2_is_subfield(self):
        """GF(2) ⊂ GF(2^6): {0, 1} — подполе GF(2^1)."""
        self.assertTrue(is_subfield(subfield_elements(1), 1))

    def test_gf4_is_subfield(self):
        """GF(4) ⊂ GF(64): подполе GF(2^2)."""
        self.assertTrue(is_subfield(subfield_elements(2), 2))

    def test_wrong_elements_not_subfield(self):
        """Произвольное множество {0,1,2} не является подполем GF(2)."""
        self.assertFalse(is_subfield([0, 1, 2], 1))

    def test_full_field_is_subfield_6(self):
        """GF(2^6) — подполе само себя."""
        self.assertTrue(is_subfield(subfield_elements(6), 6))


class TestZechLogTable(unittest.TestCase):
    """Тесты таблицы логарифмов Цеха для GF(2^6)."""

    def setUp(self):
        self.Z = build_zech_log_table()

    def test_length_is_order_minus_1(self):
        """Таблица Цеха имеет 63 записи (ORDER = 63)."""
        self.assertEqual(len(self.Z), 63)

    def test_z0_is_none(self):
        """Z[0] = None, так как g^0 + 1 = 0 в GF(2)."""
        self.assertIsNone(self.Z[0])

    def test_nonzero_entries_in_range(self):
        """Все ненулевые записи Z[k] ∈ [0, 62]."""
        for k in range(1, 63):
            if self.Z[k] is not None:
                self.assertGreaterEqual(self.Z[k], 0)
                self.assertLess(self.Z[k], 63)

    def test_zech_log_addition(self):
        """Z используется для: exp[k + Z[k']] = exp[k] XOR exp[k'] при k=1, k'=2."""
        from projects.hexgf.hexgf import _get_tables
        exp, log = _get_tables()
        # test for specific k, k' where we know Z is not None
        k = 1
        k_prime = 2
        z_val = self.Z[(k_prime - k) % 63]
        if z_val is not None:
            expected = exp[k] ^ exp[k_prime]
            actual = exp[(k + z_val) % 63]
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
