"""Тесты для hexcrypt — криптографические примитивы на Q6."""
import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from projects.hexcrypt.hexcrypt import (
    SBox, HexStream, FeistelCipher,
    identity_sbox, bit_reversal_sbox, affine_sbox,
    complement_sbox, random_sbox, yang_sort_sbox,
    evaluate_sbox, search_good_sbox,
    best_differential_characteristic, best_linear_bias,
    _popcount, _inner_product,
)


# ── SBox: базовые свойства ───────────────────────────────────────────────────

class TestSBoxBasic(unittest.TestCase):

    def test_valid_permutation(self):
        """SBox принимает правильную перестановку."""
        sb = SBox(list(range(64)))
        self.assertEqual(sb(0), 0)
        self.assertEqual(sb(63), 63)

    def test_invalid_not_permutation(self):
        """SBox отвергает не-перестановку."""
        bad = list(range(64))
        bad[0] = bad[1] = 0
        with self.assertRaises(ValueError):
            SBox(bad)

    def test_invalid_wrong_length(self):
        """SBox отвергает таблицу неправильной длины."""
        with self.assertRaises(ValueError):
            SBox(list(range(32)))

    def test_table_roundtrip(self):
        """table() возвращает копию."""
        sb = random_sbox()
        t = sb.table()
        self.assertEqual(len(t), 64)
        self.assertEqual(sorted(t), list(range(64)))

    def test_inverse_correct(self):
        """inverse()(f(x)) = x для всех x."""
        sb = random_sbox(seed=1)
        inv = sb.inverse()
        for x in range(64):
            self.assertEqual(inv(sb(x)), x)

    def test_inverse_of_inverse(self):
        """Двойное обращение = исходный S-блок."""
        sb = random_sbox(seed=5)
        inv_inv = sb.inverse().inverse()
        for x in range(64):
            self.assertEqual(inv_inv(x), sb(x))

    def test_repr_contains_nl(self):
        """SBox.__repr__ содержит нелинейность и дифф. унифомность."""
        sb = random_sbox(seed=0)
        r = repr(sb)
        self.assertIn('nl=', r)
        self.assertIn('du=', r)

    def test_component_binary(self):
        """Компонентная функция возвращает только 0 и 1."""
        sb = random_sbox()
        for u in [1, 7, 63]:
            comp = sb.component(u)
            self.assertEqual(len(comp), 64)
            self.assertTrue(all(b in (0, 1) for b in comp))

    def test_component_u_zero_raises(self):
        """component(0) — ошибка."""
        with self.assertRaises(ValueError):
            identity_sbox().component(0)


# ── стандартные S-блоки ──────────────────────────────────────────────────────

class TestStandardSBoxes(unittest.TestCase):

    def test_identity_is_identity(self):
        """identity_sbox: f(x) = x."""
        sb = identity_sbox()
        for x in range(64):
            self.assertEqual(sb(x), x)

    def test_complement_correct(self):
        """complement_sbox: f(x) = x ⊕ 63."""
        sb = complement_sbox()
        for x in range(64):
            self.assertEqual(sb(x), x ^ 63)

    def test_bit_reversal_involutory(self):
        """Переворот битов — инволюция: f(f(x)) = x."""
        sb = bit_reversal_sbox()
        for x in range(64):
            self.assertEqual(sb(sb(x)), x)

    def test_affine_is_permutation(self):
        """affine_sbox — перестановка."""
        sb = affine_sbox()
        self.assertEqual(sorted(sb(x) for x in range(64)), list(range(64)))

    def test_yang_sort_sbox_is_permutation(self):
        """yang_sort_sbox — перестановка."""
        sb = yang_sort_sbox()
        self.assertEqual(sorted(sb(x) for x in range(64)), list(range(64)))

    def test_random_sbox_reproducible(self):
        """random_sbox(seed) воспроизводима."""
        sb1 = random_sbox(seed=99)
        sb2 = random_sbox(seed=99)
        self.assertEqual(sb1.table(), sb2.table())

    def test_random_sbox_different_seeds(self):
        """Разные seed → разные S-блоки."""
        sb1 = random_sbox(seed=1)
        sb2 = random_sbox(seed=2)
        self.assertNotEqual(sb1.table(), sb2.table())


# ── нелинейность ─────────────────────────────────────────────────────────────

class TestNonlinearity(unittest.TestCase):

    def test_identity_nl_zero(self):
        """Линейный S-блок (identity) имеет nl = 0."""
        self.assertEqual(identity_sbox().nonlinearity(), 0)

    def test_affine_nl_zero(self):
        """Аффинный S-блок имеет nl = 0."""
        self.assertEqual(affine_sbox().nonlinearity(), 0)

    def test_complement_nl_zero(self):
        """Дополнение (аффинное) имеет nl = 0."""
        self.assertEqual(complement_sbox().nonlinearity(), 0)

    def test_random_sbox_nl_positive(self):
        """Случайный S-блок ожидаемо имеет nl > 0."""
        nl = random_sbox(seed=42).nonlinearity()
        self.assertGreater(nl, 0)

    def test_nl_bounded(self):
        """nl ∈ [0, 28] для 6-битного S-блока."""
        for seed in [1, 2, 3]:
            nl = random_sbox(seed=seed).nonlinearity()
            self.assertGreaterEqual(nl, 0)
            self.assertLessEqual(nl, 28)


# ── DDT и дифференциальная равномерность ─────────────────────────────────────

class TestDDT(unittest.TestCase):

    def test_ddt_row_sum(self):
        """Сумма любой строки DDT = 64."""
        sb = random_sbox(seed=3)
        ddt = sb.difference_distribution_table()
        for a in range(64):
            self.assertEqual(sum(ddt[a]), 64)

    def test_ddt_identity_diagonal(self):
        """DDT идентичности: DDT[a][a] = 64 для a≠0, иначе 0."""
        ddt = identity_sbox().difference_distribution_table()
        for a in range(1, 64):
            for b in range(64):
                expected = 64 if a == b else 0
                self.assertEqual(ddt[a][b], expected)

    def test_ddt_a0_row(self):
        """DDT[0][0] = 64, DDT[0][b] = 0 для b≠0."""
        ddt = random_sbox().difference_distribution_table()
        self.assertEqual(ddt[0][0], 64)
        for b in range(1, 64):
            self.assertEqual(ddt[0][b], 0)

    def test_identity_du(self):
        """Идентичность: DU = 64."""
        self.assertEqual(identity_sbox().differential_uniformity(), 64)

    def test_complement_du(self):
        """Дополнение: DU = 64 (аффинное)."""
        self.assertEqual(complement_sbox().differential_uniformity(), 64)

    def test_du_even(self):
        """DU всегда чётная (при n чётной, d-гомоморфизм)."""
        for seed in [1, 5, 9]:
            du = random_sbox(seed=seed).differential_uniformity()
            self.assertEqual(du % 2, 0)

    def test_not_apn_linear(self):
        """Линейный S-блок не является APN (DU=64 ≠ 2)."""
        self.assertFalse(identity_sbox().is_almost_perfect_nonlinear())


# ── алгебраическая степень ───────────────────────────────────────────────────

class TestAlgebraicDegree(unittest.TestCase):

    def test_identity_degree_one(self):
        """Identity — линейная функция, степень = 1."""
        self.assertEqual(identity_sbox().algebraic_degree(), 1)

    def test_random_degree_ge_two(self):
        """Случайный S-блок обычно имеет степень ≥ 2."""
        deg = random_sbox(seed=42).algebraic_degree()
        self.assertGreaterEqual(deg, 2)

    def test_degree_bounded(self):
        """Степень ≤ 6 для n = 6."""
        for seed in [1, 7, 13]:
            deg = random_sbox(seed=seed).algebraic_degree()
            self.assertLessEqual(deg, 6)


# ── SAC и ветвящееся число ───────────────────────────────────────────────────

class TestSACAndBranch(unittest.TestCase):

    def test_sac_matrix_shape(self):
        """SAC-матрица 6×6."""
        sac = random_sbox().sac_matrix()
        self.assertEqual(len(sac), 6)
        for row in sac:
            self.assertEqual(len(row), 6)

    def test_sac_values_in_range(self):
        """Все значения SAC-матрицы ∈ [0, 1]."""
        sac = random_sbox().sac_matrix()
        for row in sac:
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_identity_sac_not_satisfies(self):
        """Identity не удовлетворяет SAC (SAC[i][i]=1 ≠ 0.5)."""
        self.assertFalse(identity_sbox().satisfies_sac())

    def test_branch_number_identity(self):
        """Branch number тождественного = 2 (min hw(a)+hw(a)=2)."""
        self.assertEqual(identity_sbox().branch_number(), 2)

    def test_branch_number_positive(self):
        """Branch number всегда ≥ 1."""
        for sb in [random_sbox(1), random_sbox(2), affine_sbox()]:
            self.assertGreaterEqual(sb.branch_number(), 1)

    def test_autocorrelation_at_zero(self):
        """AC[0] = Σ_x 1 = 64 (скалярное произведение с собой)."""
        sb = random_sbox()
        ac = sb.autocorrelation(u=1)
        self.assertEqual(ac[0], 64)


# ── LAT и линейные аппроксимации ─────────────────────────────────────────────

class TestLAT(unittest.TestCase):

    def test_lat_identity_at_zero_row(self):
        """LAT[0][b] = 0 для b≠0 (bijection, balanced)."""
        lat = identity_sbox().linear_approximation_table()
        for b in range(1, 64):
            self.assertEqual(lat[0][b], 0)

    def test_best_linear_bias_positive(self):
        """Лучшая линейная bias > 0 для любого S-блока."""
        bias = identity_sbox().best_linear_approximation()
        self.assertGreater(bias, 0)

    def test_best_linear_bias_bounded(self):
        """bias ≤ 64."""
        bias = random_sbox().best_linear_approximation()
        self.assertLessEqual(bias, 64)


# ── evaluate_sbox ────────────────────────────────────────────────────────────

class TestEvaluate(unittest.TestCase):

    def test_evaluate_returns_all_keys(self):
        """evaluate_sbox возвращает все ожидаемые ключи."""
        ev = evaluate_sbox(random_sbox())
        for k in ('nonlinearity', 'differential_uniformity',
                  'algebraic_degree', 'satisfies_sac', 'branch_number', 'is_apn'):
            self.assertIn(k, ev)

    def test_evaluate_identity(self):
        """evaluate_sbox(identity): nl=0, deg=1."""
        ev = evaluate_sbox(identity_sbox())
        self.assertEqual(ev['nonlinearity'], 0)
        self.assertEqual(ev['algebraic_degree'], 1)

    def test_best_differential_format(self):
        """best_differential_characteristic возвращает (a, b, p)."""
        a, b, p = best_differential_characteristic(random_sbox())
        self.assertGreater(a, 0)
        self.assertGreaterEqual(b, 0)
        self.assertGreater(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_best_linear_bias_format(self):
        """best_linear_bias возвращает (a, b, bias)."""
        a, b, bias = best_linear_bias(random_sbox())
        self.assertGreater(a, 0)
        self.assertGreater(b, 0)
        self.assertGreater(bias, 0.0)
        self.assertLessEqual(bias, 1.0)


# ── search_good_sbox ─────────────────────────────────────────────────────────

class TestSearch(unittest.TestCase):

    def test_search_finds_with_low_bar(self):
        """При низком пороге nl≥8 находит S-блок."""
        result = search_good_sbox(min_nl=8, max_du=64, n_trials=100)
        self.assertIsNotNone(result)

    def test_search_result_valid(self):
        """Найденный S-блок действительно удовлетворяет условиям."""
        result = search_good_sbox(min_nl=12, n_trials=200, seed=1)
        if result is not None:
            sb, nl, du = result
            self.assertGreaterEqual(nl, 12)
            self.assertLessEqual(du, 8)

    def test_search_impossible_returns_none(self):
        """Слишком высокий nl=30 > 28 — не найти."""
        result = search_good_sbox(min_nl=30, n_trials=50)
        self.assertIsNone(result)


# ── HexStream ────────────────────────────────────────────────────────────────

class TestHexStream(unittest.TestCase):

    def test_creation_valid(self):
        """HexStream с ключом 0..63 создаётся."""
        for key in [0, 1, 31, 63]:
            hs = HexStream(key)
            self.assertIsNotNone(hs)

    def test_invalid_key(self):
        """Ключ вне [0, 63] — ошибка."""
        with self.assertRaises(ValueError):
            HexStream(64)
        with self.assertRaises(ValueError):
            HexStream(-1)

    def test_keystream_length(self):
        """keystream(n) возвращает n битов."""
        ks = HexStream(7).keystream(100)
        self.assertEqual(len(ks), 100)

    def test_keystream_bits(self):
        """Все биты ∈ {0, 1}."""
        ks = HexStream(15).keystream(200)
        self.assertTrue(all(b in (0, 1) for b in ks))

    def test_encrypt_decrypt_roundtrip(self):
        """decrypt(encrypt(msg)) = msg."""
        msg = [1, 0, 1, 1, 0, 0, 1, 0] * 4
        enc = HexStream(42).encrypt(msg)
        dec = HexStream(42).decrypt(enc)
        self.assertEqual(dec, msg)

    def test_different_keys_different_streams(self):
        """Разные ключи дают разные потоки."""
        ks1 = HexStream(0).keystream(32)
        ks2 = HexStream(1).keystream(32)
        self.assertNotEqual(ks1, ks2)

    def test_stream_not_constant(self):
        """Поток не является константой."""
        ks = HexStream(3).keystream(64)
        self.assertGreater(sum(ks), 0)
        self.assertLess(sum(ks), 64)


# ── FeistelCipher ─────────────────────────────────────────────────────────────

class TestFeistel(unittest.TestCase):

    def test_is_permutation(self):
        """FeistelCipher — биекция на Q6."""
        fc = FeistelCipher(n_rounds=4)
        self.assertTrue(fc.is_permutation())

    def test_encrypt_decrypt_all(self):
        """decrypt(encrypt(x)) = x для всех x ∈ Q6."""
        fc = FeistelCipher(n_rounds=4, seed=17)
        for x in range(64):
            self.assertEqual(fc.decrypt(fc.encrypt(x)), x)

    def test_as_sbox_valid(self):
        """as_sbox() возвращает правильный S-блок."""
        sb = FeistelCipher().as_sbox()
        self.assertIsInstance(sb, SBox)
        self.assertEqual(sorted(sb(x) for x in range(64)), list(range(64)))

    def test_encrypt_range(self):
        """Результат encrypt ∈ [0, 63]."""
        fc = FeistelCipher()
        for x in range(64):
            self.assertIn(fc.encrypt(x), range(64))

    def test_custom_round_keys(self):
        """Пользовательские ключи работают."""
        fc = FeistelCipher(n_rounds=2, round_keys=[3, 5])
        self.assertTrue(fc.is_permutation())

    def test_wrong_round_key_count(self):
        """Неправильное число ключей → ошибка."""
        with self.assertRaises(ValueError):
            FeistelCipher(n_rounds=4, round_keys=[1, 2])

    def test_more_rounds(self):
        """8 раундов: всё ещё перестановка."""
        fc = FeistelCipher(n_rounds=8, seed=99)
        self.assertTrue(fc.is_permutation())


class TestCryptCLI(unittest.TestCase):
    def _run(self, args):
        import io
        from contextlib import redirect_stdout
        from projects.hexcrypt.hexcrypt import main
        old_argv = sys.argv
        sys.argv = ['hexcrypt.py'] + args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_cmd_info_affine(self):
        out = self._run(['info', 'affine'])
        self.assertIn('affine', out)

    def test_cmd_info_identity(self):
        out = self._run(['info', 'identity'])
        self.assertGreater(len(out), 0)

    def test_cmd_table(self):
        out = self._run(['table', 'affine'])
        self.assertIn('affine', out)

    def test_cmd_stream(self):
        out = self._run(['stream', '7', '16'])
        self.assertIn('Ключ', out)

    def test_cmd_feistel_demo(self):
        out = self._run(['feistel'])
        self.assertIn('Фейстеля', out)

    def test_cmd_feistel_perm(self):
        out = self._run(['feistel', 'perm'])
        self.assertIn('перестановкой', out)

    def test_cmd_search_not_found(self):
        # high min_nl → no result found quickly
        out = self._run(['search', '100', '5'])
        self.assertIn('найден', out.lower())

    def test_cmd_sac(self):
        out = self._run(['sac', 'affine'])
        self.assertIn('SAC', out)

    def test_cmd_help(self):
        out = self._run([])
        self.assertIn('hexcrypt', out)

    def test_cmd_unknown(self):
        out = self._run(['unknown'])
        self.assertIn('hexcrypt', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
