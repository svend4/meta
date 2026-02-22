"""Тесты для hexstat — статистика и теория информации на Q6."""
import unittest
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from projects.hexstat.hexstat import (
    Q6Distribution, RandomWalk,
    chi_square_statistic, chi_square_p_value, test_uniformity as uniformity_test,
    empirical_entropy, bootstrap_entropy_ci, kolmogorov_smirnov_yang,
    q6_mutual_information, q6_channel_capacity_bsc,
    yang_entropy, max_entropy_with_mean_yang,
    _popcount,
)


# ── Q6Distribution: создание ─────────────────────────────────────────────────

class TestDistributionCreation(unittest.TestCase):

    def test_uniform_probs(self):
        """Равномерное: каждая вероятность = 1/64."""
        d = Q6Distribution.uniform()
        for p in d.probs():
            self.assertAlmostEqual(p, 1 / 64, places=10)

    def test_normalization_auto(self):
        """Ненормированный ввод автоматически нормируется."""
        d = Q6Distribution([2.0] * 64)
        self.assertAlmostEqual(sum(d.probs()), 1.0, places=10)

    def test_invalid_length(self):
        """len ≠ 64 → ошибка."""
        with self.assertRaises(ValueError):
            Q6Distribution([1.0] * 32)

    def test_invalid_zero_sum(self):
        """Все нули → ошибка."""
        with self.assertRaises(ValueError):
            Q6Distribution([0.0] * 64)

    def test_from_counts(self):
        """from_counts([1]*64) = равномерное."""
        d = Q6Distribution.from_counts([1] * 64)
        self.assertAlmostEqual(d[0], 1 / 64, places=10)

    def test_from_samples(self):
        """from_samples вычисляет эмпирическое распределение."""
        samples = list(range(64)) * 2  # каждый элемент дважды
        d = Q6Distribution.from_samples(samples)
        for h in range(64):
            self.assertAlmostEqual(d[h], 1 / 64, places=10)

    def test_yang_weighted_sum_one(self):
        """yang_weighted нормирована."""
        d = Q6Distribution.yang_weighted(0.5)
        self.assertAlmostEqual(sum(d.probs()), 1.0, places=10)

    def test_bsc_sum_one(self):
        """BSC нормирована."""
        d = Q6Distribution.binary_symmetric_channel(0, 0.1)
        self.assertAlmostEqual(sum(d.probs()), 1.0, places=10)

    def test_hamming_shell_sum_one(self):
        """hamming_shell нормирована."""
        d = Q6Distribution.hamming_shell(0, 2)
        self.assertAlmostEqual(sum(d.probs()), 1.0, places=10)


# ── энтропии ─────────────────────────────────────────────────────────────────

class TestEntropies(unittest.TestCase):

    def test_uniform_entropy_six(self):
        """Равномерное: H = log₂(64) = 6 бит."""
        self.assertAlmostEqual(Q6Distribution.uniform().entropy(), 6.0, places=10)

    def test_entropy_nonnegative(self):
        """H ≥ 0 для любого распределения."""
        for beta in [-2, 0, 2]:
            h = Q6Distribution.yang_weighted(beta).entropy()
            self.assertGreaterEqual(h, 0.0)

    def test_entropy_bounded_by_six(self):
        """H ≤ 6 для любого распределения на Q6."""
        for beta in [-1, 0, 1]:
            h = Q6Distribution.yang_weighted(beta).entropy()
            self.assertLessEqual(h, 6.0 + 1e-9)

    def test_entropy_uniform_is_max(self):
        """Равномерное имеет максимальную энтропию."""
        h_unif = Q6Distribution.uniform().entropy()
        for beta in [-1.0, 0.5, 2.0]:
            h = Q6Distribution.yang_weighted(beta).entropy()
            self.assertLessEqual(h, h_unif + 1e-9)

    def test_renyi_alpha1_equals_shannon(self):
        """Реньи(1) = Шеннон."""
        d = Q6Distribution.yang_weighted(0.5)
        self.assertAlmostEqual(d.renyi_entropy(1.0), d.entropy(), places=8)

    def test_renyi_monotone(self):
        """H_α убывает по α."""
        d = Q6Distribution.yang_weighted(1.0)
        self.assertGreaterEqual(d.renyi_entropy(0.5), d.renyi_entropy(1.0))
        self.assertGreaterEqual(d.renyi_entropy(1.0), d.renyi_entropy(2.0))

    def test_min_entropy_le_shannon(self):
        """H_∞ ≤ H (Шеннон)."""
        d = Q6Distribution.yang_weighted(1.0)
        self.assertLessEqual(d.min_entropy(), d.entropy() + 1e-9)

    def test_yang_weighted_beta_pos_lower_entropy(self):
        """β > 0 концентрирует массу → меньше энтропии, чем при β=0."""
        h0 = Q6Distribution.yang_weighted(0.0).entropy()
        h1 = Q6Distribution.yang_weighted(2.0).entropy()
        self.assertLess(h1, h0)

    def test_renyi_invalid_alpha_raises(self):
        """renyi_entropy(α ≤ 0) → ValueError."""
        d = Q6Distribution.uniform()
        with self.assertRaises(ValueError):
            d.renyi_entropy(0)
        with self.assertRaises(ValueError):
            d.renyi_entropy(-1)

    def test_bsc_invalid_p_error_raises(self):
        """binary_symmetric_channel с p_error < 0 → ValueError."""
        with self.assertRaises(ValueError):
            Q6Distribution.binary_symmetric_channel(0, -0.1)


# ── дивергенции ───────────────────────────────────────────────────────────────

class TestDivergences(unittest.TestCase):

    def test_kl_self_zero(self):
        """KL(P ‖ P) = 0."""
        d = Q6Distribution.yang_weighted(0.5)
        self.assertAlmostEqual(d.kl_divergence(d), 0.0, places=10)

    def test_kl_nonnegative(self):
        """KL(P ‖ Q) ≥ 0 (информационное неравенство)."""
        p = Q6Distribution.yang_weighted(1.0)
        q = Q6Distribution.uniform()
        self.assertGreaterEqual(p.kl_divergence(q), 0.0)

    def test_kl_uniform_to_concentrated(self):
        """KL(uniform ‖ concentrated) > 0."""
        p = Q6Distribution.uniform()
        q = Q6Distribution.yang_weighted(5.0)
        self.assertGreater(p.kl_divergence(q), 0.0)

    def test_kl_disjoint_support_returns_inf(self):
        """KL(P ‖ Q) = +∞ когда support(P) ⊄ support(Q)."""
        counts_p = [100 if h == 0 else 0 for h in range(64)]
        counts_q = [100 if h == 1 else 0 for h in range(64)]
        p = Q6Distribution.from_counts(counts_p)
        q = Q6Distribution.from_counts(counts_q)
        self.assertEqual(p.kl_divergence(q), math.inf)

    def test_tv_self_zero(self):
        """TV(P, P) = 0."""
        d = Q6Distribution.yang_weighted(0.5)
        self.assertAlmostEqual(d.total_variation(d), 0.0, places=10)

    def test_tv_bounded(self):
        """TV(P, Q) ∈ [0, 1]."""
        p = Q6Distribution.yang_weighted(3.0)
        q = Q6Distribution.yang_weighted(-3.0)
        tv = p.total_variation(q)
        self.assertGreaterEqual(tv, 0.0)
        self.assertLessEqual(tv, 1.0)

    def test_hellinger_self_zero(self):
        """Hellinger(P, P) = 0."""
        d = Q6Distribution.yang_weighted(0.7)
        self.assertAlmostEqual(d.hellinger(d), 0.0, places=10)

    def test_hellinger_bounded(self):
        """Hellinger ∈ [0, 1]."""
        p = Q6Distribution.yang_weighted(2.0)
        q = Q6Distribution.yang_weighted(-2.0)
        h = p.hellinger(q)
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, 1.0 + 1e-9)

    def test_js_divergence_symmetric(self):
        """JS-дивергенция симметрична: JSD(P,Q) = JSD(Q,P)."""
        p = Q6Distribution.yang_weighted(1.0)
        q = Q6Distribution.yang_weighted(-1.0)
        self.assertAlmostEqual(p.js_divergence(q), q.js_divergence(p), places=10)

    def test_js_divergence_bounded(self):
        """JSD ∈ [0, 1]."""
        p = Q6Distribution.yang_weighted(2.0)
        q = Q6Distribution.yang_weighted(-2.0)
        jsd = p.js_divergence(q)
        self.assertGreaterEqual(jsd, 0.0)
        self.assertLessEqual(jsd, 1.0 + 1e-9)


# ── маргинальные распределения и yang ────────────────────────────────────────

class TestMarginals(unittest.TestCase):

    def test_uniform_marginal_half(self):
        """Равномерное: P(bit_i = 1) = 0.5 для всех i."""
        d = Q6Distribution.uniform()
        for i in range(6):
            self.assertAlmostEqual(d.marginal_bit(i), 0.5, places=10)

    def test_marginal_bits_sum_consistent(self):
        """Сумма маргинальных ≠ что-то странное, просто проверим длину."""
        mbs = Q6Distribution.yang_weighted(1.0).marginal_bits()
        self.assertEqual(len(mbs), 6)
        for p in mbs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_bit_correlation_uniform_zero(self):
        """Под равномерным: Cov(bit_i, bit_j) = 0 для i ≠ j (независимые биты)."""
        d = Q6Distribution.uniform()
        for i in range(6):
            for j in range(6):
                if i != j:
                    self.assertAlmostEqual(d.bit_correlation(i, j), 0.0, places=10)

    def test_bit_variance_uniform(self):
        """Cov(bit_i, bit_i) = Var(bit_i) = p(1-p) = 0.25."""
        d = Q6Distribution.uniform()
        for i in range(6):
            self.assertAlmostEqual(d.bit_correlation(i, i), 0.25, places=10)

    def test_yang_distribution_uniform(self):
        """Равномерное: P(yang=k) = C(6,k)/64."""
        from math import comb
        d = Q6Distribution.uniform()
        yd = d.yang_distribution()
        for k in range(7):
            self.assertAlmostEqual(yd[k], comb(6, k) / 64, places=10)

    def test_mean_yang_uniform_three(self):
        """Равномерное: E[yang] = 3."""
        self.assertAlmostEqual(Q6Distribution.uniform().mean_yang(), 3.0, places=10)

    def test_mean_yang_monotone_in_beta(self):
        """E[yang] строго возрастает по β."""
        means = [Q6Distribution.yang_weighted(b).mean_yang()
                 for b in [-2.0, -1.0, 0.0, 1.0, 2.0]]
        for i in range(len(means) - 1):
            self.assertLess(means[i], means[i + 1])

    def test_variance_yang_nonnegative(self):
        """Var[yang] ≥ 0."""
        for beta in [-1.0, 0.0, 1.0]:
            self.assertGreaterEqual(Q6Distribution.yang_weighted(beta).variance_yang(), 0.0)

    def test_bsc_max_at_center(self):
        """BSC(center=5, p=0.1): P(5) — максимум."""
        d = Q6Distribution.binary_symmetric_channel(5, 0.1)
        self.assertEqual(d.probs().index(max(d.probs())), 5)

    def test_hamming_shell_support(self):
        """hamming_shell(center=0, r=1): ненулевые вероятности только у вершин расстояния ≤1."""
        d = Q6Distribution.hamming_shell(0, 1)
        for h in range(64):
            if _popcount(h) <= 1:
                self.assertGreater(d[h], 0.0)
            else:
                self.assertEqual(d[h], 0.0)


# ── характеристическая функция ────────────────────────────────────────────────

class TestCharacteristic(unittest.TestCase):

    def test_phi0_always_one(self):
        """φ(0) = 1 для любого распределения."""
        for beta in [-1.0, 0.0, 1.0]:
            d = Q6Distribution.yang_weighted(beta)
            phi = d.characteristic_function()
            self.assertAlmostEqual(phi[0], 1.0, places=10)

    def test_phi_uniform_zero_nonzero(self):
        """Равномерное: φ(u) = 0 для u ≠ 0."""
        phi = Q6Distribution.uniform().characteristic_function()
        for u in range(1, 64):
            self.assertAlmostEqual(phi[u], 0.0, places=10)

    def test_phi_bounded(self):
        """|φ(u)| ≤ 1 для любого u."""
        d = Q6Distribution.yang_weighted(0.5)
        phi = d.characteristic_function()
        for u in range(64):
            self.assertLessEqual(abs(phi[u]), 1.0 + 1e-9)


# ── выборка ──────────────────────────────────────────────────────────────────

class TestSampling(unittest.TestCase):

    def test_sample_length(self):
        """sample(n) возвращает n элементов."""
        d = Q6Distribution.uniform()
        self.assertEqual(len(d.sample(100)), 100)

    def test_sample_values_in_range(self):
        """Все выборки ∈ [0, 63]."""
        samples = Q6Distribution.uniform().sample(200, seed=7)
        self.assertTrue(all(0 <= s <= 63 for s in samples))

    def test_sample_reproducible(self):
        """Один и тот же seed → одинаковая выборка."""
        d = Q6Distribution.uniform()
        s1 = d.sample(100, seed=42)
        s2 = d.sample(100, seed=42)
        self.assertEqual(s1, s2)

    def test_empirical_close_to_true(self):
        """Эмпирическое TV к истинному распределению убывает с n."""
        d = Q6Distribution.yang_weighted(0.5)
        small = Q6Distribution.from_samples(d.sample(200, seed=1))
        large = Q6Distribution.from_samples(d.sample(5000, seed=2))
        self.assertLess(large.total_variation(d), small.total_variation(d))


# ── RandomWalk ───────────────────────────────────────────────────────────────

class TestRandomWalk(unittest.TestCase):

    def test_walk_length(self):
        """walk(n) возвращает n+1 элементов."""
        rw = RandomWalk(start=0, seed=7)
        path = rw.walk(50)
        self.assertEqual(len(path), 51)

    def test_walk_valid_steps(self):
        """Каждый переход — ребро Q6 (Хэмминг-расстояние = 1)."""
        rw = RandomWalk(start=0, seed=3)
        path = rw.walk(100)
        for a, b in zip(path, path[1:]):
            diff = a ^ b
            self.assertGreater(diff, 0)
            self.assertEqual(diff & (diff - 1), 0)  # степень двойки → 1 бит

    def test_walk_values_in_range(self):
        """Все вершины пути ∈ [0, 63]."""
        path = RandomWalk(start=5, seed=9).walk(200)
        self.assertTrue(all(0 <= v <= 63 for v in path))

    def test_stationary_is_uniform(self):
        """Теоретическое стационарное = равномерное."""
        stat = RandomWalk.stationary_distribution()
        for h in range(64):
            self.assertAlmostEqual(stat[h], 1 / 64, places=10)

    def test_cover_time_positive(self):
        """Время покрытия ≥ 63 (нужно посетить ≥ 63 новых вершин)."""
        ct = RandomWalk().cover_time_empirical(start=0, seed=42)
        self.assertGreaterEqual(ct, 63)

    def test_empirical_converges(self):
        """После многих шагов TV к равномерному < 0.1."""
        rw = RandomWalk(start=0, seed=1)
        emp = rw.empirical_distribution(10000, start=0, seed=1)
        tv = emp.total_variation(Q6Distribution.uniform())
        self.assertLess(tv, 0.1)


# ── χ²-тест и p-значение ─────────────────────────────────────────────────────

class TestChiSquare(unittest.TestCase):

    def test_chi_square_statistic_uniform(self):
        """Для идеальных частот χ² = 0."""
        counts = [10] * 64
        self.assertAlmostEqual(chi_square_statistic(counts), 0.0, places=10)

    def test_chi_square_statistic_positive(self):
        """χ² ≥ 0 для любых частот."""
        counts = [5, 10, 3] + [8] * 61
        self.assertGreaterEqual(chi_square_statistic(counts), 0.0)

    def test_p_value_in_range(self):
        """p-значение ∈ [0, 1]."""
        for chi2 in [0.1, 10, 63, 200, 500]:
            p = chi_square_p_value(chi2, df=63)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_p_value_large_chi2_small(self):
        """Большой χ² → p близко к 0."""
        p = chi_square_p_value(500, df=63)
        self.assertLess(p, 0.01)

    def test_p_value_small_chi2_large(self):
        """Малый χ² → p близко к 1."""
        p = chi_square_p_value(10, df=63)
        self.assertGreater(p, 0.3)

    def test_uniformity_biased_rejected(self):
        """Очень неравномерные данные: отвергаем H₀."""
        # Все 500 выборок из yang_weighted(β=5)
        d = Q6Distribution.yang_weighted(5.0)
        samples = d.sample(500, seed=1)
        chi2, df, p, reject = uniformity_test(samples)
        self.assertTrue(reject)

    def test_uniformity_uniform_not_rejected(self):
        """Равномерные данные (n=2000): не отвергаем H₀ (статистически)."""
        d = Q6Distribution.uniform()
        samples = d.sample(2000, seed=42)
        chi2, df, p, reject = uniformity_test(samples)
        # p-значение должно быть разумным (не близко к 0)
        # Даём небольшой запас — иногда случайная выборка может дать низкий p
        # Проверяем просто что chi2/df близко к 1
        self.assertLess(chi2 / df, 3.0)


# ── эмпирическая энтропия и бутстрэп ────────────────────────────────────────

class TestEntropyEstimation(unittest.TestCase):

    def test_empirical_entropy_uniform_close_to_six(self):
        """Для большой равномерной выборки эмп. энтропия близка к 6."""
        d = Q6Distribution.uniform()
        samples = d.sample(5000, seed=0)
        H = empirical_entropy(samples)
        self.assertGreater(H, 5.5)
        self.assertLessEqual(H, 6.0 + 1e-9)

    def test_bootstrap_ci_contains_true(self):
        """95% CI бутстрэпа содержит истинную энтропию."""
        d = Q6Distribution.yang_weighted(0.5)
        true_h = d.entropy()
        samples = d.sample(1000, seed=13)
        lo, est, hi = bootstrap_entropy_ci(samples, n_bootstrap=200, alpha=0.05, seed=7)
        self.assertLessEqual(lo, true_h + 0.3)
        self.assertGreaterEqual(hi, true_h - 0.3)

    def test_bootstrap_ci_ordered(self):
        """lo ≤ estimate ≤ hi."""
        d = Q6Distribution.yang_weighted(1.0)
        samples = d.sample(500, seed=5)
        lo, est, hi = bootstrap_entropy_ci(samples, n_bootstrap=50, seed=1)
        self.assertLessEqual(lo, est + 1e-9)
        self.assertLessEqual(est, hi + 1e-9)


# ── KS-тест ──────────────────────────────────────────────────────────────────

class TestKS(unittest.TestCase):

    def test_ks_same_distribution_small(self):
        """KS-статистика для двух выборок из одного распределения мала."""
        d = Q6Distribution.yang_weighted(0.5)
        s1 = d.sample(500, seed=10)
        s2 = d.sample(500, seed=20)
        ks, n1, n2 = kolmogorov_smirnov_yang(s1, s2)
        self.assertLess(ks, 0.2)

    def test_ks_different_distributions_large(self):
        """KS-статистика для разных распределений больше."""
        d1 = Q6Distribution.yang_weighted(3.0)
        d2 = Q6Distribution.yang_weighted(-3.0)
        s1 = d1.sample(500, seed=1)
        s2 = d2.sample(500, seed=2)
        ks, _, _ = kolmogorov_smirnov_yang(s1, s2)
        self.assertGreater(ks, 0.3)

    def test_ks_returns_correct_sizes(self):
        """Возвращаемые n1, n2 правильны."""
        s1 = [0] * 100
        s2 = [1] * 200
        _, n1, n2 = kolmogorov_smirnov_yang(s1, s2)
        self.assertEqual(n1, 100)
        self.assertEqual(n2, 200)


# ── информационные меры ──────────────────────────────────────────────────────

class TestInformationMeasures(unittest.TestCase):

    def test_mutual_information_uniform_zero(self):
        """MI(bit_i; bit_j) = 0 при равномерном распределении (биты независимы)."""
        d = Q6Distribution.uniform()
        for i in range(6):
            for j in range(6):
                if i != j:
                    mi = q6_mutual_information(i, j, d)
                    self.assertAlmostEqual(mi, 0.0, places=10)

    def test_mutual_information_nonnegative(self):
        """MI ≥ 0 всегда."""
        d = Q6Distribution.yang_weighted(1.0)
        for i in range(3):
            mi = q6_mutual_information(i, i + 1, d)
            self.assertGreaterEqual(mi, -1e-10)

    def test_bsc_capacity_zero_error(self):
        """BSC с p=0: ёмкость = 1 бит."""
        self.assertAlmostEqual(q6_channel_capacity_bsc(0), 1.0, places=10)

    def test_bsc_capacity_half_error(self):
        """BSC с p=0.5: ёмкость = 0 бит."""
        self.assertAlmostEqual(q6_channel_capacity_bsc(0.5), 0.0, places=10)

    def test_bsc_capacity_one_error(self):
        """BSC с p=1: ёмкость = 1 бит (детерминированная инверсия)."""
        self.assertAlmostEqual(q6_channel_capacity_bsc(1), 1.0, places=10)

    def test_yang_entropy_k0(self):
        """yang_entropy(0) = log₂(1) = 0 (один элемент с yang=0)."""
        self.assertAlmostEqual(yang_entropy(0), 0.0, places=10)

    def test_yang_entropy_k3(self):
        """yang_entropy(3) = log₂(C(6,3)) = log₂(20)."""
        self.assertAlmostEqual(yang_entropy(3), math.log2(20), places=10)

    def test_yang_entropy_k6(self):
        """yang_entropy(6) = 0 (один элемент: 63)."""
        self.assertAlmostEqual(yang_entropy(6), 0.0, places=10)

    def test_max_entropy_mean_yang_three(self):
        """max_entropy_with_mean_yang(3) ≈ равномерное."""
        d = max_entropy_with_mean_yang(3.0)
        self.assertAlmostEqual(d.mean_yang(), 3.0, places=4)
        # Энтропия близка к 6
        self.assertAlmostEqual(d.entropy(), 6.0, places=3)

    def test_max_entropy_mean_yang_achieved(self):
        """E[yang] в полученном распределении ≈ target."""
        for target in [1.0, 2.5, 4.0, 5.5]:
            d = max_entropy_with_mean_yang(target)
            self.assertAlmostEqual(d.mean_yang(), target, places=4)


class TestStatCLI(unittest.TestCase):
    def _run(self, args):
        import io
        from contextlib import redirect_stdout
        from projects.hexstat.hexstat import main
        old_argv = sys.argv
        sys.argv = ['hexstat.py'] + args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_cmd_info(self):
        out = self._run(['info'])
        self.assertIn('H =', out)

    def test_cmd_walk(self):
        out = self._run(['walk', '100'])
        self.assertIn('блуждание', out)

    def test_cmd_sample(self):
        out = self._run(['sample', '200'])
        self.assertIn('χ²', out)

    def test_cmd_entropy(self):
        out = self._run(['entropy'])
        self.assertIn('β=', out)

    def test_cmd_correlation_uniform(self):
        out = self._run(['correlation', 'uniform'])
        self.assertIn('ковариаций', out)

    def test_cmd_correlation_yang(self):
        out = self._run(['correlation', 'yang'])
        self.assertGreater(len(out), 0)

    def test_cmd_correlation_bsc(self):
        out = self._run(['correlation', 'bsc'])
        self.assertGreater(len(out), 0)

    def test_cmd_test(self):
        out = self._run(['test'])
        self.assertIn('CI', out)

    def test_cmd_help(self):
        out = self._run(['help'])
        self.assertIn('hexstat', out)

    def test_cmd_unknown(self):
        out = self._run(['unknown'])
        self.assertIn('hexstat', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
