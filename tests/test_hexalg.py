"""Тесты для hexalg — гармонический анализ на Q6."""
import unittest
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from projects.hexalg.hexalg import (
    character, all_characters, hadamard_matrix,
    fourier_transform, inverse_fourier_transform, verify_inverse,
    convolve, convolve_via_fft, correlate, autocorrelation,
    convolution_theorem_check,
    inner_product_spatial, inner_product_frequency, parseval_identity,
    cayley_graph, cayley_eigenvalues, hypercube_spectrum, cayley_is_connected,
    subgroup_generated, coset_decomposition, dual_subgroup, index_of_subgroup,
    pontryagin_dual_character,
    difference_multiset, is_difference_set,
    is_bent_function, bent_function_difference_set,
    group_ring_add_f2, group_ring_mul_f2, group_ring_support, indicator,
    _inner_product, _popcount,
)


# ── характеры ────────────────────────────────────────────────────────────────

class TestCharacters(unittest.TestCase):

    def test_character_pm1(self):
        """χ_u(h) ∈ {−1, +1}."""
        for u in range(64):
            for h in [0, 7, 42, 63]:
                self.assertIn(character(u, h), (-1, 1))

    def test_character_u0_always_1(self):
        """χ_0(h) = 1 для всех h (тривиальный характер)."""
        for h in range(64):
            self.assertEqual(character(0, h), 1)

    def test_character_h0_always_1(self):
        """χ_u(0) = 1 для всех u."""
        for u in range(64):
            self.assertEqual(character(u, 0), 1)

    def test_character_multiplicative(self):
        """χ_u(a ⊕ b) = χ_u(a) × χ_u(b) (мультипликативность)."""
        for u in [1, 7, 42]:
            for a, b in [(3, 5), (15, 20), (0, 63)]:
                self.assertEqual(character(u, a ^ b),
                                 character(u, a) * character(u, b))

    def test_character_orthogonality(self):
        """Σ_h χ_u(h) χ_v(h) = 64 · δ_{u,v} (ортогональность)."""
        for u in [0, 1, 7]:
            for v in [0, 1, 7]:
                dot = sum(character(u, h) * character(v, h) for h in range(64))
                expected = 64 if u == v else 0
                self.assertEqual(dot, expected)

    def test_hadamard_matrix_shape(self):
        """Матрица Адамара 64×64."""
        H = hadamard_matrix()
        self.assertEqual(len(H), 64)
        for row in H:
            self.assertEqual(len(row), 64)

    def test_hadamard_orthogonal(self):
        """H·Hᵀ = 64·I (ортогональность строк)."""
        H = hadamard_matrix()
        for i in range(5):
            for j in range(5):
                dot = sum(H[i][k] * H[j][k] for k in range(64))
                expected = 64 if i == j else 0
                self.assertEqual(dot, expected)


# ── преобразование Фурье (WHT) ────────────────────────────────────────────────

class TestFourierTransform(unittest.TestCase):

    def test_ft_delta0(self):
        """FT(δ_0) = [1, 1, ..., 1] (все единицы)."""
        f = [1 if h == 0 else 0 for h in range(64)]
        F = fourier_transform(f)
        self.assertTrue(all(abs(v - 1) < 1e-9 for v in F))

    def test_ft_constant(self):
        """FT([1,...,1]) = [64, 0, 0, ..., 0]."""
        f = [1.0] * 64
        F = fourier_transform(f)
        self.assertAlmostEqual(F[0], 64.0, places=10)
        for u in range(1, 64):
            self.assertAlmostEqual(F[u], 0.0, places=10)

    def test_ft_character_is_delta(self):
        """FT(χ_u) = 64 · δ_u (преобразование характера = дельта)."""
        u0 = 7
        f = [float(character(u0, h)) for h in range(64)]
        F = fourier_transform(f)
        self.assertAlmostEqual(F[u0], 64.0, places=10)
        for u in range(64):
            if u != u0:
                self.assertAlmostEqual(F[u], 0.0, places=10)

    def test_ift_inverse(self):
        """IFT(FT(f)) = f."""
        f = [float(h % 7) for h in range(64)]
        self.assertTrue(verify_inverse(f))

    def test_ft_linear(self):
        """FT(af + bg) = a·FT(f) + b·FT(g)."""
        f = [1.0 if h < 32 else 0.0 for h in range(64)]
        g = [float(_popcount(h)) for h in range(64)]
        Ff = fourier_transform(f)
        Fg = fourier_transform(g)
        Fafbg = fourier_transform([2 * f[h] + 3 * g[h] for h in range(64)])
        for u in range(64):
            self.assertAlmostEqual(Fafbg[u], 2 * Ff[u] + 3 * Fg[u], places=9)

    def test_ft_involutory(self):
        """FT(FT(f)) = 64 · f (обратимость с масштабом)."""
        f = [float(_popcount(h) % 2) for h in range(64)]
        FFf = fourier_transform(fourier_transform(f))
        for h in range(64):
            self.assertAlmostEqual(FFf[h], 64.0 * f[h], places=9)


# ── свёртка ──────────────────────────────────────────────────────────────────

class TestConvolution(unittest.TestCase):

    def test_convolve_delta_identity(self):
        """f * δ_0 = f (единица свёртки — δ_0)."""
        f = [float(_popcount(h)) for h in range(64)]
        delta = [1.0 if h == 0 else 0.0 for h in range(64)]
        fg = convolve(f, delta)
        for h in range(64):
            self.assertAlmostEqual(fg[h], f[h], places=10)

    def test_convolve_commutative(self):
        """f * g = g * f."""
        f = [float(h % 3) for h in range(64)]
        g = [float(_popcount(h)) for h in range(64)]
        fg = convolve(f, g)
        gf = convolve(g, f)
        for h in range(64):
            self.assertAlmostEqual(fg[h], gf[h], places=9)

    def test_convolution_theorem(self):
        """F(f*g) = F(f)·F(g) поточечно."""
        f = [float(h & 1) for h in range(64)]
        g = [float(_popcount(h) % 2) for h in range(64)]
        self.assertTrue(convolution_theorem_check(f, g))

    def test_convolution_theorem_fails_tol_zero(self):
        """convolution_theorem_check с tol=0 и случайными сигналами → False."""
        import random
        rng = random.Random(42)
        f = [rng.gauss(0, 1) for _ in range(64)]
        g = [rng.gauss(0, 1) for _ in range(64)]
        result = convolution_theorem_check(f, g, tol=0)
        self.assertFalse(result)

    def test_convolve_via_fft_matches(self):
        """convolve_via_fft == convolve (прямое вычисление)."""
        f = [float(h % 5) for h in range(64)]
        g = [1.0 if _popcount(h) == 3 else 0.0 for h in range(64)]
        fg_direct = convolve(f, g)
        fg_fft = convolve_via_fft(f, g)
        for h in range(64):
            self.assertAlmostEqual(fg_direct[h], fg_fft[h], places=9)

    def test_correlate_equals_convolve_in_z2n(self):
        """В (Z₂)^n инверсия = тождественная: correlate(f,g) = convolve(f,g)."""
        f = [float(h % 3) for h in range(64)]
        g = [1.0 if _popcount(h) == 2 else 0.0 for h in range(64)]
        corr = correlate(f, g)
        conv = convolve(f, g)
        for h in range(64):
            self.assertAlmostEqual(corr[h], conv[h], places=9)

    def test_autocorrelation_at_0(self):
        """AC_f(0) = Σ_h f(h)² = ‖f‖²."""
        f = [float(_popcount(h)) for h in range(64)]
        ac = autocorrelation(f)
        expected = sum(x ** 2 for x in f)
        self.assertAlmostEqual(ac[0], expected, places=9)

    def test_parseval_identity(self):
        """Σ|f|² = (1/64)Σ|F(f)|²."""
        f = [float(_popcount(h) % 3) for h in range(64)]
        self.assertTrue(parseval_identity(f))

    def test_inner_product_plancherel(self):
        """⟨f, g⟩ = (1/64) ⟨F(f), F(g)⟩."""
        f = [float(h % 7) for h in range(64)]
        g = [float(_popcount(h)) for h in range(64)]
        lhs = inner_product_spatial(f, g)
        rhs = inner_product_frequency(fourier_transform(f), fourier_transform(g))
        self.assertAlmostEqual(lhs, rhs, places=9)


# ── граф Кэли ────────────────────────────────────────────────────────────────

class TestCayleyGraph(unittest.TestCase):

    def test_hypercube_is_cayley(self):
        """Q6 = Cay(Q6, {e₀,...,e₅}): 192 рёбра."""
        S = [1 << i for i in range(6)]
        edges = cayley_graph(S)
        self.assertEqual(len(edges), 192)

    def test_cayley_connected(self):
        """Q6 с генераторами {e₀,...,e₅} связен."""
        S = [1 << i for i in range(6)]
        self.assertTrue(cayley_is_connected(S))

    def test_cayley_disconnected(self):
        """Cayley с S={1} (один бит) не связен (не порождает Q6)."""
        self.assertFalse(cayley_is_connected([1]))

    def test_hypercube_eigenvalues(self):
        """Спектр Q6: {6, 4, 2, 0, -2, -4, -6} с кратностями C(6,k)."""
        from math import comb
        spec = hypercube_spectrum()
        self.assertEqual(spec[6], 1)   # C(6,0) = 1
        self.assertEqual(spec[4], 6)   # C(6,1) = 6
        self.assertEqual(spec[2], 15)  # C(6,2) = 15
        self.assertEqual(spec[0], 20)  # C(6,3) = 20
        self.assertEqual(spec[-2], 15)
        self.assertEqual(spec[-4], 6)
        self.assertEqual(spec[-6], 1)

    def test_cayley_eigenvalues_q6(self):
        """Собственные значения Cay(Q6, S₁) = 6−2·yang_count(u)."""
        S = [1 << i for i in range(6)]
        evs = cayley_eigenvalues(S)
        for u in range(64):
            expected = 6 - 2 * _popcount(u)
            self.assertEqual(evs[u], expected)

    def test_cayley_eigenvalues_sum(self):
        """Σ λ_u = 0 (матрица смежности бесследная)."""
        S = [1 << i for i in range(6)]
        evs = cayley_eigenvalues(S)
        self.assertAlmostEqual(sum(evs), 0.0, places=10)


# ── подгруппы и смежные классы ───────────────────────────────────────────────

class TestSubgroups(unittest.TestCase):

    def test_subgroup_trivial(self):
        """⟨∅⟩ = {0}."""
        H = subgroup_generated([])
        self.assertEqual(H, frozenset({0}))

    def test_subgroup_single_generator(self):
        """⟨{g}⟩ = {0, g}."""
        for g in [1, 7, 42]:
            H = subgroup_generated([g])
            self.assertEqual(H, frozenset({0, g}))

    def test_subgroup_order_power_of_2(self):
        """Порядок подгруппы = степень 2."""
        for gens in [[1, 2], [1, 2, 4], [1, 2, 4, 8]]:
            H = subgroup_generated(gens)
            n = len(H)
            self.assertEqual(n & (n - 1), 0)  # степень 2

    def test_subgroup_full_q6(self):
        """⟨{1,2,4,8,16,32}⟩ = Q6."""
        H = subgroup_generated([1 << i for i in range(6)])
        self.assertEqual(len(H), 64)

    def test_coset_decomposition_partition(self):
        """Смежные классы образуют разбиение Q6."""
        H = subgroup_generated([1, 2, 4])  # |H|=8
        cosets = coset_decomposition(H)
        all_verts = set()
        for c in cosets:
            self.assertTrue(all_verts.isdisjoint(c))
            all_verts |= set(c)
        self.assertEqual(all_verts, set(range(64)))

    def test_coset_decomposition_size(self):
        """Число смежных классов = [Q6:H] = 64/|H|."""
        H = subgroup_generated([1, 2])  # |H|=4
        cosets = coset_decomposition(H)
        self.assertEqual(len(cosets), 64 // len(H))

    def test_dual_subgroup_size(self):
        """|H⊥| = 64/|H|."""
        H = subgroup_generated([1, 2, 4])
        perp = dual_subgroup(H)
        self.assertEqual(len(perp) * len(H), 64)

    def test_dual_subgroup_orthogonal(self):
        """Все u ∈ H⊥ ортогональны всем h ∈ H."""
        H = subgroup_generated([1, 2, 4])
        perp = dual_subgroup(H)
        for u in perp:
            for h in H:
                self.assertEqual(_inner_product(u, h), 0)

    def test_index_of_subgroup_trivial(self):
        """Индекс тривиальной подгруппы = 64."""
        H = subgroup_generated([0])  # {0}
        idx = index_of_subgroup(H)
        self.assertEqual(idx, 64 // len(H))

    def test_index_of_subgroup_full(self):
        """Индекс всей группы Q6 = 1."""
        H = subgroup_generated([1, 2, 4, 8, 16, 32])  # all Q6
        idx = index_of_subgroup(H)
        self.assertEqual(idx, 64 // len(H))

    def test_pontryagin_dual_length(self):
        """Двойственный характер — список длины 64."""
        for h in [0, 7, 42]:
            dc = pontryagin_dual_character(h)
            self.assertEqual(len(dc), 64)
            self.assertTrue(all(v in (-1, 1) for v in dc))


# ── разностные множества и bent-функции ──────────────────────────────────────

class TestDifferenceSets(unittest.TestCase):

    def test_difference_multiset_size(self):
        """|D(A)| = |A|×(|A|-1) (с повторениями)."""
        A = [0, 7, 42, 63]
        dm = difference_multiset(A)
        total = sum(dm.values())
        self.assertEqual(total, len(A) * (len(A) - 1))

    def test_difference_multiset_no_zero(self):
        """0 не входит в разностный мультимножество."""
        A = [0, 7, 42, 63]
        dm = difference_multiset(A)
        self.assertNotIn(0, dm)

    def test_difference_multiset_symmetric(self):
        """D(A) симметрично: count(d) = count(d) (в Z₂⁶: d = −d)."""
        A = [0, 7, 14]
        dm = difference_multiset(A)
        for d, cnt in dm.items():
            self.assertEqual(dm.get(d, 0), cnt)  # d = -d в Z₂⁶

    def test_is_not_difference_set_small(self):
        """Произвольное малое множество обычно не difference set."""
        self.assertFalse(is_difference_set([0, 1, 2]))

    def test_is_bent_function_quadratic(self):
        """Квадратичная функция f(h) = x₀x₁ ⊕ x₂x₃ ⊕ x₄x₅ — bent."""
        def bent_f(h):
            x = [(h >> i) & 1 for i in range(6)]
            return x[0] * x[1] ^ x[2] * x[3] ^ x[4] * x[5]
        f_table = [bent_f(h) for h in range(64)]
        self.assertTrue(is_bent_function(f_table))

    def test_is_bent_function_inner_product_q3(self):
        """f(h) = ⟨lower3, upper3⟩ = inner product на Q3×Q3 — bent."""
        def bent_f(h):
            lower3 = h & 7
            upper3 = (h >> 3) & 7
            return _popcount(lower3 & upper3) % 2
        f_table = [bent_f(h) for h in range(64)]
        self.assertTrue(is_bent_function(f_table))

    def test_not_bent_linear(self):
        """Линейная функция f(h) = popcount(u&h) mod 2 — не bent (NL=0)."""
        f_table = [_popcount(21 & h) % 2 for h in range(64)]
        self.assertFalse(is_bent_function(f_table))

    def test_not_bent_bit0(self):
        """Линейная функция f(h) = bit_0(h) — не bent (NL=0)."""
        f_table = [h & 1 for h in range(64)]
        self.assertFalse(is_bent_function(f_table))

    def test_bent_function_support_size(self):
        """Носитель bent-функции: |support|=28 или 36 (дисбаланс = |WHT[0]|/2 = 4)."""
        # Bent-функции НЕ сбалансированы: |WHT[0]|=8 → |support|=(64±8)/2 = 28 или 36
        def bent_f(h):
            x = [(h >> i) & 1 for i in range(6)]
            return x[0] * x[1] ^ x[2] * x[3] ^ x[4] * x[5]
        f_table = [bent_f(h) for h in range(64)]
        A, _ = bent_function_difference_set(f_table)
        self.assertIn(len(A), [28, 36])


# ── групповое кольцо F₂[Q6] ──────────────────────────────────────────────────

class TestGroupRing(unittest.TestCase):

    def test_add_self_is_zero(self):
        """f + f = 0 в F₂[Q6]."""
        f = indicator([0, 7, 42])
        zero = group_ring_add_f2(f, f)
        self.assertTrue(all(v == 0 for v in zero))

    def test_mul_by_delta0(self):
        """f · δ_0 = f (единица кольца — δ_0)."""
        f = indicator([0, 7, 42, 63])
        delta = [1 if h == 0 else 0 for h in range(64)]
        product = group_ring_mul_f2(f, delta)
        self.assertEqual(product, f)

    def test_mul_indicator_sets(self):
        """Произведение индикаторов множеств = XOR-суммирование."""
        A = frozenset([0, 1])
        B = frozenset([0, 2])
        fA = indicator(A)
        fB = indicator(B)
        fAB = group_ring_mul_f2(fA, fB)
        # (X^0 + X^1)(X^0 + X^2) = X^0 + X^2 + X^1 + X^3 over F₂
        expected_support = frozenset([0, 1, 2, 3])
        self.assertEqual(group_ring_support(fAB), expected_support)

    def test_support_indicator(self):
        """group_ring_support(indicator(A)) = A."""
        A = frozenset([3, 7, 15])
        self.assertEqual(group_ring_support(indicator(A)), A)

    def test_add_commutative(self):
        """f + g = g + f."""
        f = indicator([0, 7])
        g = indicator([1, 7, 42])
        self.assertEqual(group_ring_add_f2(f, g), group_ring_add_f2(g, f))

    def test_mul_commutative(self):
        """f·g = g·f в F₂[Q6] (abelian group!)."""
        f = indicator([0, 3])
        g = indicator([1, 5])
        self.assertEqual(group_ring_mul_f2(f, g), group_ring_mul_f2(g, f))


# ── CLI main() ───────────────────────────────────────────────────────────────

class TestCLI(unittest.TestCase):

    def _run(self, args):
        import io
        from projects.hexalg.hexalg import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ['hexalg.py'] + args
        sys.stdout = io.StringIO()
        try:
            main()
            return sys.stdout.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

    def test_cmd_characters(self):
        out = self._run(['characters'])
        self.assertIn('χ_', out)

    def test_cmd_spectrum(self):
        out = self._run(['spectrum'])
        self.assertIn('λ', out)

    def test_cmd_convolution(self):
        out = self._run(['convolution'])
        self.assertIn('Свёртка', out)

    def test_cmd_cayley_default(self):
        out = self._run(['cayley'])
        self.assertIn('Cay', out)

    def test_cmd_cayley_with_args(self):
        out = self._run(['cayley', '1', '2'])
        self.assertIn('Cay', out)

    def test_cmd_subgroup_default(self):
        out = self._run(['subgroup'])
        self.assertIn('порядок', out)

    def test_cmd_bent(self):
        out = self._run(['bent'])
        self.assertIn('bent', out.lower())

    def test_cmd_help(self):
        out = self._run(['help'])
        self.assertIn('hexalg', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
