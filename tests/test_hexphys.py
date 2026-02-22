"""Тесты для hexphys — статистическая физика на Q6."""
import math
import unittest

from projects.hexphys import (
    IsingChain, YangGas, MetropolisQ6,
    quantum_state_uniform, hadamard_on_qubit,
    apply_all_hadamard, measure_probabilities,
    entanglement_entropy,
    exact_ising_thermodynamics, compare_exact_and_mcmc,
)
from projects.hexphys.hexphys import ising_spins


# ─────────────────────────────────────────────────────────────────────────────
# helpers

def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


# ─────────────────────────────────────────────────────────────────────────────
class TestIsingSpins(unittest.TestCase):

    def test_all_zero(self):
        """h=0 → все спины −1."""
        self.assertEqual(ising_spins(0), [-1, -1, -1, -1, -1, -1])

    def test_all_one(self):
        """h=63 → все спины +1."""
        self.assertEqual(ising_spins(63), [1, 1, 1, 1, 1, 1])

    def test_single_bit(self):
        """h=1 → σ₀=+1, остальные −1."""
        spins = ising_spins(1)
        self.assertEqual(spins[0], 1)
        self.assertTrue(all(s == -1 for s in spins[1:]))

    def test_spin_values(self):
        """Все спины ∈ {−1, +1}."""
        for h in range(64):
            for s in ising_spins(h):
                self.assertIn(s, [-1, 1])


# ─────────────────────────────────────────────────────────────────────────────
class TestIsingChainEnergy(unittest.TestCase):

    def setUp(self):
        self.chain = IsingChain(J=1.0, B=0.0)

    def test_energy_all_aligned(self):
        """h=0 (все −1) и h=63 (все +1): ферромагнитная конфигурация, E = −6J."""
        # Периодическая: 6 связей, все одинакового знака
        self.assertAlmostEqual(self.chain.energy(0), -6.0)
        self.assertAlmostEqual(self.chain.energy(63), -6.0)

    def test_energy_alternating(self):
        """Чередующиеся спины: h=0b101010=42 или h=0b010101=21 → E = +6J (антиф.)."""
        # σᵢ = −1,+1,−1,+1,−1,+1 → σᵢσᵢ₊₁ = −1 для каждой связи
        self.assertAlmostEqual(self.chain.energy(0b010101), 6.0)
        self.assertAlmostEqual(self.chain.energy(0b101010), 6.0)

    def test_energy_range(self):
        """E ∈ [−6J, +6J] для всех h."""
        for h in range(64):
            e = self.chain.energy(h)
            self.assertGreaterEqual(e, -6.0 - 1e-9)
            self.assertLessEqual(e, 6.0 + 1e-9)

    def test_field_term(self):
        """С полем B: h=63 (все +1) даёт E = −6J − 6B."""
        chain = IsingChain(J=1.0, B=2.0)
        self.assertAlmostEqual(chain.energy(63), -6.0 - 12.0)

    def test_open_chain_bonds(self):
        """Открытая цепочка: 5 связей вместо 6."""
        open_chain = IsingChain(J=1.0, B=0.0, periodic=False)
        # h=0 (все −1): 5 связей × (+1) = 5, E = −5
        self.assertAlmostEqual(open_chain.energy(0), -5.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestIsingChainThermodynamics(unittest.TestCase):

    def setUp(self):
        self.chain = IsingChain(J=1.0, B=0.0)

    def test_partition_function_positive(self):
        """Z(β) > 0 для любого β."""
        for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
            self.assertGreater(self.chain.partition_function(beta), 0)

    def test_partition_function_high_T(self):
        """При β→0 все веса exp(0)=1, Z→64."""
        Z = self.chain.partition_function(1e-6)
        self.assertAlmostEqual(Z, 64.0, places=3)

    def test_free_energy_approaches_ground_state(self):
        """F(β) → E₀ = −6 при β→∞, и F < 0 всегда."""
        # F = -(1/β)lnZ возрастает с ростом β (стремится к -6 снизу)
        betas = [0.1, 0.5, 1.0, 2.0]
        fs = [self.chain.free_energy(b) for b in betas]
        # F возрастает с β
        for i in range(len(fs) - 1):
            self.assertLessEqual(fs[i], fs[i + 1] + 1e-9)
        # F < 0 (Z > 1 всегда)
        for f_val in fs:
            self.assertLess(f_val, 0.0)
        # При больших β F → -6 (минимальная энергия)
        f_large = self.chain.free_energy(20.0)
        self.assertAlmostEqual(f_large, -6.0, delta=0.1)

    def test_mean_energy_at_high_T(self):
        """При β→0 ⟨E⟩ → (1/64) Σ E(h)."""
        expected = sum(self.chain.energy(h) for h in range(64)) / 64.0
        self.assertAlmostEqual(self.chain.mean_energy(1e-6), expected, places=3)

    def test_mean_energy_at_low_T(self):
        """При β→∞ ⟨E⟩ → min E = −6 (фм. основное состояние)."""
        self.assertAlmostEqual(self.chain.mean_energy(20.0), -6.0, places=2)

    def test_free_energy_at_beta_zero(self):
        """F(0) = 0 (предел β→0: F = −(1/β)ln Z → 0)."""
        self.assertEqual(self.chain.free_energy(0), 0.0)

    def test_heat_capacity_nonneg(self):
        """C_v = β² Var[E] ≥ 0."""
        for beta in [0.1, 0.5, 1.0, 2.0]:
            self.assertGreaterEqual(self.chain.heat_capacity(beta), -1e-10)

    def test_heat_capacity_at_beta_zero(self):
        """C_v(0) = 0."""
        self.assertEqual(self.chain.heat_capacity(0), 0.0)

    def test_susceptibility_at_beta_zero(self):
        """χ(0) = 0."""
        self.assertEqual(self.chain.susceptibility(0), 0.0)

    def test_magnetization_zero_at_B0(self):
        """При B=0 ⟨M⟩=0 по симметрии."""
        for beta in [0.5, 1.0, 2.0]:
            self.assertAlmostEqual(self.chain.magnetization(beta), 0.0, places=10)

    def test_magnetization_positive_at_positive_B(self):
        """При B>0 ⟨M⟩>0."""
        chain = IsingChain(J=1.0, B=1.0)
        self.assertGreater(chain.magnetization(1.0), 0.0)

    def test_susceptibility_nonneg(self):
        """χ = β Var[M] ≥ 0."""
        for beta in [0.5, 1.0, 2.0]:
            self.assertGreaterEqual(self.chain.susceptibility(beta), -1e-10)

    def test_transfer_matrix_matches_direct(self):
        """Tr(T^6) ≈ Σ_h exp(−β E(h)) для нескольких β."""
        for beta in [0.3, 1.0, 2.5]:
            Z_direct = self.chain.partition_function(beta)
            Z_transfer = self.chain.exact_Z_transfer(beta)
            self.assertAlmostEqual(Z_direct, Z_transfer, places=6,
                                   msg=f"beta={beta}")

    def test_boltzmann_distribution_sums_to_one(self):
        """Σ_h P(h) = 1."""
        dist = self.chain.boltzmann_distribution(1.0)
        self.assertAlmostEqual(sum(dist), 1.0, places=10)

    def test_boltzmann_distribution_nonneg(self):
        """P(h) ≥ 0 для всех h."""
        dist = self.chain.boltzmann_distribution(1.0)
        for p in dist:
            self.assertGreaterEqual(p, 0.0)

    def test_low_T_concentrated_on_ground(self):
        """При большом β масса P на минимальных энергиях (h=0, h=63)."""
        dist = self.chain.boltzmann_distribution(20.0)
        self.assertGreater(dist[0] + dist[63], 0.99)


# ─────────────────────────────────────────────────────────────────────────────
class TestIsingChainCorrelators(unittest.TestCase):

    def setUp(self):
        self.chain = IsingChain(J=1.0, B=0.0)

    def test_self_correlator(self):
        """⟨σᵢ²⟩ = 1 (спины ±1)."""
        for i in range(6):
            c = self.chain.spin_correlator(1.0, i, i)
            self.assertAlmostEqual(c, 1.0, places=8)

    def test_correlator_bounds(self):
        """−1 ≤ ⟨σᵢσⱼ⟩ ≤ 1."""
        for i in range(6):
            for j in range(6):
                c = self.chain.spin_correlator(1.0, i, j)
                self.assertGreaterEqual(c, -1.0 - 1e-9)
                self.assertLessEqual(c, 1.0 + 1e-9)

    def test_correlator_positive_J(self):
        """При J>0 (ФМ) ⟨σ₀σ₁⟩ > 0."""
        c = self.chain.spin_correlator(1.0, 0, 1)
        self.assertGreater(c, 0.0)

    def test_correlator_negative_J(self):
        """При J<0 (АФМ) ⟨σ₀σ₁⟩ < 0."""
        afm = IsingChain(J=-1.0, B=0.0)
        c = afm.spin_correlator(1.0, 0, 1)
        self.assertLess(c, 0.0)

    def test_correlation_length_positive(self):
        """ξ > 0 при β > 0."""
        xi = self.chain.correlation_length(1.0)
        self.assertGreater(xi, 0.0)

    def test_correlation_length_increases_with_beta(self):
        """ξ возрастает с ростом β (цепь становится более коррелированной)."""
        xi1 = self.chain.correlation_length(0.5)
        xi2 = self.chain.correlation_length(2.0)
        self.assertGreater(xi2, xi1)

    def test_correlation_length_with_field(self):
        """correlation_length при B≠0 вычисляется численно."""
        chain_B = IsingChain(J=1.0, B=1.0)
        xi = chain_B.correlation_length(1.0)
        self.assertGreater(xi, 0.0)

    def test_correlation_length_with_field_small_beta_returns_inf(self):
        """При β→0 и B≠0 коррелятор ≈1, длина = inf."""
        import math
        chain_B = IsingChain(J=1.0, B=1.0)
        xi = chain_B.correlation_length(1e-10)
        self.assertEqual(xi, math.inf)

    def test_correlation_length_zero_J(self):
        """При J=0 корреляционная длина = 0."""
        chain_J0 = IsingChain(J=0.0, B=0.0)
        xi = chain_J0.correlation_length(1.0)
        self.assertEqual(xi, 0.0)

    def test_correlation_length_very_small_beta(self):
        """При β≈0 корреляционная длина = 0 (tanh(β·J)≈0)."""
        xi = self.chain.correlation_length(1e-10)
        self.assertEqual(xi, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestYangGas(unittest.TestCase):

    def test_partition_function_formula(self):
        """Z = (1+z)^6 где z = exp(βμ)."""
        gas = YangGas(mu=1.0)
        for beta in [0.5, 1.0, 2.0]:
            z = math.exp(beta * 1.0)
            expected = (1 + z) ** 6
            self.assertAlmostEqual(gas.partition_function(beta), expected, places=8)

    def test_mean_yang_formula(self):
        """⟨N⟩ = 6z/(1+z)."""
        gas = YangGas(mu=1.0)
        for beta in [0.5, 1.0, 2.0]:
            z = math.exp(beta * 1.0)
            expected = 6 * z / (1 + z)
            self.assertAlmostEqual(gas.mean_yang(beta), expected, places=8)

    def test_mean_yang_symmetric(self):
        """При μ=0: z=1, ⟨N⟩=3 (симметрия)."""
        gas = YangGas(mu=0.0)
        self.assertAlmostEqual(gas.mean_yang(1.0), 3.0, places=8)

    def test_mean_yang_bounds(self):
        """0 < ⟨N⟩ < 6 для конечных β, μ."""
        for mu in [-2.0, 0.0, 2.0]:
            gas = YangGas(mu=mu)
            n = gas.mean_yang(1.0)
            self.assertGreater(n, 0.0)
            self.assertLess(n, 6.0)

    def test_mean_yang_monotone_in_mu(self):
        """⟨N⟩ возрастает с μ."""
        mus = [-1.0, 0.0, 1.0, 2.0]
        ns = [YangGas(mu=mu).mean_yang(1.0) for mu in mus]
        for i in range(len(ns) - 1):
            self.assertLess(ns[i], ns[i + 1])

    def test_yang_variance_positive(self):
        """Var[N] > 0."""
        gas = YangGas(mu=1.0)
        self.assertGreater(gas.yang_variance(1.0), 0.0)

    def test_yang_distribution_sums_to_one(self):
        """Σ P(yang=k) = 1."""
        gas = YangGas(mu=0.5)
        dist = gas.yang_distribution(1.0)
        self.assertEqual(len(dist), 7)
        self.assertAlmostEqual(sum(dist), 1.0, places=10)

    def test_yang_distribution_nonneg(self):
        """P(yang=k) ≥ 0."""
        gas = YangGas(mu=1.0)
        for p in gas.yang_distribution(1.0):
            self.assertGreaterEqual(p, 0.0)

    def test_yang_distribution_mu0_binomial(self):
        """При μ=0 (z=1): P(k) = C(6,k)/64 (равномерное по k, взвешенное биномом)."""
        from math import comb
        gas = YangGas(mu=0.0)
        dist = gas.yang_distribution(1.0)
        for k in range(7):
            expected = comb(6, k) / 64.0
            self.assertAlmostEqual(dist[k], expected, places=8)

    def test_free_energy_less_than_zero_at_positive_mu(self):
        """F < 0 при μ > 0, β > 0 (Z > 1)."""
        gas = YangGas(mu=1.0)
        self.assertLess(gas.free_energy(1.0), 0.0)

    def test_free_energy_at_beta_zero(self):
        """YangGas.free_energy(0) = 0."""
        gas = YangGas(mu=1.0)
        self.assertEqual(gas.free_energy(0), 0.0)

    def test_pressure_at_beta_zero(self):
        """YangGas.pressure(0) = 0."""
        gas = YangGas(mu=1.0)
        self.assertEqual(gas.pressure(0), 0.0)

    def test_compressibility_very_negative_mu(self):
        """При μ → −∞ фугитивность → 0, ⟨N⟩ = 0, сжимаемость → inf."""
        gas = YangGas(mu=-1e300)
        import math
        result = gas.compressibility(1.0)
        self.assertEqual(result, math.inf)

    def test_chemical_potential_for_mean(self):
        """chemical_potential_for_mean находит μ с нужным ⟨N⟩."""
        gas = YangGas(mu=0.0)
        for target in [1.5, 3.0, 4.5]:
            mu_found = gas.chemical_potential_for_mean(1.0, target)
            gas2 = YangGas(mu=mu_found)
            self.assertAlmostEqual(gas2.mean_yang(1.0), target, places=5)


# ─────────────────────────────────────────────────────────────────────────────
class TestMetropolisQ6(unittest.TestCase):

    def setUp(self):
        self.chain = IsingChain(J=1.0, B=0.0)
        self.mc = MetropolisQ6(self.chain.energy, seed=42)

    def test_step_returns_valid_vertex(self):
        """Шаг Метрополиса возвращает h ∈ [0, 63]."""
        current = 0
        for _ in range(100):
            current = self.mc.step(current, beta=1.0)
            self.assertIn(current, range(64))

    def test_step_differs_by_one_bit(self):
        """Принятое предложение отличается ровно на 1 бит (или остаётся тем же)."""
        current = 0
        for _ in range(50):
            nxt = self.mc.step(current, beta=1.0)
            xor = current ^ nxt
            # xor = 0 (отклонение) или степень двойки (принятие)
            self.assertTrue(xor == 0 or (xor & (xor - 1)) == 0)
            current = nxt

    def test_run_length(self):
        """run(n) возвращает n+1 состояний."""
        traj = self.mc.run(200, beta=1.0, start=0)
        self.assertEqual(len(traj), 201)

    def test_run_all_valid(self):
        """Все состояния траектории ∈ [0, 63]."""
        traj = self.mc.run(100, beta=1.0, start=0)
        for h in traj:
            self.assertIn(h, range(64))

    def test_estimate_observable_close_to_exact(self):
        """МСМС-оценка ⟨yang⟩ близка к точной (в пределах 3σ при B=0)."""
        exact = 3.0  # при B=0 симметрия → ⟨yang⟩=3
        mc_val = self.mc.estimate_observable(
            lambda h: _popcount(h),
            n_steps=20000, beta=0.5, n_burn=2000
        )
        self.assertAlmostEqual(mc_val, exact, delta=0.3)

    def test_empirical_distribution_sums_to_one(self):
        """Σ P̂(h) = 1."""
        dist = self.mc.empirical_distribution(n_steps=1000, beta=1.0)
        self.assertAlmostEqual(sum(dist), 1.0, places=8)

    def test_empirical_distribution_nonneg(self):
        """P̂(h) ≥ 0 для всех h."""
        dist = self.mc.empirical_distribution(n_steps=1000, beta=1.0)
        for p in dist:
            self.assertGreaterEqual(p, 0.0)

    def test_low_temp_concentrated_on_minimum(self):
        """При большом β МСМС концентрируется вблизи минимума энергии."""
        mc_val = self.mc.estimate_observable(
            self.chain.energy,
            n_steps=5000, beta=10.0, n_burn=2000
        )
        # min E = −6; при β=10 должно быть очень близко
        self.assertAlmostEqual(mc_val, -6.0, delta=0.5)

    def test_high_temp_uniform(self):
        """При β→0 МСМС близок к равномерному (⟨yang⟩≈3)."""
        mc_val = self.mc.estimate_observable(
            lambda h: _popcount(h),
            n_steps=20000, beta=0.001, n_burn=1000
        )
        self.assertAlmostEqual(mc_val, 3.0, delta=0.3)


# ─────────────────────────────────────────────────────────────────────────────
class TestQuantumOperations(unittest.TestCase):

    def test_uniform_state_norm(self):
        """‖|ψ_uniform⟩‖² = 1."""
        state = quantum_state_uniform()
        norm_sq = sum(a * a for a in state)
        self.assertAlmostEqual(norm_sq, 1.0, places=10)

    def test_uniform_state_length(self):
        """Вектор состояния имеет длину 64."""
        state = quantum_state_uniform()
        self.assertEqual(len(state), 64)

    def test_hadamard_on_qubit_preserves_norm(self):
        """Оператор Адамара на один кубит сохраняет норму."""
        state = quantum_state_uniform()
        for q in range(6):
            new_state = hadamard_on_qubit(state, q)
            norm_sq = sum(a * a for a in new_state)
            self.assertAlmostEqual(norm_sq, 1.0, places=10)

    def test_hadamard_twice_is_identity(self):
        """H² = I."""
        state = quantum_state_uniform()
        for q in range(6):
            state2 = hadamard_on_qubit(hadamard_on_qubit(state, q), q)
            for a, b in zip(state, state2):
                self.assertAlmostEqual(a, b, places=10)

    def test_apply_all_hadamard_norm(self):
        """H⊗6 сохраняет норму."""
        state = quantum_state_uniform()
        new_state = apply_all_hadamard(state)
        norm_sq = sum(a * a for a in new_state)
        self.assertAlmostEqual(norm_sq, 1.0, places=10)

    def test_apply_all_hadamard_on_uniform(self):
        """H⊗6 |uniform⟩ = |0⟩ (т.к. Фурье-образ равномерного — дельта)."""
        state = quantum_state_uniform()
        new_state = apply_all_hadamard(state)
        # |0⟩ должен иметь амплитуду 1
        self.assertAlmostEqual(new_state[0], 1.0, places=8)
        # Остальные ≈ 0
        for h in range(1, 64):
            self.assertAlmostEqual(new_state[h], 0.0, places=8)

    def test_measure_probabilities_sum_to_one(self):
        """Σ P(h) = 1 для равновесного состояния."""
        state = quantum_state_uniform()
        probs = measure_probabilities(state)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_measure_probabilities_uniform(self):
        """Для |ψ_uniform⟩ все P(h) = 1/64."""
        state = quantum_state_uniform()
        probs = measure_probabilities(state)
        for p in probs:
            self.assertAlmostEqual(p, 1.0 / 64, places=10)

    def test_measure_probabilities_nonneg(self):
        """P(h) ≥ 0."""
        state = quantum_state_uniform()
        for p in measure_probabilities(state):
            self.assertGreaterEqual(p, 0.0)

    def test_entanglement_entropy_product_state(self):
        """Произведённое состояние |0⟩ имеет нулевую запутанность."""
        state = [0.0] * 64
        state[0] = 1.0  # |000000⟩
        S = entanglement_entropy(state, [0, 1])
        self.assertAlmostEqual(S, 0.0, places=8)

    def test_entanglement_entropy_maximally_entangled(self):
        """(|000000⟩ + |111111⟩)/√2: запутанность для 1 кубита = 1 бит."""
        state = [0.0] * 64
        state[0] = 1.0 / math.sqrt(2)   # |000000⟩
        state[63] = 1.0 / math.sqrt(2)  # |111111⟩
        S = entanglement_entropy(state, [0])
        self.assertAlmostEqual(S, 1.0, places=5)

    def test_entanglement_entropy_nonneg(self):
        """Запутанность ≥ 0."""
        state = quantum_state_uniform()
        S = entanglement_entropy(state, [0, 1, 2])
        self.assertGreaterEqual(S, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestIsingYangConnection(unittest.TestCase):
    """Проверить связь IsingChain ↔ YangGas при J=0."""

    def test_ising_j0_equals_yanggas(self):
        """
        IsingChain(J=0, B=μ/2): каждый спин независим.
        ⟨yang_count⟩ = Σ P(h) popcount(h) должно совпадать с YangGas(μ).

        При J=0: E(h) = −B·Σσᵢ = −B·(2·popcount(h)−6)
        P(h) ∝ exp(β·B·(2·popcount(h)−6))
        ⟨N⟩ = Σ popcount(h) P(h)

        YangGas(mu): P(N=k) ∝ C(6,k) exp(βμk)
        ⟨N⟩ = 6z/(1+z), z = exp(βμ)

        Связь: μ = 2B (т.к. σ = 2n−1, Σσ = 2N−6)
        """
        B = 1.0
        mu = 2 * B
        beta = 1.5

        chain = IsingChain(J=0.0, B=B)
        gas = YangGas(mu=mu)

        ising_mean = chain.mean_yang(beta)
        yang_mean = gas.mean_yang(beta)

        self.assertAlmostEqual(ising_mean, yang_mean, places=6)


class TestExactIsingThermodynamics(unittest.TestCase):
    """Тесты exact_ising_thermodynamics — точный расчёт термодинамики Изинга."""

    def test_returns_dict(self):
        result = exact_ising_thermodynamics(J=1.0, beta_values=[1.0])
        self.assertIsInstance(result, dict)

    def test_beta_key_in_result(self):
        beta = 1.0
        result = exact_ising_thermodynamics(J=1.0, beta_values=[beta])
        self.assertIn(beta, result)

    def test_expected_keys(self):
        result = exact_ising_thermodynamics(J=1.0, beta_values=[1.0])
        thermo = result[1.0]
        for key in ['Z', 'F', 'E', 'C_v']:
            self.assertIn(key, thermo)

    def test_partition_function_positive(self):
        """Статсумма Z > 0."""
        result = exact_ising_thermodynamics(J=1.0, beta_values=[1.0])
        self.assertGreater(result[1.0]['Z'], 0)

    def test_multiple_beta_values(self):
        betas = [0.5, 1.0, 2.0]
        result = exact_ising_thermodynamics(J=1.0, beta_values=betas)
        self.assertEqual(len(result), 3)

    def test_default_beta_values(self):
        """Без beta_values должны использоваться значения по умолчанию."""
        result = exact_ising_thermodynamics(J=1.0)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)


class TestCompareExactAndMCMC(unittest.TestCase):
    """Тесты compare_exact_and_mcmc — сравнение точных и MCMC значений."""

    def test_returns_dict(self):
        result = compare_exact_and_mcmc(J=1.0, beta=1.0, n_steps=100, seed=42)
        self.assertIsInstance(result, dict)

    def test_expected_keys(self):
        result = compare_exact_and_mcmc(J=1.0, beta=1.0, n_steps=100, seed=42)
        for key in ['exact_E', 'mcmc_E', 'error_E']:
            self.assertIn(key, result)

    def test_error_is_nonnegative(self):
        result = compare_exact_and_mcmc(J=1.0, beta=1.0, n_steps=100, seed=42)
        self.assertGreaterEqual(result['error_E'], 0)

    def test_deterministic_with_seed(self):
        """С одним seed результаты воспроизводимы."""
        r1 = compare_exact_and_mcmc(J=1.0, beta=1.0, n_steps=50, seed=7)
        r2 = compare_exact_and_mcmc(J=1.0, beta=1.0, n_steps=50, seed=7)
        self.assertEqual(r1['mcmc_E'], r2['mcmc_E'])


class TestPhysCLI(unittest.TestCase):
    def _run(self, args):
        import io
        import sys
        from contextlib import redirect_stdout
        from projects.hexphys.hexphys import main
        old_argv = sys.argv
        sys.argv = ['hexphys.py'] + args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_cmd_ising(self):
        out = self._run(['ising', '1.0'])
        self.assertIn('β=', out)

    def test_cmd_yang(self):
        out = self._run(['yang', '1.0'])
        self.assertIn('⟨N⟩', out)

    def test_cmd_mcmc(self):
        out = self._run(['mcmc', '1.0', '1.0'])
        self.assertIn('МСМС', out)

    def test_cmd_quantum(self):
        out = self._run(['quantum'])
        self.assertIn('P(|000000⟩)', out)

    def test_cmd_correlator(self):
        out = self._run(['correlator', '1.0', '1.0'])
        self.assertIn('⟨σ₀σ_', out)

    def test_cmd_help(self):
        out = self._run(['help'])
        self.assertIn('hexphys', out)

    def test_cmd_unknown(self):
        out = self._run(['unknown'])
        self.assertIn('hexphys', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
