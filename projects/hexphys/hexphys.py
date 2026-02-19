"""hexphys.py — Статистическая физика на Q6 = (Z₂)⁶ (64 гексаграммы).

Каждая гексаграмма h = (σ₀,...,σ₅) ∈ {0,1}⁶ — конфигурация 6 спинов:
  σᵢ = 2·bitᵢ − 1 ∈ {−1, +1}   (инь = −1, янь = +1)

Модели:
  • IsingChain(J, B, periodic):
      E(h) = −J Σᵢ σᵢσᵢ₊₁ − B Σᵢ σᵢ
      Z(β) = Σ_h exp(−β E(h))  (ровно 64 слагаемых — все гексаграммы)
      Точное решение через 2×2 трансфер-матрицу
  • YangGas(μ):
      Z(β) = Σ_{k=0}^{6} C(6,k) exp(βμk)  (ян-число = число частиц)
  • MetropolisQ6(energy_fn, seed):
      Алгоритм Метрополиса на Q6: предложение = случайный флип бита
  • QuantumBits: базовые операции над 6-кубитным состоянием (вектор 64 амплитуд)

Соединительные мосты с другими проектами:
  • IsingChain → hexstat.Q6Distribution (напрямую задаёт Больцмановское распределение)
  • YangGas → hexstat.yang_weighted(β·μ)
  • Метрополис → hexopt.SimulatedAnnealing (тот же алгоритм, другой контекст)
  • Трансфер-матрица → 2×2 линейная алгебра
"""
import math
import random

# ── helpers ──────────────────────────────────────────────────────────────────

def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def ising_spins(h):
    """Преобразовать гексаграмму h → 6 спинов σᵢ = 2·bitᵢ − 1 ∈ {−1, +1}."""
    return [2 * ((h >> i) & 1) - 1 for i in range(6)]


def _comb6(k):
    """C(6, k)."""
    from math import comb
    return comb(6, k)


# ── цепочка Изинга ──────────────────────────────────────────────────────────

class IsingChain:
    """
    1D цепочка Изинга на 6 спинах гексаграммы.

    Гамильтониан:
      H(h) = −J Σᵢ σᵢσᵢ₊₁ − B Σᵢ σᵢ
    где σᵢ = 2·bitᵢ(h) − 1 ∈ {−1, +1}.
    При periodic=True: индекс mod 6 (замкнутая цепочка).
    При periodic=False: 5 связей (открытая цепочка).

    Все термодинамические величины вычисляются точно
    суммированием по всем 64 гексаграммам.
    """

    def __init__(self, J=1.0, B=0.0, periodic=True):
        self.J = float(J)
        self.B = float(B)
        self.periodic = bool(periodic)

    def energy(self, h):
        """E(h) = −J Σᵢ σᵢσᵢ₊₁ − B Σᵢ σᵢ."""
        sigma = ising_spins(h)
        n = 6
        n_bonds = n if self.periodic else n - 1
        bond_sum = sum(sigma[i] * sigma[(i + 1) % n] for i in range(n_bonds))
        field_sum = sum(sigma)
        return -self.J * bond_sum - self.B * field_sum

    def _boltzmann_weights(self, beta):
        """Список весов exp(−β E(h)) для всех h ∈ Q6."""
        return [math.exp(-beta * self.energy(h)) for h in range(64)]

    def partition_function(self, beta):
        """Z(β) = Σ_h exp(−β E(h))."""
        return sum(self._boltzmann_weights(beta))

    def free_energy(self, beta):
        """F(β) = −(1/β) ln Z(β)."""
        if beta == 0:
            return 0.0
        return -math.log(self.partition_function(beta)) / beta

    def mean_energy(self, beta):
        """⟨E⟩ = Σ_h E(h) P(h) = −∂ln Z / ∂β (численно)."""
        weights = self._boltzmann_weights(beta)
        Z = sum(weights)
        return sum(self.energy(h) * weights[h] / Z for h in range(64))

    def energy_squared(self, beta):
        """⟨E²⟩ = Σ_h E²(h) P(h)."""
        weights = self._boltzmann_weights(beta)
        Z = sum(weights)
        return sum(self.energy(h) ** 2 * weights[h] / Z for h in range(64))

    def heat_capacity(self, beta):
        """C_v = β² (⟨E²⟩ − ⟨E⟩²) = β² Var[E]."""
        if beta == 0:
            return 0.0
        e1 = self.mean_energy(beta)
        e2 = self.energy_squared(beta)
        return beta ** 2 * (e2 - e1 ** 2)

    def magnetization(self, beta):
        """⟨M⟩ = ⟨(1/6) Σᵢ σᵢ⟩ = ⟨yang_frac⟩ = (⟨yang_count⟩ − 3) / 3."""
        weights = self._boltzmann_weights(beta)
        Z = sum(weights)
        return sum(((_popcount(h) - 3) / 3.0) * weights[h] / Z for h in range(64))

    def mean_yang(self, beta):
        """⟨yang_count⟩ = Σ_h popcount(h) P(h)."""
        weights = self._boltzmann_weights(beta)
        Z = sum(weights)
        return sum(_popcount(h) * weights[h] / Z for h in range(64))

    def susceptibility(self, beta):
        """χ = β (⟨M²⟩ − ⟨M⟩²) = β Var[M]."""
        if beta == 0:
            return 0.0
        weights = self._boltzmann_weights(beta)
        Z = sum(weights)

        def m(h):
            return (_popcount(h) - 3) / 3.0

        m1 = sum(m(h) * weights[h] / Z for h in range(64))
        m2 = sum(m(h) ** 2 * weights[h] / Z for h in range(64))
        return beta * (m2 - m1 ** 2)

    def spin_correlator(self, beta, i, j):
        """⟨σᵢσⱼ⟩ = Σ_h σᵢ(h)σⱼ(h) P(h)."""
        weights = self._boltzmann_weights(beta)
        Z = sum(weights)
        result = 0.0
        for h in range(64):
            sigma = ising_spins(h)
            result += sigma[i] * sigma[j] * weights[h] / Z
        return result

    def correlation_length(self, beta, tol=1e-6):
        """
        Корреляционная длина ξ для открытой цепочки:
        ξ = −1 / ln|⟨σ₀σ₁⟩| (при ⟨σ₀σ₁⟩ ≠ 0 и < 1).
        Для 1D Изинга: ξ = −1 / ln(tanh(βJ)).
        """
        if self.B != 0:
            # Численно
            c = abs(self.spin_correlator(beta, 0, 1))
            if c < tol or c >= 1:
                return math.inf
            return -1.0 / math.log(c)
        # Точная формула (B=0, J>0):
        if abs(self.J) < tol:
            return 0.0
        t = math.tanh(abs(beta * self.J))
        if t < tol:
            return 0.0
        return -1.0 / math.log(t)

    def transfer_matrix(self, beta):
        """
        2×2 трансфер-матрица T[s₁][s₂] = exp(β(J·σ₁σ₂ + (B/2)(σ₁+σ₂))).
        Индексация: s ∈ {0=инь, 1=янь}, σ = 2s−1.
        """
        J, B, b = self.J, self.B, beta
        T = [[0.0] * 2 for _ in range(2)]
        for s1 in range(2):
            for s2 in range(2):
                sg1 = 2 * s1 - 1
                sg2 = 2 * s2 - 1
                T[s1][s2] = math.exp(b * (J * sg1 * sg2 + (B / 2) * (sg1 + sg2)))
        return T

    def _mat_power(self, T, n):
        """Возведение 2×2 матрицы в степень n (рекурсивно)."""
        if n == 1:
            return T
        if n % 2 == 0:
            half = self._mat_power(T, n // 2)
            return self._mat_mul(half, half)
        return self._mat_mul(T, self._mat_power(T, n - 1))

    @staticmethod
    def _mat_mul(A, B):
        return [[A[i][0] * B[0][j] + A[i][1] * B[1][j]
                 for j in range(2)] for i in range(2)]

    def exact_Z_transfer(self, beta):
        """Z = Tr(T^6) через трансфер-матрицу."""
        T = self.transfer_matrix(beta)
        Tn = self._mat_power(T, 6)
        return Tn[0][0] + Tn[1][1]  # Tr(T^6)

    def boltzmann_distribution(self, beta):
        """Больцмановское распределение P(h) = exp(−βE(h)) / Z."""
        weights = self._boltzmann_weights(beta)
        Z = sum(weights)
        return [w / Z for w in weights]


# ── большой канонический ансамбль (ян-газ) ────────────────────────────────────

class YangGas:
    """
    «Ян-газ»: большой канонический ансамбль на Q6.
    Ян-число k = yang_count(h) = число частиц.

    Z(β, μ) = Σ_{k=0}^{6} C(6,k) exp(βμk) = (1 + exp(βμ))^6
    (биномиальный ансамбль — 6 независимых фермионных уровней).
    """

    def __init__(self, mu=1.0):
        self.mu = float(mu)

    def fugacity(self, beta):
        """Активность (fugacity): z = exp(βμ)."""
        return math.exp(beta * self.mu)

    def partition_function(self, beta):
        """Z(β) = (1 + z)^6 где z = exp(βμ)."""
        return (1 + self.fugacity(beta)) ** 6

    def mean_yang(self, beta):
        """⟨N⟩ = 6z/(1+z) = 6/(1+z⁻¹) (среднее ян-число)."""
        z = self.fugacity(beta)
        return 6 * z / (1 + z)

    def yang_variance(self, beta):
        """Var[N] = 6z/(1+z)² (дисперсия ян-числа)."""
        z = self.fugacity(beta)
        return 6 * z / (1 + z) ** 2

    def compressibility(self, beta):
        """Сжимаемость κ = Var[N] / ⟨N⟩² × ⟨N⟩ = Var[N]/⟨N⟩."""
        n = self.mean_yang(beta)
        if n == 0:
            return math.inf
        return self.yang_variance(beta) / n

    def pressure(self, beta):
        """Давление P = (1/β) ln Z = (6/β) ln(1+z)."""
        if beta == 0:
            return 0.0
        z = self.fugacity(beta)
        return math.log(1 + z) * 6 / beta

    def yang_distribution(self, beta):
        """P(yang=k) для k = 0,...,6."""
        from math import comb
        z = self.fugacity(beta)
        Z = (1 + z) ** 6
        return [comb(6, k) * z ** k / Z for k in range(7)]

    def free_energy(self, beta):
        """F = −(1/β) ln Z."""
        if beta == 0:
            return 0.0
        return -math.log(self.partition_function(beta)) / beta

    def entropy(self, beta):
        """S = β²(∂F/∂β) ≈ (E − F) × β."""
        # S = (⟨E⟩ − F) × β где ⟨E⟩ = −μ⟨N⟩
        E_mean = -self.mu * self.mean_yang(beta)
        F = self.free_energy(beta)
        return beta * (E_mean - F)

    def chemical_potential_for_mean(self, beta, target_mean, tol=1e-9):
        """
        Найти μ такое, что ⟨N⟩(β, μ) = target_mean.
        Бисекция по μ ∈ (−∞, +∞).
        """
        if not (0 <= target_mean <= 6):
            raise ValueError("target_mean must be in [0, 6]")
        if target_mean == 3:
            return 0.0
        lo, hi = -50.0, 50.0
        for _ in range(100):
            mid = (lo + hi) / 2
            gas = YangGas(mid)
            m = gas.mean_yang(beta)
            if m < target_mean:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


# ── метод Монте-Карло (Метрополис) на Q6 ─────────────────────────────────────

class MetropolisQ6:
    """
    Алгоритм Метрополиса на Q6.
    Предложение: случайный флип одного бита (переход к соседу Q6).
    Принятие: с вероятностью min(1, exp(−β·ΔE)).
    """

    def __init__(self, energy_fn, seed=42):
        self._E = energy_fn
        self._rng = random.Random(seed)

    def step(self, current, beta):
        """Один шаг Метрополиса: возвращает новое состояние."""
        bit = self._rng.randint(0, 5)
        proposal = current ^ (1 << bit)
        dE = self._E(proposal) - self._E(current)
        if dE <= 0 or self._rng.random() < math.exp(-beta * dE):
            return proposal
        return current

    def run(self, n_steps, beta, start=0):
        """Запустить цепь МСМС: возвращает траекторию из n_steps+1 состояний."""
        trajectory = [start]
        current = start
        for _ in range(n_steps):
            current = self.step(current, beta)
            trajectory.append(current)
        return trajectory

    def estimate_observable(self, obs, n_steps, beta, n_burn=1000, start=0):
        """
        Оценить ⟨obs(h)⟩ по цепи МСМС.
        obs: функция Q6 → R.
        n_burn: шаги прогрева.
        """
        current = start
        for _ in range(n_burn):
            current = self.step(current, beta)
        total, count = 0.0, 0
        for _ in range(n_steps):
            current = self.step(current, beta)
            total += obs(current)
            count += 1
        return total / count

    def estimate_energy(self, energy_fn, n_steps, beta, n_burn=1000):
        """⟨E⟩ по МСМС."""
        return self.estimate_observable(energy_fn, n_steps, beta, n_burn)

    def empirical_distribution(self, n_steps, beta, n_burn=1000, start=0):
        """Эмпирическое распределение P̂(h) по МСМС."""
        current = start
        for _ in range(n_burn):
            current = self.step(current, beta)
        counts = [0] * 64
        for _ in range(n_steps):
            current = self.step(current, beta)
            counts[current] += 1
        total = sum(counts)
        return [c / total for c in counts]


# ── квантовые операции (дискретная структура) ─────────────────────────────────

def quantum_state_uniform():
    """Равновесное квантовое состояние |ψ⟩ = (1/8) Σ_h |h⟩ (длина 64 амплитуд)."""
    amp = 1.0 / 8.0  # 1/sqrt(64)
    return [amp] * 64


def hadamard_on_qubit(state, qubit):
    """
    Применить оператор Адамара к кубиту `qubit` вектора состояния длины 64.
    |0⟩ → (|0⟩ + |1⟩)/√2, |1⟩ → (|0⟩ − |1⟩)/√2.
    """
    new_state = list(state)
    sqrt2_inv = 1.0 / math.sqrt(2)
    mask = 1 << qubit
    for h in range(64):
        if h & mask == 0:
            a = state[h]
            b = state[h | mask]
            new_state[h] = (a + b) * sqrt2_inv
            new_state[h | mask] = (a - b) * sqrt2_inv
    return new_state


def apply_all_hadamard(state):
    """WHT-состояние: применить H⊗6 к каждому кубиту."""
    for i in range(6):
        state = hadamard_on_qubit(state, i)
    return state


def measure_probabilities(state):
    """Вероятности измерения: P(h) = |ψ(h)|²."""
    return [a * a for a in state]


def entanglement_entropy(state, partition_bits):
    """
    Энтропия запутанности для раздела (partition_bits | остальные).
    Вычисляется через сингулярные числа матрицы плотности.
    Используется простая SVD-подобная процедура.
    """
    m = len(partition_bits)
    k = 6 - m
    rows = 2 ** m
    cols = 2 ** k

    # Перестройка ψ в матрицу ψ[a][b], a ∈ {0,1}^m, b ∈ {0,1}^k
    def split(h):
        a_bits = sum(((h >> p) & 1) << i for i, p in enumerate(partition_bits))
        free = [q for q in range(6) if q not in partition_bits]
        b_bits = sum(((h >> p) & 1) << i for i, p in enumerate(free))
        return a_bits, b_bits

    psi = [[0.0] * cols for _ in range(rows)]
    for h in range(64):
        a, b = split(h)
        psi[a][b] = state[h]

    # Частичный след: ρ_A[a1][a2] = Σ_b ψ[a1][b] ψ[a2][b]
    rho = [[sum(psi[a1][b] * psi[a2][b] for b in range(cols))
            for a2 in range(rows)] for a1 in range(rows)]

    # Собственные значения ρ_A (упрощённо: диагональ для декомпонованного состояния)
    # Для общего случая: S = −Σ λ ln(λ)
    # Мы используем trace(ρ ln ρ) ≈ trace(ρ log ρ) via diagonalization
    # Упрощение: для 2×2 используем точную формулу
    if rows == 2:
        tr = rho[0][0] + rho[1][1]
        det = rho[0][0] * rho[1][1] - rho[0][1] * rho[1][0]
        disc = max(0, (tr / 2) ** 2 - det)
        l1 = tr / 2 + math.sqrt(disc)
        l2 = tr / 2 - math.sqrt(disc)
        S = -sum(l * math.log2(l) for l in [l1, l2] if l > 1e-12)
        return S
    # Общий случай: диагональные элементы ρ как приближение
    diag = [rho[i][i] for i in range(rows)]
    norm = sum(diag)
    if norm < 1e-12:
        return 0.0
    diag = [d / norm for d in diag]
    return -sum(d * math.log2(d) for d in diag if d > 1e-12)


# ── утилиты ──────────────────────────────────────────────────────────────────

def exact_ising_thermodynamics(J=1.0, B=0.0, beta_values=None):
    """
    Вычислить термодинамику IsingChain для набора β.
    Возвращает dict {beta: {Z, F, <E>, C_v, <M>, χ}}.
    """
    if beta_values is None:
        beta_values = [0.1 * i for i in range(1, 21)]
    chain = IsingChain(J=J, B=B)
    results = {}
    for beta in beta_values:
        results[beta] = {
            'Z': chain.partition_function(beta),
            'F': chain.free_energy(beta),
            'E': chain.mean_energy(beta),
            'C_v': chain.heat_capacity(beta),
            'M': chain.magnetization(beta),
            'chi': chain.susceptibility(beta),
        }
    return results


def compare_exact_and_mcmc(J=1.0, B=0.0, beta=1.0, n_steps=50000, seed=42):
    """
    Сравнить точные и МСМС-оценки для IsingChain.
    """
    chain = IsingChain(J=J, B=B)
    mc = MetropolisQ6(chain.energy, seed=seed)
    exact_E = chain.mean_energy(beta)
    mc_E = mc.estimate_observable(chain.energy, n_steps, beta)
    exact_M = chain.magnetization(beta)
    mc_M = mc.estimate_observable(lambda h: (_popcount(h) - 3) / 3.0,
                                   n_steps, beta)
    return {
        'exact_E': exact_E, 'mcmc_E': mc_E,
        'exact_M': exact_M, 'mcmc_M': mc_M,
        'error_E': abs(exact_E - mc_E),
        'error_M': abs(exact_M - mc_M),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'ising':
        J = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        print(f"Цепочка Изинга (J={J}, B=0, периодическая):")
        chain = IsingChain(J=J, B=0.0)
        for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
            E = chain.mean_energy(beta)
            Cv = chain.heat_capacity(beta)
            M = chain.magnetization(beta)
            xi = chain.correlation_length(beta)
            print(f"  β={beta:.1f}: ⟨E⟩={E:+.3f}, C_v={Cv:.3f}, "
                  f"⟨M⟩={M:+.4f}, ξ={xi:.2f}")

    elif cmd == 'yang':
        mu = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        print(f"Ян-газ (μ={mu}):")
        gas = YangGas(mu=mu)
        for beta in [0.5, 1.0, 2.0]:
            N = gas.mean_yang(beta)
            var = gas.yang_variance(beta)
            P = gas.pressure(beta)
            print(f"  β={beta}: ⟨N⟩={N:.3f}, Var[N]={var:.3f}, P={P:.3f}")

    elif cmd == 'mcmc':
        J = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        beta = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        result = compare_exact_and_mcmc(J=J, beta=beta)
        print(f"Сравнение точное vs МСМС (J={J}, β={beta}):")
        print(f"  ⟨E⟩: точное={result['exact_E']:.4f}, МСМС={result['mcmc_E']:.4f}, "
              f"ошибка={result['error_E']:.4f}")
        print(f"  ⟨M⟩: точное={result['exact_M']:.4f}, МСМС={result['mcmc_M']:.4f}, "
              f"ошибка={result['error_M']:.4f}")

    elif cmd == 'quantum':
        state = quantum_state_uniform()
        print(f"Равновесное квантовое состояние |ψ⟩ = (1/8) Σ_h |h⟩:")
        print(f"  Σ P(h) = {sum(measure_probabilities(state)):.6f}")
        # Apply H⊗6
        h_state = apply_all_hadamard(list(state))
        probs = measure_probabilities(h_state)
        print(f"  После H⊗6: P(|000000⟩) = {probs[0]:.6f} (должно = 1)")
        # Запутанность для раздела [0,1,2] | [3,4,5]
        bell_state = [0.0] * 64
        bell_state[0] = 1.0 / math.sqrt(2)   # |000000⟩
        bell_state[63] = 1.0 / math.sqrt(2)  # |111111⟩
        S = entanglement_entropy(bell_state, [0, 1, 2])
        print(f"  Запутанность |000000⟩+|111111⟩ для [0-2]|[3-5]: S≈{S:.4f} бит")

    elif cmd == 'correlator':
        J = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        beta = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        chain = IsingChain(J=J, B=0.0)
        print(f"Спин-спиновые корреляторы (J={J}, β={beta}):")
        for d in range(1, 4):
            c = chain.spin_correlator(beta, 0, d)
            print(f"  ⟨σ₀σ_{d}⟩ = {c:.6f}")
        xi = chain.correlation_length(beta)
        print(f"  Корреляционная длина ξ = {xi:.4f}")

    else:
        print("hexphys.py — Статистическая физика на Q6")
        print("Команды: ising [J]  yang [μ]  mcmc [J] [β]  quantum  correlator [J] [β]")


if __name__ == '__main__':
    main()
