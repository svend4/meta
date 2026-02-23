"""hexstat.py — Статистика и теория информации на Q6 = (Z₂)⁶ (64 гексаграммы).

  • Q6Distribution — вероятностное распределение: энтропия, KL, TV, выборка
  • Момент yang_count, маргинальные распределения, корреляции битов
  • RandomWalk — случайное блуждание по Q6 (стационарное = равномерное)
  • Статистические тесты: χ²-тест, KS-тест (проекция на yang_count)
  • Информационные характеристики: взаимная информация, ёмкость BSC
  • Максимально-энтропийное распределение при заданном E[yang]
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


def _log2(x):
    return math.log2(x) if x > 0 else 0.0


def _neighbors(h):
    return [h ^ (1 << i) for i in range(6)]


# ── вероятностное распределение ──────────────────────────────────────────────

class Q6Distribution:
    """
    Вероятностное распределение на Q6 = {0, …, 63}.

    Допускает ненормированный ввод — автоматически нормируется.
    Поддерживает: энтропии, дивергенции, маргинальные распределения,
    характеристическая функция (WHT), выборку через CDF.
    """

    def __init__(self, probs):
        if len(probs) != 64:
            raise ValueError("probs must have length 64")
        total = sum(probs)
        if total <= 0:
            raise ValueError("probs must sum to a positive value")
        self._p = [x / total for x in probs]

    # ── конструкторы ─────────────────────────────────────────────────────────

    @classmethod
    def uniform(cls):
        """Равномерное распределение: P(h) = 1/64."""
        return cls([1.0] * 64)

    @classmethod
    def from_counts(cls, counts):
        """Из абсолютных частот (список 64 чисел ≥ 0)."""
        return cls(list(counts))

    @classmethod
    def from_samples(cls, samples):
        """Эмпирическое распределение из выборки."""
        counts = [0] * 64
        for s in samples:
            counts[int(s) & 63] += 1
        return cls(counts)

    @classmethod
    def yang_weighted(cls, beta):
        """P(h) ∝ exp(β × yang_count(h)) — экспоненциальное семейство."""
        return cls([math.exp(beta * _popcount(h)) for h in range(64)])

    @classmethod
    def binary_symmetric_channel(cls, center, p_error):
        """
        BSC: P(h) = p^{d(h, center)} × (1−p)^{6 − d(h, center)}.
        Распределение ошибок с центром в `center`.
        """
        if not (0.0 <= p_error <= 1.0):
            raise ValueError("p_error must be in [0, 1]")
        probs = []
        for h in range(64):
            d = _popcount(h ^ center)
            probs.append(p_error ** d * (1 - p_error) ** (6 - d))
        return cls(probs)

    @classmethod
    def hamming_shell(cls, center, radius):
        """Равномерное на шаре Хэмминга радиуса `radius` с центром `center`."""
        probs = [1.0 if _popcount(h ^ center) <= radius else 0.0
                 for h in range(64)]
        return cls(probs)

    # ── доступ ───────────────────────────────────────────────────────────────

    def probs(self):
        return list(self._p)

    def __getitem__(self, h):
        return self._p[h]

    # ── энтропии ─────────────────────────────────────────────────────────────

    def entropy(self):
        """Энтропия Шеннона в битах: H = −Σ p log₂ p."""
        return -sum(p * _log2(p) for p in self._p if p > 0)

    def renyi_entropy(self, alpha):
        """Энтропия Реньи порядка alpha (bits)."""
        if alpha == 1.0:
            return self.entropy()
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        s = sum(p ** alpha for p in self._p if p > 0)
        return _log2(s) / (1 - alpha)

    def min_entropy(self):
        """Мин-энтропия = −log₂ max(p)."""
        return -_log2(max(self._p))

    # ── дивергенции и расстояния ─────────────────────────────────────────────

    def kl_divergence(self, other):
        """KL(self ‖ other) = Σ p log₂(p/q)."""
        q = other._p
        total = 0.0
        for h in range(64):
            if self._p[h] > 0:
                if q[h] == 0:
                    return math.inf
                total += self._p[h] * _log2(self._p[h] / q[h])
        return total

    def total_variation(self, other):
        """TV(P, Q) = ½ Σ |P(h) − Q(h)|."""
        return 0.5 * sum(abs(self._p[h] - other._p[h]) for h in range(64))

    def hellinger(self, other):
        """Расстояние Хеллингера = √(½ Σ (√P − √Q)²)."""
        s = sum((math.sqrt(self._p[h]) - math.sqrt(other._p[h])) ** 2
                for h in range(64))
        return math.sqrt(s / 2)

    def js_divergence(self, other):
        """Дивергенция Дженсена–Шеннона (симметричная, ∈ [0, 1])."""
        m = Q6Distribution([(self._p[h] + other._p[h]) / 2 for h in range(64)])
        return 0.5 * (self.kl_divergence(m) + other.kl_divergence(m))

    # ── маргинальные распределения ───────────────────────────────────────────

    def marginal_bit(self, i):
        """P(bit_i = 1)."""
        return sum(self._p[h] for h in range(64) if (h >> i) & 1)

    def marginal_bits(self):
        """Список P(bit_i = 1) для i = 0, …, 5."""
        return [self.marginal_bit(i) for i in range(6)]

    def joint_two_bits(self, i, j):
        """Совместное распределение битов i и j → dict {(b_i, b_j): P}."""
        joint = {}
        for h in range(64):
            key = ((h >> i) & 1, (h >> j) & 1)
            joint[key] = joint.get(key, 0.0) + self._p[h]
        return joint

    def bit_correlation(self, i, j):
        """Cov(bit_i, bit_j) = E[b_i b_j] − E[b_i] E[b_j]."""
        joint = self.joint_two_bits(i, j)
        e_ij = joint.get((1, 1), 0.0)
        return e_ij - self.marginal_bit(i) * self.marginal_bit(j)

    def bit_correlation_matrix(self):
        """6×6 матрица ковариаций битов."""
        return [[self.bit_correlation(i, j) for j in range(6)]
                for i in range(6)]

    # ── моменты yang_count ───────────────────────────────────────────────────

    def mean_yang(self):
        """E[yang_count(h)]."""
        return sum(self._p[h] * _popcount(h) for h in range(64))

    def variance_yang(self):
        """Var[yang_count(h)]."""
        mu = self.mean_yang()
        return sum(self._p[h] * (_popcount(h) - mu) ** 2 for h in range(64))

    def yang_distribution(self):
        """P(yang_count = k) для k = 0, …, 6 (список длины 7)."""
        d = [0.0] * 7
        for h in range(64):
            d[_popcount(h)] += self._p[h]
        return d

    # ── характеристическая функция ───────────────────────────────────────────

    def characteristic_function(self):
        """
        WHT вероятностного вектора: φ(u) = Σ_h P(h) (−1)^{⟨u,h⟩}.
        φ(0) = 1; для равномерного φ(u) = 0 при u ≠ 0.
        """
        W = list(self._p)
        step = 1
        while step < 64:
            for i in range(0, 64, 2 * step):
                for j in range(i, i + step):
                    a, b = W[j], W[j + step]
                    W[j] = a + b
                    W[j + step] = a - b
            step <<= 1
        return W

    # ── выборка ──────────────────────────────────────────────────────────────

    def sample(self, n, seed=None):
        """Сгенерировать n независимых выборок (бинарный поиск по CDF)."""
        rng = random.Random(seed)
        cdf = []
        cum = 0.0
        for p in self._p:
            cum += p
            cdf.append(cum)
        result = []
        for _ in range(n):
            r = rng.random()
            lo, hi = 0, 63
            while lo < hi:
                mid = (lo + hi) // 2
                if cdf[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            result.append(lo)
        return result

    def __repr__(self):
        return (f"Q6Distribution(H={self.entropy():.3f} bits, "
                f"E[yang]={self.mean_yang():.2f})")


# ── случайное блуждание ──────────────────────────────────────────────────────

class RandomWalk:
    """
    Равномерное случайное блуждание по Q6: на каждом шаге выбираем
    один из 6 соседей (флип случайного бита).
    Стационарное распределение = равномерное (Q6 — 6-регулярный).
    """

    def __init__(self, start=0, seed=None):
        self._state = int(start)
        self._rng = random.Random(seed)

    def step(self):
        """Один шаг: перейти к случайному соседу."""
        i = self._rng.randint(0, 5)
        self._state ^= (1 << i)
        return self._state

    def walk(self, n_steps):
        """Путь длины n_steps (включая начальную вершину)."""
        path = [self._state]
        for _ in range(n_steps):
            path.append(self.step())
        return path

    @staticmethod
    def stationary_distribution():
        """Теоретическое стационарное распределение = равномерное."""
        return Q6Distribution.uniform()

    def empirical_distribution(self, n_steps, start=None, seed=None):
        """Эмпирическое распределение после n_steps шагов."""
        if start is not None:
            self._state = int(start)
        if seed is not None:
            self._rng = random.Random(seed)
        counts = [0] * 64
        for _ in range(n_steps):
            counts[self.step()] += 1
        return Q6Distribution.from_counts(counts)

    def cover_time_empirical(self, start=0, seed=None):
        """Время покрытия: число шагов до посещения всех 64 вершин."""
        rng = random.Random(seed)
        visited = {int(start)}
        state = int(start)
        steps = 0
        while len(visited) < 64:
            i = rng.randint(0, 5)
            state ^= (1 << i)
            visited.add(state)
            steps += 1
        return steps

    def mean_cover_time(self, n_trials=20, seed=None):
        """Среднее время покрытия по n_trials испытаниям."""
        rng = random.Random(seed)
        total = 0
        for _ in range(n_trials):
            total += self.cover_time_empirical(seed=rng.randint(0, 2 ** 31))
        return total / n_trials


# ── статистические тесты ────────────────────────────────────────────────────

def chi_square_statistic(counts, expected=None):
    """
    χ² = Σ (O_i − E_i)² / E_i.
    Если expected не задан — проверка на равномерность.
    """
    n = sum(counts)
    if expected is None:
        expected = [n / len(counts)] * len(counts)
    return sum((o - e) ** 2 / e for o, e in zip(counts, expected) if e > 0)


def chi_square_p_value(chi2, df):
    """
    Аппроксимация P(χ²_df > chi2) методом нормального приближения.
    """
    if chi2 <= 0:
        return 1.0
    z = (chi2 - df) / math.sqrt(2 * df)
    # P(Z > z) через аппроксимацию A&S 26.2.17
    az = abs(z)
    if az > 8:
        return 0.0 if z > 0 else 1.0
    t = 1 / (1 + 0.2316419 * az)
    poly = t * (0.319381530 + t * (-0.356563782 +
           t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    phi = math.exp(-az * az / 2) / math.sqrt(2 * math.pi)
    right_tail = phi * poly
    return right_tail if z >= 0 else 1.0 - right_tail


def check_uniformity(samples):
    """
    χ²-тест равномерности выборки из Q6.
    Возвращает (chi2, df, p_value, reject_at_05).
    """
    counts = [0] * 64
    for s in samples:
        counts[int(s) & 63] += 1
    chi2 = chi_square_statistic(counts)
    df = 63
    p = chi_square_p_value(chi2, df)
    return chi2, df, p, (p < 0.05)


def empirical_entropy(samples):
    """Энтропия эмпирического распределения по выборке (bits)."""
    return Q6Distribution.from_samples(samples).entropy()


def bootstrap_entropy_ci(samples, n_bootstrap=200, alpha=0.05, seed=42):
    """
    Доверительный интервал для энтропии методом бутстрэп.
    Возвращает (lower, estimate, upper).
    """
    rng = random.Random(seed)
    n = len(samples)
    estimate = empirical_entropy(samples)
    boot = []
    for _ in range(n_bootstrap):
        boot_sample = [samples[rng.randint(0, n - 1)] for _ in range(n)]
        boot.append(empirical_entropy(boot_sample))
    boot.sort()
    lo = boot[max(0, int(alpha / 2 * n_bootstrap))]
    hi = boot[min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))]
    return lo, estimate, hi


def kolmogorov_smirnov_yang(samples1, samples2):
    """
    KS-тест на yang_count проекции двух выборок.
    Возвращает (ks_statistic, n1, n2).
    """
    def ecdf(samples):
        counts = [0] * 7
        for s in samples:
            counts[_popcount(int(s) & 63)] += 1
        cum, c = [], 0
        for x in counts:
            c += x
            cum.append(c / len(samples))
        return cum

    cdf1 = ecdf(samples1)
    cdf2 = ecdf(samples2)
    ks = max(abs(a - b) for a, b in zip(cdf1, cdf2))
    return ks, len(samples1), len(samples2)


# ── информационные меры ──────────────────────────────────────────────────────

def q6_mutual_information(bit_i, bit_j, dist=None):
    """MI(bit_i; bit_j) = H(bit_i) + H(bit_j) − H(bit_i, bit_j)."""
    if dist is None:
        dist = Q6Distribution.uniform()
    joint = dist.joint_two_bits(bit_i, bit_j)

    def h_bit(p):
        return -sum(v * _log2(v) for v in [p, 1 - p] if v > 0)

    h_i = h_bit(dist.marginal_bit(bit_i))
    h_j = h_bit(dist.marginal_bit(bit_j))
    h_ij = -sum(p * _log2(p) for p in joint.values() if p > 0)
    return h_i + h_j - h_ij


def q6_channel_capacity_bsc(p):
    """Пропускная способность BSC с вероятностью ошибки p (bits/use)."""
    if p == 0 or p == 1:
        return 1.0
    return 1.0 + p * _log2(p) + (1 - p) * _log2(1 - p)


def yang_entropy(k):
    """Энтропия равномерного распределения на yang_count = k: log₂ C(6,k)."""
    from math import comb
    size = comb(6, k)
    return _log2(size)


def max_entropy_with_mean_yang(target_mean, tol=1e-9):
    """
    Максимально-энтропийное распределение при E[yang_count] = target_mean.
    Реализует P(h) ∝ exp(β × yang_count(h)) с подбором β бисекцией.
    """
    if not (0.0 <= target_mean <= 6.0):
        raise ValueError("target_mean must be in [0, 6]")

    def mean_for_beta(beta):
        probs = [math.exp(beta * _popcount(h)) for h in range(64)]
        Z = sum(probs)
        return sum(_popcount(h) * p / Z for h, p in enumerate(probs))

    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mean_for_beta(mid) < target_mean:
            lo = mid
        else:
            hi = mid
    return Q6Distribution.yang_weighted((lo + hi) / 2)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'info':
        dists = {
            'uniform': Q6Distribution.uniform(),
            'yang_weighted(β=0.5)': Q6Distribution.yang_weighted(0.5),
            'yang_weighted(β=-1)': Q6Distribution.yang_weighted(-1.0),
            'BSC(center=0, p=0.1)': Q6Distribution.binary_symmetric_channel(0, 0.1),
            'hamming_ball(r=2)': Q6Distribution.hamming_shell(0, 2),
        }
        for name, d in dists.items():
            print(f"\n{name}:")
            print(f"  H = {d.entropy():.4f} бит  |  E[yang] = {d.mean_yang():.4f}"
                  f"  |  Var[yang] = {d.variance_yang():.4f}"
                  f"  |  H_min = {d.min_entropy():.4f}")

    elif cmd == 'sample':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        dist = Q6Distribution.uniform()
        samples = dist.sample(n, seed=42)
        chi2, df, p, reject = check_uniformity(samples)
        print(f"Выборка n={n} из равномерного распределения:")
        print(f"  Эмп. энтропия:  {empirical_entropy(samples):.4f} бит (теория: 6.0)")
        print(f"  χ²-тест:        χ²={chi2:.2f}, df={df}, p={p:.4f}")
        print(f"  Отвергаем H₀?   {'Да' if reject else 'Нет (OK)'}")

    elif cmd == 'walk':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        rw = RandomWalk(start=0, seed=42)
        emp = rw.empirical_distribution(n, start=0, seed=42)
        unif = Q6Distribution.uniform()
        print(f"Случайное блуждание по Q6, {n} шагов от вершины 0:")
        print(f"  TV от равномерного:    {emp.total_variation(unif):.4f}")
        print(f"  KL от равномерного:    {emp.kl_divergence(unif):.4f}")
        print(f"  Эмп. энтропия:         {emp.entropy():.4f} бит")
        ct = RandomWalk().mean_cover_time(n_trials=20, seed=1)
        print(f"  Среднее время покрытия:{ct:.1f} шагов")

    elif cmd == 'entropy':
        print("Энтропия yang_weighted(β) и E[yang]:")
        for beta in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            d = Q6Distribution.yang_weighted(beta)
            print(f"  β={beta:5.1f}: H={d.entropy():.4f} бит, E[yang]={d.mean_yang():.4f}")

    elif cmd == 'correlation':
        name = sys.argv[2] if len(sys.argv) > 2 else 'uniform'
        if name == 'yang':
            dist = Q6Distribution.yang_weighted(0.5)
        elif name == 'bsc':
            dist = Q6Distribution.binary_symmetric_channel(0, 0.2)
        else:
            dist = Q6Distribution.uniform()
        print(f"Матрица ковариаций битов ({name}):")
        mat = dist.bit_correlation_matrix()
        for row in mat:
            print("  [" + "  ".join(f"{v:+.4f}" for v in row) + "]")

    elif cmd == 'test':
        dist = Q6Distribution.yang_weighted(1.0)
        samples = dist.sample(500, seed=7)
        chi2, df, p, reject = check_uniformity(samples)
        lo, est, hi = bootstrap_entropy_ci(samples, n_bootstrap=100, seed=42)
        print(f"Тест: yang_weighted(β=1), n=500")
        print(f"  Истинная H:            {dist.entropy():.4f} бит")
        print(f"  χ²-тест равномерности: p={p:.4f}, отвергаем? {'Да' if reject else 'Нет'}")
        print(f"  Бутстрэп 95% CI для H: [{lo:.3f}, {est:.3f}, {hi:.3f}]")

    else:
        print("hexstat.py — Статистика и теория информации на Q6")
        print("Команды: info  sample [n]  walk [n]  entropy  correlation [dist]  test")


if __name__ == '__main__':
    main()
