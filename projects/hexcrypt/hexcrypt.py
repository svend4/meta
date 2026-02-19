"""hexcrypt.py — Криптографические примитивы на Q6 = (Z₂)⁶ (64 гексаграммы).

Q6 как учебная площадка классической симметричной криптографии:
  • S-блок (биективный, нелинейный): NL, DDT, LAT, SAC, алгебраическая степень
  • Аффинные и случайные подстановки
  • LFSR-подобный потоковый шифр с состоянием ∈ Q6
  • Сеть Фейстеля (разбиение 3|3 бита)
  • Линейный / дифференциальный криптоанализ (академические утилиты)
"""
import math
import random

# ── вспомогательные функции ─────────────────────────────────────────────────

def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def _inner_product(a, b):
    """⟨a, b⟩ = popcount(a & b) mod 2 — скалярное произведение над GF(2)."""
    return _popcount(a & b) & 1


def _mat_mul_vec(rows, v):
    """Умножение матрицы (строки = 6-битные маски) на вектор v (6-битный int)."""
    result = 0
    for i, row in enumerate(rows):
        bit = _inner_product(row, v)
        result |= bit << i
    return result


def _wht_inplace(W):
    """Преобразование Уолша–Адамара in-place (длина W — степень двойки)."""
    n = len(W)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(i, i + step):
                a, b = W[j], W[j + step]
                W[j] = a + b
                W[j + step] = a - b
        step <<= 1


# ── S-блок ──────────────────────────────────────────────────────────────────

class SBox:
    """
    Биективный S-блок f: Q6 → Q6 (подстановка на 64 элементах).

    Основные криптографические параметры:
      nl  — нелинейность (мин. НЛ компонентных функций)
      du  — дифференциальная равномерность (макс. DDT, a ≠ 0)
      deg — алгебраическая степень (макс. по компонентным функциям)
    """

    def __init__(self, table):
        if len(table) != 64 or sorted(table) != list(range(64)):
            raise ValueError("S-box must be a permutation of 0..63")
        self._t = list(table)

    def __call__(self, x):
        return self._t[x]

    def table(self):
        return list(self._t)

    def inverse(self):
        inv = [0] * 64
        for x, y in enumerate(self._t):
            inv[y] = x
        return SBox(inv)

    def component(self, u):
        """Компонентная функция: x ↦ ⟨u, f(x)⟩ mod 2."""
        if u == 0:
            raise ValueError("u must be non-zero")
        return tuple(_inner_product(u, self._t[x]) for x in range(64))

    def _wht_component(self, u):
        """WHT компонентной функции f_u; W[a] = LAT[a][u]."""
        W = [(-1) ** _inner_product(u, self._t[x]) for x in range(64)]
        _wht_inplace(W)
        return W

    def nonlinearity(self):
        """Нелинейность S-блока = min_{u≠0} nl(f_u)."""
        min_nl = 64
        for u in range(1, 64):
            max_abs = max(abs(w) for w in self._wht_component(u))
            nl = (64 - max_abs) // 2
            if nl < min_nl:
                min_nl = nl
        return min_nl

    def difference_distribution_table(self):
        """DDT[a][b] = |{x : f(x⊕a) ⊕ f(x) = b}|."""
        ddt = [[0] * 64 for _ in range(64)]
        for a in range(64):
            for x in range(64):
                b = self._t[x ^ a] ^ self._t[x]
                ddt[a][b] += 1
        return ddt

    def differential_uniformity(self):
        """Макс. значение DDT[a][b] для a ≠ 0."""
        ddt = self.difference_distribution_table()
        return max(ddt[a][b] for a in range(1, 64) for b in range(64))

    def is_almost_perfect_nonlinear(self):
        """APN: дифференциальная равномерность = 2."""
        return self.differential_uniformity() == 2

    def linear_approximation_table(self):
        """LAT[a][b] = Σ_x (−1)^{⟨a,x⟩ ⊕ ⟨b,f(x)⟩}."""
        lat = [[0] * 64 for _ in range(64)]
        for b in range(64):
            W = self._wht_component(b)
            for a in range(64):
                lat[a][b] = W[a]
        return lat

    def best_linear_approximation(self):
        """Макс. |LAT[a][b]| для a ≠ 0, b ≠ 0."""
        best = 0
        for b in range(1, 64):
            W = self._wht_component(b)
            for a in range(1, 64):
                v = abs(W[a])
                if v > best:
                    best = v
        return best

    def algebraic_degree(self):
        """Алгебраическая степень S-блока: max по u≠0 степени f_u."""
        max_deg = 0
        for u in range(1, 64):
            anf = list(self.component(u))
            # Преобразование Мёбиуса (ANF)
            step = 1
            while step < 64:
                for i in range(0, 64, 2 * step):
                    for j in range(i, i + step):
                        anf[j + step] ^= anf[j]
                step <<= 1
            deg = max((_popcount(idx) for idx in range(64) if anf[idx]), default=0)
            if deg > max_deg:
                max_deg = deg
        return max_deg

    def sac_matrix(self):
        """
        SAC-матрица: sac[i][j] = P_x[f(x ⊕ e_i) отличается от f(x) в бите j].
        Идеал: все значения = 0.5.
        """
        sac = [[0.0] * 6 for _ in range(6)]
        for i in range(6):
            ei = 1 << i
            for x in range(64):
                diff = self._t[x] ^ self._t[x ^ ei]
                for j in range(6):
                    if (diff >> j) & 1:
                        sac[i][j] += 1
        for i in range(6):
            for j in range(6):
                sac[i][j] /= 64.0
        return sac

    def satisfies_sac(self, tolerance=0.1):
        """SAC выполнен, если все sac[i][j] ∈ [0.5 − tol, 0.5 + tol]."""
        return all(abs(v - 0.5) <= tolerance
                   for row in self.sac_matrix() for v in row)

    def branch_number(self):
        """Ветвящееся число = min_{a≠0} (hw(a) + hw(f(a) ⊕ f(0)))."""
        f0 = self._t[0]
        return min(_popcount(a) + _popcount(self._t[a] ^ f0) for a in range(1, 64))

    def autocorrelation(self, u=1):
        """AC_u[a] = Σ_x (−1)^{f_u(x) ⊕ f_u(x⊕a)} для компонентной функции u."""
        bits = [(-1) ** _inner_product(u, self._t[x]) for x in range(64)]
        return [sum(bits[x] * bits[x ^ a] for x in range(64)) for a in range(64)]

    def __repr__(self):
        return f"SBox(nl={self.nonlinearity()}, du={self.differential_uniformity()})"


# ── стандартные S-блоки ──────────────────────────────────────────────────────

def identity_sbox():
    """f(x) = x."""
    return SBox(list(range(64)))


def bit_reversal_sbox():
    """Переворот порядка битов: b₅b₄b₃b₂b₁b₀ ↦ b₀b₁b₂b₃b₄b₅."""
    def rev6(x):
        r = 0
        for i in range(6):
            r |= ((x >> i) & 1) << (5 - i)
        return r
    return SBox([rev6(x) for x in range(64)])


def affine_sbox(matrix_rows=None, shift=0):
    """
    Аффинный S-блок f(x) = A·x ⊕ b над GF(2)⁶.
    matrix_rows: 6 шестибитных масок (строки матрицы A).
    shift: 6-битный вектор b.
    """
    if matrix_rows is None:
        # циклический сдвиг битов: бит i → бит (i+1) mod 6
        matrix_rows = [(1 << ((i + 1) % 6)) for i in range(6)]
    table = [_mat_mul_vec(matrix_rows, x) ^ (shift & 63) for x in range(64)]
    return SBox(table)


def complement_sbox():
    """f(x) = x ⊕ 63 (дополнение всех битов)."""
    return SBox([x ^ 63 for x in range(64)])


def random_sbox(seed=42):
    """Случайная перестановка (воспроизводимая)."""
    rng = random.Random(seed)
    perm = list(range(64))
    rng.shuffle(perm)
    return SBox(perm)


def yang_sort_sbox():
    """Перестановка, сортирующая Q6 по (yang_count, value)."""
    groups = sorted(range(64), key=lambda x: (_popcount(x), x))
    return SBox(groups)


# ── потоковый шифр ──────────────────────────────────────────────────────────

class HexStream:
    """
    Потоковый шифр с состоянием ∈ Q6.

    Шаг генератора:
      fb     = popcount(state & poly) mod 2       (линейная обратная связь)
      idx    = popcount(state) mod 6              (индекс флипа)
      state' = sbox(state ⊕ (fb << idx))         (нелинейное обновление)
      bit    = state & 1                          (выходной бит)
    """

    def __init__(self, key, sbox=None, feedback_poly=0b101011):
        if not (0 <= key <= 63):
            raise ValueError("key must be 0..63")
        self._state = int(key)
        self._sbox = sbox if sbox is not None else affine_sbox()
        self._poly = int(feedback_poly) & 63

    def next_bit(self):
        """Следующий бит ключевого потока."""
        out = self._state & 1
        fb = _popcount(self._state & self._poly) & 1
        idx = _popcount(self._state) % 6
        self._state = self._sbox(self._state ^ (fb << idx))
        return out

    def keystream(self, n):
        """Сгенерировать n битов ключевого потока."""
        return [self.next_bit() for _ in range(n)]

    def encrypt(self, bits):
        """Зашифровать список битов (XOR с ключевым потоком)."""
        return [b ^ k for b, k in zip(bits, self.keystream(len(bits)))]

    def decrypt(self, bits):
        """Расшифровать (идентично encrypt, т.к. XOR)."""
        return self.encrypt(bits)


# ── сеть Фейстеля (3 | 3 бита) ──────────────────────────────────────────────

def _split3(x):
    return (x >> 3) & 7, x & 7


def _join3(L, R):
    return ((L & 7) << 3) | (R & 7)


class FeistelCipher:
    """
    Сеть Фейстеля на Q6: блок = 6 бит, разбиение L|R = 3|3.
    Раунд: (L, R) → (R, L ⊕ f(R ⊕ K))  где f — 3-битная S-перестановка.
    """

    def __init__(self, n_rounds=4, round_keys=None, seed=42):
        rng_k = random.Random(seed)
        if round_keys is None:
            round_keys = [rng_k.randint(0, 7) for _ in range(n_rounds)]
        if len(round_keys) != n_rounds:
            raise ValueError("len(round_keys) must equal n_rounds")
        self._rounds = n_rounds
        self._keys = [k & 7 for k in round_keys]
        rng_f = random.Random(seed * 31 + 7)
        perm = list(range(8))
        rng_f.shuffle(perm)
        self._f = perm  # 3-битный S-блок (раундовая функция)

    def _rf(self, R, K):
        return self._f[(R ^ K) & 7]

    def encrypt(self, x):
        L, R = _split3(x)
        for k in self._keys:
            L, R = R, L ^ self._rf(R, k)
        return _join3(L, R)

    def decrypt(self, x):
        L, R = _split3(x)
        for k in reversed(self._keys):
            R, L = L, R ^ self._rf(L, k)
        return _join3(L, R)

    def is_permutation(self):
        outputs = [self.encrypt(x) for x in range(64)]
        return sorted(outputs) == list(range(64))

    def as_sbox(self):
        """Представить шифр как S-блок на Q6."""
        return SBox([self.encrypt(x) for x in range(64)])


# ── анализ и поиск ──────────────────────────────────────────────────────────

def evaluate_sbox(sb):
    """Сводные криптографические свойства S-блока (dict)."""
    nl = sb.nonlinearity()
    du = sb.differential_uniformity()
    deg = sb.algebraic_degree()
    sac = sb.satisfies_sac()
    bn = sb.branch_number()
    return {
        'nonlinearity': nl,
        'differential_uniformity': du,
        'algebraic_degree': deg,
        'satisfies_sac': sac,
        'branch_number': bn,
        'is_apn': du == 2,
    }


def search_good_sbox(min_nl=16, max_du=8, n_trials=200, seed=42):
    """
    Случайный поиск S-блока с nl ≥ min_nl и du ≤ max_du.
    Возвращает (SBox, nl, du) или None.
    """
    rng = random.Random(seed)
    for _ in range(n_trials):
        perm = list(range(64))
        rng.shuffle(perm)
        sb = SBox(perm)
        nl = sb.nonlinearity()
        if nl >= min_nl:
            du = sb.differential_uniformity()
            if du <= max_du:
                return sb, nl, du
    return None


def best_differential_characteristic(sbox):
    """
    Лучшая одно-раундовая дифференциальная характеристика.
    Возвращает (input_diff, output_diff, probability).
    """
    ddt = sbox.difference_distribution_table()
    best = (1, 0, 0.0)
    best_p = 0.0
    for a in range(1, 64):
        for b in range(64):
            p = ddt[a][b] / 64.0
            if p > best_p:
                best_p = p
                best = (a, b, p)
    return best


def best_linear_bias(sbox):
    """
    Лучшая линейная аппроксимация.
    Возвращает (input_mask, output_mask, bias) где bias = |LAT|/64.
    """
    best_bias = 0.0
    best = (1, 1, 0.0)
    for b in range(1, 64):
        W = sbox._wht_component(b)
        for a in range(1, 64):
            bias = abs(W[a]) / 64.0
            if bias > best_bias:
                best_bias = bias
                best = (a, b, bias)
    return best


# ── CLI ─────────────────────────────────────────────────────────────────────

def _get_sbox(name):
    if name == 'identity':   return identity_sbox()
    if name == 'bitrev':     return bit_reversal_sbox()
    if name == 'affine':     return affine_sbox()
    if name == 'complement': return complement_sbox()
    if name == 'random':     return random_sbox()
    if name == 'yangsort':   return yang_sort_sbox()
    if name == 'feistel':    return FeistelCipher().as_sbox()
    raise ValueError(f"Неизвестный S-блок: {name}")


def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'info':
        name = sys.argv[2] if len(sys.argv) > 2 else 'affine'
        sb = _get_sbox(name)
        ev = evaluate_sbox(sb)
        print(f"S-блок: {name}")
        for k, v in ev.items():
            print(f"  {k}: {v}")
        a, b, p = best_differential_characteristic(sb)
        print(f"  best_differential: Δin={a:06b} → Δout={b:06b}, prob={p:.4f}")
        a2, b2, bias = best_linear_bias(sb)
        print(f"  best_linear_bias:  α={a2:06b}, β={b2:06b}, bias={bias:.4f}")

    elif cmd == 'table':
        name = sys.argv[2] if len(sys.argv) > 2 else 'affine'
        sb = _get_sbox(name)
        print(f"S-блок '{name}' (8×8):")
        for row in range(8):
            vals = [f"{sb(row * 8 + col):2d}" for col in range(8)]
            print("  " + " ".join(vals))

    elif cmd == 'stream':
        key = int(sys.argv[2]) & 63 if len(sys.argv) > 2 else 7
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
        ks = HexStream(key).keystream(n)
        print(f"Ключ: {key}, поток ({n} бит): " + "".join(map(str, ks)))

    elif cmd == 'feistel':
        action = sys.argv[2] if len(sys.argv) > 2 else 'demo'
        fc = FeistelCipher(n_rounds=4)
        if action == 'perm':
            print(f"Является перестановкой: {fc.is_permutation()}")
        else:
            print("Шифр Фейстеля — первые 8 значений:")
            for x in range(8):
                enc = fc.encrypt(x)
                dec = fc.decrypt(enc)
                print(f"  {x:06b} → {enc:06b} → {dec:06b}  {'OK' if dec==x else 'ERR'}")

    elif cmd == 'search':
        min_nl = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        print(f"Поиск S-блока с nl ≥ {min_nl} среди {n} случайных перестановок...")
        result = search_good_sbox(min_nl=min_nl, n_trials=n)
        if result:
            sb, nl, du = result
            print(f"  Найден: nl={nl}, du={du}, deg={sb.algebraic_degree()}")
        else:
            print("  Не найден.")

    elif cmd == 'sac':
        name = sys.argv[2] if len(sys.argv) > 2 else 'random'
        sb = _get_sbox(name)
        sac = sb.sac_matrix()
        print(f"SAC-матрица ({name}):")
        for i in range(6):
            print("  [" + "  ".join(f"{sac[i][j]:.2f}" for j in range(6)) + "]")
        print(f"SAC выполнен (±0.1): {sb.satisfies_sac()}")

    else:
        print("hexcrypt.py — Криптографические примитивы на Q6")
        print("Команды: info [sbox]  table [sbox]  stream <key> [n]")
        print("         feistel [demo|perm]  search [min_nl] [n]  sac [sbox]")
        print("S-блоки: identity  bitrev  affine  complement  random  yangsort  feistel")


if __name__ == '__main__':
    main()
