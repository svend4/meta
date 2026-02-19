"""hexalg.py — Гармонический анализ на Q6 = (Z₂)⁶ (64 гексаграммы).

Q6 как абелева группа (Z₂)⁶ с операцией XOR:
  • Характеры: χ_u(h) = (−1)^{⟨u,h⟩}  — 64 ортогональных характера
  • Преобразование Фурье = WHT: F(f)[u] = Σ_h f(h)(−1)^{⟨u,h⟩}
  • Свёртка: (f*g)(h) = Σ_a f(a)g(a⊕h)
  • Теорема о свёртке: F(f*g) = F(f)·F(g) (поточечно)
  • Тождество Парсеваля: Σ|f(h)|² = (1/64)Σ|F(f)[u]|²
  • Граф Кэли: Cay(Q6, S) с собственными значениями через WHT
  • Спектр гиперкуба: λ_k = 6−2k, кратность C(6,k)
  • Смежные классы, подгруппы, аннулятор (двойственная подгруппа)
  • Разностные множества, bent-функции как «плоский» спектр
  • Групповое кольцо F₂[Q6]: свёртка и произведение над GF(2)
"""
import math

# ── helpers ──────────────────────────────────────────────────────────────────

def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def _inner_product(a, b):
    """⟨a, b⟩ = popcount(a & b) mod 2."""
    return _popcount(a & b) & 1


def _wht_inplace(W):
    """WHT in-place (длина = степень 2)."""
    n = len(W)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(i, i + step):
                a, b = W[j], W[j + step]
                W[j] = a + b
                W[j + step] = a - b
        step <<= 1


# ── характеры Q6 ─────────────────────────────────────────────────────────────

def character(u, h):
    """Характер χ_u(h) = (−1)^{⟨u,h⟩} ∈ {−1, +1}."""
    return (-1) ** _inner_product(u, h)


def all_characters():
    """Все 64 характера Q6: таблица [u][h] = (−1)^{⟨u,h⟩}."""
    return [[character(u, h) for h in range(64)] for u in range(64)]


def hadamard_matrix():
    """64×64 матрица Адамара: H[u][h] = (−1)^{⟨u,h⟩}."""
    return all_characters()


# ── преобразование Фурье (WHT) ────────────────────────────────────────────────

def fourier_transform(f):
    """
    Дискретное преобразование Фурье на Q6 (= WHT).
    F(f)[u] = Σ_h f(h) (−1)^{⟨u,h⟩}.
    Принимает list из 64 чисел.
    """
    W = list(f)
    _wht_inplace(W)
    return W


def inverse_fourier_transform(F):
    """
    Обратное преобразование Фурье: f(h) = (1/64) Σ_u F[u] (−1)^{⟨u,h⟩}.
    Принимает список из 64 чисел (спектр).
    """
    W = list(F)
    _wht_inplace(W)
    return [w / 64.0 for w in W]


def verify_inverse(f, tol=1e-9):
    """Проверить IFT(FT(f)) ≈ f."""
    return all(abs(a - b) < tol
               for a, b in zip(f, inverse_fourier_transform(fourier_transform(f))))


# ── свёртка ──────────────────────────────────────────────────────────────────

def convolve(f, g):
    """
    Свёртка на Q6: (f*g)(h) = Σ_a f(a) g(a ⊕ h).
    Эквивалентна IFT(FT(f) · FT(g)) (теорема о свёртке).
    """
    result = [0.0] * 64
    for a in range(64):
        if f[a] == 0:
            continue
        for h in range(64):
            result[h] += f[a] * g[a ^ h]
    return result


def convolve_via_fft(f, g):
    """Свёртка через WHT: F⁻¹(F(f) · F(g))."""
    Ff = fourier_transform(f)
    Fg = fourier_transform(g)
    product = [x * y for x, y in zip(Ff, Fg)]
    return inverse_fourier_transform(product)


def correlate(f, g):
    """Кросс-корреляция: (f ⊛ g)(h) = Σ_a f(a) g(a ⊕ h)."""
    # В (Z₂)^n инверсия = тождество (a = -a), поэтому корреляция = свёртка
    return convolve(f, g)


def autocorrelation(f):
    """Автокорреляция: AC_f(h) = Σ_a f(a) f(a ⊕ h)."""
    return convolve(f, f)


def convolution_theorem_check(f, g, tol=1e-9):
    """Проверить F(f*g) = F(f)·F(g) поточечно."""
    fg = convolve(f, g)
    Ffg = fourier_transform(fg)
    Ff = fourier_transform(f)
    Fg = fourier_transform(g)
    for u in range(64):
        if abs(Ffg[u] - Ff[u] * Fg[u]) > tol:
            return False
    return True


# ── тождество Парсеваля ───────────────────────────────────────────────────────

def inner_product_spatial(f, g):
    """⟨f, g⟩_Q6 = Σ_h f(h) g(h)."""
    return sum(f[h] * g[h] for h in range(64))


def inner_product_frequency(Ff, Fg):
    """⟨F(f), F(g)⟩ / 64 = Σ_u F(f)[u] F(g)[u] / 64."""
    return sum(Ff[u] * Fg[u] for u in range(64)) / 64.0


def parseval_identity(f, tol=1e-9):
    """Проверить Σ|f|² = (1/64)Σ|F(f)|²."""
    lhs = sum(x * x for x in f)
    Ff = fourier_transform(f)
    rhs = sum(x * x for x in Ff) / 64.0
    return abs(lhs - rhs) < tol


# ── граф Кэли ────────────────────────────────────────────────────────────────

def cayley_graph(connection_set):
    """
    Рёбра графа Кэли Cay(Q6, S): множество пар {h, h⊕s} для h∈Q6, s∈S.
    S должно быть замкнуто относительно инверсии (в (Z₂)⁶ s = s⁻¹ всегда).
    Возвращает frozenset пар (a, b) с a < b.
    """
    edges = set()
    S = [s for s in connection_set if s != 0]
    for h in range(64):
        for s in S:
            n = h ^ s
            edges.add((min(h, n), max(h, n)))
    return frozenset(edges)


def cayley_eigenvalues(connection_set):
    """
    Собственные значения графа Кэли Cay(Q6, S):
    λ_u = Σ_{s∈S} χ_u(s) = Σ_{s∈S} (−1)^{⟨u,s⟩}.
    Возвращает list λ[u] для u = 0..63.
    """
    S = list(connection_set)
    return [sum(character(u, s) for s in S) for u in range(64)]


def hypercube_spectrum():
    """
    Спектр гиперкуба Q6 (граф Кэли с S = {e₀,...,e₅}):
    λ_k = 6 − 2k, кратность C(6, k) для k = yang_count(u).
    Возвращает dict {eigenvalue: multiplicity}.
    """
    from math import comb
    spec = {}
    for k in range(7):
        ev = 6 - 2 * k
        spec[ev] = comb(6, k)
    return spec


def cayley_is_connected(connection_set):
    """
    Граф Кэли связен ↔ S порождает Q6.
    Проверяется BFS из 0.
    """
    S = [s for s in connection_set if s != 0]
    visited = {0}
    queue = [0]
    while queue:
        h = queue.pop()
        for s in S:
            n = h ^ s
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return len(visited) == 64


# ── подгруппы и смежные классы ───────────────────────────────────────────────

def subgroup_generated(generators):
    """
    Подгруппа ⟨generators⟩ ≤ Q6, порождённая подмножеством generators.
    В (Z₂)⁶: подгруппа = линейное подпространство над GF(2).
    """
    subgroup = {0}
    queue = [0]
    gen_list = [g for g in generators if g != 0]
    while queue:
        h = queue.pop()
        for g in gen_list:
            n = h ^ g
            if n not in subgroup:
                subgroup.add(n)
                queue.append(n)
    return frozenset(subgroup)


def coset_decomposition(subgroup):
    """
    Разложение Q6 на смежные классы по H:
    Q6 / H = {h ⊕ H : h ∈ Q6}.
    Возвращает list frozenset'ов.
    """
    H = frozenset(subgroup)
    representatives = set()
    cosets = []
    for h in range(64):
        if h not in representatives:
            coset = frozenset(h ^ g for g in H)
            cosets.append(coset)
            representatives |= coset
    return cosets


def dual_subgroup(subgroup):
    """
    Аннулятор H⊥ = {u ∈ Q6 : ⟨u, h⟩ = 0 для всех h ∈ H}.
    В (Z₂)⁶: H⊥ — ортогональное дополнение по GF(2).
    |H| × |H⊥| = 64.
    """
    H = list(subgroup)
    perp = frozenset(u for u in range(64)
                     if all(_inner_product(u, h) == 0 for h in H))
    return perp


def index_of_subgroup(subgroup):
    """Индекс [Q6 : H] = |Q6| / |H| = 64 / |H|."""
    return 64 // len(subgroup)


# ── Понтрягинова двойственность ────────────────────────────────────────────────

def pontryagin_dual_character(h):
    """
    Элемент h ∈ Q6 задаёт характер χ_h ∈ Q̂6: u ↦ (−1)^{⟨h,u⟩}.
    Q6 самодвойственна: Q6 ≅ Q̂6.
    """
    return [character(h, u) for u in range(64)]


# ── разностные множества ──────────────────────────────────────────────────────

def difference_multiset(subset):
    """
    Разностный мультимножество D(A) = {a ⊕ b : a, b ∈ A, a ≠ b}.
    Возвращает dict {diff: count}.
    """
    A = list(subset)
    counts = {}
    for a in A:
        for b in A:
            if a != b:
                d = a ^ b
                counts[d] = counts.get(d, 0) + 1
    return counts


def is_difference_set(subset):
    """
    Проверить, является ли subset (v, k, λ)-разностным множеством в Q6:
    каждый ненулевой элемент Q6 встречается в D(A) одинаковое число раз λ.
    v=64, k=|subset|, λ=k(k-1)/63.
    """
    A = list(subset)
    k = len(A)
    dm = difference_multiset(A)
    if len(dm) != 63:
        return False
    counts = list(dm.values())
    return len(set(counts)) == 1 and counts[0] * 63 == k * (k - 1)


def is_bent_function(f_table):
    """
    Булева функция f: Q6 → {0,1} является bent ↔ |WHT(f)[u]| = 8 для всех u.
    Нелинейность = (64 − 8)/2 = 28 (максимальная для n=6).
    """
    f_pm = [(-1) ** b for b in f_table]  # {0,1} → {+1, −1}
    W = fourier_transform(f_pm)
    return all(abs(abs(w) - 8) < 1e-9 for w in W)


def bent_function_difference_set(f_table):
    """
    Bent-функция f задаёт разностное множество:
    A = {h : f(h) = 1} является (64, 32, 16)-разностным множеством.
    """
    A = frozenset(h for h in range(64) if f_table[h])
    return A, is_difference_set(A)


# ── групповое кольцо F₂[Q6] ──────────────────────────────────────────────────

def group_ring_add_f2(f, g):
    """Сложение в F₂[Q6]: поточечный XOR коэффициентов."""
    return [f[h] ^ g[h] for h in range(64)]


def group_ring_mul_f2(f, g):
    """
    Умножение в F₂[Q6]: (f·g)(h) = Σ_{a⊕b=h} f(a)g(b) mod 2.
    Это свёртка над GF(2).
    """
    result = [0] * 64
    for a in range(64):
        if not f[a]:
            continue
        for b in range(64):
            if g[b]:
                result[a ^ b] ^= 1
    return result


def group_ring_support(f):
    """Носитель элемента f ∈ F₂[Q6]: {h : f[h] = 1}."""
    return frozenset(h for h in range(64) if f[h])


def indicator(subset):
    """Индикаторная функция подмножества → элемент F₂[Q6]."""
    return [1 if h in set(subset) else 0 for h in range(64)]


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'characters':
        print("Первые 5 характеров Q6 (строки матрицы Адамара):")
        for u in range(5):
            row = [character(u, h) for h in range(8)]
            print(f"  χ_{u}: {row}...")

    elif cmd == 'spectrum':
        print("Спектр гиперкуба Q6 (граф Кэли, S = все ребра):")
        S = [1 << i for i in range(6)]
        evs = cayley_eigenvalues(S)
        spec = {}
        for v in evs:
            spec[v] = spec.get(v, 0) + 1
        for ev in sorted(spec.keys(), reverse=True):
            print(f"  λ = {ev:+3d}, кратность = {spec[ev]}")

    elif cmd == 'convolution':
        # Пример: свёртка двух индикаторных функций
        f = [1 if h == 0 else 0 for h in range(64)]  # δ_0
        g = [1 if _popcount(h) <= 1 else 0 for h in range(64)]  # B(0,1)
        fg = convolve(f, g)
        print("Свёртка δ_0 * 1_{B(0,1)}: первые 8 значений =", fg[:8])
        ok = convolution_theorem_check(f, g)
        print(f"Теорема о свёртке F(f*g)=F(f)·F(g): {ok}")
        print(f"Тождество Парсеваля для f: {parseval_identity(f)}")

    elif cmd == 'cayley':
        S = sys.argv[2:] if len(sys.argv) > 2 else None
        if S is None:
            S = [1, 2, 4]  # биты 0,1,2
        else:
            S = [int(x) for x in S]
        edges = cayley_graph(S)
        evs = cayley_eigenvalues(S)
        print(f"Cay(Q6, S={S}):")
        print(f"  Рёбра: {len(edges)}")
        print(f"  Связен: {cayley_is_connected(S)}")
        spec = {}
        for v in evs:
            spec[v] = spec.get(v, 0) + 1
        for ev in sorted(spec.keys(), reverse=True):
            print(f"  λ = {ev:+3d}, кратность = {spec[ev]}")

    elif cmd == 'subgroup':
        import ast
        gen_str = sys.argv[2] if len(sys.argv) > 2 else '[1, 2, 4]'
        gens = ast.literal_eval(gen_str)
        H = subgroup_generated(gens)
        perp = dual_subgroup(H)
        cosets = coset_decomposition(H)
        print(f"⟨{gens}⟩ имеет порядок {len(H)}")
        print(f"  Аннулятор H⊥: {len(perp)} элементов")
        print(f"  Индекс [Q6:H] = {len(cosets)} (смежных классов)")
        print(f"  |H|×|H⊥| = {len(H)*len(perp)} (должно быть 64)")

    elif cmd == 'bent':
        # Внутреннее произведение (классическая bent-функция)
        f_table = [_popcount(((h & 7) << 3) ^ ((h >> 3) & 7)) % 2
                   for h in range(64)]
        is_bent = is_bent_function(f_table)
        A, is_ds = bent_function_difference_set(f_table)
        print(f"Тест bent-функции (inner product on Q3×Q3):")
        print(f"  Is bent: {is_bent}")
        print(f"  |A| = {len(A)}, is_difference_set: {is_ds}")

    else:
        print("hexalg.py — Гармонический анализ на Q6")
        print("Команды: characters  spectrum  convolution  cayley [S...]  subgroup [gens]  bent")


if __name__ == '__main__':
    main()
