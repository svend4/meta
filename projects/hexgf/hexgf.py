"""hexgf.py — Поле Галуа GF(2^6) на 64 элементах гексаграмм Q6.

GF(2^6) = GF(2)[x] / p(x),  p(x) = x^6 + x + 1  (примитивный неприводимый)

Каждый элемент ≡ 6-битное число 0..63:
  bit k = коэффициент x^k в многочлене степени ≤ 5

Операции:
  • Сложение:   gf_add(a, b) = a ⊕ b  (совпадает с операцией группы Q6!)
  • Умножение:  gf_mul(a, b) — беспереносное умножение + редукция mod p(x)
  • Обратный:   gf_inv(a) = gf_pow(a, 61) = a^{q−2} (теорема Ферма)
  • Примитивный элемент: g = 2 (= x), ord(g) = 63 = 2^6 − 1

Дополнительные структуры:
  • Таблицы exp/log: gf_exp[k] = g^k, gf_log[a] = k
  • Абсолютный след: Tr(a) = a ⊕ a^2 ⊕ a^4 ⊕ a^8 ⊕ a^16 ⊕ a^32 ∈ {0,1}
  • Циклотомические классы: {a, a^2, a^4,...} (задают минимальные многочлены)
  • Минимальные многочлены элементов над GF(2)
  • Подполя: GF(2^1) ≤ GF(2^2) ≤ GF(2^3) ≤ GF(2^6)
  • Логарифмы Цеха: Z[k] = log(1 + g^k)
  • Аддитивные характеры: ψ_b(a) = (−1)^{Tr(b·a)} ∈ {+1,−1}

Связь с другими проектами:
  • hexalg:  аддитивные характеры ψ_b = WHT-характеры χ_b (идентичны!)
  • hexcode: BCH-коды через нули в GF(2^6); коды Рида–Соломона
  • hexring: минимальные многочлены → неприводимые факторы над GF(2)
  • hexcrypt: MDS-коды и S-блоки через поле Галуа
"""

# ── примитивный многочлен ─────────────────────────────────────────────────────

# p(x) = x^6 + x + 1 = 0b1000011 = 67
# Редукция: x^6 ≡ x + 1, т.е. при появлении бита 6 XOR с 67 (убираем x^6, добавляем x+1)
POLY = 0x43        # = 67 = 1000011₂ (полный многочлен включая x^6)
POLY_REDUCE = 0x03 # = 3 = 000011₂ (только x+1, без x^6 — для XOR при редукции)
ORDER = 63         # |GF(2^6)*| = 2^6 − 1
SIZE = 64          # |GF(2^6)| = 2^6

# Примитивный элемент g = x = 2 (порядок 63 в GF(2^6) с p(x)=x^6+x+1)
PRIMITIVE = 2


# ── базовые операции ──────────────────────────────────────────────────────────

def gf_add(a, b):
    """Сложение в GF(2^6): a + b = a ⊕ b (XOR)."""
    return a ^ b


def gf_sub(a, b):
    """Вычитание = сложение в характеристике 2: a − b = a ⊕ b."""
    return a ^ b


def gf_mul(a, b):
    """
    Умножение в GF(2^6) = GF(2)[x] / (x^6 + x + 1).
    Алгоритм: беспереносное умножение + редукция.
    При появлении x^6: заменяем на x+1 (т.к. x^6 = x+1 mod p(x)).
    """
    if a == 0 or b == 0:
        return 0
    result = 0
    aa = a
    while b:
        if b & 1:
            result ^= aa
        b >>= 1
        aa <<= 1
        if aa & 64:          # появился x^6
            aa ^= POLY       # XOR с 67: убирает бит 6, добавляет биты 0,1 (= x+1)
    return result


def gf_pow(a, n):
    """Возведение в степень в GF(2^6): a^n (бинарное возведение)."""
    if n == 0:
        return 1
    if a == 0:
        return 0
    n = n % ORDER   # a^63 = 1 для a ≠ 0
    if n == 0:
        return 1
    result = 1
    base = a
    while n:
        if n & 1:
            result = gf_mul(result, base)
        base = gf_mul(base, base)
        n >>= 1
    return result


def gf_inv(a):
    """Обратный элемент: a^{−1} = a^{61} = a^{q−2} (следствие теоремы Ферма)."""
    if a == 0:
        raise ZeroDivisionError("Обратный к 0 не существует в GF(2^6)")
    return gf_pow(a, ORDER - 1)


def gf_div(a, b):
    """Деление: a / b = a · b^{−1}."""
    return gf_mul(a, gf_inv(b))


# ── таблицы степеней и логарифмов ─────────────────────────────────────────────

def build_exp_log_tables():
    """
    Построить таблицы:
      exp_table[k] = g^k ∈ GF(2^6)*,  k = 0..62
      log_table[a] = k если g^k = a;   log_table[0] = -1 (условно)
    """
    exp_table = [0] * ORDER
    log_table = [-1] * SIZE
    a = 1
    for k in range(ORDER):
        exp_table[k] = a
        log_table[a] = k
        a = gf_mul(a, PRIMITIVE)
    return exp_table, log_table


# Глобальные таблицы (инициализируются один раз)
_EXP_TABLE = None
_LOG_TABLE = None


def _get_tables():
    global _EXP_TABLE, _LOG_TABLE
    if _EXP_TABLE is None:
        _EXP_TABLE, _LOG_TABLE = build_exp_log_tables()
    return _EXP_TABLE, _LOG_TABLE


def gf_exp(k):
    """g^k: примитивный элемент в степени k (k по модулю 63)."""
    exp, _ = _get_tables()
    return exp[k % ORDER]


def gf_log(a):
    """Дискретный логарифм: log_g(a) для a ≠ 0."""
    if a == 0:
        raise ValueError("log(0) не определён")
    _, log = _get_tables()
    return log[a]


def gf_mul_via_log(a, b):
    """Умножение через таблицы log/exp: a·b = g^{log(a)+log(b)}."""
    if a == 0 or b == 0:
        return 0
    _, log = _get_tables()
    exp, _ = _get_tables()
    return exp[(log[a] + log[b]) % ORDER]


# ── след и норма ───────────────────────────────────────────────────────────────

def gf_trace(a):
    """
    Абсолютный след Tr_{GF(2^6)/GF(2)}(a) = a ⊕ a^2 ⊕ a^4 ⊕ a^8 ⊕ a^16 ⊕ a^32 ∈ {0,1}.
    Tr — GF(2)-линейное отображение GF(2^6) → GF(2).
    Tr(a+b) = Tr(a) + Tr(b),  Tr(ca) = c·Tr(a) для c ∈ GF(2).
    Tr(a^2) = Tr(a) (стабильность относительно Фробениуса).
    """
    result = a
    sq = gf_mul(a, a)       # a^2
    result ^= sq
    sq = gf_mul(sq, sq)     # a^4
    result ^= sq
    sq = gf_mul(sq, sq)     # a^8
    result ^= sq
    sq = gf_mul(sq, sq)     # a^16
    result ^= sq
    sq = gf_mul(sq, sq)     # a^32
    result ^= sq
    return result & 1       # ∈ {0, 1}


def gf_norm(a):
    """
    Норма N_{GF(2^6)/GF(2)}(a) = a^{1+2+4+8+16+32} = a^63.
    N(0) = 0;  N(a) = 1 для всех a ≠ 0 (т.к. a^63 = 1 в GF(2^6)*).
    """
    if a == 0:
        return 0
    return gf_pow(a, ORDER)  # = 1 для всех a ≠ 0


def trace_bilinear(a, b):
    """
    Трейс-билинейная форма: (a, b) ↦ Tr(a · b) ∈ {0, 1}.
    Невырожденная симметричная билинейная форма GF(2^6) × GF(2^6) → GF(2).
    """
    return gf_trace(gf_mul(a, b))


# ── примитивность ─────────────────────────────────────────────────────────────

def element_order(a):
    """Порядок элемента a ∈ GF(2^6)*. Делитель 63."""
    if a == 0:
        raise ValueError("Порядок 0 не определён")
    if a == 1:
        return 1
    _, log = _get_tables()
    k = log[a]
    # ord(a) = ord(g^k) = 63 / gcd(k, 63)
    from math import gcd
    return ORDER // gcd(k, ORDER)


def is_primitive(a):
    """Является ли a примитивным элементом GF(2^6)* (ord = 63)?"""
    if a == 0 or a == 1:
        return False
    return element_order(a) == ORDER


def primitive_elements():
    """Все примитивные элементы GF(2^6)* (генераторы мультипликативной группы)."""
    return [a for a in range(1, SIZE) if is_primitive(a)]


def count_primitive():
    """Число примитивных элементов = φ(63) = φ(9)·φ(7) = 6·6 = 36."""
    from math import gcd
    return sum(1 for k in range(1, ORDER) if gcd(k, ORDER) == 1)


# ── циклотомические классы ────────────────────────────────────────────────────

def cyclotomic_coset_of_exp(k):
    """
    Циклотомический класс показателя k: {k, 2k, 4k, ...} mod 63.
    Для k=0: {0} (соответствует элементу 1 = g^0).
    """
    if k == 0:
        return frozenset([0])
    coset = set()
    i = k % ORDER
    while i not in coset:
        coset.add(i)
        i = (2 * i) % ORDER
    return frozenset(coset)


def cyclotomic_coset_of(a):
    """
    Циклотомический класс элемента a ∈ GF(2^6)*:
    {a, a^2, a^4, ..., a^{2^{k−1}}} — множество сопряжённых элементов.
    Все элементы класса имеют одинаковый минимальный многочлен над GF(2).
    """
    if a == 0:
        return frozenset([0])
    _, log = _get_tables()
    exp, _ = _get_tables()
    k = log[a]
    indices = cyclotomic_coset_of_exp(k)
    return frozenset(exp[i] for i in indices)


def all_cyclotomic_cosets():
    """
    Все циклотомические классы показателей в {0, 1,...,62}.
    Разбивают {0,...,62} на классы (размеры: делители 6).
    """
    remaining = set(range(ORDER))
    cosets = []
    while remaining:
        k = min(remaining)
        coset = cyclotomic_coset_of_exp(k)
        cosets.append(coset)
        remaining -= coset
    return cosets


# ── минимальные многочлены ────────────────────────────────────────────────────

def minimal_polynomial(a):
    """
    Коэффициенты минимального многочлена m_a(x) ∈ GF(2)[x] над GF(2).
    m_a(x) = Π_{c ∈ cyclotomic_coset(a)} (x + c) (в GF(2): x−c = x+c).
    Возвращает [c_0, c_1, ..., c_d] где m_a(x) = c_0 + c_1·x + ... + c_d·x^d.
    Коэффициенты ∈ {0,1}.
    """
    if a == 0:
        return [0, 1]   # m_0(x) = x
    coset = list(cyclotomic_coset_of(a))
    # poly = (x + coset[0])(x + coset[1])...
    poly = [1]           # начинаем с многочлена 1
    for root in coset:
        # умножить на (x + root) = [root, 1]
        new_poly = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new_poly[i] ^= gf_mul(c, root)
            new_poly[i + 1] ^= c
        # проверяем что все коэффициенты ∈ {0,1} (должно быть так, т.к. m_a ∈ GF(2)[x])
        poly = new_poly
    return poly


def poly_eval_gf(coeffs, a):
    """Вычислить многочлен над GF(2^6) в точке a (схема Горнера)."""
    result = 0
    for c in reversed(coeffs):
        result = gf_add(gf_mul(result, a), c)
    return result


# ── подполя ───────────────────────────────────────────────────────────────────

def subfield_elements(d):
    """
    Элементы подполя GF(2^d) ≤ GF(2^6) для d | 6 (d ∈ {1, 2, 3, 6}).
    GF(2^d) = {a ∈ GF(2^6) : a^{2^d} = a}.
    |GF(2^d)| = 2^d.
    """
    if 6 % d != 0:
        raise ValueError(f"GF(2^{d}) не является подполем GF(2^6) (6 не делится на {d})")
    q_d = 1 << d    # 2^d
    return frozenset(a for a in range(SIZE) if gf_pow(a, q_d) == a)


# ── логарифмы Цеха ────────────────────────────────────────────────────────────

def build_zech_log_table():
    """
    Таблица логарифмов Цеха Z[k] = log_g(1 + g^k) для k = 0..62.
    Если 1 + g^k = 0, то Z[k] = None (только k = 0 для g = примитивного... нет).
    Применение: log(a + b) = log(a) + Z[(log(b) − log(a)) mod 63].
    """
    exp, log = _get_tables()
    Z = [None] * ORDER
    for k in range(ORDER):
        val = exp[k] ^ 1    # g^k + 1 в GF(2^6) (= g^k XOR 1)
        if val == 0:
            Z[k] = None     # 1 + g^k = 0 ↔ g^k = 1 = g^0, т.е. k=0
        else:
            Z[k] = log[val]
    return Z


# ── аддитивные характеры ──────────────────────────────────────────────────────

def additive_character(a):
    """
    Аддитивный характер ψ(a) = (−1)^{Tr(a)} ∈ {+1, −1}.
    Гомоморфизм (GF(2^6), ⊕) → {+1, −1}.
    Совпадает с χ_1 из hexalg (WHT-характер с параметром u=1).
    """
    return 1 - 2 * gf_trace(a)


def additive_character_b(b, a):
    """
    ψ_b(a) = (−1)^{Tr(b·a)} — аддитивный характер с параметром b ∈ GF(2^6).
    64 ортогональных характера (b = 0 даёт тривиальный характер ψ_0 ≡ 1).
    Совпадает с WHT-характером χ_b(a) = (−1)^{⟨b,a⟩} из hexalg.
    """
    return 1 - 2 * gf_trace(gf_mul(b, a))


def character_sum(b, subset):
    """Сумма характеров Σ_{a ∈ S} ψ_b(a). Для b=0: |S|."""
    return sum(additive_character_b(b, a) for a in subset)


# ── BCH и оценки ──────────────────────────────────────────────────────────────

def bch_zeros(d_design):
    """
    Нули BCH-кода с проектным расстоянием d:
    {g^1, g^2, ..., g^{d−1}} ⊂ GF(2^6).
    Код порождается как аннулятор этих элементов над GF(2).
    """
    exp, _ = _get_tables()
    return [exp[i % ORDER] for i in range(1, d_design)]


def bch_generator_degree(d_design):
    """
    Степень порождающего многочлена BCH(d): количество отдельных циклотомических коэффициентов
    = число бит в порождающем многочлене.
    Точная степень = |объединение циклотомических классов нулей|.
    """
    zeros = bch_zeros(d_design)
    all_indices = set()
    for z in zeros:
        coset = cyclotomic_coset_of(z)
        exp, log = _get_tables()
        all_indices |= {log[c] for c in coset}
    return len(all_indices)


# ── вспомогательные функции ───────────────────────────────────────────────────

def is_subfield(elements, d):
    """Проверить, что множество elements является подполем GF(2^d)."""
    expected = subfield_elements(d)
    return frozenset(elements) == expected


def all_elements():
    """Все 64 элемента GF(2^6): [0, 1, 2, ..., 63]."""
    return list(range(SIZE))


def nonzero_elements():
    """Все 63 ненулевых элемента GF(2^6)*."""
    return list(range(1, SIZE))


def frobenius(a):
    """Отображение Фробениуса: φ(a) = a^2 (автоморфизм GF(2^6))."""
    return gf_mul(a, a)


def frobenius_orbit(a):
    """Орбита Фробениуса {a, a^2, a^4, ...} = циклотомический класс элемента a."""
    return cyclotomic_coset_of(a)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'info':
        exp, log = _get_tables()
        print(f"GF(2^6): p(x) = x^6 + x + 1, SIZE={SIZE}, ORDER={ORDER}")
        print(f"Примитивный элемент g = {PRIMITIVE} (= x)")
        n_prim = len(primitive_elements())
        print(f"Примитивных элементов: {n_prim} (φ(63) = 36)")
        cosets = all_cyclotomic_cosets()
        print(f"Циклотомических классов: {len(cosets)}")
        sizes = sorted(set(len(c) for c in cosets))
        print(f"Размеры классов: {sizes}")
        # Подполя
        for d in [1, 2, 3, 6]:
            sf = subfield_elements(d)
            print(f"  GF(2^{d}): {sorted(sf)[:8]}{'...' if len(sf) > 8 else ''}")

    elif cmd == 'mul':
        a = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        b = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        print(f"GF(2^6): {a} * {b} = {gf_mul(a, b)}")
        print(f"  {a}^{{-1}} = {gf_inv(a)}")
        print(f"  Tr({a}) = {gf_trace(a)}")

    elif cmd == 'power':
        g = PRIMITIVE
        print(f"Первые 12 степеней g = {g}:")
        for k in range(13):
            print(f"  g^{k} = {gf_pow(g, k)}")

    elif cmd == 'cosets':
        cosets = all_cyclotomic_cosets()
        print(f"Циклотомические классы ({len(cosets)} классов):")
        for i, c in enumerate(sorted(cosets, key=min)):
            exp, _ = _get_tables()
            elems = sorted(exp[k] for k in c) if c != frozenset([0]) else [1]
            print(f"  C_{min(c)}: показатели={sorted(c)}, элементы≈{elems[:4]}")

    elif cmd == 'minpoly':
        a = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        poly = minimal_polynomial(a)
        terms = []
        for i, c in enumerate(poly):
            if c:
                terms.append(f"x^{i}" if i > 0 else "1")
        print(f"Мин. многочлен m_{a}(x) = " + " + ".join(reversed(terms)))
        print(f"  Степень: {len(poly) - 1}")
        print(f"  Проверка m_{a}({a}) = {poly_eval_gf(poly, a)}")

    elif cmd == 'trace':
        print("Распределение следа Tr: GF(2^6) → {0, 1}:")
        t0 = sum(1 for a in range(SIZE) if gf_trace(a) == 0)
        t1 = sum(1 for a in range(SIZE) if gf_trace(a) == 1)
        print(f"  Tr=0: {t0} элементов, Tr=1: {t1} элементов (должно быть 32 каждого)")

    else:
        print("hexgf.py — Поле Галуа GF(2^6)")
        print("Команды: info  mul [a] [b]  power  cosets  minpoly [a]  trace")


if __name__ == '__main__':
    main()
