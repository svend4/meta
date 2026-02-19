"""hexlat — булева решётка: Q6 как частично упорядоченное множество.

Q6 = (Z₂)⁶ — это булева решётка B₆, изоморфная степени множества 2^{0,...,5}.
Частичный порядок: x ≤ y ⟺ (x & y) == x  (побитовое включение).
Это решётка с meet x∧y = x&y и join x∨y = x|y.

Структура:
  - Минимум: 0 (пустое множество)
  - Максимум: 63 = 0b111111 (полное множество)
  - Ранг элемента: popcount(x) (размер подмножества)
  - Уровни: B₆ имеет 7 уровней (ранги 0..6), мощности C(6,0)..C(6,6)
  - Высота: 6
"""
import sys
import os
from math import comb
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Частичный порядок и базовые операции решётки
# ---------------------------------------------------------------------------

def leq(x: int, y: int) -> bool:
    """Частичный порядок: x ≤ y ⟺ x является подмножеством y."""
    return (x & y) == x


def meet(x: int, y: int) -> int:
    """Нижняя грань (meet, inf): x ∧ y = x & y."""
    return x & y


def join(x: int, y: int) -> int:
    """Верхняя грань (join, sup): x ∨ y = x | y."""
    return x | y


def complement(x: int) -> int:
    """Дополнение элемента в B₆: ~x & 63."""
    return (~x) & 63


def rank(x: int) -> int:
    """Ранг элемента = popcount(x) = размер подмножества."""
    return bin(x).count('1')


def rank_elements(r: int) -> list:
    """Все элементы ранга r (0 ≤ r ≤ 6)."""
    return [x for x in range(64) if rank(x) == r]


def whitney_numbers() -> list:
    """Числа Уитни W_k = |{x : rank(x) = k}| = C(6,k).

    Возвращает список [W_0, W_1, ..., W_6].
    """
    return [comb(6, k) for k in range(7)]


# ---------------------------------------------------------------------------
# Цепи и антицепи
# ---------------------------------------------------------------------------

def is_chain(elements: list) -> bool:
    """True, если все элементы попарно сравнимы (цепь)."""
    s = sorted(elements)
    for i in range(len(s) - 1):
        if not leq(s[i], s[i + 1]):
            return False
    return True


def is_antichain(elements: list) -> bool:
    """True, если никакие два элемента не сравнимы (антицепь)."""
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if leq(elements[i], elements[j]) or leq(elements[j], elements[i]):
                return False
    return True


def maximal_chains() -> list:
    """Все максимальные цепи (от 0 до 63) длиной 7 (ранги 0..6).

    Это перестановки: каждая цепь соответствует порядку добавления битов.
    Число максимальных цепей = 6! = 720.
    """
    result = []

    def _build(current: int, chain: list):
        if current == 63:
            result.append(chain + [63])
            return
        added_any = False
        for bit in range(6):
            if not (current & (1 << bit)):
                _build(current | (1 << bit), chain + [current])
                added_any = True
        if not added_any:
            result.append(chain)

    _build(0, [])
    return result


def largest_antichain() -> list:
    """Наибольшая антицепь (уровень ранга 3, по теореме Дилворта/Шпернера).

    |A| = C(6,3) = 20.
    """
    return rank_elements(3)


def dilworth_theorem() -> dict:
    """Теорема Дилворта: ширина = минимальное покрытие цепями.

    В B₆: ширина = C(6,3) = 20, нужно ровно 20 цепей для покрытия.
    """
    width = len(largest_antichain())  # C(6,3) = 20
    return {
        'width': width,
        'sperner_antichain': largest_antichain(),
        'description': 'Ширина B6 = C(6,3) = 20 (теорема Шпернера)',
    }


# ---------------------------------------------------------------------------
# Функция Мёбиуса
# ---------------------------------------------------------------------------

def mobius(x: int, y: int) -> int:
    """Функция Мёбиуса булевой решётки.

    μ(x, y) = (-1)^{rank(y) - rank(x)} если x ≤ y, иначе 0.
    Эквивалентно: μ(x, y) = (-1)^{popcount(y ^ x)} для x ≤ y.
    """
    if not leq(x, y):
        return 0
    return (-1) ** rank(y ^ x)


def zeta_function(x: int, y: int) -> int:
    """Дзета-функция: ζ(x, y) = 1 если x ≤ y, иначе 0."""
    return 1 if leq(x, y) else 0


def mobius_inversion_check() -> bool:
    """Проверка формулы обращения Мёбиуса: Σ_{x≤z≤y} μ(x,z) = [x==y]."""
    for x in range(64):
        for y in range(64):
            total = sum(mobius(x, z) for z in range(64) if leq(x, z) and leq(z, y))
            expected = 1 if x == y else 0
            if total != expected:
                return False
    return True


def euler_characteristic() -> int:
    """Эйлерова характеристика B₆ = Σ_{k=0}^{6} (-1)^k C(6,k) = (1-1)^6 = 0."""
    return sum((-1) ** k * comb(6, k) for k in range(7))


# ---------------------------------------------------------------------------
# Интервалы, фильтры, идеалы
# ---------------------------------------------------------------------------

def interval(x: int, y: int) -> list:
    """Все элементы z такие, что x ≤ z ≤ y."""
    if not leq(x, y):
        return []
    return [z for z in range(64) if leq(x, z) and leq(z, y)]


def principal_filter(x: int) -> list:
    """Главный фильтр: {z : z ≥ x}."""
    return [z for z in range(64) if leq(x, z)]


def principal_ideal(x: int) -> list:
    """Главный идеал: {z : z ≤ x}."""
    return [z for z in range(64) if leq(z, x)]


def upset_closure(elements: list) -> list:
    """Замыкание вверх (up-set): {z : ∃x ∈ S, z ≥ x}."""
    result = set()
    for x in elements:
        result.update(principal_filter(x))
    return sorted(result)


def downset_closure(elements: list) -> list:
    """Замыкание вниз (down-set): {z : ∃x ∈ S, z ≤ x}."""
    result = set()
    for x in elements:
        result.update(principal_ideal(x))
    return sorted(result)


# ---------------------------------------------------------------------------
# Покрытие (Hasse-диаграмма)
# ---------------------------------------------------------------------------

def covers(x: int, y: int) -> bool:
    """True, если y покрывает x: x < y и нет z с x < z < y.

    В булевой решётке: x ⋖ y ⟺ y = x | (1<<bit) для некоторого бита.
    """
    if not leq(x, y) or x == y:
        return False
    diff = y ^ x
    return diff != 0 and (diff & (diff - 1)) == 0  # diff — степень двойки


def hasse_edges() -> list:
    """Все рёбра диаграммы Хассе (пары покрытия x ⋖ y)."""
    edges = []
    for x in range(64):
        for bit in range(6):
            if not (x & (1 << bit)):
                y = x | (1 << bit)
                edges.append((x, y))
    return edges


def atom_decomposition(x: int) -> list:
    """Разложение x в атомы (элементы ранга 1, покрывающие 0)."""
    return [1 << bit for bit in range(6) if x & (1 << bit)]


# ---------------------------------------------------------------------------
# Многочлены решётки
# ---------------------------------------------------------------------------

def rank_generating_function() -> list:
    """Ранг-образующий многочлен f(q) = Σ_k W_k * q^k = (1+q)^6.

    Возвращает коэффициенты [W_0, W_1, ..., W_6] (коэффициент при q^k).
    """
    return whitney_numbers()


def poincare_polynomial() -> list:
    """Полином Пуанкаре (μ от 0 до x, суммируя).

    π(B₆, t) = Σ_{x} |μ(0, x)| t^{rank(x)} = Σ_k C(6,k) t^k.
    Коэффициент при t^k = C(6,k).
    """
    return [comb(6, k) for k in range(7)]


def characteristic_polynomial() -> list:
    """Характеристический многочлен решётки.

    χ(B₆, t) = Σ_{x≤1} μ(0,x) t^{6-rank(x)} = Σ_k C(6,k)(-1)^k t^{6-k}
             = (t-1)^6.
    Возвращает коэффициенты: [1, -6, 15, -20, 15, -6, 1] (от t^6 до t^0).
    """
    # (t-1)^6 = Σ_k C(6,k) (-1)^{6-k} t^k
    coeffs = [0] * 7
    for k in range(7):
        coeffs[6 - k] = (-1) ** k * comb(6, k)
    return coeffs


def zeta_polynomial(n: int) -> int:
    """Дзета-многочлен Z(n): число цепей длиной n-1 в B₆.

    Z(1) = 64, Z(2) = число пар x ≤ y, и т.д.
    Z(n) = Σ_{x_0 ≤ x_1 ≤ ... ≤ x_{n-1}} 1.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 64
    # Рекуррентно через матричные степени (для небольших n)
    # Z(n) = (Zeta^{n-1})_{все пары} сумма
    from functools import reduce
    # Вычислим итеративно: count[x] = число цепей длиной n-1 из 0, кончающихся в x
    count = [1] * 64  # n=1: один элемент
    for _ in range(n - 1):
        new_count = [0] * 64
        for y in range(64):
            new_count[y] = sum(count[x] for x in range(64) if leq(x, y))
        count = new_count
    return sum(count)


# ---------------------------------------------------------------------------
# Граф Q6 в контексте решётки
# ---------------------------------------------------------------------------

def lattice_diameter() -> int:
    """Диаметр диаграммы Хассе как неориентированного графа = 6."""
    return 6


def comparable_pairs_count() -> int:
    """Число сравнимых пар (x, y) с x ≤ y (включая x=y).

    Для каждого y число элементов x ≤ y равно 2^{rank(y)}.
    Итого: Σ_{k=0}^{6} C(6,k) * 2^k = (1+2)^6 = 3^6 = 729.
    """
    return sum(comb(6, k) * (2 ** k) for k in range(7))


def incomparable_pairs_count() -> int:
    """Число несравнимых пар {x, y} (x≠y, x и y несравнимы)."""
    total_pairs = 64 * 63 // 2  # все пары {x,y} с x≠y = 2016
    # comparable_pairs_count() считает упорядоченные (x,y) с x≤y.
    # Вычитая диагональ (64 пары x=y), получаем число неупорядоченных
    # сравнимых пар {x,y} с x<y (каждая пара встречается ровно один раз).
    comp = comparable_pairs_count() - 64
    return total_pairs - comp


def sublattice_boolean(k: int) -> list:
    """Список всех булевых подрешёток B_k в B_6.

    B_k ⊆ B_6 задаётся выбором k битов из 6.
    Возвращает список кортежей (маска битов, элементы подрешётки).
    """
    result = []
    for mask in range(64):
        if rank(mask) != k:
            continue
        bits = [b for b in range(6) if mask & (1 << b)]
        # Элементы: все подмножества bits
        elements = []
        for sub in range(1 << k):
            elem = 0
            for i, b in enumerate(bits):
                if sub & (1 << i):
                    elem |= (1 << b)
            elements.append(elem)
        result.append((mask, elements))
    return result


# ---------------------------------------------------------------------------
# Теорема о цепях (Dilworth / Mirsky)
# ---------------------------------------------------------------------------

def mirsky_decomposition() -> list:
    """Разложение B₆ в антицепи (Mirsky: слои по рангам).

    Возвращает [A_0, A_1, ..., A_6] где A_k = rank_elements(k).
    """
    return [rank_elements(k) for k in range(7)]


def chain_partition_count() -> int:
    """Минимальное покрытие цепями = ширина = C(6,3) = 20."""
    return comb(6, 3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_info(_args):
    print("Булева решётка B₆ = Q6 как частично упорядоченное множество")
    print("=" * 55)
    wn = whitney_numbers()
    print("Числа Уитни W_k = C(6,k):")
    for k, w in enumerate(wn):
        print(f"  W_{k} = {w}")
    print(f"Итого: {sum(wn)} элементов")
    print()
    ec = euler_characteristic()
    print(f"Эйлерова характеристика: {ec}")
    d = dilworth_theorem()
    print(f"Ширина (антицепь Шпернера): {d['width']}")
    print(f"Число максимальных цепей: 6! = 720")
    print(f"Диаметр Хассе-диаграммы: {lattice_diameter()}")
    print()
    print("Характеристический многочлен (t-1)^6:")
    cp = characteristic_polynomial()
    terms = []
    for i, c in enumerate(cp):
        if c == 0:
            continue
        power = 6 - i
        if power == 0:
            terms.append(str(c))
        elif power == 1:
            terms.append(f"{c}t" if c != 1 else "t")
        else:
            terms.append(f"{c}t^{power}" if c != 1 else f"t^{power}")
    print("  " + " + ".join(terms).replace("+ -", "- "))


def _cmd_interval(args):
    if len(args) < 2:
        print("Использование: interval <x> <y>")
        return
    x, y = int(args[0]), int(args[1])
    iv = interval(x, y)
    print(f"Интервал [{x}, {y}] в B₆: {len(iv)} элементов")
    print(f"Элементы: {iv}")


def _cmd_mobius(args):
    if len(args) < 2:
        print("Использование: mobius <x> <y>")
        return
    x, y = int(args[0]), int(args[1])
    mu = mobius(x, y)
    print(f"μ({x}, {y}) = {mu}")
    print(f"  x ≤ y: {leq(x, y)}")
    if leq(x, y):
        print(f"  rank(y^x) = {rank(y ^ x)}")


def _cmd_chains(_args):
    chains = maximal_chains()
    print(f"Максимальные цепи (0 → 63): {len(chains)}")
    print("Первые 3:")
    for ch in chains[:3]:
        print(f"  {ch}")


def _cmd_antichain(_args):
    ac = largest_antichain()
    print(f"Наибольшая антицепь (ранг 3, теорема Шпернера): {len(ac)} элементов")
    print(f"  {ac}")


def main():
    import sys
    if len(sys.argv) < 2:
        print("Использование: hexlat.py <команда> [аргументы]")
        print("Команды: info, interval <x> <y>, mobius <x> <y>, chains, antichain")
        return
    cmd = sys.argv[1]
    args = sys.argv[2:]
    commands = {
        'info': _cmd_info,
        'interval': _cmd_interval,
        'mobius': _cmd_mobius,
        'chains': _cmd_chains,
        'antichain': _cmd_antichain,
    }
    if cmd not in commands:
        print(f"Неизвестная команда: {cmd}")
        print(f"Доступные: {list(commands)}")
        return
    commands[cmd](args)


if __name__ == '__main__':
    main()
