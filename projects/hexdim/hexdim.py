"""hexdim.py — Q6 как 6-мерный гиперкуб: подпространства, проекции, псевдо-QR.

Q6 = (Z₂)⁶ — шестимерный двоичный гиперкуб. Каждая гексаграмма = вершина Q6.
Данный модуль исследует его разложения и проекции:

  Подкубы:
    Q0 (1 вершина)  — C(6,0)=1
    Q1 (ребро)      — C(6,1)=6
    Q2 (квадрат)    — C(6,2)=15
    Q3 (куб)        — C(6,3)=20   ← каждая триграмма порождает куб
    Q4 (тессеракт)  — C(6,4)=15
    Q5 (пятимерный) — C(6,5)=6

  Произведения:
    Q6 = Q1×Q5 = Q2×Q4 = Q3×Q3 = Q2×Q2×Q1×Q1 = …
    Q3×Q3: нижняя триграмма × верхняя триграмма (классическая И-Цзин структура)

  Псевдо-QR: гексаграмма как 2D-код для 6D-навигации
    • 6 линий = 6 бит = 6D-координата
    • 8×8 сетка = 2D-карта всего Q6
    • Код Грея: гамильтонов путь по Q6, упорядочивающий все 64 гексаграммы
    • Добавление состояний: 4 состояния/линия → Q12 (4⁶ = 4096 = 2¹²)

  Проекции:
    Q6 → R²: различные 2D-проекции (по Грею, по весу, по триграммам)
    Q6 → R³: 3D-координаты (прямое вложение ℤ₂³ первых 3 бит)
    Q6 → Q4: проекция фиксацией 2 координат
"""
import math
import itertools

# ── helpers ──────────────────────────────────────────────────────────────────

def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def hamming(a, b):
    return _popcount(a ^ b)


# ── подкубы Qₖ ───────────────────────────────────────────────────────────────

def subcube(free_axes, base=0):
    """
    Подкуб Qₖ с k = len(free_axes) свободными осями.

    free_axes: подмножество {0,1,2,3,4,5} — индексы свободных битов.
    base: 6-битное число, значения фиксированных битов.
    Возвращает frozenset из 2^k вершин.
    """
    free_axes = list(free_axes)
    k = len(free_axes)
    # Зафиксировать фиксированные биты из base, перебрать свободные
    fixed_mask = ((1 << 6) - 1) ^ sum(1 << a for a in free_axes)
    fixed_val = base & fixed_mask
    vertices = set()
    for bits in range(1 << k):
        h = fixed_val
        for idx, axis in enumerate(free_axes):
            if (bits >> idx) & 1:
                h |= (1 << axis)
        vertices.add(h)
    return frozenset(vertices)


def all_subcubes(k):
    """
    Все подкубы Qₖ в Q6.
    Возвращает список (free_axes, base, vertices) — всего C(6,k) × 2^{6-k} наборов,
    но уникальных по множеству вершин = C(6,k) × 2^{6-k}/2^{6-k}... нет:
    фактически фиксированных бит 2^{6-k} вариантов → итого C(6,k) × 2^{6-k} подкубов.
    """
    from math import comb
    result = []
    for axes in itertools.combinations(range(6), k):
        free_axes = list(axes)
        fixed_axes = [a for a in range(6) if a not in free_axes]
        n_fixed = len(fixed_axes)
        for fixed_bits in range(1 << n_fixed):
            base = 0
            for idx, fa in enumerate(fixed_axes):
                if (fixed_bits >> idx) & 1:
                    base |= (1 << fa)
            verts = subcube(free_axes, base)
            result.append((free_axes, base, verts))
    return result


def subcube_count(k):
    """Число подкубов Qₖ в Q6 = C(6,k) × 2^{6-k}."""
    from math import comb
    return comb(6, k) * (2 ** (6 - k))


def find_subcube_of(vertices):
    """
    Минимальный подкуб Q6, содержащий данное множество вершин.
    Возвращает (free_axes, base, size).
    Вычисляется через XOR-замыкание (аффинное подпространство).
    """
    vertices = list(vertices)
    if not vertices:
        return [], 0, 0
    anchor = vertices[0]
    # Свободные оси — позиции, в которых вершины отличаются от anchor
    diff_bits = 0
    for v in vertices:
        diff_bits |= (v ^ anchor)
    free_axes = [i for i in range(6) if (diff_bits >> i) & 1]
    base = anchor & ~diff_bits
    return free_axes, base, len(free_axes)


# ── тессеракт Q4 ─────────────────────────────────────────────────────────────

def tesseract_subgraph(free_axes, base=0):
    """Q4-подграф (тессеракт): 4 свободных оси → 16 вершин."""
    if len(free_axes) != 4:
        raise ValueError("tesseract needs exactly 4 free axes")
    return subcube(free_axes, base)


def all_tesseracts():
    """
    Все C(6,4)×2^2 = 15×4 = 60 тессерактов (Q4-подграфов) в Q6.
    Возвращает список из 60 frozenset по 16 вершин.
    """
    return [verts for _, _, verts in all_subcubes(4)]


def tesseract_count():
    """Число Q4-подграфов в Q6 = C(6,4) × 2^2 = 60."""
    return subcube_count(4)


def is_tesseract(vertices):
    """Проверить, является ли набор из 16 вершин Q4-подграфом."""
    vertices = frozenset(vertices)
    if len(vertices) != 16:
        return False
    axes, base, k = find_subcube_of(vertices)
    return k == 4 and subcube(axes, base) == vertices


# ── куб Q3 ───────────────────────────────────────────────────────────────────

def all_cubes():
    """Все Q3-подграфы (кубы) в Q6: C(6,3)×2^3 = 20×8 = 160 штук."""
    return [verts for _, _, verts in all_subcubes(3)]


def cube_count():
    """Число Q3-подграфов = 160."""
    return subcube_count(3)


# ── разложение Q6 = Q3 × Q3 (триграммы) ─────────────────────────────────────

def trigram_decomposition(h):
    """
    Разложить гексаграмму h на (нижнюю, верхнюю) триграммы.

    Нижняя триграмма (земля): биты 0,1,2 → {0..7}
    Верхняя триграмма (небо): биты 3,4,5 → {0..7}

    Q6 = Q3_lower × Q3_upper — классическая структура И-Цзин.
    """
    lower = h & 7          # биты 0-2
    upper = (h >> 3) & 7   # биты 3-5
    return lower, upper


def from_trigrams(lower, upper):
    """Собрать гексаграмму из двух триграмм."""
    return (lower & 7) | ((upper & 7) << 3)


def trigram_name(t):
    """Традиционное название триграммы (☰ ≡ 7, …)."""
    names = {
        7: '☰ Цянь (Небо)',   0: '☷ Кунь (Земля)',
        5: '☱ Дуй (Озеро)',   6: '☲ Ли (Огонь)',
        3: '☳ Чжэнь (Гром)',  4: '☴ Сунь (Ветер)',
        2: '☵ Кань (Вода)',   1: '☶ Гэнь (Гора)',
    }
    return names.get(t, f'?({t:03b})')


def hexagram_structure(h):
    """Полная структура гексаграммы: биты, триграммы, вес Хэмминга."""
    lower, upper = trigram_decomposition(h)
    return {
        'value': h,
        'bits': f'{h:06b}',
        'yang_count': _popcount(h),
        'lower_trigram': lower,
        'upper_trigram': upper,
        'lower_name': trigram_name(lower),
        'upper_name': trigram_name(upper),
        'antipodal': h ^ 63,
    }


# ── произведения и разложения Q6 ─────────────────────────────────────────────

def product_decomposition(axes1, axes2):
    """
    Проверить, что axes1 и axes2 — дизъюнктное разбиение {0..5},
    задающее разложение Q6 = Q_{|axes1|} × Q_{|axes2|}.
    Возвращает (factor1_vertices, factor2_vertices).
    """
    if set(axes1) | set(axes2) != set(range(6)) or set(axes1) & set(axes2):
        raise ValueError("axes1 and axes2 must partition {0,1,2,3,4,5}")
    f1 = subcube(axes1, 0)  # 2^|axes1| вершин с нулевыми axes2
    f2 = subcube(axes2, 0)  # 2^|axes2| вершин с нулевыми axes1
    return f1, f2


def all_partitions_into_two():
    """
    Все разбиения {0..5} на два непустых подмножества (без учёта порядка).
    Задают разложения Q6 = Q_k × Q_{6-k}.
    """
    from math import comb
    result = []
    for k in range(1, 6):
        for axes1 in itertools.combinations(range(6), k):
            axes2 = tuple(a for a in range(6) if a not in axes1)
            if axes1 < axes2:  # избежать дублирования
                result.append((list(axes1), list(axes2)))
    return result


# ── проекции Q6 → низшие пространства ────────────────────────────────────────

def project_to_qk(h, free_axes):
    """Проекция вершины h на Qₖ: оставить только free_axes биты."""
    result = 0
    for idx, axis in enumerate(free_axes):
        if (h >> axis) & 1:
            result |= (1 << idx)
    return result


def project_to_q4(h, fixed_axes, fixed_values=None):
    """
    Проекция Q6 → Q4: зафиксировать 2 оси.
    fixed_axes: 2 индекса осей для фиксации.
    fixed_values: значения фиксированных битов (по умолчанию 0).
    """
    free_axes = [a for a in range(6) if a not in fixed_axes]
    return project_to_qk(h, free_axes)


def q6_to_grid_coords(h, method='trigram'):
    """
    Отобразить вершину h в 2D-координаты для визуализации.

    Методы:
      'trigram' : (lower_trigram, upper_trigram) — 8×8 сетка
      'yang'    : (yang_count, value_within_level) — треугольная форма
      'gray'    : (gray_row, gray_col) — упорядочение по коду Грея
    """
    if method == 'trigram':
        lower, upper = trigram_decomposition(h)
        return lower, upper  # 0..7 × 0..7

    elif method == 'yang':
        w = _popcount(h)
        # Внутри уровня yang_count=w: порядковый номер среди вершин того же веса
        same_weight = sorted(v for v in range(64) if _popcount(v) == w)
        pos = same_weight.index(h)
        return w, pos

    elif method == 'gray':
        # Код Грея: i → i ⊕ (i >> 1)
        gray_order = [i ^ (i >> 1) for i in range(64)]
        pos = gray_order.index(h)
        return pos // 8, pos % 8

    else:
        raise ValueError(f"Unknown method: {method}")


def q6_to_3d_coords(h):
    """
    Вложение Q6 → R³ (не изометрическое): суммируем пары битов.
    x = bit0 + bit1, y = bit2 + bit3, z = bit4 + bit5 (каждая ∈ {0,1,2}).
    """
    x = (h & 1) + ((h >> 1) & 1)
    y = ((h >> 2) & 1) + ((h >> 3) & 1)
    z = ((h >> 4) & 1) + ((h >> 5) & 1)
    return x, y, z


def q6_to_r6_coords(h):
    """Стандартное вложение в R⁶: {0,1}⁶."""
    return tuple((h >> i) & 1 for i in range(6))


# ── код Грея ─────────────────────────────────────────────────────────────────

def gray_code_sequence():
    """
    Последовательность кода Грея Q6: i ↦ i ⊕ (i >> 1).
    Гамильтонов путь по Q6: каждый шаг меняет ровно 1 бит.
    """
    return [i ^ (i >> 1) for i in range(64)]


def gray_code_position(h):
    """Позиция гексаграммы h в стандартном коде Грея."""
    # Обратный код Грея
    pos = h
    mask = h >> 1
    while mask:
        pos ^= mask
        mask >>= 1
    return pos


def gray_code_step(i):
    """Какой бит меняется на шаге i → i+1 кода Грея."""
    return (i + 1) ^ i ^ ((i + 1) ^ ((i + 1) >> 1)) ^ (i ^ (i >> 1))
    # Проще: flip bit = trailing zeros of (i+1)
    v = i + 1
    bit = 0
    while v & 1 == 0:
        v >>= 1
        bit += 1
    return bit


def gray_code_step_axis(i):
    """Индекс оси (0..5), меняющейся на шаге i → i+1 кода Грея."""
    v = i + 1
    bit = 0
    while v & 1 == 0:
        v >>= 1
        bit += 1
    return bit


# ── псевдо-QR-код: гексаграмма как 2D-бинарный код ───────────────────────────

def hexagram_as_barcode(h, style='unicode'):
    """
    Гексаграмма как 6-битный «штрих-код» (линейный бинарный код).

    style='unicode': использовать ─── (ян) и ─ ─ (инь)
    style='binary' : строки '1' и '0'
    style='grid'   : 2D-матрица 6×1 из 0/1
    """
    lines = []
    for i in range(6):  # снизу вверх
        bit = (h >> i) & 1
        if style == 'unicode':
            lines.append('───' if bit else '─ ─')
        elif style == 'binary':
            lines.append('1' if bit else '0')
        else:
            lines.append([bit])
    lines.reverse()  # верхняя линия первая
    return lines


def q6_as_8x8_grid(ordering='trigram'):
    """
    Разместить все 64 гексаграммы в сетке 8×8.

    ordering='trigram': строка=lower, столбец=upper (X = (Z₂)³ × (Z₂)³)
    ordering='gray'   : строка=row, столбец=col в коде Грея
    ordering='natural': строка=h//8, столбец=h%8
    """
    grid = [[None] * 8 for _ in range(8)]
    for h in range(64):
        if ordering == 'trigram':
            lower, upper = trigram_decomposition(h)
            grid[upper][lower] = h
        elif ordering == 'gray':
            row, col = q6_to_grid_coords(h, 'gray')
            grid[row][col] = h
        else:
            grid[h // 8][h % 8] = h
    return grid


def grid_to_string(grid, show_bits=False):
    """Напечатать 8×8 сетку гексаграмм (ASCII)."""
    lines = []
    for row in grid:
        parts = []
        for h in row:
            if h is None:
                parts.append('  ?  ')
            elif show_bits:
                parts.append(f'{h:06b}')
            else:
                parts.append(f' {h:2d}  ')
        lines.append('│'.join(parts))
    return '\n'.join(lines)


# ── расширенные состояния: Q12 через 4-уровневые линии ───────────────────────

_STATE_NAMES = {0: 'старая инь  ⚋⚋', 1: 'молодая инь  ─ ─',
                2: 'молодая янь  ───', 3: 'старый янь  ⊗──'}


def q12_hexagram(states):
    """
    Гексаграмма с 4-уровневыми линиями (как в традиции ярлычных монет).
    states: список из 6 значений {0,1,2,3}.
    Кодирует 6 × 2 бита = 12 бит → вершина Q12 (4096 состояний).

    Традиция: 0=старая-инь (→ янь), 1=молодая-инь, 2=молодая-янь, 3=старый-янь (→ инь)
    'Изменяющиеся' линии: состояния 0 и 3.
    """
    if len(states) != 6 or not all(0 <= s <= 3 for s in states):
        raise ValueError("states: список из 6 значений {0,1,2,3}")
    # Кодирование: каждое состояние = 2 бита
    code = 0
    for i, s in enumerate(states):
        code |= (s & 3) << (2 * i)
    return code  # 12-битный код


def q12_to_hexagram(code):
    """Q12-код → текущая гексаграмма (по молодым линиям, биты 1 каждой пары)."""
    h = 0
    for i in range(6):
        state = (code >> (2 * i)) & 3
        yang_bit = (state >> 1) & 1  # молодая-янь (2) или старый-янь (3) → бит1=1 → янь
        h |= yang_bit << i
    return h


def q12_transformed(code):
    """Q12-код → преобразованная гексаграмма (изменяющиеся линии меняют полярность)."""
    h = 0
    for i in range(6):
        state = (code >> (2 * i)) & 3
        # 0=старая-инь → становится янь, 3=старый-янь → становится инь
        if state == 3:
            yang = 0  # превращается в инь
        elif state == 0:
            yang = 1  # превращается в янь
        elif state == 2:
            yang = 1  # молодая-янь остаётся янь
        else:
            yang = 0  # молодая-инь остаётся инь
        h |= yang << i
    return h


def q12_changing_lines(code):
    """Индексы 'изменяющихся' линий (состояния 0 или 3) в Q12-коде."""
    return [i for i in range(6) if ((code >> (2 * i)) & 3) in (0, 3)]


# ── информация о размерности ──────────────────────────────────────────────────

def dimension_info():
    """Сводная информация о размерностях Q6."""
    from math import comb
    info = {}
    for k in range(7):
        info[k] = {
            'name': f'Q{k}',
            'vertices': 2 ** k,
            'edges': k * (2 ** (k - 1)) if k > 0 else 0,
            'count_in_q6': subcube_count(k),
            'unique_axes': comb(6, k),
        }
    return info


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'info':
        print("Q6 = (Z₂)⁶ — структура подкубов:")
        info = dimension_info()
        for k, d in info.items():
            if k == 0:
                continue
            print(f"  Q{k}: {d['vertices']:3d} вершин, C(6,{k})×2^{6-k}"
                  f" = {d['unique_axes']}×{2**(6-k)} = {d['count_in_q6']} подкубов")
        print(f"\nQ6 = Q3×Q3: нижняя триграмма (биты 0-2) × верхняя (биты 3-5)")
        print(f"Q6 = Q4×Q2: тессеракт × квадрат  (15 тессерактов разных ориентаций)")
        print(f"\n4-уровневые линии: 4^6 = {4**6} состояний = 2^12 = Q12")

    elif cmd == 'hexagram':
        h = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        info = hexagram_structure(h)
        print(f"Гексаграмма {h}:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("\n  Штрих-код (снизу вверх = биты 0..5):")
        for line in hexagram_as_barcode(h):
            print(f"    {line}")

    elif cmd == 'tesseracts':
        tess = all_tesseracts()
        print(f"Число Q4-подграфов (тессерактов) в Q6: {len(tess)}")
        # Показать первые 5
        for i, t in enumerate(tess[:5]):
            verts = sorted(t)
            print(f"  #{i+1}: {verts[:4]}... ({len(t)} вершин)")

    elif cmd == 'grid':
        ordering = sys.argv[2] if len(sys.argv) > 2 else 'trigram'
        grid = q6_as_8x8_grid(ordering)
        print(f"Q6 как 8×8 сетка (упорядочение: {ordering}):")
        print(grid_to_string(grid))

    elif cmd == 'gray':
        seq = gray_code_sequence()
        print(f"Код Грея Q6 (гамильтонов путь, {len(seq)} шагов):")
        for i in range(0, 64, 8):
            row = [f"{seq[j]:06b}" for j in range(i, min(i + 8, 64))]
            print("  " + " ".join(row))

    elif cmd == 'q12':
        h = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        import random
        rng = random.Random(h)
        states = [rng.randint(0, 3) for _ in range(6)]
        code = q12_hexagram(states)
        current = q12_to_hexagram(code)
        transformed = q12_transformed(code)
        changing = q12_changing_lines(code)
        print(f"Q12 пример (seed={h}):")
        for i, s in enumerate(states):
            print(f"  Линия {i+1}: {_STATE_NAMES[s]}")
        print(f"  Текущая гексаграмма:       {current:06b} = {current}")
        print(f"  Преобразованная гексаграмма:{transformed:06b} = {transformed}")
        print(f"  Изменяющиеся линии:         {changing}")
        print(f"  Q12-код (12 бит):           {code:012b}")

    elif cmd == 'projection':
        print("Проекции нескольких вершин Q6:")
        for h in [0, 7, 21, 42, 63]:
            lower, upper = trigram_decomposition(h)
            x, y, z = q6_to_3d_coords(h)
            gp = gray_code_position(h)
            print(f"  {h:06b}: "
                  f"триграммы=({lower},{upper}), "
                  f"3D=({x},{y},{z}), "
                  f"серое_место={gp}")

    else:
        print("hexdim.py — Q6 как 6D-гиперкуб: размерности, проекции, псевдо-QR")
        print("Команды:")
        print("  info              — структура подкубов Q6")
        print("  hexagram [h]      — разложение гексаграммы h")
        print("  tesseracts        — Q4-подграфы (тессеракты) в Q6")
        print("  grid [ordering]   — 8×8 карта Q6 (trigram/gray/natural)")
        print("  gray              — код Грея (гамильтонов путь)")
        print("  q12 [seed]        — Q12: 4-уровневые линии")
        print("  projection        — примеры проекций вершин Q6")


if __name__ == '__main__':
    main()
