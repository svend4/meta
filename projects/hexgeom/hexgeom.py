"""hexgeom.py — Метрическая геометрия на Q6 = (Z₂)⁶ (64 гексаграммы).

Q6 как метрическое пространство с расстоянием Хэмминга:
  • Шары и сферы Хэмминга, размеры, пересечения
  • Диаграмма Вороного: разбиение Q6 по ближайшим центрам
  • Граф Делоне (двойственный к Вороного)
  • Геодезические интервалы: {h : d(u,h)+d(h,v)=d(u,v)}
  • Точка Штейнера / медиана (минимизация суммы расстояний)
  • Равноудалённые множества, антиподальные пары
  • Пакинг (packing) и покрытие (covering), числа κ(r) и ρ(r)
  • Распределение расстояний в коде (distance distribution)
  • Граница Джонсона, изоперметрическое неравенство для Q6
"""
import math
import itertools
import random

# ── helpers ──────────────────────────────────────────────────────────────────

def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def hamming(a, b):
    """Расстояние Хэмминга d(a, b) = popcount(a ⊕ b)."""
    return _popcount(a ^ b)


def _neighbors(h):
    return [h ^ (1 << i) for i in range(6)]


# ── шары и сферы Хэмминга ────────────────────────────────────────────────────

def hamming_ball(center, radius):
    """B(center, r) = {h ∈ Q6 : d(h, center) ≤ r}."""
    return frozenset(h for h in range(64) if hamming(h, center) <= radius)


def hamming_sphere(center, radius):
    """S(center, r) = {h ∈ Q6 : d(h, center) = r}."""
    return frozenset(h for h in range(64) if hamming(h, center) == radius)


def ball_size(radius):
    """|B(c, r)| = Σ_{k=0}^{r} C(6, k)."""
    from math import comb
    return sum(comb(6, k) for k in range(min(radius, 6) + 1))


def sphere_size(radius):
    """|S(c, r)| = C(6, r)."""
    from math import comb
    return comb(6, min(radius, 6))


def ball_intersection(c1, r1, c2, r2):
    """B(c1, r1) ∩ B(c2, r2)."""
    return hamming_ball(c1, r1) & hamming_ball(c2, r2)


def ball_union_size(centers, radius):
    """Размер объединения шаров с одним радиусом."""
    union = set()
    for c in centers:
        union |= hamming_ball(c, radius)
    return len(union)


# ── диаграмма Вороного ───────────────────────────────────────────────────────

def voronoi_cells(centers):
    """
    Диаграмма Вороного: Vor(s) = {h : d(h, s) ≤ d(h, s') для всех s' ∈ centers}.
    При равных расстояниях — относим h к наименьшему центру.
    Возвращает dict {center: frozenset(вершины)}.
    """
    centers = list(centers)
    cells = {c: set() for c in centers}
    for h in range(64):
        dists = [(hamming(h, c), c) for c in centers]
        min_dist, nearest = min(dists)
        cells[nearest].add(h)
    return {c: frozenset(v) for c, v in cells.items()}


def voronoi_sizes(centers):
    """Размеры ячеек Вороного."""
    cells = voronoi_cells(centers)
    return {c: len(v) for c, v in cells.items()}


def delaunay_graph(centers):
    """
    Граф Делоне: два центра связаны, если их ячейки имеют общую границу
    (существует вершина Q6 на равном расстоянии от обоих).
    Возвращает set пар (c1, c2) с c1 < c2.
    """
    centers = list(centers)
    edges = set()
    cells = voronoi_cells(centers)
    # h ∈ Q6 на границе двух ячеек → соответствующие центры смежны
    for h in range(64):
        dists = sorted((hamming(h, c), c) for c in centers)
        if len(dists) >= 2 and dists[0][0] == dists[1][0]:
            c1, c2 = min(dists[0][1], dists[1][1]), max(dists[0][1], dists[1][1])
            edges.add((c1, c2))
    return edges


# ── метрические интервалы ────────────────────────────────────────────────────

def metric_interval(u, v):
    """
    I(u, v) = {h : d(u, h) + d(h, v) = d(u, v)} — геодезический интервал.
    Это множество вершин, лежащих на кратчайшем пути между u и v.
    В Q6: I(u, v) = {h : bits(h) согласованы с u и v по всем отличающимся битам}.
    """
    d = hamming(u, v)
    return frozenset(h for h in range(64)
                     if hamming(u, h) + hamming(h, v) == d)


def gate(u, v):
    """
    Вершины, смежные с u и ближе к v (один шаг вперёд).
    Аналог "следующего измерения" на пути u → v.
    """
    du = hamming(u, v)
    return frozenset(n for n in _neighbors(u) if hamming(n, v) == du - 1)


def median(points):
    """
    Медиана множества точек: argmin_{h ∈ Q6} Σ d(h, p).
    В (Z₂)ⁿ это побитовое большинство.
    """
    bit_counts = [0] * 6
    for p in points:
        for i in range(6):
            if (p >> i) & 1:
                bit_counts[i] += 1
    n = len(points)
    result = 0
    for i in range(6):
        if bit_counts[i] * 2 > n:
            result |= (1 << i)
    return result


def steiner_point(points):
    """
    Точка Штейнера: argmin_{h ∈ Q6} Σ d(h, p).
    В (Z₂)ⁿ это побитовое большинство (совпадает с медианой).
    """
    return median(points)


def sum_of_distances(h, points):
    """Σ d(h, p) для p ∈ points."""
    return sum(hamming(h, p) for p in points)


def find_1_median(points):
    """Минимизировать Σ d(h, p) по всем h ∈ Q6 (l₁-медиана)."""
    best = min(range(64), key=lambda h: sum_of_distances(h, points))
    return best, sum_of_distances(best, points)


# ── специальные множества ─────────────────────────────────────────────────────

def antipodal_pair(h):
    """Антиподальная пара: (h, h ⊕ 63), расстояние = 6."""
    return h, h ^ 63


def all_antipodal_pairs():
    """Все 32 антиподальные пары в Q6."""
    seen = set()
    pairs = []
    for h in range(64):
        if h not in seen:
            a = h ^ 63
            pairs.append((h, a))
            seen.add(h)
            seen.add(a)
    return pairs


def equidistant_set(target_dist, seed=42):
    """
    Поиск максимального подмножества Q6 с попарным расстоянием = target_dist.
    Использует жадный алгоритм с рестартами.
    """
    rng = random.Random(seed)
    best = []
    for _ in range(100):
        candidates = list(range(64))
        rng.shuffle(candidates)
        current = []
        for c in candidates:
            if all(hamming(c, x) == target_dist for x in current):
                current.append(c)
        if len(current) > len(best):
            best = current[:]
    return best


def clique_number_at_distance(d):
    """Максимальный размер d-равноудалённого множества."""
    return len(equidistant_set(d))


# ── пакинг и покрытие ────────────────────────────────────────────────────────

def packing_number(radius):
    """
    Число пакинга κ(r): максимальный размер r-пакинга
    (множества с попарными расстояниями > 2r).
    Жадный алгоритм.
    """
    centers = []
    candidates = list(range(64))
    random.shuffle(candidates)
    for h in candidates:
        if all(hamming(h, c) > 2 * radius for c in centers):
            centers.append(h)
    return centers


def covering_number_greedy(radius):
    """
    Минимальное r-покрытие: минимальное множество центров,
    такое что каждая вершина Q6 покрыта некоторым шаром.
    Жадный алгоритм (не оптимальный).
    """
    uncovered = set(range(64))
    centers = []
    while uncovered:
        # Выбрать вершину, покрывающую максимум непокрытых
        best = max(range(64),
                   key=lambda h: len(hamming_ball(h, radius) & uncovered))
        centers.append(best)
        uncovered -= hamming_ball(best, radius)
    return centers


def is_perfect_code(centers, radius):
    """
    Является ли множество centers совершенным r-кодом?
    (Шары попарно не пересекаются и покрывают всё Q6.)
    """
    balls = [hamming_ball(c, radius) for c in centers]
    # Проверить непересечение
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            if balls[i] & balls[j]:
                return False
    # Проверить покрытие
    return set().union(*balls) == set(range(64))


# ── распределение расстояний ─────────────────────────────────────────────────

def distance_distribution(code):
    """
    Распределение расстояний кода C:
    A[d] = |{(c1, c2) ∈ C² : d(c1, c2) = d}| / |C|.
    A[0] = 1 всегда.
    """
    code = list(code)
    n = len(code)
    A = [0] * 7
    for c1 in code:
        for c2 in code:
            A[hamming(c1, c2)] += 1
    return [a / n for a in A]


def dual_distance_distribution(code):
    """
    Двойственное распределение расстояний (через WHT).
    B[d] = (1/|C|) Σ_{c1,c2} K_d(d(c1,c2)) / C(6,d)
    где K_d — полиномы Кравчука.
    """
    code = list(code)
    n = len(code)
    A = distance_distribution(code)
    # Матрица Кравчука для Q6
    # K_j(i) = sum_{s=0}^{j} (-1)^s C(i,s) C(6-i,j-s)
    from math import comb

    def krawtchouk(j, i, m=6):
        return sum((-1) ** s * comb(i, s) * comb(m - i, j - s)
                   for s in range(j + 1)
                   if s <= i and j - s <= m - i)

    B = []
    for j in range(7):
        b = sum(A[i] * krawtchouk(j, i) for i in range(7))
        B.append(b / comb(6, j) if comb(6, j) > 0 else 0.0)
    return B


def minimum_distance(code):
    """Минимальное ненулевое расстояние в коде."""
    code = list(code)
    if len(code) < 2:
        return 0
    return min(hamming(c1, c2) for c1 in code for c2 in code if c1 != c2)


def diameter_of_set(S):
    """Диаметр множества: max пара расстояний."""
    S = list(S)
    if len(S) < 2:
        return 0
    return max(hamming(a, b) for a in S for b in S)


# ── изоперметрическое неравенство (вертекс-расширение) ───────────────────────

def vertex_boundary(S):
    """
    Граница множества S: вершины Q6 вне S, смежные с S.
    ∂S = N(S) \ S.
    """
    boundary = set()
    for h in S:
        for n in _neighbors(h):
            if n not in S:
                boundary.add(n)
    return frozenset(boundary)


def edge_boundary(S):
    """
    Рёберная граница: рёбра между S и Q6 \ S.
    Возвращает список (u, v) с u ∈ S, v ∉ S.
    """
    S_set = set(S)
    return [(h, n) for h in S for n in _neighbors(h) if n not in S_set]


def isoperimetric_ratio(S):
    """
    Изоперметрическое отношение: |∂S| / |S|.
    Минимум для Q6 при |S|=32 равен 1 (ширина бисекции = 32).
    """
    if not S:
        return 0.0
    return len(vertex_boundary(S)) / len(S)


# ── границы для кодов ────────────────────────────────────────────────────────

def johnson_bound(d, n=6):
    """
    Граница Джонсона: верхняя оценка числа кодовых слов
    A(n, d) ≤ floor(n/d × floor((n-1)/(d-1) × ... )).
    Для Q6 используем рекуррентную форму.
    """
    from math import comb
    if d == 0:
        return 2 ** n
    if d == 1:
        return 2 ** n
    # LP-граница (простое рекурсивное определение через объём шаров Хэмминга)
    # A(n, d) ≤ 2^n / V(n, floor((d-1)/2))
    t = (d - 1) // 2
    vol = ball_size(t)
    return (2 ** n) // vol if vol > 0 else 2 ** n


def singleton_bound(d, n=6):
    """Граница Синглтона: A(n, d) ≤ 2^{n-d+1}."""
    return 2 ** max(0, n - d + 1)


def plotkin_bound(d, n=6):
    """Граница Плоткина: при d > n/2: A(n, d) ≤ 2d / (2d - n)."""
    if 2 * d <= n:
        return None  # не применима
    return (2 * d) // (2 * d - n)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'ball':
        c = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        r = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        b = hamming_ball(c, r)
        print(f"B({c:06b}, {r}): {len(b)} вершин")
        print(f"  Ожидаемый размер: {ball_size(r)}")
        print(f"  Совершенный код? {is_perfect_code([c], r)}")

    elif cmd == 'voronoi':
        import ast
        centers_str = sys.argv[2] if len(sys.argv) > 2 else '[0, 21, 42, 63]'
        centers = ast.literal_eval(centers_str)
        cells = voronoi_cells(centers)
        print(f"Диаграмма Вороного ({len(centers)} центров):")
        for c, verts in cells.items():
            print(f"  center={c:06b}: {len(verts)} вершин")
        dg = delaunay_graph(centers)
        print(f"Граф Делоне: {len(dg)} рёбер")

    elif cmd == 'interval':
        u = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        v = int(sys.argv[3]) if len(sys.argv) > 3 else 63
        I = metric_interval(u, v)
        print(f"I({u:06b}, {v:06b}): {len(I)} вершин, d={hamming(u, v)}")

    elif cmd == 'packing':
        for r in range(4):
            p = packing_number(r)
            c = covering_number_greedy(r)
            print(f"r={r}: пакинг={len(p)}, покрытие={len(c)}, "
                  f"совершенный? {is_perfect_code(p, r)}")

    elif cmd == 'dist':
        import ast
        code_str = sys.argv[2] if len(sys.argv) > 2 else '[0,21,42,63]'
        code = ast.literal_eval(code_str)
        A = distance_distribution(code)
        print(f"Распределение расстояний ({len(code)} кодовых слов):")
        for d, a in enumerate(A):
            print(f"  d={d}: {a:.4f}")
        print(f"  min_distance: {minimum_distance(code)}")

    elif cmd == 'bounds':
        print(f"Границы A(6, d) — максимальный код с минимальным расстоянием d:")
        for d in range(1, 7):
            jb = johnson_bound(d)
            sb = singleton_bound(d)
            pb = plotkin_bound(d)
            print(f"  d={d}: Johnson≤{jb}, Singleton≤{sb}",
                  f", Plotkin≤{pb}" if pb else "")

    else:
        print("hexgeom.py — Метрическая геометрия на Q6")
        print("Команды: ball [c] [r]  voronoi [centers]  interval [u] [v]")
        print("         packing  dist [code]  bounds")


if __name__ == '__main__':
    main()
