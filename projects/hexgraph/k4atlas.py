"""k4atlas — Атлас подграфов K₄ с визуализацией через систему глифов.

Ключевое наблюдение: каждый глиф (0..63) — подграф K₄.

Четыре вершины K₄ — углы квадрата (= вершины glyph-рамки):
    TL(0)──TR(1)
     │  ╲  ╱  │
     │  ╱  ╲  │
    BL(2)──BR(3)

Шесть рёбер K₄ = шесть сегментов глифа:
    бит 0 = ребро (TL,TR)  верхняя перекладина    ─
    бит 1 = ребро (BL,BR)  нижняя перекладина     ─
    бит 2 = ребро (TL,BL)  левая стойка           │
    бит 3 = ребро (TR,BR)  правая стойка          │
    бит 4 = ребро (TL,BR)  диагональ              ╲
    бит 5 = ребро (TR,BL)  диагональ              ╱

XOR двух глифов = симметричная разность рёберных множеств.
Расстояние Хэмминга = число рёбер, по которым подграфы различаются.
Ян-счёт (yang_count) = число рёбер = плотность подграфа.

Классификация:
    11 классов изоморфизма (совпадает с числом незамощаемых нетов куба!)
    19 классов геометрической эквивалентности (орбиты D₄)
"""

from __future__ import annotations
import sys
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, apply_permutation, SIZE
from projects.hexvis.hexvis import render_glyph, _YANG_ANSI, _RESET, _BOLD

# ---------------------------------------------------------------------------
# Структура K₄: рёбра, соответствующие битам глифа
# ---------------------------------------------------------------------------

# Вершины: TL=0, TR=1, BL=2, BR=3
VERTICES = (0, 1, 2, 3)
VERTEX_NAMES = {0: 'TL', 1: 'TR', 2: 'BL', 3: 'BR'}

# Ребро для каждого бита (индекс бита → пара вершин)
BIT_EDGE = [
    (0, 1),  # бит 0: TL─TR
    (2, 3),  # бит 1: BL─BR
    (0, 2),  # бит 2: TL─BL
    (1, 3),  # бит 3: TR─BR
    (0, 3),  # бит 4: TL─BR
    (1, 2),  # бит 5: TR─BL
]

# Обратное: пара вершин → номер бита
EDGE_BIT = {edge: bit for bit, edge in enumerate(BIT_EDGE)}
EDGE_BIT.update({(v, u): bit for bit, (u, v) in enumerate(BIT_EDGE)})


def edges_of(h: int) -> list[tuple[int, int]]:
    """Список рёбер подграфа, соответствующего глифу h."""
    return [BIT_EDGE[b] for b in range(6) if (h >> b) & 1]


def degree_sequence(h: int) -> tuple[int, ...]:
    """Отсортированная последовательность степеней вершин (по убыванию)."""
    deg = [0, 0, 0, 0]
    for u, v in edges_of(h):
        deg[u] += 1
        deg[v] += 1
    return tuple(sorted(deg, reverse=True))


# ---------------------------------------------------------------------------
# Анализ связности (Union-Find)
# ---------------------------------------------------------------------------

def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], x: int, y: int) -> None:
    parent[_find(parent, x)] = _find(parent, y)


def components(h: int) -> list[frozenset[int]]:
    """Список связных компонент подграфа h (только вершины со степенью ≥ 0)."""
    parent = list(range(4))
    active = set()
    for u, v in edges_of(h):
        active.add(u)
        active.add(v)
        _union(parent, u, v)
    # Изолированные вершины тоже в своих компонентах
    for v in range(4):
        active.add(v)
    groups: dict[int, set[int]] = defaultdict(set)
    for v in active:
        groups[_find(parent, v)].add(v)
    return [frozenset(g) for g in groups.values()]


def is_connected(h: int) -> bool:
    """Подграф h связен (все 4 вершины в одной компоненте)?"""
    if h == 0:
        return False  # пустой граф несвязен
    comps = components(h)
    # Связен если все 4 вершины в одной компоненте
    return len(comps) == 1 and len(comps[0]) == 4


def count_triangles(h: int) -> int:
    """Число треугольников (K₃) в подграфе h."""
    edgeset = set(edges_of(h))
    edgeset.update((v, u) for u, v in list(edgeset))
    count = 0
    verts = list(range(4))
    for a, b, c in combinations(verts, 3):
        if ((a, b) in edgeset and (b, c) in edgeset and (a, c) in edgeset):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Изоморфизм: канонический инвариант
# ---------------------------------------------------------------------------

def _graph_signature(h: int) -> tuple:
    """
    Каноническая сигнатура подграфа для определения изоморфизма.
    Перебираем все 24 перестановки вершин, берём минимальное 6-битное число.
    """
    from itertools import permutations
    edgeset = set(edges_of(h))
    best = SIZE  # 64
    for perm in permutations(range(4)):
        # perm[v] = новый номер для старой вершины v
        new_h = 0
        for bit, (u, v) in enumerate(BIT_EDGE):
            # ребро (u,v) в новом = ребро (perm^{-1}[u], perm^{-1}[v]) в старом
            pu, pv = perm[u], perm[v]
            canonical_edge = (min(pu, pv), max(pu, pv))
            orig_edge = tuple(sorted((BIT_EDGE[bit][0], BIT_EDGE[bit][1])))
            # Проще: проверить по обратной перестановке
            pass
        # Альтернативный способ: перенумеровать вершины и записать adj matrix
        # Биты нового графа:
        new_h = 0
        inv_perm = [0] * 4
        for i, p in enumerate(perm):
            inv_perm[p] = i
        for bit, (u, v) in enumerate(BIT_EDGE):
            # В новом графе ребро (u,v) присутствует iff в старом присутствует
            # ребро (inv_perm[u], inv_perm[v])
            old_u, old_v = inv_perm[u], inv_perm[v]
            old_edge = (min(old_u, old_v), max(old_u, old_v))
            if old_edge in edgeset or (old_edge[1], old_edge[0]) in edgeset:
                new_h |= (1 << bit)
        if new_h < best:
            best = new_h
    return best


# Предвычислим сигнатуры всех 64 глифов
_SIGNATURES: dict[int, int] = {h: _graph_signature(h) for h in range(64)}

# Изоморфизм-классы: словарь {канон → список элементов}
_ISO_CLASSES: dict[int, list[int]] = defaultdict(list)
for _h in range(64):
    _ISO_CLASSES[_SIGNATURES[_h]].append(_h)


def iso_class(h: int) -> int:
    """Канонический представитель изоморфизм-класса для h."""
    return _SIGNATURES[h]


def same_iso_class(a: int, b: int) -> bool:
    """a и b изоморфны как подграфы K₄?"""
    return _SIGNATURES[a] == _SIGNATURES[b]


# ---------------------------------------------------------------------------
# Имена изоморфизм-классов
# ---------------------------------------------------------------------------

def _classify_iso(h: int) -> str:
    """Имя класса изоморфизма по числу рёбер и инвариантам."""
    e = yang_count(h)
    ds = degree_sequence(h)
    tri = count_triangles(h)
    conn = is_connected(h)

    if e == 0:
        return '∅ (пустой граф)'
    if e == 1:
        return 'K₂ (одно ребро)'
    if e == 2:
        if ds == (1, 1, 1, 1):
            return 'M₂ (паросочетание)'
        return 'P₃ (путь 3 вершины)'
    if e == 3:
        if ds == (3, 1, 1, 1):
            return 'K₁,₃ (звезда)'
        if ds == (2, 2, 1, 1):
            return 'P₄ (путь 4 вершины)'
        if ds == (2, 2, 2, 0):
            return 'K₃ (треугольник)'
        return f'3-рёбер ds={ds}'
    if e == 4:
        if ds == (2, 2, 2, 2):
            return 'C₄ (цикл 4)'
        if ds == (3, 2, 2, 1):
            return '⊳ (лапа: K₃+рёбро)'
        return f'4-рёбер ds={ds}'
    if e == 5:
        return 'K₄−e (алмаз)'
    if e == 6:
        return 'K₄ (полный граф)'
    return f'{e} рёбер'


# Строим карту канон → имя
_ISO_NAMES: dict[int, str] = {
    canon: _classify_iso(canon)
    for canon in _ISO_CLASSES
}


def iso_name(h: int) -> str:
    """Имя изоморфизм-класса для h."""
    return _ISO_NAMES[_SIGNATURES[h]]


# ---------------------------------------------------------------------------
# Визуализация: таблица атласа
# ---------------------------------------------------------------------------

_RESET_STR = _RESET
_YANG_COLORS = _YANG_ANSI


def render_iso_class_row(elements: list[int], name: str, color: bool = True) -> str:
    """Одна строка атласа: имя класса + все его глифы (3 строки каждый)."""
    lines: list[str] = []
    label = f"{name:28s}  ({len(elements):2d})  "
    pad = ' ' * len(label)

    glyphs = [render_glyph(h) for h in elements]

    for ri in range(3):
        parts: list[str] = []
        for gi, h in enumerate(elements):
            cell = glyphs[gi][ri]
            if color:
                yc = yang_count(h)
                c = _YANG_COLORS[yc]
                cell = c + cell + _RESET_STR
            parts.append(cell)
        prefix = label if ri == 1 else pad
        lines.append(prefix + ' '.join(parts))

    return '\n'.join(lines)


def render_atlas(color: bool = True) -> str:
    """Полный атлас: все 11 классов изоморфизма, по числу рёбер."""
    out: list[str] = []
    out.append('═' * 72)
    out.append('  АТЛАС ПОДГРАФОВ K₄  (11 классов изоморфизма, 64 подграфа)')
    out.append('  Каждый глиф = подграф K₄  •  XOR глифов = Δ рёберных множеств')
    out.append('═' * 72)

    # Группируем по числу рёбер
    by_edges: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for canon, elems in sorted(_ISO_CLASSES.items()):
        e = yang_count(canon)
        by_edges[e].append((canon, sorted(elems)))

    for e in range(7):
        if e not in by_edges:
            continue
        out.append(f'\n── {e} рёбер ──────────────────────────────────────')
        for canon, elems in by_edges[e]:
            name = _ISO_NAMES[canon]
            out.append(render_iso_class_row(elems, name, color=color))
            out.append('')

    out.append('─' * 72)
    out.append(f'  Итого: {len(_ISO_CLASSES)} классов изоморфизма')
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Детальный вывод одного подграфа
# ---------------------------------------------------------------------------

def describe_subgraph(h: int, color: bool = True) -> str:
    """Подробное описание подграфа: глиф + все свойства."""
    lines: list[str] = []
    glyph = render_glyph(h)
    yc = yang_count(h)
    c = (_YANG_COLORS[yc] + _BOLD) if color else ''
    rst = _RESET_STR if color else ''

    lines.append(f'  Подграф h={h:2d}  ({h:06b})  {iso_name(h)}')
    lines.append(f'  ┌───┐')
    for row in glyph:
        lines.append(f'  │{c}{row}{rst}│')
    lines.append(f'  └───┘')
    lines.append(f'  Рёбра ({yc}): {[f"{VERTEX_NAMES[u]}─{VERTEX_NAMES[v]}" for u,v in edges_of(h)]}')
    lines.append(f'  Степени вершин: TL={degree_sequence(h)!r}')
    lines.append(f'  Треугольников: {count_triangles(h)}')
    lines.append(f'  Связен: {"да" if is_connected(h) else "нет"}  '
                 f'(компонент: {len(components(h))})')
    lines.append(f'  Антипод h⊕63={h ^ 63:2d}: {iso_name(h ^ 63)}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Статистика классов изоморфизма в Хассе-диаграмме
# ---------------------------------------------------------------------------

def render_hasse_by_class(color: bool = True) -> str:
    """Диаграмма Хассе, где каждый глиф подписан номером изоморфизм-класса."""
    # Построим нумерацию классов
    canon_to_idx = {c: i + 1 for i, c in enumerate(sorted(_ISO_CLASSES))}

    lines: list[str] = ['  Диаграмма Хассе B₆ (число в глифе = № изоморфизм-класса)']
    by_rank: list[list[int]] = [[] for _ in range(7)]
    for h in range(64):
        by_rank[yc := yang_count(h)].append(h)

    max_n = 20
    cw = 3
    sw = 1
    total_w = max_n * cw + (max_n - 1) * sw

    for k, elems in enumerate(by_rank):
        n = len(elems)
        row_w = n * cw + (n - 1) * sw
        pad = ' ' * ((total_w - row_w) // 2)
        glyphs = [render_glyph(h) for h in elems]
        for ri in range(3):
            parts: list[str] = []
            for gi, h in enumerate(elems):
                cell = glyphs[gi][ri]
                if color:
                    yc2 = yang_count(h)
                    cell = _YANG_COLORS[yc2] + cell + _RESET_STR
                parts.append(cell)
            lines.append(pad + ' '.join(parts))
        # Строка с номерами классов
        idx_row = ' '.join(
            f'{canon_to_idx[_SIGNATURES[h]]:>3d}' for h in elems
        )
        lines.append(pad + idx_row)
        lines.append('')

    # Легенда
    lines.append('  Легенда (класс → тип):')
    for canon, elems in sorted(_ISO_CLASSES.items()):
        idx = canon_to_idx[canon]
        lines.append(f'    [{idx:2d}] {_ISO_NAMES[canon]}  (|класс|={len(elems)})')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='k4atlas — Атлас подграфов K₄ через глифы Q6'
    )
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('atlas', help='Полный атлас всех 11 классов изоморфизма')
    p_show = sub.add_parser('show', help='Описание конкретного подграфа')
    p_show.add_argument('h', type=int, help='Номер глифа (0..63)')
    sub.add_parser('hasse', help='Диаграмма Хассе с классами')
    sub.add_parser('stats', help='Сводная статистика')

    for p in [sub.choices.get('atlas'), sub.choices.get('hasse'),
              sub.choices.get('stats'), sub.choices.get('show')]:
        if p:
            p.add_argument('--no-color', action='store_true')

    args = parser.parse_args()
    color = not getattr(args, 'no_color', False)

    if args.cmd == 'atlas' or args.cmd is None:
        print(render_atlas(color=color))

    elif args.cmd == 'show':
        print(describe_subgraph(args.h, color=color))

    elif args.cmd == 'hasse':
        print(render_hasse_by_class(color=color))

    elif args.cmd == 'stats':
        print('Статистика K₄-атласа:')
        print(f'  Всего подграфов: 64')
        print(f'  Классов изоморфизма: {len(_ISO_CLASSES)}')
        print()
        print(f'  {"Класс":<30} {"Размер":>6}  {"Канон":>6}  Степени')
        print('  ' + '─' * 56)
        for canon in sorted(_ISO_CLASSES):
            elems = _ISO_CLASSES[canon]
            name = _ISO_NAMES[canon]
            ds = degree_sequence(canon)
            print(f'  {name:<30} {len(elems):>6}  {canon:>6}  {ds}')
        print()
        # Проверка: сумма размеров = 64
        total = sum(len(e) for e in _ISO_CLASSES.values())
        print(f'  Сумма: {total} = 64 ✓' if total == 64 else f'  ОШИБКА: {total} ≠ 64')
