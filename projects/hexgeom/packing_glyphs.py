"""packing_glyphs — Упаковка шаров Хэмминга на Q6 через глифы.

Каждый глиф (0..63) — вершина Q6 = (Z₂)⁶.
Расстояние Хэмминга d(x, y) = popcount(x ⊕ y).

Упаковка и покрытие:
  • r-упаковка: множество с попарными расстояниями > 2r
    (шары радиуса r не пересекаются)
  • r-покрытие: каждая вершина Q6 в шаре радиуса r от некоторого центра
  • Совершенный код: 1-упаковка + 1-покрытие одновременно

Ключевые факты для Q6 (n=6):
  • Граница Хэмминга: |C| ≤ 2^6 / V(6,t) где t = ⌊(d−1)/2⌋
  • Для d=3 (1-исправляющий код): |C| ≤ 64/7 ≈ 9.1 → максимум 9
  • Совершенного 1-кода не существует (64 не делится на 7)
  • Для d=3 на Q6: лучшие коды имеют 9-10 кодовых слов

Визуализация:
  • packing  — 1-упаковка на Q6: глифы-центры ярко, остальные тёмно
  • voronoi  — диаграмма Вороного для набора центров
  • balls    — шары Хэмминга из центра r=0..3
  • bound    — граница Хэмминга vs фактические коды

Команды CLI:
  packing [--radius r]  — жадная упаковка радиуса r
  voronoi <centers...>  — диаграмма Вороного
  balls   <center> [r]  — шар Хэмминга из центра
  bound               — таблица границ для Q6
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexgeom.hexgeom import (
    hamming, hamming_ball, hamming_sphere, ball_size, sphere_size,
    packing_number, voronoi_cells, is_perfect_code, distance_distribution,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# 1. Упаковка шаров Хэмминга
# ---------------------------------------------------------------------------

def render_packing(radius: int = 1, color: bool = True) -> str:
    """
    8×8 сетка глифов: центры упаковки ярко, покрытые — их цветом, остальные тёмно.
    """
    centers = packing_number(radius)
    centers_set = set(centers)

    # Ближайший центр для каждой вершины
    nearest: dict[int, int] = {}
    for h in range(64):
        best = min(centers, key=lambda c: hamming(h, c))
        nearest[h] = best

    # Цвета центров по их yang_count
    center_colors = {}
    for i, c in enumerate(centers):
        center_colors[c] = _YANG_ANSI[yang_count(c)]

    is_perfect = is_perfect_code(centers, radius)
    bound = 64 // ball_size(radius)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Упаковка шаров радиуса r={radius} на Q6')
    lines.append(f'  Найдено центров: {len(centers)}   '
                 f'Граница Хэмминга ≤ {bound}   '
                 f'{"СОВЕРШЕННЫЙ КОД!" if is_perfect else "не совершенный"}')
    lines.append(f'  |B(c,{radius})| = {ball_size(radius)}   '
                 f'{len(centers)} × {ball_size(radius)} = {len(centers)*ball_size(radius)} / 64')
    lines.append('  Яркий глиф = центр упаковки')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                if h in centers_set:
                    yc = yang_count(h)
                    c_str = _YANG_BG[yc] + _BOLD
                else:
                    nc = nearest[h]
                    dist = hamming(h, nc)
                    if dist <= radius:
                        yc = yang_count(nc)
                        c_str = _YANG_ANSI[yc]
                    else:
                        c_str = _YANG_ANSI[0]  # тёмный — вне зоны покрытия
                rows3 = [c_str + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            nc = nearest[h]
            d = hamming(h, nc)
            is_c = h in centers_set
            if color:
                if is_c:
                    c_str = _YANG_ANSI[5]
                    lbl.append(f'{c_str}C:{h:02d}{_RESET}')
                else:
                    c_str = _YANG_ANSI[max(0, radius + 1 - d)]
                    lbl.append(f'{c_str}d={d}{_RESET}')
            else:
                lbl.append(f'{"C" if is_c else "d"}={d if not is_c else h}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    lines.append(f'  Центры упаковки: {sorted(centers)}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Диаграмма Вороного
# ---------------------------------------------------------------------------

def render_voronoi(centers: list[int], color: bool = True) -> str:
    """
    8×8 сетка: каждый глиф раскрашен по ближайшему центру Вороного.
    """
    if not centers:
        return '  Нет центров Вороного.'
    cells = voronoi_cells(centers)
    # cells: dict center → frozenset

    # Цвет для каждого центра по его yang_count
    center_to_color = {}
    for i, c in enumerate(centers):
        center_to_color[c] = yang_count(c)

    # Для каждой вершины — её центр
    vertex_to_center = {}
    for c, cell in cells.items():
        for h in cell:
            vertex_to_center[h] = c

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Диаграмма Вороного (метрика Хэмминга) на Q6')
    lines.append(f'  Центры: {sorted(centers)}  ({len(centers)} областей)')
    lines.append('  Цвет глифа = область Вороного ближайшего центра')
    lines.append('═' * 64)
    lines.append('')

    for i, c in enumerate(centers):
        cell_size = len(cells[c])
        if color:
            cc = _YANG_ANSI[center_to_color[c]]
            lines.append(f'  {cc}Центр {c:2d} (yang={yang_count(c)}): '
                         f'{cell_size} вершин{_RESET}')
        else:
            lines.append(f'  Центр {c:2d}: {cell_size} вершин')
    lines.append('')

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                nc = vertex_to_center.get(h, centers[0])
                yc = center_to_color[nc]
                is_center = (h in centers)
                c_str = (_YANG_BG[yc] + _BOLD) if is_center else _YANG_ANSI[yc]
                rows3 = [c_str + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            nc = vertex_to_center.get(h, centers[0])
            d = hamming(h, nc)
            if color:
                yc = center_to_color[nc]
                c_str = _YANG_ANSI[yc]
                lbl.append(f'{c_str}→{nc:02d}d{d}{_RESET}')
            else:
                lbl.append(f'→{nc:02d}d{d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Шары Хэмминга
# ---------------------------------------------------------------------------

def render_balls(center: int, max_r: int = 3, color: bool = True) -> str:
    """
    Показать шары B(center, 0), B(center, 1), ..., B(center, max_r).

    Каждый шар = все вершины Q6 на расстоянии ≤ r от центра.
    """
    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Шары Хэмминга из центра h={center} '
                 f'(yang={yang_count(center)}, bits={format(center,"06b")})')
    lines.append('═' * 64)

    for r in range(max_r + 1):
        sphere_r = hamming_sphere(center, r)
        ball_r = hamming_ball(center, r)
        lines.append(f'\n  B({center},{r}): |шар|={len(ball_r)}  '
                     f'|сфера S({center},{r})|={len(sphere_r)}  '
                     f'C(6,{r})={math.comb(6,r)}')

        glyphs = [render_glyph(h) for h in sorted(sphere_r)]
        if color:
            yc = r  # расстояние = слой
            c_str = _YANG_BG[yc] + _BOLD if r == 0 else _YANG_ANSI[min(yc, 6)]
            glyphs = [[c_str + row + _RESET for row in g] for g in glyphs]

        # Выводим в строку (до 16 глифов)
        chunk = glyphs[:16]
        for ri in range(3):
            lines.append('    ' + '  '.join(g[ri] for g in chunk))
        nums = [
            (_YANG_ANSI[min(r, 6)] + f'{h:02d}' + _RESET if color else f'{h:02d}')
            for h in sorted(sphere_r)[:16]
        ]
        lines.append('    ' + '  '.join(nums))

    lines.append('')
    lines.append('  Спектр расстояний от центра:')
    lines.append('  ' + ', '.join(
        f'S({r})={sphere_size(r)}' for r in range(7)
    ))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Граница Хэмминга для Q6
# ---------------------------------------------------------------------------

def render_bounds(color: bool = True) -> str:
    """
    Таблица: для каждого d (минимального расстояния),
    граница Хэмминга и лучшие известные коды на Q6 (n=6).
    """
    # Лучшие известные [6, k, d]₂ коды
    # (n=6, binary)
    known = {
        1: {'k': 6, 'size': 64, 'name': 'trivial'},
        2: {'k': 5, 'size': 32, 'name': '[6,5,2] even-weight'},
        3: {'k': 3, 'size':  8, 'name': '[6,3,4] dual Hamming'},  # d=3→4 actually
        4: {'k': 3, 'size':  8, 'name': '[6,3,4]₂'},
        5: {'k': 1, 'size':  2, 'name': '[6,1,6] repetition'},
        6: {'k': 1, 'size':  2, 'name': '[6,1,6]'},
    }

    lines: list[str] = []
    lines.append('╔' + '═' * 62 + '╗')
    lines.append('║  Границы кодирования для Q6 = (n=6 бит)' + ' ' * 22 + '║')
    lines.append('║  Граница Хэмминга: |C| ≤ 2^6 / V(6, ⌊(d−1)/2⌋)' + ' ' * 13 + '║')
    lines.append('╚' + '═' * 62 + '╝')
    lines.append('')
    lines.append(f'  {"d":>3}  {"t=⌊(d-1)/2⌋":>12}  {"V(6,t)":>8}  '
                 f'{"Граница":>8}  {"Лучший код":>10}  Название')
    lines.append('  ' + '─' * 60)

    for d in range(1, 7):
        t = (d - 1) // 2
        v = ball_size(t)
        bound = 64 // v
        info = known.get(d, {'size': '?', 'name': '?'})
        size = info['size']
        name = info['name']

        # Совершенный код: size == bound
        is_perfect = (isinstance(size, int) and size == bound)

        if color:
            c = _YANG_ANSI[d - 1]
            pfx = _BOLD if is_perfect else ''
            lines.append(
                f'  {c}{pfx}{d:>3}  {t:>12}  {v:>8}  '
                f'{bound:>8}  {str(size):>10}  {name}{_RESET}'
            )
        else:
            lines.append(
                f'  {d:>3}  {t:>12}  {v:>8}  '
                f'{bound:>8}  {str(size):>10}  {name}'
            )

    lines.append('')
    lines.append('  V(6,t) = Σ_{k=0}^{t} C(6,k) — объём шара радиуса t')
    lines.append('  Совершенный 1-код: 64/7 ≈ 9.1 — не целое → не существует!')
    lines.append('')
    lines.append('  Шары Хэмминга: |B(h,r)| = V(6,r):')
    for r in range(7):
        v = ball_size(r)
        if color:
            c = _YANG_ANSI[r]
            lines.append(f'  {c}  r={r}: V(6,{r}) = {v}{_RESET}')
        else:
            lines.append(f'    r={r}: V(6,{r}) = {v}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='packing_glyphs',
        description='Упаковка шаров Хэмминга на Q6 через глифы',
    )
    p.add_argument('--no-color', action='store_true', help='без ANSI-цветов')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('packing', help='жадная упаковка на Q6')
    s.add_argument('--radius', '-r', type=int, default=1, help='радиус шаров')

    s = sub.add_parser('voronoi', help='диаграмма Вороного для центров')
    s.add_argument('centers', type=int, nargs='+', help='центры (целые 0..63)')

    s = sub.add_parser('balls', help='шары Хэмминга из центра')
    s.add_argument('center', type=int, help='центр (0..63)')
    s.add_argument('--max-r', type=int, default=3, help='максимальный радиус')

    sub.add_parser('bound', help='таблица границ Хэмминга для Q6')

    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'packing':
        print(render_packing(radius=args.radius, color=color))
    elif args.cmd == 'voronoi':
        print(render_voronoi(args.centers, color=color))
    elif args.cmd == 'balls':
        print(render_balls(args.center, max_r=args.max_r, color=color))
    elif args.cmd == 'bound':
        print(render_bounds(color=color))


if __name__ == '__main__':
    main()
