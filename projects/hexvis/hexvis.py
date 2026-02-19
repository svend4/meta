"""
hexvis — визуализация Q6: ASCII-арт, DOT/Graphviz, SVG

Форматы вывода:
  ascii   — текстовая решётка 8×8 всех 64 гексаграмм с цветовым кодированием
  path    — вывод пути через Q6 в виде последовательности гексаграмм
  dot     — Graphviz DOT для произвольного подграфа/пути
  svg     — SVG-диаграмма пути или подграфа

Цветовой код (ANSI, 256-color):
  yang=0 → тёмно-серый, yang=6 → яркий желтый
"""

from __future__ import annotations
import sys
from typing import Sequence

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    neighbors, hamming, yang_count, to_bits, render, SIZE,
    gray_code, antipode, shortest_path,
)


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_RESET = '\033[0m'
_BOLD = '\033[1m'

# yang 0..6 → foreground colour
_YANG_ANSI = [
    '\033[38;5;238m',   # 0  very dark grey
    '\033[38;5;27m',    # 1  blue
    '\033[38;5;39m',    # 2  light blue
    '\033[38;5;82m',    # 3  green
    '\033[38;5;208m',   # 4  orange
    '\033[38;5;196m',   # 5  red
    '\033[38;5;226m',   # 6  bright yellow
]

_YANG_BG = [
    '\033[48;5;238m',
    '\033[48;5;27m',
    '\033[48;5;39m',
    '\033[48;5;82m',
    '\033[48;5;208m',
    '\033[48;5;196m',
    '\033[48;5;226m',
]


def _ansi_hex(h: int, highlight: bool = False, bg: bool = False) -> str:
    """Раскрасить номер гексаграммы по ян-счёту."""
    c = _YANG_BG[yang_count(h)] if bg else _YANG_ANSI[yang_count(h)]
    bold = _BOLD if highlight else ''
    return f"{c}{bold}{h:2d}{_RESET}"


# ---------------------------------------------------------------------------
# ASCII-решётка 8×8
# ---------------------------------------------------------------------------

# Gray-код для 3 бит: [0,1,3,2,6,7,5,4]
_GRAY3 = [i ^ (i >> 1) for i in range(8)]
_GRAY3_LABELS = [format(g, '03b') for g in _GRAY3]


def render_grid(
    highlights: set[int] | None = None,
    labels: dict[int, str] | None = None,
    color: bool = True,
    title: str = '',
) -> str:
    """
    8×8 сетка всех 64 гексаграмм.
    Строки = x5x4x3 (Gray-код), столбцы = x2x1x0 (Gray-код).
    Соседние клетки по вертикали/горизонтали отличаются ровно 1 битом.

    highlights : множество гексаграмм для выделения
    labels     : {hexagram: метка} для подписей
    """
    if highlights is None:
        highlights = set()

    lines = []
    if title:
        lines.append(f"  {title}")

    # Заголовок столбцов (x2x1x0)
    header = '      ' + '  '.join(f'{lb}' for lb in _GRAY3_LABELS)
    lines.append(header)
    lines.append('      ' + '─' * (len(_GRAY3_LABELS) * 4 - 1))

    for row_g in _GRAY3:
        row_label = format(row_g, '03b')
        cells = []
        for col_g in _GRAY3:
            # x5x4x3 = row_g (биты 5,4,3), x2x1x0 = col_g (биты 2,1,0)
            h = (row_g << 3) | col_g
            if h in highlights:
                if color:
                    cells.append(_ansi_hex(h, highlight=True, bg=True))
                else:
                    cells.append(f'*{h:02d}')
            else:
                if color:
                    cells.append(_ansi_hex(h))
                else:
                    cells.append(f' {h:02d}')
        lines.append(f'  {row_label} │ ' + '  '.join(cells))

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Визуализация пути
# ---------------------------------------------------------------------------

def render_path(
    path: list[int],
    color: bool = True,
    show_bits: bool = False,
    show_yang: bool = True,
) -> str:
    """
    Текстовый вывод пути через Q6.
    Показывает каждый шаг с гексаграммой, изменённым битом и ян-счётом.
    """
    if not path:
        return '(пустой путь)'

    lines = [f"Путь длиной {len(path) - 1} шагов:"]
    for i, h in enumerate(path):
        yc = yang_count(h)
        prefix = f"  [{i:2d}] "
        hex_str = f"{h:2d}"
        if color:
            hex_str = _ansi_hex(h, highlight=(i == 0 or i == len(path) - 1))

        yang_str = f"  ян={yc}" if show_yang else ''
        bits_str = f"  {to_bits(h)}" if show_bits else ''

        line = f"{prefix}{hex_str}{yang_str}{bits_str}"

        if i < len(path) - 1:
            next_h = path[i + 1]
            diff = h ^ next_h
            changed_bit = diff.bit_length() - 1
            direction = '↑' if (next_h >> changed_bit) & 1 else '↓'
            line += f"  →(бит {changed_bit} {direction})"

        lines.append(line)

    lines.append(f"  Расстояние: {len(path) - 1}")
    if len(path) >= 2:
        lines.append(f"  Хэмминг(start, end) = {hamming(path[0], path[-1])}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# DOT (Graphviz)
# ---------------------------------------------------------------------------

_YANG_COLORS_DOT = [
    '#555555',  # 0
    '#1155BB',  # 1
    '#2299DD',  # 2
    '#22BB44',  # 3
    '#FF8800',  # 4
    '#DD2222',  # 5
    '#FFDD00',  # 6
]


def to_dot(
    vertices: set[int],
    edges: set[tuple[int, int]] | None = None,
    highlights: set[int] | None = None,
    path: list[int] | None = None,
    title: str = 'Q6 subgraph',
    directed: bool = False,
) -> str:
    """
    Сгенерировать Graphviz DOT для подграфа Q6.

    vertices   : множество вершин
    edges      : рёбра (если None — все Q6-рёбра между vertices)
    highlights : выделенные вершины
    path       : путь (рёбра выделяются жирным)
    directed   : если True — ориентированный граф
    """
    highlights = highlights or set()
    if edges is None:
        edges = set()
        for u in vertices:
            for v in neighbors(u):
                if v in vertices and u < v:
                    edges.add((u, v))

    path_edges: set[tuple[int, int]] = set()
    if path:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            path_edges.add((min(a, b), max(a, b)) if not directed else (a, b))

    gtype = 'digraph' if directed else 'graph'
    sep = '->' if directed else '--'

    lines = [
        f'{gtype} "{title}" {{',
        '  graph [rankdir=LR, fontname="monospace"];',
        '  node [shape=circle, fontname="monospace", fontsize=10, width=0.4];',
        '  edge [fontname="monospace", fontsize=8];',
        '',
    ]

    for v in sorted(vertices):
        yc = yang_count(v)
        color = _YANG_COLORS_DOT[yc]
        style = 'filled'
        penwidth = '3' if v in highlights else '1'
        border = '#000000' if v in highlights else color
        label = to_bits(v)
        lines.append(
            f'  {v} [label="{v}\\n{label}", style={style}, '
            f'fillcolor="{color}", color="{border}", penwidth={penwidth}];'
        )

    lines.append('')

    for (u, v) in sorted(edges):
        is_path_edge = (u, v) in path_edges or (v, u) in path_edges
        changed_bit = (u ^ v).bit_length() - 1
        style = 'bold' if is_path_edge else 'solid'
        color = '#000000' if is_path_edge else '#999999'
        penwidth = '2.5' if is_path_edge else '1'
        lines.append(
            f'  {u} {sep} {v} [label="b{changed_bit}", style={style}, '
            f'color="{color}", penwidth={penwidth}];'
        )

    lines.append('}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# SVG (standalone, без зависимостей)
# ---------------------------------------------------------------------------

_SVG_YANG_COLORS = [
    '#888888',  # 0
    '#2255CC',  # 1
    '#22AADD',  # 2
    '#33BB55',  # 3
    '#FF8800',  # 4
    '#EE3333',  # 5
    '#FFCC00',  # 6
]


def _positions_grid(vertices: list[int]) -> dict[int, tuple[float, float]]:
    """Позиции вершин по 8×8 решётке Q6."""
    pos = {}
    for h in vertices:
        col = _GRAY3.index(h & 0b111)
        row = _GRAY3.index((h >> 3) & 0b111)
        pos[h] = (60 + col * 70, 60 + row * 70)
    return pos


def _positions_circle(vertices: list[int]) -> dict[int, tuple[float, float]]:
    """Позиции вершин по окружности."""
    import math
    n = len(vertices)
    pos = {}
    for i, h in enumerate(sorted(vertices)):
        angle = 2 * math.pi * i / n - math.pi / 2
        r = min(240, 40 * n / (2 * math.pi) + 40)
        cx, cy = 300, 300
        pos[h] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
    return pos


def to_svg(
    vertices: set[int],
    edges: set[tuple[int, int]] | None = None,
    highlights: set[int] | None = None,
    path: list[int] | None = None,
    title: str = 'Q6 subgraph',
    layout: str = 'grid',
    width: int = 640,
    height: int = 640,
) -> str:
    """
    Сгенерировать SVG для подграфа Q6.

    layout: 'grid' (8×8) или 'circle' (по окружности)
    """
    highlights = highlights or set()
    if edges is None:
        edges = set()
        for u in vertices:
            for v in neighbors(u):
                if v in vertices and u < v:
                    edges.add((u, v))

    path_edges: set[tuple[int, int]] = set()
    if path:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            path_edges.add((min(a, b), max(a, b)))

    verts_list = list(vertices)
    if layout == 'grid':
        pos = _positions_grid(verts_list)
        w, h = 640, 640
    else:
        pos = _positions_circle(verts_list)
        w, h = 640, 640
    width, height = w, h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'  <title>{title}</title>',
        f'  <rect width="{width}" height="{height}" fill="#1a1a2e"/>',
    ]

    # Рёбра
    for (u, v) in sorted(edges):
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        is_path = (u, v) in path_edges
        stroke = '#ffffff' if is_path else '#444466'
        sw = '2.5' if is_path else '0.8'
        lines.append(
            f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    # Вершины
    for h in sorted(vertices):
        if h not in pos:
            continue
        x, y = pos[h]
        yc = yang_count(h)
        fill = _SVG_YANG_COLORS[yc]
        stroke = '#ffffff' if h in highlights else '#000000'
        sw = '3' if h in highlights else '1'
        r = 16 if h in highlights else 13

        lines.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        label_color = '#000000' if yc in (3, 4, 6) else '#ffffff'
        lines.append(
            f'  <text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            f'font-family="monospace" font-size="10" fill="{label_color}">{h}</text>'
        )

    # Путь: нумерация шагов
    if path:
        for i, h in enumerate(path):
            if h not in pos:
                continue
            x, y = pos[h]
            lines.append(
                f'  <text x="{x + 14:.1f}" y="{y - 10:.1f}" font-family="monospace" '
                f'font-size="9" fill="#aaaaff">{i}</text>'
            )

    # Заголовок
    lines.append(
        f'  <text x="10" y="20" font-family="monospace" font-size="12" '
        f'fill="#aaaacc">{title}</text>'
    )

    lines.append('</svg>')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Glyph encoding: each Q6 element as a 3×3 ASCII icon (6 line segments)
# ---------------------------------------------------------------------------
#
# Bit → segment mapping (all 64 elements produce distinct glyphs):
#   bit 0  top bar        row 0 centre = '_'
#   bit 1  bottom bar     row 2 centre = '_'
#   bit 2  left bar       row 1 left   = '|'
#   bit 3  right bar      row 1 right  = '|'
#   bit 4  '\' diagonal   row 1 centre = '\\'
#   bit 5  '/' diagonal   row 1 centre = '/'   (both set → 'X')


def render_glyph(h: int) -> list[str]:
    """Return three 3-char strings representing h as a line-segment glyph.

    Each of the 6 bits controls one visible segment in a small square frame:
      bit 0 = top bar, bit 1 = bottom bar,
      bit 2 = left bar, bit 3 = right bar,
      bit 4 = backslash diagonal, bit 5 = slash diagonal.
    All 64 values produce distinct glyphs.
    """
    b = [(h >> i) & 1 for i in range(6)]
    row0 = ' ' + ('_' if b[0] else ' ') + ' '
    dc = ('X' if b[4] and b[5] else '\\' if b[4] else '/' if b[5] else ' ')
    row1 = ('|' if b[2] else ' ') + dc + ('|' if b[3] else ' ')
    row2 = ' ' + ('_' if b[1] else ' ') + ' '
    return [row0, row1, row2]


def render_hasse_glyphs(
    color: bool = True,
    show_numbers: bool = False,
    highlights: set[int] | None = None,
) -> str:
    """Hasse diagram of B₆ as an isosceles triangle of 3×3 glyphs.

    Rank 0 (element 0, no segments) sits at the top; rank 6 (element 63,
    all segments) at the bottom.  Elements within each rank are sorted
    numerically.  Each glyph is 3 chars wide; ranks are centred relative
    to the widest row (rank 3, C(6,3)=20 elements).

    Parameters
    ----------
    color       : colour glyphs by yang-count (ANSI 256-colour).
    show_numbers: prepend a row of decimal indices above each rank band.
    highlights  : elements to render with background colour.
    """
    highlights = highlights or set()

    rank_elems: list[list[int]] = [[] for _ in range(7)]
    for h in range(64):
        rank_elems[bin(h).count('1')].append(h)

    cw = 3          # cell width (chars)
    sw = 1          # separator width (1 space)
    max_n = 20      # C(6,3) — widest rank
    total_w = max_n * cw + (max_n - 1) * sw   # = 79

    lines: list[str] = []

    for k, elems in enumerate(rank_elems):
        n = len(elems)
        row_w = n * cw + (n - 1) * sw
        pad = ' ' * ((total_w - row_w) // 2)

        if show_numbers:
            lines.append(pad + ' '.join(f'{h:3d}' for h in elems))

        glyphs = [render_glyph(h) for h in elems]

        for ri in range(3):
            parts: list[str] = []
            for gi, h in enumerate(elems):
                cell = glyphs[gi][ri]
                if color:
                    yc = yang_count(h)
                    ansi = (_YANG_BG[yc] + _BOLD) if h in highlights else _YANG_ANSI[yc]
                    cell = ansi + cell + _RESET
                parts.append(cell)
            lines.append(pad + ' '.join(parts))

        lines.append('')   # blank line between ranks

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Специализированные диаграммы
# ---------------------------------------------------------------------------

def render_hexagram(h: int, color: bool = True) -> str:
    """Гексаграмма в виде ASCII-арт символа с номером и битами."""
    bits = to_bits(h)
    yc = yang_count(h)
    col = _YANG_ANSI[yc] if color else ''
    rst = _RESET if color else ''
    lines_hex = render(h).split('\n')
    header = f"{col}#{h:02d}  {bits}  ян={yc}{rst}"
    return header + '\n' + '\n'.join(f"  {l}" for l in lines_hex)


def render_transition(a: int, b: int, color: bool = True) -> str:
    """Переход a → b: два символа гексаграммы рядом с указанием изменённого бита."""
    if hamming(a, b) != 1:
        raise ValueError(f"a={a} и b={b} не являются соседями в Q6")
    changed = (a ^ b).bit_length() - 1
    bits_a = to_bits(a)
    bits_b = to_bits(b)

    # Выделить изменённый бит в строках
    def mark_bit(bits: str, bit: int, is_yang: bool) -> str:
        pos = 5 - bit   # to_bits: бит 5 слева
        if color:
            mark = _BOLD + _YANG_ANSI[1] if is_yang else _BOLD + _YANG_ANSI[4]
            return bits[:pos] + mark + bits[pos] + _RESET + bits[pos + 1:]
        return bits[:pos] + '[' + bits[pos] + ']' + bits[pos + 1:]

    bits_a_str = mark_bit(bits_a, changed, bool((a >> changed) & 1))
    bits_b_str = mark_bit(bits_b, changed, bool((b >> changed) & 1))

    col_a = _YANG_ANSI[yang_count(a)] if color else ''
    col_b = _YANG_ANSI[yang_count(b)] if color else ''
    rst = _RESET if color else ''

    return (
        f"{col_a}#{a:02d} {bits_a_str}{rst}  →  "
        f"{col_b}#{b:02d} {bits_b_str}{rst}  "
        f"(бит {changed} {'↑' if (b >> changed) & 1 else '↓'})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='hexvis — визуализация Q6')
    sub = parser.add_subparsers(dest='cmd')

    p_grid = sub.add_parser('grid', help='Показать 8×8 решётку Q6')
    p_grid.add_argument('--highlight', nargs='*', type=int, default=[],
                        help='Гексаграммы для выделения')
    p_grid.add_argument('--no-color', action='store_true')

    p_path = sub.add_parser('path', help='Показать путь')
    p_path.add_argument('nodes', nargs='+', type=int, help='Узлы пути')
    p_path.add_argument('--bits', action='store_true')
    p_path.add_argument('--no-color', action='store_true')
    p_path.add_argument('--grid', action='store_true', help='Показать путь на решётке')

    p_auto = sub.add_parser('auto', help='Кратчайший путь start → goal')
    p_auto.add_argument('start', type=int)
    p_auto.add_argument('goal', type=int)
    p_auto.add_argument('--grid', action='store_true')
    p_auto.add_argument('--dot', action='store_true', help='Вывод в DOT')
    p_auto.add_argument('--svg', metavar='FILE', help='Сохранить SVG')

    p_hasse = sub.add_parser('hasse', help='Диаграмма Хассе B₆ как треугольник глифов')
    p_hasse.add_argument('--highlight', nargs='*', type=int, default=[],
                         help='Гексаграммы для выделения')
    p_hasse.add_argument('--numbers', action='store_true',
                         help='Показать номера над глифами')
    p_hasse.add_argument('--no-color', action='store_true')

    p_hex = sub.add_parser('hexagram', help='Показать гексаграмму')
    p_hex.add_argument('h', type=int)
    p_hex.add_argument('--no-color', action='store_true')

    p_dot = sub.add_parser('dot', help='DOT вывод подграфа')
    p_dot.add_argument('vertices', nargs='+', type=int)
    p_dot.add_argument('--out', metavar='FILE', help='Файл вывода')

    p_svg_cmd = sub.add_parser('svg', help='SVG вывод подграфа')
    p_svg_cmd.add_argument('vertices', nargs='+', type=int)
    p_svg_cmd.add_argument('--out', metavar='FILE', default='hexvis.svg')
    p_svg_cmd.add_argument('--layout', choices=['grid', 'circle'], default='grid')
    p_svg_cmd.add_argument('--path', nargs='*', type=int)

    args = parser.parse_args()

    if args.cmd == 'hasse':
        print(render_hasse_glyphs(
            color=not args.no_color,
            show_numbers=args.numbers,
            highlights=set(args.highlight),
        ))

    elif args.cmd == 'grid':
        print(render_grid(
            highlights=set(args.highlight),
            color=not args.no_color,
            title='Q6 — 64 гексаграммы (8×8 в коде Грея)',
        ))

    elif args.cmd == 'hexagram':
        print(render_hexagram(args.h, color=not args.no_color))

    elif args.cmd == 'path':
        path = args.nodes
        color = not args.no_color
        print(render_path(path, color=color, show_bits=args.bits))
        if args.grid:
            print()
            print(render_grid(highlights=set(path), color=color,
                              title='Путь на Q6'))

    elif args.cmd == 'auto':
        path = shortest_path(args.start, args.goal)
        color = True
        print(render_path(path, color=color))
        if args.grid:
            print()
            print(render_grid(highlights=set(path), color=color,
                              title=f'Путь {args.start}→{args.goal}'))
        if args.dot:
            print()
            all_verts = set(range(SIZE))
            print(to_dot(set(path), path=path, highlights={args.start, args.goal},
                         title=f'Path {args.start}→{args.goal}'))
        if args.svg:
            svg = to_svg(set(path), path=path,
                         highlights={args.start, args.goal},
                         title=f'Path {args.start}→{args.goal}',
                         layout='circle')
            with open(args.svg, 'w') as f:
                f.write(svg)
            print(f"SVG сохранён: {args.svg}")

    elif args.cmd == 'dot':
        verts = set(args.vertices)
        print(to_dot(verts, title='Q6 subgraph'))

    elif args.cmd == 'svg':
        verts = set(args.vertices)
        path = args.path or None
        svg = to_svg(verts, path=path, layout=args.layout,
                     title='Q6 subgraph')
        with open(args.out, 'w') as f:
            f.write(svg)
        print(f"SVG сохранён: {args.out} ({len(verts)} вершин)")

    else:
        parser.print_help()
