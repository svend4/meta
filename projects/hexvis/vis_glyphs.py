"""hexvis/vis_glyphs.py — мета-визуализация: витрина возможностей hexvis.

hexvis — базовая библиотека визуализации Q6. Этот модуль демонстрирует
все режимы визуализации на одном месте: глифы, решётки, диаграммы Хассе, пути.

Каждый глиф h ∈ {0,...,63} определяется своими 6 битами:
    bit 0 = верхняя черта      bit 1 = нижняя черта
    bit 2 = левая черта        bit 3 = правая черта
    bit 4 = диагональ ╲        bit 5 = диагональ ╱

Пример глифа h=21 (010101):
    bits: 1,0,1,0,1,0 → верх + лев + диаг.╲
    ┌ _   ┐
    │\    │
    └     ┘

Диаграмма Хассе B₆: ян=0 (h=0) вверху, ян=6 (h=63) внизу.
8×8 решётка Gray-кода: соседние ячейки отличаются ровно на 1 бит.

Визуализация:
  glyph   [--h n]          — ASCII-арт конкретного глифа + все соседи
  grid    [--highlights hs] — стандартная 8×8 решётка с выделением
  hasse   [--numbers]      — диаграмма Хассе B₆ (ян-уровни 0..6)
  path    [--start s --end e] — путь по Q6 + визуализация

Команды CLI:
  glyph  [--h n]
  grid   [--highlights h1,h2,...]
  hasse  [--numbers]
  path   [--start s] [--end e]
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    yang_count, hamming, neighbors, to_bits, antipode, shortest_path,
)
from projects.hexvis.hexvis import (
    render_glyph,
    render_grid,
    render_hasse_glyphs,
    render_path,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

_GRAY3 = [i ^ (i >> 1) for i in range(8)]


def _header(title: str, subtitle: str = '') -> list[str]:
    lines = ['═' * 66, f'  {title}']
    if subtitle:
        lines.append(f'  {subtitle}')
    lines.append('═' * 66)
    return lines


# ---------------------------------------------------------------------------
# 1. ASCII-арт одного глифа
# ---------------------------------------------------------------------------

def render_glyph_view(h: int = 21, color: bool = True) -> str:
    """Детальный показ одного глифа h: ASCII-арт + биты + соседи.

    Формат глифа (3×3 символа):
      row0: ' ' + ('_' if bit0) + ' '
      row1: ('|' if bit2) + диаг + ('|' if bit3)
      row2: ' ' + ('_' if bit1) + ' '

    Диагональ: 'X' if bit4 and bit5, '\\' if bit4, '/' if bit5, ' ' else.
    """
    rows = render_glyph(h)
    bits = [(h >> i) & 1 for i in range(6)]
    nb   = sorted(neighbors(h))
    yc   = yang_count(h)
    ant  = antipode(h)

    lines = _header(
        f'Глиф h={h} ({to_bits(h)})  ян={yc}',
        f'Соседи: {nb}  Антипод: {ant} ({to_bits(ant)})',
    )
    lines.append('')

    # ASCII-арт глифа (3 строки, растянуто)
    c = _YANG_ANSI[yc] if color else ''
    r = _RESET if color else ''
    for i, row in enumerate(rows):
        lines.append(f'  {c}{row * 3}{r}')
    lines.append('')

    # Биты
    bit_names = ['верх', 'низ ', 'лев ', 'прав', 'диаг╲', 'диаг╱']
    lines.append('  Биты:')
    for i in range(6):
        bv = bits[i]
        bc = _YANG_ANSI[bv * 6] if color else ''
        lines.append(f'    bit{i} ({bit_names[i]}): {bc}{bv}{_RESET if color else ""}')

    lines.append('')
    lines.append(f'  Ян-уровень: {yc}  (из 6 битов {yc} единиц)')
    lines.append(f'  Антипод:    {ant} ({to_bits(ant)})  расстояние = 6')
    lines.append('')

    # Соседи с ASCII-артом
    lines.append('  Соседи (6 рёбер Q6):')
    for nb_h in nb:
        nb_rows = render_glyph(nb_h)
        diff    = h ^ nb_h
        bit_flipped = diff.bit_length() - 1
        cn = _YANG_ANSI[yang_count(nb_h)] if color else ''
        lines.append(
            f'    h={nb_h:2d} ({to_bits(nb_h)}) ян={yang_count(nb_h)}'
            f'  flip bit{bit_flipped}: '
            f'{cn}{nb_rows[0]}│{nb_rows[1]}│{nb_rows[2]}{_RESET if color else ""}'
        )

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Стандартная 8×8 решётка
# ---------------------------------------------------------------------------

def render_grid_view(highlights: set[int] | None = None,
                     color: bool = True) -> str:
    """Полная 8×8 решётка Q6 с опциональным выделением."""
    highlights = highlights or set()

    lines = _header(
        'Q6: стандартная 8×8 решётка (Gray-код)',
        'Строки = x₅x₄x₃ (Gray)  Столбцы = x₂x₁x₀ (Gray)',
    )
    lines.append('')
    lines.append(render_grid(highlights=highlights, color=color,
                              title='Все 64 гексаграммы Q6'))
    lines.append('')

    # Статистика
    lines.append('  Ян-распределение (C(6,k)):')
    for k in range(7):
        cnt = sum(1 for h in range(64) if yang_count(h) == k)
        bar = '█' * cnt
        c   = _YANG_ANSI[k] if color else ''
        r   = _RESET if color else ''
        lines.append(f'  ян={k}: {c}{bar} {cnt}{r}')

    if highlights:
        lines.append('')
        lines.append(f'  Выделено: {len(highlights)} глифов')
        lines.append(f'  Ян-распределение выделения:')
        for k in range(7):
            cnt = sum(1 for h in highlights if yang_count(h) == k)
            if cnt:
                c = _YANG_ANSI[k] if color else ''
                r = _RESET if color else ''
                lines.append(f'    ян={k}: {c}{cnt}{r}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Диаграмма Хассе B₆
# ---------------------------------------------------------------------------

def render_hasse_view(show_numbers: bool = False, color: bool = True) -> str:
    """Диаграмма Хассе булева куба B₆ в виде треугольника глифов.

    Ранг k = yang_count(h) = число единиц.
    Ранг 0: {0}, ранг 6: {63}; максимальная ширина при ранге 3 (C(6,3)=20).
    """
    lines = _header(
        'Диаграмма Хассе B₆: 64 глифа по ян-уровням 0..6',
        'Ранг = yang_count(h) = число единиц в 6-битном представлении',
    )
    lines.append('')
    lines.append(render_hasse_glyphs(color=color, show_numbers=show_numbers))
    lines.append('')

    from math import comb
    lines.append('  Размеры рангов:')
    for k in range(7):
        cnt = comb(6, k)
        c   = _YANG_ANSI[k] if color else ''
        r   = _RESET if color else ''
        lines.append(f'  ранг {k}: {c}C(6,{k})={cnt:2d} элементов{r}')

    lines.append('')
    lines.append('  Общее число рёбер Хассе: 6×C(6,1) + ... = 6·2⁵ = 192')
    lines.append('  Диаметр Q6 = 6  Радиус = 3  Хроматическое число = 2')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Путь в Q6
# ---------------------------------------------------------------------------

def render_path_view(start: int = 0, end: int = 63,
                     color: bool = True) -> str:
    """Путь start→end в Q6 по кратчайшему маршруту BFS.

    Показывает:
    1. 8×8 карту с выделением вершин пути
    2. Пошаговое описание пути с флипами битов
    """
    path = shortest_path(start, end)

    lines = _header(
        f'Q6 путь: h={start} → h={end}',
        f'Длина = {len(path)-1}  Хэмминг = {hamming(start, end)}',
    )
    lines.append('')

    # 8×8 карта с путём
    highlights = set(path)
    lines.append(render_grid(
        highlights=highlights, color=color,
        title=f'Путь {start}→{end} ({len(path)-1} шагов)',
    ))
    lines.append('')

    # Детальный путь
    lines.append(render_path(path, color=color, show_bits=True, show_yang=True))
    lines.append('')

    # Анализ пути
    bit_changes = {}
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        bit = (a ^ b).bit_length() - 1
        bit_changes[bit] = bit_changes.get(bit, 0) + 1

    lines.append('  Биты, изменённые вдоль пути:')
    for bit, cnt in sorted(bit_changes.items()):
        c = _YANG_ANSI[min(6, cnt * 2)] if color else ''
        r = _RESET if color else ''
        lines.append(f'    бит {bit}: {c}{cnt} раз(а){r}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='vis_glyphs',
        description='hexvis: витрина визуализации Q6.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')

    sg = sub.add_parser('glyph', help='ASCII-арт конкретного глифа')
    sg.add_argument('--h', type=int, default=21, metavar='H',
                    help='номер глифа 0..63 (умолч. 21)')

    sgr = sub.add_parser('grid', help='8×8 решётка всех глифов')
    sgr.add_argument('--highlights', type=str, default='',
                     metavar='H1,H2,...',
                     help='выделить глифы (через запятую)')

    shs = sub.add_parser('hasse', help='диаграмма Хассе B₆')
    shs.add_argument('--numbers', action='store_true',
                     help='показать числовые индексы над глифами')

    spt = sub.add_parser('path', help='BFS путь start→end')
    spt.add_argument('--start', type=int, default=0, metavar='S')
    spt.add_argument('--end',   type=int, default=63, metavar='E')

    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'glyph':
        h = max(0, min(63, args.h))
        print(render_glyph_view(h, color))

    elif args.cmd == 'grid':
        hs: set[int] = set()
        if args.highlights:
            for part in args.highlights.split(','):
                try:
                    hs.add(int(part.strip()))
                except ValueError:
                    pass
        print(render_grid_view(hs, color))

    elif args.cmd == 'hasse':
        print(render_hasse_view(show_numbers=args.numbers, color=color))

    elif args.cmd == 'path':
        s = max(0, min(63, args.start))
        e = max(0, min(63, args.end))
        print(render_path_view(s, e, color))

    else:
        p.print_help()


if __name__ == '__main__':
    main()
