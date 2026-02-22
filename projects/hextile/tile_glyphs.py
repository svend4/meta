"""hextile/tile_glyphs.py — Q6 глифы через апериодические мозаики Германна.

Каждый глиф h через yang_count(h) = k ∈ {0,...,6} отображается на элемент
апериодической мозаики (ромбы Пенроза-Германна):

  yang 0..2 → тонкий ромб (острый угол α = 36°)
  yang = 3  → «стрела Германна» (HermanArrow: тонкий + толстый ромб)
  yang 4..6 → толстый ромб (острый угол β = 72°)

Квазикристаллическая мозаика:
  QuasicrystalMosaic(polygon_sides = yang+3, divisions=5)
  симметрия порядка = yang+3 (т.к. divisions=5 нечётное)

Золотое сечение φ = (1+√5)/2 ≈ 1.618 — отношение тонких/толстых ромбов
в правильной мозаике Пенроза (φ²:1).

Визуализация (8×8, Gray-код Q6):
  rhombus  — тип ромба для каждого глифа (T/A/K)
  sym      — порядок симметрии квазикристалла (yang+3)
  area     — площадь ромба (пропорциональна sin(острого угла))
  mosaic   — ASCII-арт квазикристалла для каждого ян-слоя

Команды CLI:
  rhombus
  sym
  area
  mosaic  [--yang k]
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hextile.hextile import (
    Rhombus,
    HermanArrow,
    AperiodicTiling,
    AperiodicTiler,
    QuasicrystalMosaic,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_PHI   = (1 + math.sqrt(5)) / 2      # золотое сечение
_ALPHA = 36 * math.pi / 180          # угол тонкого ромба
_BETA  = 72 * math.pi / 180          # угол толстого ромба

# Тип тайла по ян-слою
_TILE_TYPE = {
    0: 'T',   # тонкий (thin)
    1: 'T',
    2: 'T',
    3: 'A',   # стрела (arrow)
    4: 'K',   # толстый (thick / кrunный)
    5: 'K',
    6: 'K',
}

# Цвета для типов
_THIN_COLOR  = '\033[38;5;39m'    # голубой = тонкий
_ARROW_COLOR = '\033[38;5;226m'   # жёлтый  = стрела
_THICK_COLOR = '\033[38;5;208m'   # оранжевый = толстый

_TYPE_COLOR = {
    'T': _THIN_COLOR,
    'A': _ARROW_COLOR,
    'K': _THICK_COLOR,
}

_GRAY3 = [i ^ (i >> 1) for i in range(8)]

# Предвычисленные площади для 3 типов
_AREA_THIN  = 1.0 * 1.0 * math.sin(_ALPHA)   # side=1
_AREA_THICK = 1.0 * 1.0 * math.sin(_BETA)
_AREA_ARROW = _AREA_THIN + _AREA_THICK


def _header(title: str, subtitle: str = '') -> list[str]:
    lines = ['═' * 66, f'  {title}']
    if subtitle:
        lines.append(f'  {subtitle}')
    lines.append('═' * 66)
    col_hdr = '  '.join(format(g, '03b') for g in _GRAY3)
    lines.append(f'        {col_hdr}')
    lines.append('        ' + '─' * len(col_hdr))
    return lines


# ---------------------------------------------------------------------------
# 1. Тип ромба
# ---------------------------------------------------------------------------

def render_rhombus(color: bool = True) -> str:
    """8×8 сетка: тип ромба для каждого глифа по ян-уровню.

    T = тонкий ромб (α=36°), yang 0..2   → 42 глифа
    A = стрела Германна (тонкий+толстый), yang=3 → 20 глифов
    K = толстый ромб (β=72°), yang 4..6  → 22 глифа
    """
    lines = _header(
        'Tile: тип ромба по ян-слою',
        'T=тонкий(36°,голуб.)  A=стрела(арр.,жёлт.)  K=толстый(72°,оранж.)',
    )

    counts = {'T': 0, 'A': 0, 'K': 0}
    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            tp = _TILE_TYPE[k]
            counts[tp] += 1
            if color:
                c = _TYPE_COLOR[tp]
                cell = f'{c}{tp}{_RESET}'
            else:
                cell = tp
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    for tp, label, angle in [
        ('T', 'тонкий ромб',   '36°'),
        ('A', 'стрела Герм.',  '36°+72°'),
        ('K', 'толстый ромб',  '72°'),
    ]:
        c = _TYPE_COLOR[tp] if color else ''
        r = _RESET if color else ''
        yang_vals = [k for k in range(7) if _TILE_TYPE[k] == tp]
        lines.append(
            f'  {c}{tp}{r}: {counts[tp]:2d} глифов  '
            f'{label} ({angle})  ян={yang_vals}'
        )

    lines.append('')
    lines.append(f'  φ = (1+√5)/2 ≈ {_PHI:.6f}  (золотое сечение)')
    lines.append(f'  Соотношение площадей тонкий/толстый = sin(36°)/sin(72°)')
    ratio = math.sin(_ALPHA) / math.sin(_BETA)
    lines.append(f'    = {ratio:.6f}  ≈ 1/φ = {1/_PHI:.6f}')
    lines.append('  В правильной мозаике Пенроза соотношение тонких:толстых = φ²:1')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Порядок симметрии квазикристалла
# ---------------------------------------------------------------------------

def render_sym(color: bool = True) -> str:
    """8×8 сетка: порядок симметрии квазикристалла для данного ян-слоя.

    QuasicrystalMosaic(polygon_sides = yang+3, divisions=5)
    symmetry_order = yang+3  (т.к. divisions=5 нечётное).
    """
    lines = _header(
        'Tile: порядок симметрии квазикристалла yang+3',
        'Цифра = porядок_симметрии = yang_count(h)+3  (3..9)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h   = (row_g << 3) | col_g
            k   = yang_count(h)
            sym = str(k + 3)
            if color:
                c = _YANG_ANSI[k]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Порядок симметрии = число сторон многоугольника = yang+3:')
    for k in range(7):
        sides = k + 3
        qm = QuasicrystalMosaic(polygon_sides=sides, divisions=5)
        qm.generate()
        sym_ord = qm.symmetry_order()
        n_seg   = qm.segment_count()
        n_vtx   = qm.vertex_count()
        cnt     = sum(1 for h in range(64) if yang_count(h) == k)
        c = _YANG_ANSI[k] if color else ''
        r = _RESET if color else ''
        lines.append(
            f'  ян={k}: {c}{sides}-уголь{r}  '
            f'симм.={sym_ord}  сегм.={n_seg}  вершин={n_vtx}  глифов={cnt}'
        )

    lines.append('')
    lines.append('  Теорема: для нечётного числа разбиений k порядок = n,')
    lines.append('  для чётного k порядок = 2n  (где n = polygon_sides).')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Площадь ромба
# ---------------------------------------------------------------------------

def render_area(color: bool = True) -> str:
    """8×8 сетка: площадь ромба A(h) = f(yang_count(h)).

    Площадь стороны=1:
      тонкий ромб  (36°): S = sin(36°) ≈ 0.588
      толстый ромб (72°): S = sin(72°) ≈ 0.951
      стрела        (36°+72°): S = sin(36°)+sin(72°) ≈ 1.539
    """
    lines = _header(
        'Tile: площадь ромба (сторона = 1)',
        'Ярлык = 1 знак после десятичной точки площади',
    )

    _AREA = {
        0: _AREA_THIN,  1: _AREA_THIN,  2: _AREA_THIN,
        3: _AREA_ARROW,
        4: _AREA_THICK, 5: _AREA_THICK, 6: _AREA_THICK,
    }

    # Нормированная площадь: min=THIN, max=ARROW
    a_min = _AREA_THIN
    a_max = _AREA_ARROW

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            ar = _AREA[k]
            # Отображение в 0..6 для ян-цвета
            norm = int(round((ar - a_min) / (a_max - a_min) * 6))
            norm = max(0, min(6, norm))
            tp   = _TILE_TYPE[k]
            sym  = f'{ar:.1f}'[0]   # первая цифра (0 или 1)
            if color:
                c = _TYPE_COLOR[tp]
                cell = f'{c}{tp}{_RESET}'
            else:
                cell = tp
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Площади ромбов (сторона = 1):')
    for k, area, tp, name in [
        (0, _AREA_THIN,  'T', 'тонкий 36°'),
        (3, _AREA_ARROW, 'A', 'стрела 36°+72°'),
        (4, _AREA_THICK, 'K', 'толстый 72°'),
    ]:
        c = _TYPE_COLOR[tp] if color else ''
        r = _RESET if color else ''
        lines.append(f'  {c}{tp}{r} {name:16s}: S = {area:.6f}')

    ratio = _AREA_THICK / _AREA_THIN
    lines.append('')
    lines.append(f'  S(толст.)/S(тонк.) = {ratio:.6f}')
    lines.append(f'  φ                  = {_PHI:.6f}')
    lines.append(f'  sin(72°)/sin(36°) = {ratio:.6f} ≈ φ  (золотое сечение!)')
    lines.append(f'  S(стрела) = S(тонк.) + S(толст.) = {_AREA_ARROW:.6f}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. ASCII-арт квазикристалла
# ---------------------------------------------------------------------------

def render_mosaic(yang: int = 0, color: bool = True) -> str:
    """ASCII-арт QuasicrystalMosaic для заданного ян-уровня yang.

    polygon_sides = yang + 3, divisions = 5.
    Показывает также шапку 8×8 глифов с выделенным ян-слоем.
    """
    sides = yang + 3
    qm = QuasicrystalMosaic(polygon_sides=sides, divisions=5)
    qm.generate()

    lines = ['═' * 66]
    lines.append(f'  Tile: квазикристалл yang={yang}  ({sides}-угольник, divisions=5)')
    lines.append(f'  Порядок симметрии = {qm.symmetry_order()}')
    lines.append(f'  Сегментов: {qm.segment_count()}  Вершин: {qm.vertex_count()}')
    lines.append('═' * 66)

    # 8×8 карта: выделить ян-слой yang
    col_hdr = '  '.join(format(g, '03b') for g in _GRAY3)
    lines.append(f'        {col_hdr}')
    lines.append('        ' + '─' * len(col_hdr))

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            if k == yang:
                tp  = _TILE_TYPE[k]
                sym = tp
                if color:
                    c = f'{_BOLD}{_TYPE_COLOR[tp]}'
                    cell = f'{c}{sym}{_RESET}'
                else:
                    cell = f'[{sym}]'[:1]
            else:
                sym = '·'
                if color:
                    c = _YANG_ANSI[k]
                    cell = f'{c}{sym}{_RESET}'
                else:
                    cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  {sides}-угольная квазикристаллическая мозаика:')

    # ASCII-визуализация (берём первые 20 строк)
    ascii_art = qm.to_ascii(width=60)
    for ln in ascii_art.splitlines()[:20]:
        lines.append(f'  {ln}')

    lines.append('')
    lines.append(f'  Глифов в ян={yang}: {sum(1 for h in range(64) if yang_count(h)==yang)}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='tile_glyphs',
        description='Q6 глифы через апериодические мозаики Германна.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('rhombus', help='тип ромба T/A/K по ян-слою')
    sub.add_parser('sym',     help='порядок симметрии квазикристалла')
    sub.add_parser('area',    help='площадь ромба (sin(угол))')
    sm = sub.add_parser('mosaic',  help='ASCII квазикристалл для ян-слоя')
    sm.add_argument('--yang', type=int, default=3, metavar='K',
                    help='ян-уровень 0..6 (по умолч. 3 = стрела)')
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'rhombus':
        print(render_rhombus(color))
    elif args.cmd == 'sym':
        print(render_sym(color))
    elif args.cmd == 'area':
        print(render_area(color))
    elif args.cmd == 'mosaic':
        k = max(0, min(6, args.yang))
        print(render_mosaic(yang=k, color=color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
