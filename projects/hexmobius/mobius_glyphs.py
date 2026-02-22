"""hexmobius/mobius_glyphs.py — Q6 glyphs through Möbius surface topology.

Каждый глиф h отображается на поверхность Мёбиуса M_N с N = yang_count(h) скрутками.

Топология обобщённой поверхности Мёбиуса M_N:
  euler_characteristic   = 0  (всегда)
  is_orientable          = True только при N = 0 (цилиндр)
  num_boundary_components:
      N=0 → 2 края   (два отдельных ребра у цилиндра)
      N нечётное → 1 край  (лента Мёбиуса и обобщения)
      N чётное ≥ 2 → 0 краёв  (бутылка Кляйна и обобщения)
  writhing_number  = N / 2
  surface_class    = "Cylinder" / "Möbius band" / "Klein bottle" / "Generalized Möbius (N=k)"

Таблица по ян-слоям (N = yang_count(h)):
  yang=0 (N=0): Цилиндр, χ=0, ориент., 2 края
  yang=1 (N=1): Лента Мёбиуса, χ=0, неориент., 1 край
  yang=2 (N=2): Бутылка Кляйна, χ=0, неориент., 0 краёв
  yang=3 (N=3): Обобщённая, χ=0, неориент., 1 край
  yang=4 (N=4): Обобщённая, χ=0, неориент., 0 краёв
  yang=5 (N=5): Обобщённая, χ=0, неориент., 1 край
  yang=6 (N=6): Обобщённая, χ=0, неориент., 0 краёв

Визуализация (8×8, строки/столбцы = Gray-код Q6):
  twists   — число скруток (ян-счёт), окраска по ян-слоям
  orient   — ориентируемость: O = да (только h=0), N = нет
  boundary — число компонент границы: 0 / 1 / 2
  surface  — класс: C=Цилиндр M=Мёбиус K=Кляйн G=Обобщённая

Команды CLI:
  twists
  orient
  boundary
  surface
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexmobius.hexmobius import MobiusSurface
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

_GRAY3 = [i ^ (i >> 1) for i in range(8)]

# Цвета для числа компонент границы
_BC_COLOR = {
    0: '\033[38;5;238m',   # серый = 0 краёв (замкнутая)
    1: '\033[38;5;208m',   # оранжевый = 1 край (Мёбиус)
    2: '\033[38;5;82m',    # зелёный = 2 края (цилиндр)
}

# Цвета для ориентируемости
_ORIENT_FG  = '\033[38;5;82m'    # зелёный = ориентируема
_NONOR_FG   = '\033[38;5;196m'   # красный = неориентируема

# Однобуквенные ярлыки классов
_CLASS_LABEL = {
    'Cylinder':     'C',
    'Möbius band':  'M',
    'Klein bottle': 'K',
}

# Предвычисленные свойства по ян-уровню (0..6)
_SURFACES = [MobiusSurface(R=3.0, width=1.0, twists=k) for k in range(7)]


def _header(title: str, subtitle: str = '') -> list[str]:
    lines = [
        '═' * 66,
        f'  {title}',
    ]
    if subtitle:
        lines.append(f'  {subtitle}')
    lines.append('═' * 66)
    col_hdr = '  '.join(format(g, '03b') for g in _GRAY3)
    lines.append(f'        {col_hdr}')
    lines.append('        ' + '─' * len(col_hdr))
    return lines


def _row_pref(row_g: int, color: bool) -> str:
    return f'  {format(row_g, "03b")} │ '


# ---------------------------------------------------------------------------
# 1. Число скруток = yang_count(h)
# ---------------------------------------------------------------------------

def render_twists(color: bool = True) -> str:
    """8×8 сетка: каждый глиф раскрашен по числу скруток N = yang_count(h).

    N=0 (серый) → цилиндр.  N растёт — Möbius, Klein, обобщённые поверхности.
    """
    lines = _header(
        'Möbius: число скруток N = yang_count(h)',
        'N=0→Цилиндр  N=1→Мёбиус  N=2→Кляйн  N≥3→Обобщённая  χ=0 всегда',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            glyph = render_glyph(h)          # ['_', '|x|', '_'] 3-char per row
            label = str(k)
            if color:
                cell = f'{_YANG_ANSI[k]}{label}{_RESET}'
            else:
                cell = label
            cells.append(cell)
        lines.append(_row_pref(row_g, color) + '  '.join(cells))

    lines.append('')
    lines.append('  Топология по ян-слоям (N = yang_count):')
    for k in range(7):
        ms = _SURFACES[k]
        cls = ms.surface_class()
        bc  = ms.num_boundary_components()
        wr  = ms.writhing_number()
        cnt = sum(1 for h in range(64) if yang_count(h) == k)
        lbl = _CLASS_LABEL.get(cls, 'G')
        c = _YANG_ANSI[k] if color else ''
        r = _RESET if color else ''
        lines.append(
            f'  ян={k} N={k}: {c}[{lbl}] {cls}{r}'
            f'  края={bc}  writhe={wr:.1f}  глифов={cnt}'
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Ориентируемость
# ---------------------------------------------------------------------------

def render_orient(color: bool = True) -> str:
    """8×8 сетка: ориентируемость поверхности M_{yang(h)}.

    O = ориентируема (только h=0, yang=0, цилиндр).
    N = неориентируема (все остальные).
    """
    lines = _header(
        'Möbius: ориентируемость M_{yang(h)}',
        'O = ориентируема (yang=0)  N = неориентируема (yang≥1)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            ok = _SURFACES[k].is_orientable()
            sym = 'O' if ok else 'N'
            if color:
                fg = _ORIENT_FG if ok else _YANG_ANSI[k]
                cell = f'{fg}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(_row_pref(row_g, color) + '  '.join(cells))

    lines.append('')
    n_or = sum(1 for h in range(64) if yang_count(h) == 0)
    lines.append(f'  Ориентируемых глифов : {n_or}/64  (h=0, yang=0)')
    lines.append(f'  Неориентируемых      : {64 - n_or}/64')
    lines.append('  N=0 (цилиндр) → χ=0 + ориентируема')
    lines.append('  N≥1           → χ=0 + неориентируема')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Компоненты границы
# ---------------------------------------------------------------------------

def render_boundary(color: bool = True) -> str:
    """8×8 сетка: число компонент границы (0 / 1 / 2).

    yang=0: 2 края (цилиндр).
    yang нечётное (1,3,5): 1 край.
    yang чётное ≥ 2 (2,4,6): 0 краёв (замкнутая поверхность).
    """
    lines = _header(
        'Möbius: компоненты границы M_{yang(h)}',
        '0=замкнутая(зелёный)  1=один край(оранж.)  2=два края(жёлтый)',
    )

    # Соответствие количества краёв и цвета символа
    bc_sym = {0: '0', 1: '1', 2: '2'}

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            bc = _SURFACES[k].num_boundary_components()
            sym = bc_sym[bc]
            if color:
                c = _BC_COLOR[bc]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(_row_pref(row_g, color) + '  '.join(cells))

    lines.append('')
    for bc_val in range(3):
        cnt = sum(
            1 for h in range(64)
            if _SURFACES[yang_count(h)].num_boundary_components() == bc_val
        )
        yang_list = [k for k in range(7)
                     if _SURFACES[k].num_boundary_components() == bc_val]
        c = _BC_COLOR[bc_val] if color else ''
        r = _RESET if color else ''
        lines.append(
            f'  {c}краёв={bc_val}{r}: {cnt} глифов  '
            f'(ян = {", ".join(map(str, yang_list))})'
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Класс поверхности
# ---------------------------------------------------------------------------

def render_surface(color: bool = True) -> str:
    """8×8 сетка: класс поверхности Мёбиуса M_{yang(h)}.

    C = Cylinder  M = Möbius band  K = Klein bottle  G = Generalized (N≥3)
    """
    lines = _header(
        'Möbius: класс поверхности (C/M/K/G)',
        'C=Цилиндр  M=Мёбиус  K=Кляйн  G=Обобщённая N≥3',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h   = (row_g << 3) | col_g
            k   = yang_count(h)
            cls = _SURFACES[k].surface_class()
            lbl = _CLASS_LABEL.get(cls, 'G')
            if color:
                cell = f'{_YANG_ANSI[k]}{lbl}{_RESET}'
            else:
                cell = lbl
            cells.append(cell)
        lines.append(_row_pref(row_g, color) + '  '.join(cells))

    lines.append('')
    lines.append('  Детальная информация по ян-слоям:')
    for k in range(7):
        ms  = _SURFACES[k]
        cls = ms.surface_class()
        bc  = ms.num_boundary_components()
        wr  = ms.writhing_number()
        ax  = ms.axial_length()
        lbl = _CLASS_LABEL.get(cls, 'G')
        c   = _YANG_ANSI[k] if color else ''
        r   = _RESET if color else ''
        lines.append(
            f'  ян={k}: {c}[{lbl}] {cls}{r}'
            f'  writhe={wr:.1f}  L_oc={ax:.2f}  χ=0'
        )
    lines.append('')
    lines.append('  Euler χ=0 для всех поверхностей Мёбиуса (независимо от N).')
    lines.append('  Ориентируемая + χ=0 ↔ тор; неориентируемая + χ=0 ↔ Klein/Möbius.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='mobius_glyphs',
        description='Q6 глифы через топологию поверхностей Мёбиуса.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('twists',   help='число скруток N = yang_count(h)')
    sub.add_parser('orient',   help='ориентируемость поверхности')
    sub.add_parser('boundary', help='компоненты границы (0/1/2)')
    sub.add_parser('surface',  help='класс: C/M/K/G')
    args = p.parse_args(argv)
    color = not args.no_color

    dispatch = {
        'twists':   render_twists,
        'orient':   render_orient,
        'boundary': render_boundary,
        'surface':  render_surface,
    }
    if args.cmd in dispatch:
        print(dispatch[args.cmd](color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
