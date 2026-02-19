"""intermed_glyphs — Промежуточный ряд H Германа через глифы Q6.

Промежуточный ряд H: в каждом интервале (n², (n+1)²) есть ровно одно
кратное (2n+1). Этот элемент — h(k).

Замкнутая формула: h(k) = (2k+1) · (2k+1 − (−1)^k) / 4

Ряд H = 3, 5, 14, 18, 33, 39, 60, 68, 95, 105, ...

Среди первых 64 глифов Q6 ряд H даёт: {3, 5, 14, 18, 33, 39, 60}.
Это 7 значений — ровно по одному на n=1..7.

Связь с Q6:
  Гексаграмма h(k) → yang_count(h(k)) = число единичных битов числа h(k).
  Индекс k ↔ yang-слой: h(k) нечётное ↔ нечётный yang.

Визуализация:
  series    — выделить элементы ряда H среди 0..63
  factors   — факторизация h(k) = (2k+1) · q(k)
  envelopes — положение h(k) в интервале (n², (n+1)²)
  partial   — частичные суммы S(n) = Σ h(k)

Команды CLI:
  series
  factors
  envelopes
  partial
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexintermed.hexintermed import IntermediateSeries
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_hs = IntermediateSeries()

# Элементы ряда H ≤ 63
_H_IN_64 = [h for h in _hs.generate(50) if h <= 63]
_H_SET_64 = set(_H_IN_64)

# Полный ряд до 100
_H_FULL = _hs.generate(40)


def _h_index(val: int) -> int:
    """Индекс k такой, что h(k) = val, или -1 если не в ряду."""
    for k, h in enumerate(_H_FULL, start=1):
        if h == val:
            return k
    return -1


def _interval_n(val: int) -> int:
    """n такое, что n² < val < (n+1)²."""
    if val <= 0:
        return 0
    return int(math.isqrt(val - 1))


# ---------------------------------------------------------------------------
# 1. Ряд H среди глифов 0..63
# ---------------------------------------------------------------------------

def render_series(color: bool = True) -> str:
    """
    8×8 сетка: выделить элементы ряда H среди глифов 0..63.

    H∩[0,63] = {3, 5, 14, 18, 33, 39, 60} — 7 элементов.
    Каждый соответствует промежутку (n², (n+1)²) для n=1..7.
    """
    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Промежуточный ряд H Германа: h(k) ∈ (n², (n+1)²), кратно (2k+1)')
    lines.append(f'  h(k) = (2k+1)·(2k+1 − (−1)^k) / 4')
    lines.append(f'  H∩[0,63] = {sorted(_H_SET_64)}  ({len(_H_SET_64)} элементов)')
    lines.append('  Жирный = элемент ряда H   Цвет = yang_count')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            in_H = h in _H_SET_64
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_BG[yc] + _BOLD if in_H else _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            in_H = h in _H_SET_64
            k = _h_index(h)
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc] if in_H else _YANG_ANSI[0]
                lbl.append(f'{c}{"H"+str(k) if in_H else "  "}{_RESET}')
            else:
                lbl.append(f'H{k}' if in_H else '  ')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Таблица
    lines.append('  Элементы ряда H в [0,63]:')
    for k, h in enumerate(_H_FULL[:10], start=1):
        n = _interval_n(h)
        in_64 = h <= 63
        yc = yang_count(h) if in_64 else -1
        mark = '▶' if in_64 else ' '
        if color:
            c = _YANG_ANSI[yc] if in_64 else _YANG_ANSI[0]
            lines.append(f'  {c}{mark} k={k:2d}: h={h:3d}  n={n}  '
                         f'n²={n**2}  (n+1)²={(n+1)**2}'
                         f'{"  yang="+str(yc) if in_64 else ""}{_RESET}')
        else:
            lines.append(f'  {mark} k={k:2d}: h={h:3d}  n={n}  '
                         f'n²={n**2}  (n+1)²={(n+1)**2}'
                         f'{"  yang="+str(yc) if in_64 else ""}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Факторизация h(k) = (2k+1)·q(k)
# ---------------------------------------------------------------------------

def render_factors(color: bool = True) -> str:
    """
    8×8 сетка: для глифов ∈ H — показать факторизацию h(k).

    Каждый h(k) = (2k+1) · d, где d = (2k+1 ∓ 1)/4.
    """
    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Факторизация элементов ряда H: h(k) = x·y')
    lines.append('  y = 2k+1 (нечётное),  x = h(k) / (2k+1)')
    lines.append('  Цвет = yang_count(h),  жирный = в ряду H')
    lines.append('═' * 66)

    # факторизация каждого H-элемента
    facts = {}
    for k, h in enumerate(_H_FULL[:15], start=1):
        f = _hs.factorize(k)
        facts[h] = (k, f['x'], f['y'], f.get('check', False))

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            in_H = h in _H_SET_64
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_BG[yc] + _BOLD if in_H else _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            in_H = h in _H_SET_64
            if in_H and h in facts:
                k_idx, x, y, ok = facts[h]
                tag = f'{x}×{y}'
            else:
                tag = '    '
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc] if in_H else _YANG_ANSI[0]
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    lines.append('  Факторизации (h = x × y, y = 2k+1):')
    for k, h in enumerate(_H_FULL[:10], start=1):
        f = _hs.factorize(k)
        ok_mark = '✓' if f.get('check') else '?'
        if color:
            yc = yang_count(h) if h <= 63 else 0
            c = _YANG_ANSI[yc]
            lines.append(f'  {c}  h({k:2d}) = {h:3d} = {f["x"]:2d} × {f["y"]:2d}  {ok_mark}{_RESET}')
        else:
            lines.append(f'    h({k:2d}) = {h:3d} = {f["x"]:2d} × {f["y"]:2d}  {ok_mark}')

    # Графический ряд
    lines.append('\n  Ряд H (ASCII):')
    lines.append(_hs.plot(n_max=8, width=48))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Огибающие n² и (n+1)²
# ---------------------------------------------------------------------------

def render_envelopes(color: bool = True) -> str:
    """
    8×8 сетка: положение глифов h относительно огибающих n², (n+1)².

    Для каждого h: n = floor(sqrt(h-1)), classify по:
      h == n²: квадрат
      h == (n+1)²: следующий квадрат
      h ∈ H: элемент ряда H (нечётное, кратное (2n+1))
      иначе: обычное число
    """
    def classify(h: int):
        if h == 0:
            return 'zero'
        n = _interval_n(h)
        if h == n * n:
            return 'square'
        if h == (n + 1) * (n + 1):
            return 'square'
        if h in _H_SET_64:
            return 'H'
        if h % 2 == 0:
            return 'even'
        return 'odd'

    cls_color = {
        'zero':   _YANG_ANSI[0],
        'square': _YANG_BG[3] + _BOLD,
        'H':      _YANG_BG[5] + _BOLD,
        'even':   _YANG_ANSI[2],
        'odd':    _YANG_ANSI[1],
    }

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Огибающие ряда H: h(k) ∈ (n², (n+1)²)')
    lines.append('  Жирный/фон = точный квадрат n²')
    lines.append('  Яркий      = элемент ряда H')
    lines.append('  Синий      = чётное   Жёлтый = нечётное')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            cl = classify(h)
            rows3 = render_glyph(h)
            if color:
                c = cls_color.get(cl, _YANG_ANSI[0])
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            cl = classify(h)
            n = _interval_n(h)
            if cl == 'square':
                tag = f'n={n}'
            elif cl == 'H':
                k_idx = _h_index(h)
                tag = f'H{k_idx}'
            elif cl == 'zero':
                tag = ' 0 '
            elif cl == 'even':
                tag = 'ev '
            else:
                tag = 'od '
            if color:
                c = cls_color.get(cl, _YANG_ANSI[0])
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Таблица квадратов
    lines.append('  Квадраты n² до 63:')
    for n in range(9):
        sq = n * n
        if sq > 63:
            break
        h_next = _H_FULL[n - 1] if n >= 1 else None
        h_next_str = f'  h({n})={h_next}' if h_next and h_next <= 63 else ''
        if color:
            c = _YANG_ANSI[n % 7]
            lines.append(f'  {c}  n={n}: n²={sq:2d}  (n+1)²={(n+1)**2:2d}{h_next_str}{_RESET}')
        else:
            lines.append(f'    n={n}: n²={sq:2d}  (n+1)²={(n+1)**2:2d}{h_next_str}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Частичные суммы S(n)
# ---------------------------------------------------------------------------

def render_partial(color: bool = True) -> str:
    """
    8×8 сетка: раскраска по частичной сумме S(yang_count(h)).

    S(k) = Σ_{i=1}^{k} h(i) — сумма первых k элементов ряда H.
    Формула: S(n) = (n+1)·(4(n+1)²−1−3(−1)^n) / 12
    """
    partial_sums = [_hs.partial_sum(k) for k in range(7)]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Частичные суммы ряда H: S(n) = Σ_{k=1}^{n} h(k)')
    lines.append('  Формула: S(n) = (n+1)·(4(n+1)²−1−3(−1)^n) / 12')
    lines.append('  Цвет = yang_count(h),  число = S(yang_count)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            yc = yang_count(h)
            rows3 = render_glyph(h)
            if color:
                c = _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            yc = yang_count(h)
            s = partial_sums[yc]
            if color:
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}S={s:3d}{_RESET}')
            else:
                lbl.append(f'S={s:3d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    lines.append('  Таблица частичных сумм:')
    for k in range(7):
        s = partial_sums[k]
        terms = _H_FULL[:k]
        direct = sum(terms)
        ok = '✓' if s == direct else f'✗(direct={direct})'
        if color:
            c = _YANG_ANSI[k]
            lines.append(f'  {c}  S({k}) = {s:5d}   = '
                         f'{" + ".join(str(h) for h in terms) or "0"}  {ok}{_RESET}')
        else:
            lines.append(f'    S({k}) = {s:5d}   = '
                         f'{" + ".join(str(h) for h in terms) or "0"}  {ok}')

    # Многоугольные числа
    lines.append('\n  Многоугольные числа (IntermediateSeries.polygonal):')
    for k in [3, 4, 5, 6]:
        row_vals = [_hs.polygonal(n, k) for n in range(1, 9)]
        in_64 = [v for v in row_vals if v <= 63]
        if color:
            c = _YANG_ANSI[k % 7]
            lines.append(f'  {c}  {k}-угольные: {row_vals}   в[0,63]={in_64}{_RESET}')
        else:
            lines.append(f'    {k}-угольные: {row_vals}   в[0,63]={in_64}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='intermed_glyphs',
        description='Промежуточный ряд H Германа через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('series',    help='ряд H среди глифов 0..63')
    sub.add_parser('factors',   help='факторизация h(k) = x·y')
    sub.add_parser('envelopes', help='положение h(k) в (n², (n+1)²)')
    sub.add_parser('partial',   help='частичные суммы S(n)')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'series':
        print(render_series(color))
    elif args.cmd == 'factors':
        print(render_factors(color))
    elif args.cmd == 'envelopes':
        print(render_envelopes(color))
    elif args.cmd == 'partial':
        print(render_partial(color))


if __name__ == '__main__':
    main()
