"""hexellipse/ellipse_glyphs.py — Q6 глифы через анализ эллипса и линии катастрофы.

Каждый глиф h отображается на эллипс E(k) через yang_count(h) = k ∈ 0..6:
    a(k) = k + 1   (большая полуось: 1..7)
    b(k) = 7 − k   (малая полуось:  6..1)

При k=0: a=1, b=6 → вытянутый вертикальный (EllipseAnalysis меняет, если a<b → a=6,b=1)
При k=3: a=4, b=4 → окружность (e=0)
При k=6: a=7, b=1 → сильно вытянутый

Основные параметры:
    c = √(a²−b²)   фокусное расстояние
    e = c/a         эксцентриситет ∈ [0, 1)
    p₁ = b²/a       фокальный параметр (радиус кривизны в вершинах ±a)
    p₂ = a²/b       радиальный параметр (вписанные окружности)
    p = ab/c        полупараметр фокальной хорды

Инвариант: a·b = c·p (всегда)

Линия катастрофы: параллельная кривая на расстоянии q = p₁ имеет особые точки (cusps).

Визуализация (8×8, Gray-код Q6):
  eccentric  — эксцентриситет e(h) по ян-слоям
  focal      — фокальный параметр p₁ = b²/a
  inscribed  — отношение R/r = (a/b)² вписанных окружностей
  catastrophe — тип катастрофы для каждого эллипса

Команды CLI:
  eccentric
  focal
  inscribed
  catastrophe
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexellipse.hexellipse import EllipseAnalysis
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Отображение yang → эллипс
# ---------------------------------------------------------------------------

def _ellipse(k: int) -> EllipseAnalysis:
    """EllipseAnalysis для ян-уровня k (a=k+1, b=7-k, swap if needed)."""
    a = float(k + 1)
    b = float(7 - k)
    # EllipseAnalysis сам разберётся с порядком a, b
    return EllipseAnalysis(a, b)


# Предвычисление параметров по 7 ян-уровням
_E: list[EllipseAnalysis] = [_ellipse(k) for k in range(7)]

_GRAY3 = [i ^ (i >> 1) for i in range(8)]

# Диапазоны для нормировки цвета
_E_MIN = _E[3].eccentricity()   # 0 (окружность)
_E_MAX = _E[0].eccentricity()   # максимум (k=0 → a=6,b=1 после swap)


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
# 1. Эксцентриситет
# ---------------------------------------------------------------------------

def render_eccentric(color: bool = True) -> str:
    """8×8 сетка: эксцентриситет e(h) = c/a по ян-уровням.

    yang=3 → окружность (e=0).
    yang→0 или →6 → всё более вытянутый эллипс (e→1).
    Ярлык — 1-я цифра после запятой эксцентриситета.
    """
    lines = _header(
        'Эллипс: эксцентриситет e = c/a  (ян→форма)',
        'yang=3→окружность(e=0)  yang=0,6→вытянутый(e→1)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            e = _E[k].eccentricity()
            # Ярлык: 1 цифра (10-кратное округление)
            sym = str(int(round(e * 9)))[-1]
            if color:
                c = _YANG_ANSI[k]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Эксцентриситет по ян-слоям  (a=k+1 или b=7-k, swap если a<b):')
    for k in range(7):
        ea = _E[k]
        e  = ea.eccentricity()
        a  = ea.a
        b  = ea.b
        cnt = sum(1 for h in range(64) if yang_count(h) == k)
        c  = _YANG_ANSI[k] if color else ''
        r  = _RESET if color else ''
        lines.append(
            f'  ян={k}: {c}a={a:.0f} b={b:.0f} e={e:.6f}{r}  глифов={cnt}'
        )
    lines.append('')
    lines.append('  yang=3 → a=b=4 → окружность (e=0, c=0).')
    lines.append('  Зеркальная симметрия: e(k) = e(6−k) (т.к. swap a,b).')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Фокальный параметр
# ---------------------------------------------------------------------------

def render_focal(color: bool = True) -> str:
    """8×8 сетка: фокальный параметр p₁ = b²/a.

    p₁ — радиус кривизны в вершинах ±a эллипса.
    Ярлык: однозначное целое (округлено до 1 знака).
    """
    lines = _header(
        'Эллипс: фокальный параметр p₁ = b²/a',
        'p₁ = радиус кривизны в вершинах ±a; при q=p₁ → катастрофа (cusps)',
    )

    # Диапазон p₁ для цветовой нормировки
    p1_vals = [_E[k].focal_parameter() for k in range(7)]
    p1_min  = min(p1_vals)
    p1_max  = max(p1_vals)

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            p1 = _E[k].focal_parameter()
            # Нормируем в 0..6 для цвета
            if p1_max > p1_min:
                norm = int(round((p1 - p1_min) / (p1_max - p1_min) * 6))
            else:
                norm = 3
            norm = max(0, min(6, norm))
            sym  = str(int(round(p1)))[-1]   # последняя цифра округлённого p₁
            if color:
                c = _YANG_ANSI[k]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Параметры по ян-слоям:')
    for k in range(7):
        ea = _E[k]
        p1 = ea.focal_parameter()
        p2 = ea.radial_parameter()
        c  = ea.c()
        e  = ea.eccentricity()
        cnt = sum(1 for h in range(64) if yang_count(h) == k)
        clr = _YANG_ANSI[k] if color else ''
        r   = _RESET if color else ''
        lines.append(
            f'  ян={k}: {clr}p₁={p1:.4f}  p₂={p2:.4f}  c={c:.4f}  e={e:.4f}{r}'
            f'  глиф={cnt}'
        )

    lines.append('')
    lines.append('  p₁·p₂ = (b²/a)·(a²/b) = ab  (произведение параметров = a·b)')
    lines.append('  a·b = c·p₁  (инвариант эллипса)')

    # Проверка инварианта
    ok_all = all(_E[k].verify_identity()['ok'] for k in range(7))
    lines.append(f'  Проверка a·b = c·p для всех k: {"✓" if ok_all else "✗"}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Вписанные окружности
# ---------------------------------------------------------------------------

def render_inscribed(color: bool = True) -> str:
    """8×8 сетка: отношение R/r = p₂/p₁ = (a/b)² вписанных окружностей.

    Ярлык: однозначное представление отношения R/r (масштаб).
    """
    lines = _header(
        'Эллипс: вписанные окружности R/r = (a/b)²',
        'R=p₂=a²/b (внешняя)  r=p₁=b²/a (внутренняя)  R/r=(a/b)²',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            ea = _E[k]
            sol = ea.inscribed_system_solution()
            ratio = sol['R/r']
            # Цифра: log scale
            if ratio >= 1.0:
                sym = str(min(9, int(math.log(ratio, 2))))
            else:
                sym = '0'
            if color:
                c = _YANG_ANSI[k]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Система вписанных окружностей по ян-слоям:')
    for k in range(7):
        ea  = _E[k]
        sol = ea.inscribed_system_solution()
        R   = sol['R_outer']
        r   = sol['r_inner']
        ratio = sol['R/r']
        prod  = sol['product']
        c   = _YANG_ANSI[k] if color else ''
        rst = _RESET if color else ''
        lines.append(
            f'  ян={k}: {c}R={R:.4f}  r={r:.4f}  R/r={ratio:.4f}'
            f'  R·r={prod:.4f}{rst}'
        )

    lines.append('')
    lines.append('  Для окружности (ян=3, a=b=4): R/r = 1, R = r = 4.')
    lines.append('  При ян=0 или ян=6 (наибольший эксцентриситет): R/r максимально.')
    lines.append('  R·r = p₂·p₁ = a²/b · b²/a = ab  (произведение = a·b).')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Тип катастрофы
# ---------------------------------------------------------------------------

def render_catastrophe(color: bool = True) -> str:
    """8×8 сетка: тип особой точки линии катастрофы (cusp / circle).

    Линия катастрофы — параллельная кривая на расстоянии q = p₁.
    Для эллипса (e>0) появляются cusps (cusps); для окружности (e=0) — нет.
    Ярлык: C = cusp (особая точка), O = circle (без катастрофы).
    """
    _CUSP_COLOR   = '\033[38;5;196m'   # красный = cusp
    _CIRCLE_COLOR = '\033[38;5;82m'    # зелёный = circle

    lines = _header(
        'Эллипс: тип линии катастрофы (q = p₁)',
        'C=cusp особая точка (e>0)  O=окружность без катастрофы (e=0)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            ea = _E[k]
            info = ea.at_catastrophe()
            is_cusp = info.get('type', '') == 'cusp'
            sym = 'C' if is_cusp else 'O'
            if color:
                c = _CUSP_COLOR if is_cusp else _CIRCLE_COLOR
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Информация о катастрофах по ян-слоям:')
    for k in range(7):
        ea   = _E[k]
        info = ea.at_catastrophe()
        tp   = info.get('type', '?')
        q    = info.get('q', 0.0)
        cnt  = sum(1 for h in range(64) if yang_count(h) == k)
        c    = _YANG_ANSI[k] if color else ''
        r    = _RESET if color else ''
        sp   = info.get('t_special', [])
        n_sp = len(sp) if isinstance(sp, list) else 0
        lines.append(
            f'  ян={k}: {c}тип={tp}  q={q:.4f}  особых_точек={n_sp}{r}'
            f'  глифов={cnt}'
        )

    lines.append('')
    lines.append('  Линия катастрофы: параллельная кривая E+q·n = E_q')
    lines.append('  При q = p₁ кривая E_q самопересекается → cusps.')
    lines.append('  Число cusps = 4 для эллипса (2 вблизи каждого фокуса).')

    # ASCII-визуализация для ян=5 (интересный случай)
    lines.append('')
    lines.append('  ASCII-проекция эллипса ян=5 (a=6,b=2) с катастрофой:')
    try:
        art = _E[5].plot_ascii(width=50)
        for ln in art.splitlines()[:12]:
            lines.append(f'  {ln}')
    except Exception:
        lines.append('  (визуализация недоступна)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='ellipse_glyphs',
        description='Q6 глифы через анализ эллипса и линию катастрофы.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('eccentric',    help='эксцентриситет e = c/a')
    sub.add_parser('focal',        help='фокальный параметр p₁ = b²/a')
    sub.add_parser('inscribed',    help='вписанные окружности R/r = (a/b)²')
    sub.add_parser('catastrophe',  help='тип линии катастрофы (cusp / circle)')
    args = p.parse_args(argv)
    color = not args.no_color

    dispatch = {
        'eccentric':   render_eccentric,
        'focal':       render_focal,
        'inscribed':   render_inscribed,
        'catastrophe': render_catastrophe,
    }
    if args.cmd in dispatch:
        print(dispatch[args.cmd](color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
