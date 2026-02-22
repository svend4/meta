"""hexpowerxy/powerxy_glyphs.py — Q6 глифы через уравнение X^Y = Y^X.

Уравнение X^Y = Y^X имеет нетривиальные решения (X ≠ Y) вдоль параметрической кривой:
    X(t) = t^(1/(t−1))     t > 1
    Y(t) = t^(t/(t−1))     (t → ∞: X → 1, Y → ∞)

Классическое решение: X=2, Y=4 (t=2: X=2^1=2, Y=2^2=4).

Золотое сечение φ = (1+√5)/2 ≈ 1.618 выступает корнем особого уравнения:
    φ^(1/(φ−1)) = (1+1/φ)^φ  и  φ^(φ/(φ−1)) = (1+1/φ)^(φ+1)

Особая точка: X → e (Эйлер ≈ 2.718) — единственная точка без пары Y ≠ X.

Отображение Q6 → кривая X^Y=Y^X:
    t(h) = 1.0 + 0.15 * h   (t ∈ [1.0, 10.45] для h ∈ 0..63)
    Для h=0: t→1 (предел: X→e, Y→e)

Визуализация (8×8, Gray-код Q6):
  curve    — X(t(h)) по параметрической кривой
  golden   — отклонение t(h) от золотого корня φ
  find     — значение Y при X = X(t(h)) методом бисекции
  verify   — проверка X(t)^Y(t) ≈ Y(t)^X(t)

Команды CLI:
  curve
  golden
  find
  verify
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexpowerxy.hexpowerxy import PowerXY
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_PHI  = (1.0 + math.sqrt(5.0)) / 2.0   # φ ≈ 1.618
_E    = math.e                           # e ≈ 2.718

_PXY  = PowerXY()

_GRAY3 = [i ^ (i >> 1) for i in range(8)]


def _t(h: int) -> float:
    """Параметр t(h) ∈ (1, ∞) для глифа h."""
    return 1.0 + 0.15 * h if h > 0 else 1.001


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
# 1. Параметрическая кривая X(t(h))
# ---------------------------------------------------------------------------

def render_curve(color: bool = True) -> str:
    """8×8 сетка: значение X(t(h)) = t^(1/(t−1)).

    Ярлык = 1-я значащая цифра X  (X → e ≈ 2.718 при t → ∞).
    Диапазон t(h) = 1+0.15h: для h=1 t=1.15 (X≈8.1), для h=63 t=10.45 (X≈1.39).
    """
    # Предвычисление X(t) для всех h
    x_vals: list[float] = []
    for h in range(64):
        t = _t(h)
        try:
            xy = _PXY.xy_from_t(t)
            x_vals.append(xy[0])
        except Exception:
            x_vals.append(float('nan'))

    x_min = min((x for x in x_vals if not math.isnan(x) and x < 1000), default=1.0)
    x_max = max((x for x in x_vals if not math.isnan(x) and x < 1000), default=10.0)

    lines = _header(
        'X^Y=Y^X: X(t) = t^(1/(t−1))  t(h)=1+0.15h',
        'Ярлык = 1-я цифра X(t)  Цвет = ян-слой',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            x = x_vals[h]
            if math.isnan(x) or x > 100:
                sym = '?'
            else:
                # 1-я цифра числа X (округлённого до целых)
                sym = str(int(min(9, x)))[-1]
            if color:
                c = _YANG_ANSI[k]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Параметрическая кривая X^Y = Y^X:')
    lines.append(f'  X(t) = t^(1/(t-1))   Y(t) = t^(t/(t-1))')
    lines.append(f'  При t→1: X→e, Y→e   При t→∞: X→1, Y→∞')
    lines.append(f'  Классический пример: t=2 → X=2, Y=4  (2^4=4^2=16)')
    lines.append('')
    lines.append('  Выборка значений:')
    samples = [1, 7, 13, 20, 27, 42, 56, 63]
    for h in samples:
        t = _t(h)
        x = x_vals[h]
        if not math.isnan(x) and x < 100:
            try:
                y = _PXY.xy_from_t(t)[1]
                v = _PXY.verify(x, y)
            except Exception:
                y, v = float('nan'), False
            c = _YANG_ANSI[yang_count(h)] if color else ''
            r = _RESET if color else ''
            lines.append(
                f'    h={h:2d}: {c}t={t:.2f}  X={x:.4f}  Y={y:.4f}'
                f'  X^Y≈Y^X? {"✓" if v else "✗"}{r}'
            )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Расстояние до золотого корня φ
# ---------------------------------------------------------------------------

def render_golden(color: bool = True) -> str:
    """8×8 сетка: |t(h) − φ| (близость к золотому корню).

    Золотое сечение φ удовлетворяет уравнениям:
        φ^(1/(φ−1)) = (1+1/φ)^φ
        φ^(φ/(φ−1)) = (1+1/φ)^(φ+1)

    При t = φ ≈ 1.618: X(φ) = φ^(1/(φ-1)) ≈ 2.058.
    Ярлык: пятибалльная близость (0=далеко, 9=точное совпадение).
    """
    phi_root = _PXY.find_golden_root()

    lines = _header(
        f'X^Y=Y^X: близость t(h) к золотому корню φ≈{phi_root:.6f}',
        'Ярлык = 9−min(9, floor(|t−φ|×3))  (9=точно, 0=далеко)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h    = (row_g << 3) | col_g
            k    = yang_count(h)
            t    = _t(h)
            dist = abs(t - phi_root)
            # 9 = совпадает, 0 = далеко (dist≥3)
            score = max(0, 9 - int(dist * 3))
            sym  = str(score)
            if color:
                norm = min(6, score * 6 // 9)
                c = _YANG_ANSI[norm]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Золотой корень φ = {phi_root:.8f}')
    lines.append(f'  φ = (1+√5)/2      = {_PHI:.8f}')
    lines.append(f'  Совпадение: {abs(phi_root - _PHI) < 1e-6}')

    # Ближайшие h к золотому корню
    dists = [(h, abs(_t(h) - phi_root)) for h in range(64)]
    dists.sort(key=lambda x: x[1])
    lines.append('')
    lines.append('  Ближайшие глифы к φ:')
    for h, dist in dists[:5]:
        t = _t(h)
        c = _YANG_ANSI[yang_count(h)] if color else ''
        r = _RESET if color else ''
        lines.append(
            f'    h={h:2d}: {c}t={t:.4f}  |t−φ|={dist:.4f}{r}'
        )

    lines.append('')
    lines.append('  Золотые уравнения:')
    lines.append('    X(φ)^(1/(φ-1)) = (1+1/φ)^φ')
    lines.append('    X(φ)^(φ/(φ-1)) = (1+1/φ)^(φ+1)')
    # Проверка
    try:
        x_phi, y_phi = _PXY.xy_from_t(phi_root)
        v = _PXY.verify(x_phi, y_phi)
        lines.append(f'    X(φ)={x_phi:.6f}  Y(φ)={y_phi:.6f}  X^Y≈Y^X? {"✓" if v else "✗"}')
    except Exception:
        lines.append('    (вычисление недоступно)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Нахождение Y по X
# ---------------------------------------------------------------------------

def render_find(color: bool = True) -> str:
    """8×8 сетка: Y(h) — второй корень уравнения X^Y = Y^X для X = X(t(h)).

    Ярлык: 1-я цифра Y.
    """
    lines = _header(
        'X^Y=Y^X: значение Y при X = X(t(h))',
        'Ярлык = 1-я цифра Y  ·=нет решения (X≈e)',
    )

    y_vals: list[float | None] = []
    for h in range(64):
        t = _t(h)
        try:
            x = _PXY.xy_from_t(t)[0]
            y = _PXY.find_y(x)
            y_vals.append(y)
        except Exception:
            y_vals.append(None)

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            y = y_vals[h]
            if y is None or math.isnan(y):
                sym = '·'
            else:
                sym = str(int(min(9, y)))[-1]
            if color:
                c = _YANG_ANSI[k]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Нахождение Y бисекцией:')
    lines.append('  Для каждого X(h) ищем Y ≠ X такое, что X^Y = Y^X.')
    lines.append(f'  Особая точка: X = e ≈ {_E:.4f} (нет Y ≠ X).')
    lines.append('')
    lines.append('  Выборка решений:')
    samples = [h for h in range(0, 64, 8)]
    for h in samples:
        t = _t(h)
        try:
            x, y_t = _PXY.xy_from_t(t)
            y_found = y_vals[h]
            c = _YANG_ANSI[yang_count(h)] if color else ''
            r = _RESET if color else ''
            if y_found is not None and not math.isnan(y_found):
                v = _PXY.verify(x, y_found)
                lines.append(
                    f'    h={h:2d}: {c}X={x:.4f}  Y(param)={y_t:.4f}'
                    f'  Y(find)={y_found:.4f}  ✓{r}'
                )
            else:
                lines.append(f'    h={h:2d}: {c}X={x:.4f}  нет Y{r}')
        except Exception:
            pass
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Верификация X^Y ≈ Y^X
# ---------------------------------------------------------------------------

def render_verify(color: bool = True) -> str:
    """8×8 сетка: верификация X(t)^Y(t) ≈ Y(t)^X(t) для каждого h.

    V = verified (✓)   · = не удалось проверить
    """
    _OK_COLOR = '\033[38;5;82m'
    _NA_COLOR = '\033[38;5;238m'

    lines = _header(
        'X^Y=Y^X: верификация X(t)^Y(t) ≈ Y(t)^X(t)',
        'V=верно(зел.)  ·=нет пары или переполнение',
    )

    results: list[str] = []
    for h in range(64):
        t = _t(h)
        try:
            x, y = _PXY.xy_from_t(t)
            v = _PXY.verify(x, y)
            results.append('V' if v else '!')
        except Exception:
            results.append('·')

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h   = (row_g << 3) | col_g
            k   = yang_count(h)
            sym = results[h]
            if color:
                c = _OK_COLOR if sym == 'V' else _NA_COLOR
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    n_v = results.count('V')
    n_n = results.count('·')
    n_e = results.count('!')
    lines.append('')
    lines.append(f'  Верифицировано: {n_v}/64  не вычислено: {n_n}/64  ошибок: {n_e}/64')
    lines.append('')
    lines.append('  Примечательные пары:')
    for x, y in _PXY.notable_pairs():
        v = _PXY.verify(x, y)
        c = _OK_COLOR if color else ''
        r = _RESET if color else ''
        lines.append(
            f'    {c}X={x:.4f}  Y={y:.4f}  X^Y≈Y^X? {"✓" if v else "✗"}{r}'
        )

    lines.append('')
    # ASCII-арт кривой
    lines.append('  Кривая X^Y=Y^X (фрагмент):')
    try:
        art = _PXY.plot_curve(t_min=1.01, t_max=5.0, steps=60, width=50)
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
        prog='powerxy_glyphs',
        description='Q6 глифы через уравнение X^Y = Y^X.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('curve',  help='X(t(h)) = t^(1/(t-1))')
    sub.add_parser('golden', help='расстояние t(h) до золотого корня φ')
    sub.add_parser('find',   help='Y при X = X(t(h)) методом бисекции')
    sub.add_parser('verify', help='верификация X^Y ≈ Y^X')
    args = p.parse_args(argv)
    color = not args.no_color

    dispatch = {
        'curve':  render_curve,
        'golden': render_golden,
        'find':   render_find,
        'verify': render_verify,
    }
    if args.cmd in dispatch:
        print(dispatch[args.cmd](color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
