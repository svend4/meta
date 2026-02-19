"""buffon_glyphs — Обобщённая задача Бюффона через глифы Q6.

Каждый глиф h (0..63) интерпретируется как параметр мозаики:
  yang_count(h) = k → «активных» плиток k из 6 видов.

Формула Германа (обобщение Бюффона):
  W = L · U / (π · F)
  L — длина иглы,  U — периметр плитки,  F — площадь плитки

Золотой прямоугольник (φ = (1+√5)/2):
  Для иглы длиной L = a·e/2 и плитки с a=b/φ:
  W = φ·e/π ≈ 1.4008  (константа, не зависит от a!)

Соответствие Q6:
  Бит i = «включена плитка i»
  yang_count(h) = число активных плиток
  W(h) = Σ W_i · бит_i — взвешенное число пересечений

Визуализация:
  tiles    — W для 6 типов мозаик (квадрат, прямоуг., шестиугольник...)
  golden   — золотое тождество φ·e/π
  crossing — W(h) для каждого глифа (взвешенная сумма)
  simulate — Монте-Карло: оценка W для каждого yang-слоя

Команды CLI:
  tiles
  golden
  crossing
  simulate
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexbuffon.hexbuffon import BuffonParquet, buffon_general
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_PHI = (1 + math.sqrt(5)) / 2
_E   = math.e
_PI  = math.pi

_bp = BuffonParquet()

# 6 типов плиток: W = L·U/(π·F) для L=1, разные U и F
_TILES = [
    ('Квадрат 1×1',      4.0,   1.0,    _bp.square(a=1, needle=1)),
    ('Прямоугольник 1×2',6.0,   2.0,    _bp.rectangular(a=1, b=2, needle=1)),
    ('Квадрат 2×2',      8.0,   4.0,    _bp.square(a=2, needle=1)),
    ('Шестиугольник r=1',6*1.0, 3*math.sqrt(3)/2, _bp.hexagonal(r=1, needle=1)),
    ('Прямоугольник 1×3',8.0,   3.0,    _bp.rectangular(a=1, b=3, needle=1)),
    ('Золотой прямоуг.', None,  None,   _bp.golden_rectangle(a=1)),
]


# ---------------------------------------------------------------------------
# 1. Сравнение типов мозаик
# ---------------------------------------------------------------------------

def render_tiles(color: bool = True) -> str:
    """
    8×8 сетка: каждый глиф h раскрашен по типу мозаики (бит i).

    6 битов = 6 типов плиток. Цвет = тип самой «активной» плитки.
    """
    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Задача Бюффона: 6 типов мозаик ↔ 6 битов Q6')
    lines.append('  Формула: W = L·U/(π·F)   (L=игла, U=периметр, F=площадь)')
    lines.append('  Каждый бит h[i]=1 означает «активна плитка типа i»')
    lines.append('═' * 66)
    lines.append('')

    # Таблица плиток
    lines.append('  Типы мозаик (для L=1):')
    for i, (name, U, F, W) in enumerate(_TILES):
        u_str = f'{U:.2f}' if U is not None else '?'
        f_str = f'{F:.4f}' if F is not None else '?'
        if color:
            c = _YANG_ANSI[i + 1]
            lines.append(f'  {c}  бит {i}: {name:<22} W={W:.4f}{_RESET}')
        else:
            lines.append(f'    бит {i}: {name:<22} W={W:.4f}')
    lines.append('')

    # Для каждого глифа: «доминирующий» тип = бит с наибольшим W среди активных
    W_vals = [t[3] for t in _TILES]

    def dominant_tile(h: int) -> int:
        active = [(i, W_vals[i]) for i in range(6) if (h >> i) & 1]
        if not active:
            return -1
        return max(active, key=lambda x: x[1])[0]

    def total_W(h: int) -> float:
        return sum(W_vals[i] for i in range(6) if (h >> i) & 1)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            dt = dominant_tile(h)
            rows3 = render_glyph(h)
            if color:
                if dt < 0:
                    c = _YANG_ANSI[0]
                else:
                    c = _YANG_ANSI[dt + 1]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            tw = total_W(h)
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}{tw:.2f}{_RESET}')
            else:
                lbl.append(f'{tw:.2f}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Золотое тождество φ·e/π
# ---------------------------------------------------------------------------

def render_golden(color: bool = True) -> str:
    """
    Визуализация золотого тождества Германа: W = φ·e/π.

    Для золотого прямоугольника a×(a/φ) с иглой L = a·e/2:
    W = φ·e/π ≈ 1.4008  (независимо от a!)

    8×8 карта: глифы раскрашены по отклонению W(h) от φ·e/π.
    """
    golden_W = _PHI * _E / _PI

    # Для каждого h: W(h) = golden_rectangle(a = yang_count(h)+1)
    W_by_yang = [_bp.golden_rectangle(a=k + 1) if k >= 0 else 0.0 for k in range(7)]

    # Отклонение от теоретического значения
    def deviation(h: int) -> float:
        k = yang_count(h)
        return abs(W_by_yang[k] - golden_W)

    verify = _bp.golden_rectangle_verify()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Золотое тождество Германа: W = φ·e/π')
    lines.append(f'  φ = (1+√5)/2 = {_PHI:.6f}')
    lines.append(f'  e = {_E:.6f}')
    lines.append(f'  π = {_PI:.6f}')
    lines.append(f'  φ·e/π = {golden_W:.6f}   (вычислено: {verify["exact"]:.6f}   ✓={verify["ok"]})')
    lines.append('  8×8 карта: цвет = yang_count = «вес» плитки')
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
            k = yang_count(h)
            w = W_by_yang[k]
            if color:
                c = _YANG_ANSI[k]
                lbl.append(f'{c}{w:.3f}{_RESET}')
            else:
                lbl.append(f'{w:.3f}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # По yang-слоям
    lines.append('  W(a) = φ·e/π для золотого прямоугольника (по yang-слоям):')
    for k in range(7):
        w = W_by_yang[k]
        diff = w - golden_W
        import math as _math
        bar_len = int(w * 15)
        bar = '█' * bar_len
        if color:
            c = _YANG_ANSI[k]
            lines.append(f'  {c}  yang={k}: W={w:.6f}  diff={diff:+.2e}  {bar}{_RESET}')
        else:
            lines.append(f'    yang={k}: W={w:.6f}  diff={diff:+.2e}  {bar}')

    lines.append(f'\n  Все W = φ·e/π (одна константа) — золотое тождество Германа!')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. W(h) — взвешенное число пересечений
# ---------------------------------------------------------------------------

def render_crossing(color: bool = True) -> str:
    """
    8×8 сетка: для каждого глифа h — W(h) = Σ W_tile_i · bit_i.

    Суммируем числа пересечений для каждой активной плитки.
    Максимум при h=63 (все 6 плиток активны).
    """
    W_vals = [t[3] for t in _TILES]

    def total_W(h: int) -> float:
        return sum(W_vals[i] for i in range(6) if (h >> i) & 1)

    W_all = [total_W(h) for h in range(64)]
    W_max = max(W_all)
    W_min = min(w for w in W_all if w > 0)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  W(h) = взвешенное число пересечений Бюффона')
    lines.append('  W(h) = Σᵢ W_i · bit_i(h)   (L=1, по активным плиткам)')
    lines.append(f'  Мин W(h>0) = {W_min:.4f}   Макс W(63) = {W_max:.4f}')
    lines.append('  Цвет = yang_count(h),  число = W(h)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            w = W_all[h]
            yc = yang_count(h)
            rows3 = render_glyph(h)
            if color:
                if h == 63:
                    c = _YANG_BG[yc] + _BOLD
                elif h == 0:
                    c = _YANG_ANSI[0]
                else:
                    c = _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            w = W_all[h]
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}{w:.2f}{_RESET}')
            else:
                lbl.append(f'{w:.2f}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Среднее W по yang-слоям
    lines.append('  Средний W по yang-слоям:')
    import math
    for k in range(7):
        layer = [W_all[h] for h in range(64) if yang_count(h) == k]
        avg = sum(layer) / len(layer) if layer else 0
        binom = math.comb(6, k)
        bar = '█' * int(avg * 8)
        if color:
            c = _YANG_ANSI[k]
            lines.append(f'  {c}  yang={k} ({binom:2d} гл.): avg_W={avg:.4f}  {bar}{_RESET}')
        else:
            lines.append(f'    yang={k} ({binom:2d} гл.): avg_W={avg:.4f}  {bar}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Монте-Карло
# ---------------------------------------------------------------------------

def render_simulate(color: bool = True) -> str:
    """
    Монте-Карло оценка W для квадратной мозаики, по yang-слоям.

    Для каждого yang-слоя k: игла длиной L=yang_count(h)/6,
    плитка = квадрат со стороной 1.
    """
    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Монте-Карло: оценка W для задачи Бюффона')
    lines.append('  Плитка: квадрат 1×1   Длина иглы L = yang_count(h) / 6')
    lines.append('  Точное W(L) = 4L/π   (Бюффон, квадратная мозаика)')
    lines.append('  n=10 000 испытаний на слой')
    lines.append('═' * 66)
    lines.append('')

    yang_layers = []
    for k in range(7):
        L = k / 6.0 if k > 0 else 0.001
        exact_W = _bp.square(a=1, needle=L)
        if k > 0:
            sim = _bp.simulate('square', needle=L, n=10000, seed=k * 7, a=1)
            est_W = sim['estimated_W']
            err = sim['error_pct']
        else:
            est_W = 0.0
            err = 0.0
        yang_layers.append((k, L, exact_W, est_W, err))

        bar_exact = '█' * int(exact_W * 12)
        bar_est   = '░' * int(est_W * 12)
        if color:
            c = _YANG_ANSI[k]
            lines.append(f'  {c}yang={k}: L={L:.3f}  '
                         f'W_exact={exact_W:.4f}  '
                         f'W_MC={est_W:.4f}  '
                         f'err={err:.2f}%{_RESET}')
            lines.append(f'  {c}  точное: {bar_exact}{_RESET}')
            lines.append(f'  {c}  MC:     {bar_est}{_RESET}')
        else:
            lines.append(f'  yang={k}: L={L:.3f}  '
                         f'W_exact={exact_W:.4f}  '
                         f'W_MC={est_W:.4f}  '
                         f'err={err:.2f}%')
            lines.append(f'    точное: {bar_exact}')
            lines.append(f'    MC:     {bar_est}')
        lines.append('')

    # 8×8 карта: цвет = yang_count
    lines.append('  8×8 карта: глифы раскрашены по yang (= L = длина иглы/6)')
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
            k = yang_count(h)
            _, L, exact_W, _, _ = yang_layers[k]
            if color:
                c = _YANG_ANSI[k]
                lbl.append(f'{c}{exact_W:.2f}{_RESET}')
            else:
                lbl.append(f'{exact_W:.2f}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='buffon_glyphs',
        description='Обобщённая задача Бюффона через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('tiles',    help='6 типов мозаик как 6 битов Q6')
    sub.add_parser('golden',   help='золотое тождество φ·e/π')
    sub.add_parser('crossing', help='W(h) = взвешенное число пересечений')
    sub.add_parser('simulate', help='Монте-Карло оценка W по yang-слоям')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'tiles':
        print(render_tiles(color))
    elif args.cmd == 'golden':
        print(render_golden(color))
    elif args.cmd == 'crossing':
        print(render_crossing(color))
    elif args.cmd == 'simulate':
        print(render_simulate(color))


if __name__ == '__main__':
    main()
