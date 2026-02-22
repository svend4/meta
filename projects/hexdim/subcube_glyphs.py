"""subcube_glyphs — Размерная структура Q6 через глифы.

Каждый глиф (0..63) — вершина 6-мерного гиперкуба Q6.
Q6 содержит богатую иерархию подкубов:
  k-подкуб: 2^k вершин, все на расстоянии ≤k друг от друга
  Фиксированные биты определяют «адрес» подкуба.

Число k-подкубов = C(6,k) · 2^(6-k):
  k=1: 6·32 = 192   (рёбра)
  k=2: 15·16 = 240  (квадраты)
  k=3: 20·8 = 160   (трёхмерные кубы)
  k=4: 15·4 = 60    (тессеракты, 4D-кубы)
  k=5: 6·2 = 12     (5D-подкубы)
  k=6: 1             (весь Q6)

Разложение гексаграмм:
  Тригра́мма = 3-битная часть → верхний и нижний триграм
  Q6 = Q3 × Q3 (произведение двух 3-кубов)

Визуализация:
  subcubes [--k 3]  — выделить все k-подкубы (раскраска по индексу)
  tesseracts        — все 60 тессерактов Q6
  trigrams          — разложение на верхний/нижний триграм
  layers  [--k 3]   — уровни Хэмминга шара из центра 0

Команды CLI:
  subcubes  [--k 3]
  tesseracts
  trigrams
  layers    [--k 4]
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexdim.hexdim import (
    all_subcubes, all_tesseracts, all_cubes,
    trigram_decomposition, subcube_count, tesseract_count, cube_count,
    trigram_name,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Цвета для подкубов (по индексу)
# ---------------------------------------------------------------------------

_SUBCUBE_COLORS = [
    '\033[38;5;27m',   '\033[38;5;82m',   '\033[38;5;196m',
    '\033[38;5;208m',  '\033[38;5;201m',  '\033[38;5;226m',
    '\033[38;5;39m',   '\033[38;5;238m',  '\033[38;5;46m',
    '\033[38;5;51m',   '\033[38;5;160m',  '\033[38;5;11m',
    '\033[38;5;93m',   '\033[38;5;130m',  '\033[38;5;50m',
    '\033[38;5;200m',
]


# ---------------------------------------------------------------------------
# 1. k-подкубы: раскраска по принадлежности
# ---------------------------------------------------------------------------

def render_subcubes(k: int = 3, color: bool = True) -> str:
    """
    8×8 сетка: каждый глиф раскрашен по принадлежности одному из k-подкубов.

    Если вершина принадлежит нескольким подкубам — берётся первый.
    """
    subcubes = all_subcubes(k)
    total = subcube_count(k)

    # Для каждой вершины — номер первого содержащего подкуба
    vertex_cube: dict[int, int] = {}
    for idx, (free_axes, fixed_val, verts) in enumerate(subcubes):
        for h in verts:
            if h not in vertex_cube:
                vertex_cube[h] = idx

    # Подсчёт числа подкубов, содержащих каждую вершину
    vertex_count = [0] * 64
    for _, _, verts in subcubes:
        for h in verts:
            vertex_count[h] += 1

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  k-подкубы Q6 для k={k}')
    lines.append(f'  Число {k}-подкубов = C(6,{k})·2^(6−{k}) = '
                 f'{math.comb(6,k)}·{2**(6-k)} = {total}')
    lines.append(f'  Каждая вершина принадлежит C(6,{k}) = {math.comb(6,k)} '
                 f'различным {k}-подкубам')
    lines.append(f'  Цвет = номер первого содержащего подкуба')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            ci = vertex_cube.get(h, 0)
            rows3 = render_glyph(h)
            if color:
                c = _SUBCUBE_COLORS[ci % len(_SUBCUBE_COLORS)]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            cnt = vertex_count[h]
            if color:
                ci = vertex_cube.get(h, 0)
                c = _SUBCUBE_COLORS[ci % len(_SUBCUBE_COLORS)]
                lbl.append(f'{c}×{cnt:2d}{_RESET}')
            else:
                lbl.append(f'×{cnt:2d}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Первые 4 подкуба
    lines.append(f'  Примеры {k}-подкубов (первые 4 из {total}):')
    for idx, (free_axes, fixed_val, verts) in enumerate(subcubes[:4]):
        mask = format(fixed_val, '06b') if isinstance(fixed_val, int) else str(fixed_val)
        v_list = sorted(verts)[:4]
        if color:
            c = _SUBCUBE_COLORS[idx % len(_SUBCUBE_COLORS)]
            lines.append(f'  {c}  [{idx}] оси={free_axes}  '
                         f'вершины={v_list}...{_RESET}')
        else:
            lines.append(f'    [{idx}] оси={free_axes}  вершины={v_list}...')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Тессеракты
# ---------------------------------------------------------------------------

def render_tesseracts(color: bool = True) -> str:
    """
    Показать первые несколько тессерактов (4D-подкубов) Q6.

    Каждый тессеракт = 16 вершин Q6 на 4-мерном подкубе.
    Число тессерактов = C(6,4)·2² = 15·4 = 60.
    """
    tesseracts = all_tesseracts()
    n_tc = tesseract_count()

    # Для каждой вершины — номер её тессеракта (первый)
    vertex_tc: dict[int, int] = {}
    tc_counts = [0] * 64
    for idx, verts in enumerate(tesseracts):
        for h in verts:
            tc_counts[h] += 1
            if h not in vertex_tc:
                vertex_tc[h] = idx

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Тессеракты (4D-подкубы) в Q6')
    lines.append(f'  Число тессерактов = C(6,4)·4 = {n_tc}')
    lines.append(f'  Каждый тессеракт: 16 вершин, 32 ребра, 24 грани')
    lines.append(f'  Каждая вершина принадлежит C(6,4) = 15 тессерактам')
    lines.append('═' * 64)

    # Карта: раскраска по принадлежности тессеракту
    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            ti = vertex_tc.get(h, 0)
            rows3 = render_glyph(h)
            if color:
                c = _SUBCUBE_COLORS[ti % len(_SUBCUBE_COLORS)]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            cnt = tc_counts[h]
            if color:
                c = _YANG_ANSI[min(6, cnt // 3)]
                lbl.append(f'{c}t{cnt:2d}{_RESET}')
            else:
                lbl.append(f't{cnt:2d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Первый тессеракт подробно
    t0 = sorted(tesseracts[0])
    lines.append(f'  Тессеракт 0: {t0}')
    glyphs_t0 = [render_glyph(h) for h in t0[:8]]
    if color:
        glyphs_t0 = [
            [_SUBCUBE_COLORS[0] + r + _RESET for r in g]
            for g in glyphs_t0
        ]
    for ri in range(3):
        lines.append('  ' + '  '.join(g[ri] for g in glyphs_t0) + '  ...')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Разложение на триграммы
# ---------------------------------------------------------------------------

def render_trigrams(color: bool = True) -> str:
    """
    Разложить Q6 на нижний (биты 0-2) и верхний (биты 3-5) триграм.

    Q6 ≅ Q3 × Q3: каждая гексаграмма = пара триграмм.
    Цвет = yang_count верхнего или нижнего триграма.
    """
    # Разбиение по верхнему триграму (биты 3-5)
    upper_group: dict[int, list[int]] = {}
    for h in range(64):
        lo, hi = trigram_decomposition(h)   # (lower, upper)
        if hi not in upper_group:
            upper_group[hi] = []
        upper_group[hi].append(h)

    lines: list[str] = []
    lines.append('╔' + '═' * 62 + '╗')
    lines.append('║  Разложение Q6 → Q3 × Q3: верхний × нижний триграм' + ' ' * 10 + '║')
    lines.append('║  h = (upper << 3) | lower,  upper,lower ∈ {0,...,7}' + ' ' * 9 + '║')
    lines.append('╚' + '═' * 62 + '╝')
    lines.append('')

    for hi in range(8):
        members = sorted(upper_group.get(hi, []))
        hi_name = trigram_name(hi) if trigram_name(hi) else f'tri{hi}'
        yc_hi = yang_count(hi)

        if color:
            c = _YANG_ANSI[yc_hi]
            lines.append(f'  {c}Верхний={hi:3d} ({format(hi,"03b")}) = '
                         f'{hi_name:<12} yang={yc_hi}{_RESET}')
        else:
            lines.append(f'  Верхний={hi:3d} ({format(hi,"03b")}) = '
                         f'{hi_name:<12} yang={yc_hi}')

        glyphs_m = [render_glyph(h) for h in members]
        if color:
            glyphs_m = [
                [_YANG_ANSI[yc_hi] + r + _RESET for r in g]
                for g in glyphs_m
            ]
        for ri in range(3):
            lines.append('    ' + '  '.join(g[ri] for g in glyphs_m))
        lower_names = []
        for h in members:
            lo, _ = trigram_decomposition(h)
            ln = trigram_name(lo) if trigram_name(lo) else f'tri{lo}'
            lower_names.append(f'{lo}({ln})')
        lines.append('    нижн: ' + '  '.join(lower_names))
        lines.append('')

    lines.append('  Q6 = Q3 × Q3: 8 групп по 8 гексаграмм')
    lines.append('  Верхний и нижний триграмы независимы (произведение графов)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Уровни Хэмминга (шары)
# ---------------------------------------------------------------------------

def render_layers(max_k: int = 4, color: bool = True) -> str:
    """
    Уровни сферы Хэмминга от центра 0: S(0,k) = {h: yang_count(h)=k}.

    Показать нарастающий шар B(0,0) ⊂ B(0,1) ⊂ ... ⊂ Q6.
    """
    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Шары Хэмминга B(0,k) в Q6  k=0..{max_k}')
    lines.append(f'  |B(0,k)| = Σ_{{i=0}}^{{k}} C(6,i)')
    lines.append('═' * 64)

    cumulative = 0
    for k in range(max_k + 1):
        sphere_k = [h for h in range(64) if yang_count(h) == k]
        cumulative += len(sphere_k)
        binom_k = math.comb(6, k)

        if color:
            c = _YANG_ANSI[k]
            lines.append(f'\n  {c}S(0,{k}): C(6,{k})={binom_k} вершин   '
                         f'|B(0,{k})|={cumulative}{_RESET}')
        else:
            lines.append(f'\n  S(0,{k}): C(6,{k})={binom_k} вершин   '
                         f'|B(0,{k})|={cumulative}')

        glyphs_s = [render_glyph(h) for h in sphere_k]
        if color:
            glyphs_s = [
                [_YANG_ANSI[k] + r + _RESET for r in g]
                for g in glyphs_s
            ]
        for ri in range(3):
            lines.append('  ' + '  '.join(g[ri] for g in glyphs_s))
        lines.append('  ' + '  '.join(format(h, '06b') for h in sphere_k))

    lines.append('')
    lines.append('  Спектр расстояний Q6: [1, 6, 15, 20, 15, 6, 1]')
    lines.append('  = [C(6,k) for k in 0..6]')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='subcube_glyphs',
        description='Размерная структура Q6 через глифы гексаграмм',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('subcubes', help='k-подкубы Q6')
    s.add_argument('--k', type=int, default=3, choices=[1, 2, 3, 4, 5])

    sub.add_parser('tesseracts', help='все 60 тессерактов Q6')
    sub.add_parser('trigrams', help='разложение Q6 = Q3 × Q3')

    s = sub.add_parser('layers', help='уровни шара Хэмминга от 0')
    s.add_argument('--k', type=int, default=4, choices=[1, 2, 3, 4, 5, 6])

    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'subcubes':
        print(render_subcubes(k=args.k, color=color))
    elif args.cmd == 'tesseracts':
        print(render_tesseracts(color=color))
    elif args.cmd == 'trigrams':
        print(render_trigrams(color=color))
    elif args.cmd == 'layers':
        print(render_layers(max_k=args.k, color=color))


if __name__ == '__main__':
    main()
