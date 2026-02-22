"""factoradic_glyphs — Система факториального счисления и перестановки через глифы.

Каждый глиф (0..63) — число в факториальной системе счисления,
соответствующее перестановке из первых 64 перестановок {1,...,6}.

Факториальная система (система Лемера):
  n = a₅·5! + a₄·4! + a₃·3! + a₂·2! + a₁·1! + a₀·0!
  где 0 ≤ aᵢ ≤ i  (т.е. a₅∈0..5, a₄∈0..4, ..., a₀∈0..0)

  Это взаимно однозначное кодирование рангов перестановок.
  6! = 720 перестановок {1,...,6},  первые 64 рангов → 64 глифа.

Анализ перестановок:
  • Неподвижные точки (fixed points): σ(i) = i
  • Беспорядки (derangements): σ(i) ≠ i для всех i
  • Инверсии: пары (i,j) с i<j но σ(i)>σ(j)
  • Циклический тип: (λ₁, λ₂, ...) длины циклов

Визуализация:
  rank     — 8×8 карта рангов 0..63: глиф h = ранг σₕ
  fixed    — раскраска по числу неподвижных точек
  cycles   — раскраска по числу циклов в перестановке
  drange   — беспорядки среди первых 64 перестановок

Команды CLI:
  rank
  fixed
  cycles
  drange
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexperms.hexperms import PermutationEngine
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _get_perms_64() -> list[list[int]]:
    """Первые 64 перестановок {1,...,6} в лексикографическом порядке."""
    pe = PermutationEngine(6)
    return [pe.unrank(r) for r in range(64)]


def _count_fixed(perm: list[int]) -> int:
    """Число неподвижных точек σ(i) = i."""
    return sum(1 for i, p in enumerate(perm, start=1) if p == i)


def _count_cycles(perm: list[int]) -> int:
    """Число циклов в циклическом разложении."""
    n = len(perm)
    visited = [False] * (n + 1)
    cycles = 0
    for start in range(1, n + 1):
        if not visited[start]:
            cycles += 1
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cur = perm[cur - 1]
    return cycles


def _count_inversions(perm: list[int]) -> int:
    """Число инверсий: |{(i,j): i<j, σ(i)>σ(j)}|."""
    n = len(perm)
    return sum(1 for i in range(n) for j in range(i + 1, n)
               if perm[i] > perm[j])


def _perm_to_str(perm: list[int]) -> str:
    return '[' + ','.join(str(p) for p in perm) + ']'


# ---------------------------------------------------------------------------
# 1. Карта рангов
# ---------------------------------------------------------------------------

def render_rank(color: bool = True) -> str:
    """
    8×8 сетка: глиф h = ранг h в лексикографическом порядке перестановок.

    Цвет = yang_count(h) — не зависит от перестановки, показывает структуру Q6.
    Под каждым глифом — сама перестановка.
    """
    perms = _get_perms_64()

    lines: list[str] = []
    lines.append('═' * 68)
    lines.append('  Первые 64 перестановки {1,...,6} в лексикографическом порядке')
    lines.append('  Глиф h ↔ перестановка ранга h (факториальная нумерация)')
    lines.append('  6! = 720 перестановок всего,  здесь отображены ранги 0..63')
    lines.append('═' * 68)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            perm = perms[h]
            short = ''.join(str(p) for p in perm)
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}{short}{_RESET}')
            else:
                lbl.append(short)
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    lines.append('  Факториальная система: ранг h → перестановка через цифры Лемера')
    lines.append('  Пример: ранг 0 = [1,2,3,4,5,6] (тождественная перестановка)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Неподвижные точки
# ---------------------------------------------------------------------------

def render_fixed(color: bool = True) -> str:
    """
    8×8 сетка: глиф h раскрашен по числу неподвижных точек σₕ.

    Беспорядки (0 фикс. точек) выделены особо.
    """
    perms = _get_perms_64()
    fixed_counts = [_count_fixed(p) for p in perms]
    derangements = [h for h in range(64) if fixed_counts[h] == 0]

    # Число D(6) = 265 беспорядков из 720, среди первых 64 их меньше
    n_dera_64 = len(derangements)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append('  Неподвижные точки σ(i)=i в первых 64 перестановках')
    lines.append(f'  Беспорядков (fixed=0) среди 64: {n_dera_64}')
    lines.append(f'  D(6)={265}  — всего беспорядков из 6! = 720')
    lines.append('  Цвет = число неподвижных точек (0..6)')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            fp = fixed_counts[h]
            rows3 = render_glyph(h)
            if color:
                is_dera = (fp == 0)
                c = (_YANG_BG[fp] + _BOLD) if is_dera else _YANG_ANSI[fp]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            fp = fixed_counts[h]
            perm = perms[h]
            short = ''.join(str(p) for p in perm)
            if color:
                c = _YANG_ANSI[fp]
                lbl.append(f'{c}f{fp}:{short}{_RESET}')
            else:
                lbl.append(f'f{fp}:{short}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Формула включений-исключений
    lines.append(f'  D(n) = n! · Σ_{{k=0}}^{{n}} (-1)^k / k!')
    lines.append(f'  D(6) = 720 · (1 - 1 + 1/2 - 1/6 + 1/24 - 1/120 + 1/720) = {265}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Циклическая структура
# ---------------------------------------------------------------------------

def render_cycles(color: bool = True) -> str:
    """
    8×8 сетка: глиф h раскрашен по числу циклов в σₕ.

    Тождественная = 6 циклов (1)(2)(3)(4)(5)(6).
    Транспозиция = 5 циклов (один 2-цикл + четыре 1-цикла).
    """
    perms = _get_perms_64()
    cycle_counts = [_count_cycles(p) for p in perms]
    inv_counts = [_count_inversions(p) for p in perms]

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append('  Циклическая структура первых 64 перестановок {1,...,6}')
    lines.append('  Число циклов: 1 = один большой цикл ... 6 = тождественная')
    lines.append('  Инверсии: |{(i<j): σ(i)>σ(j)}|')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            nc = cycle_counts[h]
            rows3 = render_glyph(h)
            if color:
                c = _YANG_ANSI[nc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            nc = cycle_counts[h]
            ni = inv_counts[h]
            if color:
                c = _YANG_ANSI[nc]
                lbl.append(f'{c}c{nc}i{ni:2d}{_RESET}')
            else:
                lbl.append(f'c{nc}i{ni:2d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    from collections import Counter
    cycle_hist = Counter(cycle_counts)
    lines.append('  Распределение по числу циклов:')
    for nc in sorted(cycle_hist):
        cnt = cycle_hist[nc]
        if color:
            c = _YANG_ANSI[nc]
            lines.append(f'  {c}  {nc} цикл(а): {cnt} перестановок{_RESET}')
        else:
            lines.append(f'    {nc} цикл(а): {cnt} перестановок')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Беспорядки
# ---------------------------------------------------------------------------

def render_drange(color: bool = True) -> str:
    """
    Визуализировать беспорядки (derangements) среди первых 64 перестановок.

    Беспорядок = нет неподвижных точек: σ(i) ≠ i для всех i.
    """
    perms = _get_perms_64()
    is_dera = [_count_fixed(p) == 0 for p in perms]
    dera_indices = [h for h in range(64) if is_dera[h]]
    n_dera = len(dera_indices)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Беспорядки среди рангов 0..63  (из 6! = 720)')
    lines.append(f'  D(6) = 265  ≈ 720/e ≈ 264.9')
    lines.append(f'  Среди первых 64: {n_dera} беспорядков '
                 f'({n_dera/64*100:.1f}%)')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            dera = is_dera[h]
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                if dera:
                    c = _YANG_BG[yc] + _BOLD
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            dera = is_dera[h]
            perm = perms[h]
            fp = _count_fixed(perm)
            if color:
                c = _YANG_ANSI[5] if dera else _YANG_ANSI[0]
                tag = 'D!' if dera else f'f={fp}'
                lbl.append(f'{c}{tag:>4}{_RESET}')
            else:
                lbl.append(f'{"D!" if dera else f"f={fp}":>4}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Первые 6 беспорядков
    lines.append(f'  Первые {min(6, n_dera)} беспорядков:')
    for h in dera_indices[:6]:
        perm = perms[h]
        inv = _count_inversions(perm)
        nc = _count_cycles(perm)
        if color:
            c = _YANG_ANSI[yang_count(h)]
            lines.append(f'  {c}  ранг={h:2d}: {_perm_to_str(perm)}'
                         f'  inv={inv}  cycles={nc}{_RESET}')
        else:
            lines.append(f'    ранг={h:2d}: {_perm_to_str(perm)}'
                         f'  inv={inv}  cycles={nc}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='factoradic_glyphs',
        description='Система факториального счисления через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('rank',   help='ранг каждого глифа = перестановка')
    sub.add_parser('fixed',  help='неподвижные точки перестановок')
    sub.add_parser('cycles', help='цикличная структура перестановок')
    sub.add_parser('drange', help='беспорядки среди первых 64')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'rank':
        print(render_rank(color))
    elif args.cmd == 'fixed':
        print(render_fixed(color))
    elif args.cmd == 'cycles':
        print(render_cycles(color))
    elif args.cmd == 'drange':
        print(render_drange(color))


if __name__ == '__main__':
    main()
