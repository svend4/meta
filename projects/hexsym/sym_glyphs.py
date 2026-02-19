"""sym_glyphs — Группа автоморфизмов Aut(Q6) через глифы.

Каждый глиф (0..63) — вершина Q6.
Aut(Q6) = (Z₂)⁶ ⋊ S₆:  биты-отражения × перестановки битов.

|Aut(Q6)| = 2⁶ · 6! = 64 · 720 = 46 080

Структура группы:
  • (Z₂)⁶ — 64 отображения «XOR маска»: v ↦ v ⊕ mask
  • S₆     — 720 перестановок битов: v ↦ π(v)
  • Вместе: |Aut| = 64 · 720 = 46 080

Yang-орбиты (оbits числа единичных битов):
  |S(0,k)| = C(6,k): 1, 6, 15, 20, 15, 6, 1  — ровно 7 орбит.
  Aut(Q6) действует транзитивно на каждом слое yang=k.

Визуализация:
  yang      — 7 орбит по числу единичных битов (yang_count)
  fixed     — неподвижные точки генераторов Aut(Q6)
  antipodal — антиподальные пары {h, h⊕63}
  burnside  — таблица Бернсайда/Полиа для n цветов

Команды CLI:
  yang
  fixed
  antipodal
  burnside  [--colors n]
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexsym.hexsym import (
    identity_aut, bit_flip_single, bit_permutation, bit_transposition,
    aut_generators, s6_generators,
    yang_orbits, antipodal_orbits,
    fixed_points, cycle_decomposition, cycle_count,
    burnside_count, burnside_subset, polya_count,
    Automorphism,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_AUT_ORDER = 64 * 720   # |Aut(Q6)|


# ---------------------------------------------------------------------------
# 1. Yang-орбиты Aut(Q6)
# ---------------------------------------------------------------------------

def render_yang(color: bool = True) -> str:
    """
    8×8 сетка: 7 орбит Aut(Q6) по yang_count.

    Aut(Q6) действует транзитивно на каждом слое {h : yang_count(h)=k}.
    Это 7 орбит размеров C(6,0)..C(6,6) = 1,6,15,20,15,6,1.
    """
    import math
    yo = yang_orbits()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Орбиты Aut(Q6) на Q6: 7 слоёв по yang_count')
    lines.append(f'  |Aut(Q6)| = 2⁶·6! = {_AUT_ORDER}')
    lines.append('  Aut(Q6) транзитивно на каждом yang-слое')
    lines.append('  Цвет = yang_count(h) = число единичных битов')
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
            if color:
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}y={yc}{_RESET}')
            else:
                lbl.append(f'y={yc}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    lines.append('  Yang-орбиты (|орбиты| = C(6,k)):')
    for k, orb in enumerate(yo):
        binom = math.comb(6, k)
        stab = _AUT_ORDER // len(orb)
        if color:
            c = _YANG_ANSI[k]
            lines.append(f'  {c}  k={k}: {len(orb):2d} вершин = C(6,{k})={binom}  '
                         f'|Stab|={stab}{_RESET}')
        else:
            lines.append(f'    k={k}: {len(orb):2d} вершин = C(6,{k})={binom}  '
                         f'|Stab|={stab}')
    lines.append(f'  Σ|орбит| = {sum(len(o) for o in yo)} = 64 ✓')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Неподвижные точки генераторов
# ---------------------------------------------------------------------------

def render_fixed(color: bool = True) -> str:
    """
    8×8 сетка: для каждого из 6 генераторов Aut(Q6) — его неподвижные точки.

    Генераторы:
      g_i = bit_transposition(i, i+1)  (i=0..4) — 5 транспозиций соседних битов
      g_5 = bit_flip_single(0)          — отражение первого бита
    """
    gens = aut_generators()

    # Для каждой вершины: сколько генераторов её фиксируют
    fix_by_gen: list[list[int]] = []
    for g in gens:
        fp = set(fixed_points(g))
        fix_by_gen.append([1 if h in fp else 0 for h in range(64)])

    fix_total = [sum(fix_by_gen[i][h] for i in range(len(gens)))
                 for h in range(64)]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Неподвижные точки генераторов Aut(Q6)')
    lines.append(f'  {len(gens)} генераторов: {len(gens)-1} транспозиций битов + 1 отражение')
    lines.append('  fix(h) = числo генераторов, сохраняющих вершину h')
    lines.append('  Жирный = фиксируется всеми генераторами (h=0: все нули)')
    lines.append('═' * 66)

    max_fix = max(fix_total)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            ft = fix_total[h]
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_BG[yc] + _BOLD if ft == max_fix else _YANG_ANSI[ft]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            ft = fix_total[h]
            if color:
                c = _YANG_ANSI[min(ft, 6)]
                lbl.append(f'{c}f{ft}{_RESET}')
            else:
                lbl.append(f'f{ft}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Подробно по каждому генератору
    lines.append('  Генераторы Aut(Q6):')
    for i, g in enumerate(gens):
        fp = fixed_points(g)
        cc = cycle_count(g)
        if color:
            c = _YANG_ANSI[i % 7]
            lines.append(f'  {c}  g{i}: {g}   '
                         f'Fix={len(fp):2d}   циклов={cc}{_RESET}')
        else:
            lines.append(f'    g{i}: {g}   Fix={len(fp):2d}   циклов={cc}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Антиподальные пары
# ---------------------------------------------------------------------------

def render_antipodal(color: bool = True) -> str:
    """
    8×8 сетка: антиподальные пары {h, h⊕63}.

    Антипод h* = h ⊕ 0b111111 = h XOR 63 (инверсия всех битов).
    Расстояние Хэмминга d(h, h*) = 6 (максимальное).
    Есть ровно 32 антиподальные пары.
    """
    ao = antipodal_orbits()   # список пар frozenset

    # Для каждой вершины — индекс её пары
    pair_idx = [-1] * 64
    for idx, pair in enumerate(ao):
        for h in pair:
            pair_idx[h] = idx

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Антиподальные пары Q6: {h, h⊕63}')
    lines.append('  Антипод h* = h XOR 63 — дополнение всех битов')
    lines.append(f'  32 пары   d(h, h*) = 6   yang(h) + yang(h*) = 6')
    lines.append('  Цвет пары: yang_count ∈ {0..3} (пара содержит both части)')
    lines.append('═' * 66)

    _PAIR_COLORS = [
        '\033[38;5;27m',  '\033[38;5;82m',  '\033[38;5;196m',
        '\033[38;5;208m', '\033[38;5;201m', '\033[38;5;226m',
        '\033[38;5;39m',  '\033[38;5;46m',  '\033[38;5;51m',
        '\033[38;5;160m', '\033[38;5;11m',  '\033[38;5;93m',
        '\033[38;5;130m', '\033[38;5;50m',  '\033[38;5;200m',
        '\033[38;5;238m',
    ]

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            pi = pair_idx[h]
            rows3 = render_glyph(h)
            if color:
                antip = h ^ 63
                # Цвет: по yang меньшего элемента пары
                yc = min(yang_count(h), yang_count(antip))
                is_lower = (h < antip)   # меньший в паре = ярче
                c = _YANG_BG[yc] + _BOLD if is_lower else _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            antip = h ^ 63
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}↔{antip:02d}{_RESET}')
            else:
                lbl.append(f'↔{antip:02d}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    lines.append('  Первые 8 антиподальных пар {h, h*=h⊕63}:')
    for idx, pair in enumerate(ao[:8]):
        h1, h2 = sorted(pair)
        if color:
            yc = yang_count(h1)
            c = _YANG_ANSI[yc]
            lines.append(f'  {c}  {{{h1:02d},{h2:02d}}}  '
                         f'{format(h1,"06b")} ↔ {format(h2,"06b")}  '
                         f'yang={yang_count(h1)}+{yang_count(h2)}=6{_RESET}')
        else:
            lines.append(f'    {{{h1:02d},{h2:02d}}}  '
                         f'{format(h1,"06b")} ↔ {format(h2,"06b")}  '
                         f'yang={yang_count(h1)}+{yang_count(h2)}=6')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Бернсайд/Полиа: число раскрасок
# ---------------------------------------------------------------------------

def render_burnside(n_colors: int = 2, color: bool = True) -> str:
    """
    Таблица Бернсайда/Полиа: число раскрасок Q6 при n цветах.

    Polya(n) = (1/|G|) Σ_g n^{cycle_count(g)}
    — число существенно различных раскрасок Q6 при действии Aut(Q6).
    """
    gens = aut_generators()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Теорема Бернсайда/Полиа для Aut(Q6)')
    lines.append(f'  Polya(n) = (1/{_AUT_ORDER}) Σ_g n^{{c(g)}}')
    lines.append(f'  = число раскрасок вершин Q6 в n цветов с точн. до Aut(Q6)')
    lines.append('═' * 66)
    lines.append('')

    # Цикловой индекс: считаем числа циклов для каждого генератора
    lines.append(f'  Цикловой индекс генераторов (c(g) = число циклов на Q6):')
    for i, g in enumerate(gens):
        cc = cycle_count(g)
        fp = fixed_points(g)
        if color:
            c = _YANG_ANSI[i % 7]
            lines.append(f'  {c}  g{i}: цiklов={cc}   fix={len(fp)}{_RESET}')
        else:
            lines.append(f'    g{i}: циклов={cc}   fix={len(fp)}')

    lines.append('')
    lines.append('  Polya(n) для малых n:')
    for n in range(2, min(n_colors + 1, 8)):
        pc = polya_count(n)
        if color:
            c = _YANG_ANSI[(n - 1) % 7]
            lines.append(f'  {c}  Polya({n}) = {pc:,}{_RESET}')
        else:
            lines.append(f'    Polya({n}) = {pc:,}')

    lines.append('')
    lines.append(f'  Число раскрасок k единичных вершин (k-подмножества):')
    lines.append('  burnside_subset(k) = число орбит на k-элементных подмн. Q6')
    for k in [0, 1, 2, 3, 6, 10, 16, 32]:
        bs = burnside_subset(k, gens)
        if color:
            c = _YANG_ANSI[min(k % 7, 6)]
            lines.append(f'  {c}  k={k:2d}: {bs:8,} различных подмн.{_RESET}')
        else:
            lines.append(f'    k={k:2d}: {bs:8,} различных подмн.')

    # 8×8 карта: для каждого h, сколько генераторов его фиксируют
    lines.append('\n  8×8 карта: yang_count как орбитный индекс')

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
            if color:
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}y{yc}{_RESET}')
            else:
                lbl.append(f'y{yc}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='sym_glyphs',
        description='Группа автоморфизмов Aut(Q6) через глифы',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('yang',      help='7 орбит по yang_count')
    sub.add_parser('fixed',     help='неподвижные точки генераторов')
    sub.add_parser('antipodal', help='антиподальные пары {h, h⊕63}')

    s = sub.add_parser('burnside', help='теорема Полиа для n цветов')
    s.add_argument('--colors', type=int, default=4,
                   help='максимальное число цветов (default=4)')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'yang':
        print(render_yang(color))
    elif args.cmd == 'fixed':
        print(render_fixed(color))
    elif args.cmd == 'antipodal':
        print(render_antipodal(color))
    elif args.cmd == 'burnside':
        print(render_burnside(n_colors=args.colors, color=color))


if __name__ == '__main__':
    main()
