"""screw_glyphs — Группа-винт B₇ и её действие на Q6 через глифы.

Группа-винт B₇ = {σ ∈ S₇ | σ(1) = 1} — подгруппа S₇, сохраняющая элемент 1.
|B₇| = 6! = 720  ≅  S₆

B₇ действует на Q6 (6-битные гексаграммы), переставляя биты 0..5:
  позиции 2..7 перестановки → биты 0..5 гексаграммы

Теория Германа (спин):
  left_spin(σ)  = sgn(σ) = +1 (чётная) / -1 (нечётная)
  right_spin(σ) = произведение длин орбит на {2..7} mod 2

Визуализация:
  orbits  [--elem e]   — орбиты Q6 под σ из B₇
  spin                 — раскраска Q6 по числу инверсий спинорного действия
  fixed   [--cc cc]    — неподвижные точки σ для каждого класса сопряжённости
  classes              — сопряжённые классы B₇ и их орбиты на Q6

Команды CLI:
  orbits  [--elem e]
  spin
  fixed
  classes
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexscrew.hexscrew import ScrewGroup
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _act_b7(perm: list[int], h: int) -> int:
    """Действие B₇ на Q6: позиции 2..7 переставляют биты 0..5 гексаграммы.

    Коррект­ное отображение: position i (2≤i≤7) → bit (i−2).
    """
    result = 0
    for i in range(6):          # бит 0..5
        src_pos = i + 2         # позиция в перестановке (2..7)
        dst_pos = perm[src_pos - 1]  # куда отправляется эта позиция
        dst_bit = dst_pos - 2   # целевой бит (0..5)
        bit_val = (h >> i) & 1
        result |= (bit_val << dst_bit)
    return result


def _get_orbits(perm: list[int]) -> list[list[int]]:
    """Разбить Q6 на орбиты под действием итераций perm."""
    visited = [False] * 64
    orbits: list[list[int]] = []
    for h0 in range(64):
        if not visited[h0]:
            orb: list[int] = []
            cur = h0
            while not visited[cur]:
                visited[cur] = True
                orb.append(cur)
                cur = _act_b7(perm, cur)
            orbits.append(orb)
    return orbits


def _orbit_index(perm: list[int]) -> list[int]:
    """Для каждой вершины h — номер её орбиты."""
    result = [-1] * 64
    for idx, orb in enumerate(_get_orbits(perm)):
        for h in orb:
            result[h] = idx
    return result


def _fixed_points(perm: list[int]) -> list[int]:
    """Неподвижные вершины: h такие, что σ(h)=h."""
    return [h for h in range(64) if _act_b7(perm, h) == h]


# ---------------------------------------------------------------------------
# Цветовая палитра для орбит
# ---------------------------------------------------------------------------

_ORBIT_COLORS = [
    '\033[38;5;27m',  '\033[38;5;82m',  '\033[38;5;196m',
    '\033[38;5;208m', '\033[38;5;201m', '\033[38;5;226m',
    '\033[38;5;39m',  '\033[38;5;238m', '\033[38;5;46m',
    '\033[38;5;51m',  '\033[38;5;160m', '\033[38;5;11m',
    '\033[38;5;93m',  '\033[38;5;130m', '\033[38;5;50m',
    '\033[38;5;200m',
]


# ---------------------------------------------------------------------------
# 1. Орбиты Q6 под одним элементом B₇
# ---------------------------------------------------------------------------

def render_orbits(elem_idx: int = 1, color: bool = True) -> str:
    """
    8×8 сетка: глифы раскрашены по принадлежности орбите σ ∈ B₇.

    elem_idx — индекс в лексикографическом порядке элементов B₇ (0..719).
    """
    sg = ScrewGroup(7)
    elems = list(sg.elements())
    perm = elems[elem_idx % len(elems)]

    orbits = _get_orbits(perm)
    orb_idx = _orbit_index(perm)

    # Орбитная сигнатура
    from collections import Counter
    size_dist = Counter(len(o) for o in orbits)

    ct = sg.cycle_type(perm)
    ls = sg.left_spin(perm)
    rs = sg.right_spin(perm)
    fp = _fixed_points(perm)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Орбиты B₇ на Q6: элемент σ[{elem_idx}] = {perm}')
    lines.append(f'  Тип цикла: {ct}   left_spin={ls:+d}   right_spin={rs:+d}')
    lines.append(f'  Орбит: {len(orbits)}   Неподвижных вершин: {len(fp)}')
    lines.append(f'  Орбитная сигнатура: { {k: v for k, v in sorted(size_dist.items())} }')
    lines.append('  Цвет = номер орбиты')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8   # type: ignore
        for col in range(8):
            h = row * 8 + col
            oi = orb_idx[h]
            rows3 = render_glyph(h)
            if color:
                is_fp = (_act_b7(perm, h) == h)
                c = _ORBIT_COLORS[oi % len(_ORBIT_COLORS)]
                if is_fp:
                    c = _YANG_BG[yang_count(h)] + _BOLD
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            oi = orb_idx[h]
            nh = _act_b7(perm, h)
            if color:
                c = _ORBIT_COLORS[oi % len(_ORBIT_COLORS)]
                lbl.append(f'{c}→{nh:02d}{_RESET}')
            else:
                lbl.append(f'→{nh:02d}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Первые 5 орбит
    lines.append('  Первые 5 орбит:')
    for idx, orb in enumerate(sorted(orbits, key=lambda o: -len(o))[:5]):
        path = ' → '.join(f'{h:02d}' for h in orb[:8])
        if len(orb) > 8:
            path += ' → ...'
        if color:
            c = _ORBIT_COLORS[idx % len(_ORBIT_COLORS)]
            lines.append(f'  {c}  [{idx+1}] длина={len(orb)}: {path}{_RESET}')
        else:
            lines.append(f'    [{idx+1}] длина={len(orb)}: {path}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Спиновое действие: раскраска по числу нечётных орбит
# ---------------------------------------------------------------------------

def render_spin(color: bool = True) -> str:
    """
    Для каждой вершины h вычислить «спиновый вес» —
    количество элементов σ ∈ B₇, для которых σ(h) ≠ h (h нестационарна).

    Вершины Q6 с наибольшей подвижностью = «горячие точки» действия группы.
    """
    sg = ScrewGroup(7)
    elems = list(sg.elements())

    # Для каждого h: сколько σ его двигают
    move_count = [0] * 64
    fix_count = [0] * 64
    for perm in elems:
        for h in range(64):
            if _act_b7(perm, h) == h:
                fix_count[h] += 1
            else:
                move_count[h] += 1

    # Орбита всей группы: для каждого h — группная орбита Gh
    group_orbit = [set() for _ in range(64)]
    for perm in elems:
        for h in range(64):
            group_orbit[h].add(_act_b7(perm, h))

    orbit_sizes = [len(group_orbit[h]) for h in range(64)]
    max_os = max(orbit_sizes)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Спиновое действие B₇ на Q6: орбита вершины под всей группой')
    lines.append(f'  |B₇| = {len(elems)} = 6!   Максимальный размер орбиты = {max_os}')
    lines.append('  Цвет = yang_count(h),  фон = размер орбиты (большие = светлее)')
    lines.append('  Под глифом: fix=число σ, фиксирующих h  |Gh|=размер орбиты')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            yc = yang_count(h)
            os_h = orbit_sizes[h]
            rows3 = render_glyph(h)
            if color:
                # Орбиты размера 64 = транзитивное действие (максимум)
                if os_h == max_os:
                    c = _YANG_BG[yc] + _BOLD
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
            fc = fix_count[h]
            os_h = orbit_sizes[h]
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}f{fc:3d}{_RESET}')
            else:
                lbl.append(f'f{fc:3d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Распределение размеров орбит
    from collections import Counter
    orbit_dist = Counter(orbit_sizes)
    lines.append('  Распределение размеров групповых орбит |Gh|:')
    for sz in sorted(orbit_dist):
        cnt = orbit_dist[sz]
        # По теореме Бернсайда: |орбита| · |стабилизатор| = |G|
        stab = len(elems) // sz
        if color:
            c = _YANG_ANSI[min(6, sz // 10)]
            lines.append(f'  {c}  |Gh|={sz:2d}: {cnt:2d} вершин  '
                         f'|Stab(h)|={stab}{_RESET}')
        else:
            lines.append(f'    |Gh|={sz:2d}: {cnt:2d} вершин  |Stab(h)|={stab}')

    # Число орбит по Бернсайду
    total_fixed = sum(fix_count)
    n_orbits_burnside = total_fixed // len(elems)
    lines.append(f'\n  Теорема Бернсайда: |G\\ Q6| = (1/|G|)·Σ|Fix(σ)| '
                 f'= {total_fixed}//{len(elems)} = {n_orbits_burnside}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Неподвижные точки по классам сопряжённости
# ---------------------------------------------------------------------------

def render_fixed(cc_name: str = 'transposition', color: bool = True) -> str:
    """
    8×8 сетка: глифы раскрашены по числу σ ∈ B₇, их фиксирующих.

    Показывает, какие вершины Q6 наиболее «устойчивы» к действию группы.
    """
    sg = ScrewGroup(7)
    elems = list(sg.elements())

    # Считаем fix_count[h] = |{σ ∈ B₇ : σ(h)=h}| = |Stab(h)|
    fix_count = [0] * 64
    for perm in elems:
        for h in range(64):
            if _act_b7(perm, h) == h:
                fix_count[h] += 1

    # Классы сопряжённости → один представитель
    cc = sg.conjugacy_classes()
    cc_names = list(cc.keys())

    # Фиксированные точки каждого класса
    cc_fix: dict[tuple, list[int]] = {}
    for ctype, members in cc.items():
        rep = members[0]
        cc_fix[ctype] = _fixed_points(rep)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Неподвижные точки B₇ на Q6: |Stab(h)| для каждой вершины')
    lines.append(f'  |B₇| = {len(elems)}   Классов сопряжённости: {len(cc)}')
    lines.append('  По орбитно-стабилизаторной теореме: |Orb(h)| · |Stab(h)| = |G|')
    lines.append('  Цвет = yang_count(h),  жирный = максимальный стабилизатор')
    lines.append('═' * 66)

    max_fc = max(fix_count)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            fc = fix_count[h]
            yc = yang_count(h)
            rows3 = render_glyph(h)
            if color:
                if fc == max_fc:
                    c = _YANG_BG[yc] + _BOLD
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
            fc = fix_count[h]
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}{fc:3d}{_RESET}')
            else:
                lbl.append(f'{fc:3d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Неподвижные точки каждого класса
    lines.append('  Неподвижные точки по классам сопряжённости:')
    for ctype, members in sorted(cc.items(), key=lambda x: len(x[1]), reverse=True)[:8]:
        fp = cc_fix[ctype]
        if color:
            lines.append(f'  {_YANG_ANSI[2]}  цикл={ctype}'
                         f'  |класс|={len(members):3d}'
                         f'  |Fix|={len(fp):2d}'
                         f'  fix={sorted(fp)[:6]}{"..." if len(fp)>6 else ""}'
                         f'{_RESET}')
        else:
            lines.append(f'    цикл={ctype}'
                         f'  |класс|={len(members):3d}'
                         f'  |Fix|={len(fp):2d}'
                         f'  fix={sorted(fp)[:6]}{"..." if len(fp)>6 else ""}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Структура классов сопряжённости
# ---------------------------------------------------------------------------

def render_classes(color: bool = True) -> str:
    """
    Для каждого класса сопряжённости B₇ нарисовать орбитную структуру на Q6.

    B₇ ≅ S₆ имеет 11 классов сопряжённости (по числу разбиений 6).
    """
    sg = ScrewGroup(7)
    elems = list(sg.elements())
    cc = sg.conjugacy_classes()

    lines: list[str] = []
    lines.append('╔' + '═' * 64 + '╗')
    lines.append('║  Классы сопряжённости B₇ ≅ S₆ и орбитная структура на Q6' + ' ' * 6 + '║')
    lines.append(f'║  |B₇| = {len(elems)} = 6!   Классов: {len(cc)}' + ' ' * 44 + '║')
    lines.append('╚' + '═' * 64 + '╝')
    lines.append('')

    for idx, (ctype, members) in enumerate(
            sorted(cc.items(), key=lambda x: sum(x[0]))):
        rep = members[0]
        orbits = _get_orbits(rep)
        from collections import Counter
        orb_sig = Counter(len(o) for o in orbits)
        fp = _fixed_points(rep)
        ls = sg.left_spin(rep)

        if color:
            c = _YANG_ANSI[idx % 7]
            lines.append(f'  {c}Класс {idx+1}: цикл-тип {ctype}'
                         f'   |класс|={len(members):3d}'
                         f'   spin={ls:+d}{_RESET}')
            lines.append(f'  {c}  Орбиты на Q6: {dict(orb_sig)}'
                         f'   |Fix|={len(fp)}{_RESET}')
        else:
            lines.append(f'  Класс {idx+1}: цикл-тип {ctype}'
                         f'   |класс|={len(members):3d}'
                         f'   spin={ls:+d}')
            lines.append(f'    Орбиты на Q6: {dict(orb_sig)}'
                         f'   |Fix|={len(fp)}')

        # Глифы первой орбиты (нетривиальной)
        big_orbs = sorted(orbits, key=lambda o: -len(o))
        if big_orbs and len(big_orbs[0]) > 1:
            orb0 = big_orbs[0][:8]
            glyphs_o = [render_glyph(h) for h in orb0]
            if color:
                c = _YANG_ANSI[idx % 7]
                glyphs_o = [[c + r + _RESET for r in g] for g in glyphs_o]
            for ri in range(3):
                lines.append('    ' + ' → '.join(g[ri] for g in glyphs_o)
                             + (' → ...' if len(big_orbs[0]) > 8 else ''))
        lines.append('')

    # Итог: число орбит (формула Бернсайда)
    total_fixed_total = sum(
        len(cc_fix_members)
        for _, members in cc.items()
        for cc_fix_members in [_fixed_points(members[0])]
        for _ in range(len(members))
    )
    n_orbits = total_fixed_total // len(elems)
    lines.append(f'  Число B₇-орбит на Q6 (Бернсайд): {n_orbits}')
    lines.append(f'  Проверка: C(6,0)+C(6,1)+...+C(6,6) = 64 = |Q6|')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='screw_glyphs',
        description='Группа-винт B₇ ≅ S₆ и её действие на Q6',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('orbits', help='орбиты Q6 под элементом B₇')
    s.add_argument('--elem', type=int, default=1,
                   help='индекс элемента в B₇ (0..719)')

    sub.add_parser('spin',    help='спиновое действие: орбита каждой вершины')
    sub.add_parser('fixed',   help='неподвижные точки: |Stab(h)|')
    sub.add_parser('classes', help='классы сопряжённости и орбитная структура')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'orbits':
        print(render_orbits(elem_idx=args.elem, color=color))
    elif args.cmd == 'spin':
        print(render_spin(color=color))
    elif args.cmd == 'fixed':
        print(render_fixed(color=color))
    elif args.cmd == 'classes':
        print(render_classes(color=color))


if __name__ == '__main__':
    main()
