"""d4orbits — Орбиты группы D₄ на 64 глифах Q6.

Группа симметрий квадрата D₄ (8 элементов) действует на глифах,
переставляя 6 сегментов как рёбра K₄ (см. k4atlas.py).

Элементы D₄ как перестановки битов:
    e    = [0,1,2,3,4,5]  — тождественная
    r    = [3,2,0,1,5,4]  — поворот 90° по часовой стрелке
    r²   = [1,0,3,2,4,5]  — поворот 180°
    r³   = [2,3,1,0,5,4]  — поворот 270° (= 90° против ч.с.)
    s    = [0,1,3,2,5,4]  — отражение горизонтально (лево–право)
    sr   = [2,3,0,1,4,5]  — отражение по диагонали ╲
    sr²  = [1,0,2,3,5,4]  — отражение вертикально (верх–низ)
    sr³  = [3,2,1,0,4,5]  — отражение по антидиагонали ╱

Числа фиксированных точек (лемма Бёрнсайда):
    |Fix(e)|  = 64
    |Fix(r)|  = 4   (h: биты 0=1=2=3, биты 4=5)
    |Fix(r²)| = 16  (h: бит0=бит1, бит2=бит3)
    |Fix(r³)| = 4
    |Fix(s)|  = 16  (h: бит2=бит3, бит4=бит5)
    |Fix(sr)| = 16  (h: бит0=бит2, бит1=бит3)
    |Fix(sr²)|= 16  (h: бит0=бит1, бит4=бит5)
    |Fix(sr³)|= 16  (h: бит0=бит3, бит1=бит2)

Число орбит = (64+4+16+4+16+16+16+16) / 8 = 152/8 = 19.
"""

from __future__ import annotations
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, apply_permutation, SIZE
from projects.hexvis.hexvis import render_glyph, _YANG_ANSI, _YANG_BG, _RESET, _BOLD

# ---------------------------------------------------------------------------
# Генераторы D₄ и все 8 элементов группы
# ---------------------------------------------------------------------------

def _compose(p: list[int], q: list[int]) -> list[int]:
    """Композиция перестановок: сначала q, потом p.  result[i] = p[q[i]]."""
    return [p[q[i]] for i in range(6)]


# Базовые перестановки (задают генераторы)
_PERM_E   = [0, 1, 2, 3, 4, 5]   # тождественная
_PERM_R   = [3, 2, 0, 1, 5, 4]   # поворот 90° CW
_PERM_S   = [0, 1, 3, 2, 5, 4]   # горизонтальное отражение

# Все 8 элементов D₄
_PERM_R2  = _compose(_PERM_R, _PERM_R)
_PERM_R3  = _compose(_PERM_R, _PERM_R2)
_PERM_SR  = _compose(_PERM_S, _PERM_R)
_PERM_SR2 = _compose(_PERM_S, _PERM_R2)
_PERM_SR3 = _compose(_PERM_S, _PERM_R3)

D4_ELEMENTS: dict[str, list[int]] = {
    'e':   _PERM_E,
    'r':   _PERM_R,
    'r²':  _PERM_R2,
    'r³':  _PERM_R3,
    's':   _PERM_S,
    'sr':  _PERM_SR,
    'sr²': _PERM_SR2,
    'sr³': _PERM_SR3,
}

D4_DESCRIPTIONS: dict[str, str] = {
    'e':   'тождественная',
    'r':   'поворот 90° CW',
    'r²':  'поворот 180°',
    'r³':  'поворот 270° CW',
    's':   'отражение горизонтально',
    'sr':  'отражение по диагонали ╲',
    'sr²': 'отражение вертикально',
    'sr³': 'отражение по антидиагонали ╱',
}


def d4_act(h: int, name: str) -> int:
    """Применить элемент D₄ к глифу h."""
    return apply_permutation(h, D4_ELEMENTS[name])


def d4_orbit(h: int) -> list[int]:
    """Орбита глифа h под всей группой D₄."""
    orbit: set[int] = set()
    for perm in D4_ELEMENTS.values():
        orbit.add(apply_permutation(h, perm))
    return sorted(orbit)


# ---------------------------------------------------------------------------
# Вычисление всех 19 орбит
# ---------------------------------------------------------------------------

def all_d4_orbits() -> list[list[int]]:
    """Все орбиты D₄ на {0..63}, отсортированные по наименьшему элементу."""
    unvisited = set(range(64))
    orbits: list[list[int]] = []
    while unvisited:
        h = min(unvisited)
        orb = d4_orbit(h)
        orbits.append(orb)
        unvisited -= set(orb)
    return sorted(orbits, key=lambda o: (yang_count(o[0]), o[0]))


_ORBITS: list[list[int]] = all_d4_orbits()

# Номер орбиты для каждого глифа
_ORBIT_INDEX: dict[int, int] = {}
for _idx, _orb in enumerate(_ORBITS):
    for _h in _orb:
        _ORBIT_INDEX[_h] = _idx


def orbit_index(h: int) -> int:
    """Индекс D₄-орбиты (0..18) для глифа h."""
    return _ORBIT_INDEX[h]


def orbit_size(h: int) -> int:
    """Размер орбиты, содержащей h."""
    return len(_ORBITS[_ORBIT_INDEX[h]])


# ---------------------------------------------------------------------------
# Описание орбиты
# ---------------------------------------------------------------------------

def _orbit_description(orb: list[int]) -> str:
    """Краткое описание орбиты по геометрическим свойствам."""
    e = yang_count(orb[0])
    size = len(orb)
    rep = orb[0]
    bits = [format(h, '06b') for h in orb]

    if size == 1:
        return f'фиксированная (симметрия D₄)'
    if size == 2:
        return f'ось симметрии второго порядка'
    if size == 4:
        return f'полуорбита (стабилизатор Z₂)'
    if size == 8:
        return f'свободная орбита (тривиальный стабилизатор)'
    return f'орбита размера {size}'


def _stabilizer_name(orb: list[int]) -> str:
    """Определить стабилизатор (подгруппу D₄, фиксирующую орбиту)."""
    rep = orb[0]
    fixed_by = [name for name, perm in D4_ELEMENTS.items()
                if apply_permutation(rep, perm) == rep]
    return f'Stab = {{{", ".join(sorted(fixed_by))}}} (|Stab|={len(fixed_by)})'


# ---------------------------------------------------------------------------
# Визуализация орбит
# ---------------------------------------------------------------------------

def render_orbit_row(orb: list[int], idx: int, color: bool = True) -> str:
    """Одна строка таблицы орбит: номер орбиты + все глифы орбиты."""
    glyphs = [render_glyph(h) for h in orb]
    size = len(orb)
    e = yang_count(orb[0])
    label_main = f'  Орбита {idx+1:2d}  (рёбер={e}, |орб|={size})  '
    pad = ' ' * len(label_main)

    lines: list[str] = []
    for ri in range(3):
        parts: list[str] = []
        for gi, h in enumerate(orb):
            cell = glyphs[gi][ri]
            if color:
                yc = yang_count(h)
                cell = _YANG_ANSI[yc] + cell + _RESET
            parts.append(cell)
        prefix = label_main if ri == 1 else pad
        lines.append(prefix + ' '.join(parts))

    # Стабилизатор (подпись)
    lines.append(pad + _stabilizer_name(orb))
    return '\n'.join(lines)


def render_all_orbits(color: bool = True) -> str:
    """Таблица всех 19 D₄-орбит на 64 глифах."""
    out: list[str] = []
    out.append('═' * 72)
    out.append('  D₄-ОРБИТЫ НА Q6  (19 орбит, 64 элемента)')
    out.append('  Действие: симметрии квадрата переставляют сегменты глифа')
    out.append('═' * 72)

    current_e = -1
    for idx, orb in enumerate(_ORBITS):
        e = yang_count(orb[0])
        if e != current_e:
            out.append(f'\n── {e} рёбер ────────────────────────────────────────')
            current_e = e
        out.append(render_orbit_row(orb, idx, color=color))
        out.append('')

    out.append('─' * 72)
    out.append(f'  Итого: {len(_ORBITS)} орбит')
    out.append(f'  Бёрнсайд: (64+4+16+4+16+16+16+16)/8 = 152/8 = 19 ✓')
    sizes = defaultdict(int)
    for orb in _ORBITS:
        sizes[len(orb)] += 1
    for sz in sorted(sizes):
        out.append(f'    |орбита|={sz}: {sizes[sz]} орбит(ы)')
    return '\n'.join(out)


def render_hasse_by_orbit(color: bool = True) -> str:
    """Диаграмма Хассе B₆, где каждый глиф подписан номером D₄-орбиты."""
    lines: list[str] = ['  Диаграмма Хассе B₆  (число под глифом = № D₄-орбиты)']
    by_rank: list[list[int]] = [[] for _ in range(7)]
    for h in range(64):
        by_rank[yang_count(h)].append(h)

    max_n = 20
    cw = 3
    sw = 1
    total_w = max_n * cw + (max_n - 1) * sw

    for k, elems in enumerate(by_rank):
        n = len(elems)
        row_w = n * cw + (n - 1) * sw
        pad = ' ' * ((total_w - row_w) // 2)
        glyphs = [render_glyph(h) for h in elems]
        for ri in range(3):
            parts: list[str] = []
            for gi, h in enumerate(elems):
                cell = glyphs[gi][ri]
                if color:
                    yc = yang_count(h)
                    c = _YANG_BG[yc] if h in {orb[0] for orb in _ORBITS} else _YANG_ANSI[yc]
                    cell = c + cell + _RESET
                parts.append(cell)
            lines.append(pad + ' '.join(parts))
        # Номера орбит
        idx_row = ' '.join(f'{_ORBIT_INDEX[h]+1:>3d}' for h in elems)
        lines.append(pad + idx_row)
        lines.append('')

    return '\n'.join(lines)


def render_group_action_table(color: bool = True) -> str:
    """Таблица действия: для каждого элемента D₄ показываем его перестановку битов."""
    lines: list[str] = []
    lines.append('  Таблица действия D₄ (перестановки битов глифа):')
    lines.append(f'  {"Элемент":6s}  {"Биты→":6s}  [б0→б?, б1→?, б2→?, б3→?, б4→?, б5→?]  Описание')
    lines.append('  ' + '─' * 70)
    for name, perm in D4_ELEMENTS.items():
        perm_str = f'[{",".join(str(p) for p in perm)}]'
        desc = D4_DESCRIPTIONS[name]
        # Считаем фиксированные точки
        fixed = sum(1 for h in range(64) if apply_permutation(h, perm) == h)
        lines.append(f'  {name:6s}   {perm_str:20s}  |Fix|={fixed:2d}   {desc}')
    lines.append(f'\n  Число орбит (Бёрнсайд) = (64+4+16+4+16+16+16+16)/8 = 19')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='d4orbits — Орбиты симметрий квадрата D₄ на глифах Q6'
    )
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('orbits', help='Все 19 D₄-орбит')
    sub.add_parser('hasse', help='Диаграмма Хассе с орбитами')
    sub.add_parser('table', help='Таблица действия D₄')

    p_show = sub.add_parser('show', help='Орбита конкретного глифа')
    p_show.add_argument('h', type=int, help='Номер глифа (0..63)')

    p_act = sub.add_parser('act', help='Применить элемент D₄ к глифу')
    p_act.add_argument('h', type=int, help='Номер глифа')
    p_act.add_argument('g', help='Элемент D₄ (e,r,r²,r³,s,sr,sr²,sr³)')

    for p in sub.choices.values():
        p.add_argument('--no-color', action='store_true')

    args = parser.parse_args()
    color = not getattr(args, 'no_color', False)

    if args.cmd == 'orbits' or args.cmd is None:
        print(render_all_orbits(color=color))

    elif args.cmd == 'hasse':
        print(render_hasse_by_orbit(color=color))

    elif args.cmd == 'table':
        print(render_group_action_table(color=color))

    elif args.cmd == 'show':
        orb = d4_orbit(args.h)
        print(f'D₄-орбита глифа h={args.h}  (орбита #{_ORBIT_INDEX[args.h]+1}):')
        print(render_orbit_row(orb, _ORBIT_INDEX[args.h], color=color))

    elif args.cmd == 'act':
        if args.g not in D4_ELEMENTS:
            print(f'Элемент {args.g!r} не в D₄. Допустимые: {list(D4_ELEMENTS)}')
        else:
            result = d4_act(args.h, args.g)
            print(f'  {args.g}({args.h}) = {result}')
            glyphs_before = render_glyph(args.h)
            glyphs_after = render_glyph(result)
            for ri in range(3):
                print(f'    {glyphs_before[ri]}  →  {glyphs_after[ri]}')
            print(f'  ({D4_DESCRIPTIONS[args.g]})')
