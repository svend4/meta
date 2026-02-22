"""poly_glyphs — Правильные многогранники и Эк-додекаэдр через глифы Q6.

Каждый глиф (0..63) — вершина Q6.

Правильные (платоновы) тела — 5 штук, соответствие через гиперкуб:
  Теtraedр:   4 вершины  → yang=1 (C(6,1)=6, берём 4)  → биты 1-активные
  Куб:        8 вершин   → yang=1..2 подмножество
  Октаэдр:    6 вершин   → yang=1 (6 граней куба ↔ 6 вершин октаэдра)
  Додекаэдр: 20 вершин   → глифы 0..19
  Икосаэдр:  12 вершин   → yang=2 (C(6,2)=15, берём 12)

Эк-додекаэдр Германа:
  32 вершины, 24 грани, 54 ребра, χ=2
  32 = 64/2 → ровно половина вершин Q6!
  Грани: 8 треугольников, 12 ромбов, 4 пятиугольника
  Объём: V = (a³/2)(4 + 3φ),  φ = (1+√5)/2 (золотое сечение)

Визуализация:
  platonic  — 5 платоновых тел как подграфы Q6
  euler     — формула Эйлера V−E+F=χ для всех тел
  exdodeca  — 32 вершины Эк-додекаэдра в Q6
  diagonals — число диагоналей: V(V−1)/2 − E

Команды CLI:
  platonic
  euler
  exdodeca
  diagonals
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexpolyenum.hexpolyenum import (
    PolyhedronRecord, PolyhedronEnumerator, ExDodecahedron,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_PHI = (1 + math.sqrt(5)) / 2   # золотое сечение


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _solid_color(name: str) -> str:
    """Цвет ANSI для тела по имени."""
    palette = {
        'Тетраэдр':   '\033[38;5;196m',
        'Куб':         '\033[38;5;27m',
        'Октаэдр':    '\033[38;5;82m',
        'Додекаэдр':  '\033[38;5;208m',
        'Икосаэдр':   '\033[38;5;201m',
    }
    return palette.get(name, _YANG_ANSI[1])


# ---------------------------------------------------------------------------
# 1. Платоновы тела как подграфы Q6
# ---------------------------------------------------------------------------

def render_platonic(color: bool = True) -> str:
    """
    8×8 сетка: вершины Q6 раскрашены по принадлежности платоновым телам.

    Соответствие: тела вложены в Q6 через yang-слои:
      Тетраэдр  (4 верш):  yang=1, первые 4: {1,2,4,8}
      Октаэдр   (6 верш):  yang=1 (все 6)
      Куб       (8 верш):  yang=3 (все C(6,3)=20, берём 8 с чётным попарным расст.)
      Додекаэдр (20 верш): yang=3 (все 20)
      Икосаэдр  (12 верш): yang=2 (12 из 15)
    """
    pe = PolyhedronEnumerator()
    solids = pe.enumerate_spherical()

    # Вложения: yang-слои
    # Тетраэдр — 4 вершины, yang=1: глифы 1,2,4,8 (только 4 из 6)
    tetra = [1, 2, 4, 8]
    # Октаэдр — 6 вершин, yang=1: все глифы с 1 битом
    octa = [h for h in range(64) if yang_count(h) == 1]
    # Куб — 8 вершин, yang=0 or 6 or выбор из yang=3
    cube_8 = [0, 9, 18, 27, 36, 45, 54, 63]  # красивый набор
    # Додекаэдр — 20 вершин, yang=3
    dodeca = [h for h in range(64) if yang_count(h) == 3][:20]
    # Икосаэдр — 12 вершин, yang=2
    icosa = [h for h in range(64) if yang_count(h) == 2][:12]

    vertex_sets = [
        (solids[0], set(tetra)),
        (solids[2], set(octa)),
        (solids[1], set(cube_8)),
        (solids[3], set(dodeca)),
        (solids[4], set(icosa)),
    ]

    # Назначение: первое совпадение
    vertex_solid: dict[int, PolyhedronRecord] = {}
    for solid, vset in vertex_sets:
        for h in vset:
            if h not in vertex_solid:
                vertex_solid[h] = solid

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  5 Платоновых тел как вложения в Q6')
    lines.append('  Вершины тел ↔ глифы Q6 по yang-слоям')
    lines.append('  Тетраэдр(4)≤Октаэдр(6)≤Куб(8)≤Додекаэдр(20)≤Икосаэдр(12)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            solid = vertex_solid.get(h)
            rows3 = render_glyph(h)
            if color:
                if solid:
                    c = _solid_color(solid.name)
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            solid = vertex_solid.get(h)
            if color:
                c = _solid_color(solid.name) if solid else _YANG_ANSI[0]
                lbl.append(f'{c}{solid.name[:3] if solid else "   "}{_RESET}')
            else:
                lbl.append(solid.name[:3] if solid else '   ')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Таблица тел
    lines.append('  Платоновы тела:')
    for solid, vset in vertex_sets:
        if color:
            c = _solid_color(solid.name)
            lines.append(f'  {c}  {solid.name:<12} '
                         f'V={solid.vertices:2d}  E={solid.edges:2d}  F={solid.faces:2d}  '
                         f'χ={solid.euler()}  '
                         f'вложение: yang={yang_count(list(vset)[0])}{_RESET}')
        else:
            lines.append(f'    {solid.name:<12} '
                         f'V={solid.vertices:2d}  E={solid.edges:2d}  F={solid.faces:2d}  '
                         f'χ={solid.euler()}  '
                         f'вложение: yang={yang_count(list(vset)[0])}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Формула Эйлера
# ---------------------------------------------------------------------------

def render_euler(color: bool = True) -> str:
    """
    Визуализация формулы Эйлера V−E+F=χ для тел.

    8×8 сетка Q6 раскрашена по значению yang_count(h) = число вершин mod 7.
    Для каждого тела: V−E+F = 2 (сфера).
    """
    pe = PolyhedronEnumerator()
    solids = pe.enumerate_spherical()
    ex = ExDodecahedron()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Формула Эйлера V − E + F = χ для выпуклых многогранников')
    lines.append('  Для сферических (платоновых) тел: χ = 2')
    lines.append('  Для тороидальных: χ = 0')
    lines.append('═' * 66)
    lines.append('')

    all_bodies = [(s.name, s.vertices, s.edges, s.faces, s.euler())
                  for s in solids]
    all_bodies.append(('Эк-додекаэдр', ex.vertices(), ex.edges(), ex.faces(), ex.euler()))

    for i, (name, V, E, F, chi) in enumerate(all_bodies):
        check = '✓' if V - E + F == chi else '✗'
        diag = V * (V - 1) // 2 - E
        if color:
            c = _solid_color(name) if name != 'Эк-додекаэдр' else '\033[38;5;226m'
            lines.append(f'  {c}  {name:<14} '
                         f'V={V:3d}  E={E:3d}  F={F:3d}  '
                         f'V−E+F={V-E+F}={chi} {check}  '
                         f'диагоналей={diag}{_RESET}')
        else:
            lines.append(f'    {name:<14} '
                         f'V={V:3d}  E={E:3d}  F={F:3d}  '
                         f'V−E+F={V-E+F}={chi} {check}  '
                         f'диагоналей={diag}')

    lines.append('')
    lines.append('  8×8 карта Q6: цвет = yang_count(h) = число единичных битов')
    lines.append('  C(6,k) вершин на слое k: 1 6 15 20 15 6 1')

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

    lines.append('  Аналог Эйлера для Q6:  V=64  E=192  F=240(квадраты)  3D=160')
    lines.append('  χ(Q6 как клеточный комплекс) = 0  (Q6 ≅ тор в некотором смысле)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Эк-додекаэдр (32 вершины в Q6)
# ---------------------------------------------------------------------------

def render_exdodeca(color: bool = True) -> str:
    """
    8×8 сетка: выделить 32 вершины Эк-додекаэдра Германа в Q6.

    32 = 64/2 → ровно половина всех вершин Q6.
    Выбираем yang∈{2,3,4} (наиболее «центральные» слои).
    """
    ex = ExDodecahedron()

    # 32 вершины: yang ∈ {2,3,4} → C(6,2)+C(6,3)+C(6,4) = 15+20+15 = 50, берём первые 32
    # Или: yang=3 (20 верш) + часть yang=2 (6 верш) + часть yang=4 (6 верш)
    yang3 = [h for h in range(64) if yang_count(h) == 3]        # 20
    yang2 = [h for h in range(64) if yang_count(h) == 2][:6]    # 6
    yang4 = [h for h in range(64) if yang_count(h) == 4][:6]    # 6
    exdodeca_verts = set(yang3 + yang2 + yang4)                  # 32

    face_types = ex.face_types()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Эк-додекаэдр Германа: 32 вершины ⊂ Q6')
    lines.append(f'  V={ex.vertices()}  E={ex.edges()}  F={ex.faces()}  χ={ex.euler()}')
    lines.append(f'  Грани: треугольников={face_types["triangle"]}'
                 f'  ромбов={face_types["rhombus"]}'
                 f'  пятиугольников={face_types["pentagon"]}')
    lines.append(f'  Объём = (a³/2)(4+3φ) ≈ {ex.volume():.4f}   φ=(1+√5)/2≈{_PHI:.4f}')
    lines.append(f'  Диагоналей: {ex.diagonal_count()}   32вершин = 64/2 = |Q6|/2')
    lines.append('  Жирный = вершина Эк-додекаэдра,  Цвет = yang_count')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            in_ex = h in exdodeca_verts
            yc = yang_count(h)
            rows3 = render_glyph(h)
            if color:
                c = _YANG_BG[yc] + _BOLD if in_ex else _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            in_ex = h in exdodeca_verts
            yc = yang_count(h)
            if color:
                c = _YANG_ANSI[yc] if in_ex else _YANG_ANSI[0]
                lbl.append(f'{c}{"ED" if in_ex else "  "}{_RESET}')
            else:
                lbl.append('ED' if in_ex else '  ')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Шаги построения
    lines.append('  Конструкция Эк-додекаэдра:')
    for i, step in enumerate(ex.construction_steps()):
        if color:
            c = _YANG_ANSI[(i + 1) % 7]
            lines.append(f'  {c}  {i+1}. {step}{_RESET}')
        else:
            lines.append(f'    {i+1}. {step}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Диагонали многогранников
# ---------------------------------------------------------------------------

def render_diagonals(color: bool = True) -> str:
    """
    Число диагоналей D = V(V−1)/2 − E для каждого тела.

    Диагональ = отрезок между несмежными вершинами.
    8×8 сетка: глифы раскрашены по числу вершин модули 7.
    """
    pe = PolyhedronEnumerator()
    solids = pe.enumerate_spherical()
    ex = ExDodecahedron()

    bodies = [(s.name, s.vertices, s.edges, s.diagonal_count()) for s in solids]
    bodies.append(('Эк-додекаэдр', ex.vertices(), ex.edges(), ex.diagonal_count()))

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Диагонали многогранников: D = V(V−1)/2 − E')
    lines.append('  Диагональ = прямая между несмежными вершинами')
    lines.append('═' * 66)
    lines.append('')
    lines.append(f'  {"Тело":<15} {"V":>4} {"E":>4} {"V(V-1)/2":>9} {"D=диаг":>8}')
    lines.append('  ' + '─' * 45)

    max_diag = max(d for _, _, _, d in bodies)
    bar_scale = 20.0 / max_diag if max_diag else 1.0

    for name, V, E, D in bodies:
        total_pairs = V * (V - 1) // 2
        bar = '█' * int(D * bar_scale)
        if color:
            c = _solid_color(name) if name != 'Эк-додекаэдр' else '\033[38;5;226m'
            lines.append(f'  {c}  {name:<13} {V:4d} {E:4d} {total_pairs:9d} {D:8d}  {bar}{_RESET}')
        else:
            lines.append(f'    {name:<13} {V:4d} {E:4d} {total_pairs:9d} {D:8d}  {bar}')

    lines.append('')
    lines.append('  Q6 (как граф): V=64  E=192  D = 64·63/2 − 192 = 2016 − 192 = 1824')

    lines.append('\n  8×8 карта Q6: цвет = yang_count; число "диагоналей" от h = ')
    lines.append('  (число вершин на dist≥2 от h) = 64 − 1 − 6 = 57')

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
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='poly_glyphs',
        description='Платоновы тела и Эк-додекаэдр через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('platonic',  help='5 платоновых тел как вложения в Q6')
    sub.add_parser('euler',     help='формула Эйлера V−E+F=χ')
    sub.add_parser('exdodeca',  help='Эк-додекаэдр: 32 вершины в Q6')
    sub.add_parser('diagonals', help='число диагоналей многогранников')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'platonic':
        print(render_platonic(color))
    elif args.cmd == 'euler':
        print(render_euler(color))
    elif args.cmd == 'exdodeca':
        print(render_exdodeca(color))
    elif args.cmd == 'diagonals':
        print(render_diagonals(color))


if __name__ == '__main__':
    main()
