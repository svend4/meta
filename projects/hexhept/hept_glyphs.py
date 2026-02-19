"""hept_glyphs — Гептаэдр (модель RP²) через глифы Q6.

Гептаэдр Германа: V=6, F=7, E=12, χ=1 — полиэдральная модель RP².
6 вершин гептаэдра ↔ 6 битов гексаграммы.

Характеристика Эйлера:
  Сфера:   χ = V − E + F = 2  (платоновы тела)
  RP²:     χ = V − E + F = 1  (гептаэдр)
  Тор:     χ = 0
  Кляйн:   χ = 0

Грани гептаэдра: 4 треугольника + 3 квадрата.
Каждая вершина ↔ бит Q6 (0..5) → глиф 1,2,4,8,16,32.

Кандидаты RP²: все (V,F,E) с χ=1, V≥4, степень ≥3.
Среди 0..63: 34 комбинации (V,F,E) с χ=1.

Визуализация:
  heptahedron — 6 вершин как глифы Q6, 7 граней
  rp2         — кандидаты RP² по (V,F,E) → глиф V⊕F⊕E
  euler       — χ=V−E+F для каждого глифа (h=вершины, yang=E)
  faces       — 7 граней гептаэдра как подмножества Q6

Команды CLI:
  heptahedron
  rp2
  euler
  faces
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexhept.hexhept import Heptahedron, RP2Checker
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_hept = Heptahedron(edge_length=1.0)
_checker = RP2Checker()

# 6 вершин гептаэдра ↔ 6 однобитных глифов
_HEPT_VERTS = [1 << i for i in range(6)]   # [1, 2, 4, 8, 16, 32]
_HEPT_VERT_SET = set(_HEPT_VERTS)


# ---------------------------------------------------------------------------
# 1. Гептаэдр
# ---------------------------------------------------------------------------

def render_heptahedron(color: bool = True) -> str:
    """
    8×8 сетка: 6 вершин гептаэдра как однобитные глифы Q6.

    V=6: глифы 1,2,4,8,16,32  (yang=1)
    Грани гептаэдра закодированы как XOR соседних вершин.
    """
    ft = _hept.face_types()
    sa = _hept.surface_area()
    pv = _hept.pseudo_volume()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Гептаэдр Германа: V=6  F=7  E=12  χ=1 = χ(RP²)')
    lines.append(f'  Грани: {ft["triangle"]} треугольников + {ft["square"]} квадрата')
    lines.append(f'  Площадь поверхности ≈ {sa:.4f}   Псевдообъём ≈ {pv:.4f}')
    lines.append('  6 вершин ↔ 6 битов Q6: глифы 1,2,4,8,16,32 (yang=1)')
    lines.append('  Жирный = вершина гептаэдра,  Цвет = yang_count')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            is_vert = h in _HEPT_VERT_SET
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_BG[yc] + _BOLD if is_vert else _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            is_vert = h in _HEPT_VERT_SET
            yc = yang_count(h)
            if is_vert:
                bit = next(b for b in range(6) if h == (1 << b))
                tag = f'V{bit}'
            else:
                tag = '  '
            if color:
                c = _YANG_ANSI[yc] if is_vert else _YANG_ANSI[0]
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Вершины
    lines.append('  6 вершин гептаэдра → 6 битов Q6:')
    for i, h in enumerate(_HEPT_VERTS):
        if color:
            c = _YANG_ANSI[1]
            lines.append(f'  {c}  V{i}: бит{i} → глиф {h:2d} = {format(h,"06b")}{_RESET}')
        else:
            lines.append(f'    V{i}: бит{i} → глиф {h:2d} = {format(h,"06b")}')

    lines.append('')
    lines.append(f'  Гептаэдр — единственный выпуклый многогранник с χ=1.')
    lines.append('  Теорема Германа об единственности гептаэдра.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Кандидаты RP²
# ---------------------------------------------------------------------------

def render_rp2(color: bool = True) -> str:
    """
    8×8 сетка: кандидаты (V,F,E) с χ=1 → глиф V⊕F⊕E (XOR).

    Показывает, какие комбинации (V,F,E) удовлетворяют χ=1.
    """
    cands = _checker.enumerate_rp2_candidates(max_vertices=10)

    # Для каждого кандидата: глиф = V ⊕ F (XOR, truncated to 6 bits)
    cand_glyphs = {}
    for c in cands:
        V, F, E = c['vertices'], c['faces'], c['edges']
        g = ((V - 1) ^ (F - 1)) % 64
        if g not in cand_glyphs:
            cand_glyphs[g] = []
        cand_glyphs[g].append((V, F, E))

    # Гептаэдр
    hept_glyph = ((_hept.vertices() - 1) ^ (_hept.faces() - 1)) % 64

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Кандидаты RP²: (V,F,E) с χ = V−E+F = 1')
    lines.append(f'  Всего кандидатов V≤10: {len(cands)}')
    lines.append('  Глиф = (V−1) XOR (F−1) mod 64')
    lines.append('  Жирный = гептаэдр (V=6,F=7,E=12)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            has_cand = h in cand_glyphs
            is_hept = (h == hept_glyph)
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                if is_hept:
                    c = _YANG_BG[yc] + _BOLD
                elif has_cand:
                    c = _YANG_ANSI[yc]
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
            has_cand = h in cand_glyphs
            is_hept = (h == hept_glyph)
            if is_hept:
                tag = 'RP²'
            elif has_cand:
                V, F, E = cand_glyphs[h][0]
                tag = f'{V},{F}'
            else:
                tag = '   '
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc] if has_cand else _YANG_ANSI[0]
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Первые 10 кандидатов
    lines.append('  Первые 10 кандидатов (V,F,E) с χ=1:')
    for c2 in cands[:10]:
        V, F, E = c2['vertices'], c2['faces'], c2['edges']
        chi = V - E + F
        is_hept = (V == 6 and F == 7 and E == 12)
        mark = ' ← ГЕПТАЭДР!' if is_hept else ''
        if color:
            col = _YANG_ANSI[V % 7]
            lines.append(f'  {col}  V={V:2d}  F={F:2d}  E={E:2d}  χ={chi}{mark}{_RESET}')
        else:
            lines.append(f'    V={V:2d}  F={F:2d}  E={E:2d}  χ={chi}{mark}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Характеристика Эйлера χ = h − yang − something
# ---------------------------------------------------------------------------

def render_euler(color: bool = True) -> str:
    """
    8×8 сетка: интерпретация глифа h как χ = yang_count(h) − something.

    Для каждого h: «поверхность» с V=yang_count(h)+1, E=6−yang_count(h), F=h%7.
    Показывает разнообразие χ по Q6.
    """
    def euler_for(h: int) -> int:
        V = yang_count(h) + 1   # 1..7
        E = 6 - yang_count(h)   # 0..6
        F = (h % 7) + 1         # 1..7
        return V - E + F

    chis = [euler_for(h) for h in range(64)]
    chi_min, chi_max = min(chis), max(chis)

    # Нормировка для цвета: сдвигаем к 0..6
    def chi_color(chi: int) -> int:
        return min(max(chi - chi_min, 0), 6)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Характеристика Эйлера χ = V−E+F по глифам Q6')
    lines.append('  Интерпретация: V=yang+1, E=6−yang, F=h mod 7 + 1')
    lines.append(f'  χ(RP²)=1,  χ(сфера)=2,  χ(тор)=0')
    lines.append(f'  Диапазон χ на Q6: {chi_min}..{chi_max}')
    lines.append('  Жирный = χ=1 (RP²-тип),  Цвет = χ')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            chi = chis[h]
            rows3 = render_glyph(h)
            if color:
                cc = chi_color(chi)
                is_rp2 = (chi == 1)
                c = _YANG_BG[cc] + _BOLD if is_rp2 else _YANG_ANSI[cc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            chi = chis[h]
            if color:
                cc = chi_color(chi)
                c = _YANG_ANSI[cc]
                lbl.append(f'{c}χ={chi:+d}{_RESET}')
            else:
                lbl.append(f'χ={chi:+d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    from collections import Counter
    chi_dist = Counter(chis)
    lines.append('  Распределение χ по Q6:')
    for chi_val in sorted(chi_dist):
        cnt = chi_dist[chi_val]
        mark = ''
        if chi_val == 1:
            mark = ' ← RP²'
        elif chi_val == 2:
            mark = ' ← сфера'
        elif chi_val == 0:
            mark = ' ← тор'
        if color:
            c = _YANG_ANSI[chi_color(chi_val)]
            lines.append(f'  {c}  χ={chi_val:+d}: {cnt:2d} глифов{mark}{_RESET}')
        else:
            lines.append(f'    χ={chi_val:+d}: {cnt:2d} глифов{mark}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Грани гептаэдра
# ---------------------------------------------------------------------------

def render_faces(color: bool = True) -> str:
    """
    8×8 сетка: 7 граней гептаэдра как XOR подмножеств вершин Q6.

    Грань = подмножество {V_i, V_j, ...} → XOR = V_i ⊕ V_j ⊕ ...
    """
    # 7 граней: 4 треугольника (3 вершины) + 3 квадрата (4 вершины)
    # Вершины: V0=1, V1=2, V2=4, V3=8, V4=16, V5=32
    triangles = [
        (0, 1, 2),   # V0,V1,V2
        (0, 3, 4),   # V0,V3,V4
        (1, 3, 5),   # V1,V3,V5
        (2, 4, 5),   # V2,V4,V5
    ]
    squares = [
        (0, 1, 5, 3),  # V0,V1,V5,V3
        (0, 2, 5, 4),  # V0,V2,V5,V4
        (1, 2, 4, 3),  # V1,V2,V4,V3 — квадрат «дна»
    ]

    # XOR каждой грани
    face_xors = []
    face_descs = []
    for tri in triangles:
        xor_val = 0
        for vi in tri:
            xor_val ^= (1 << vi)
        face_xors.append(xor_val)
        face_descs.append(f'△V{tri[0]}V{tri[1]}V{tri[2]} ={xor_val}')
    for sq in squares:
        xor_val = 0
        for vi in sq:
            xor_val ^= (1 << vi)
        face_xors.append(xor_val)
        face_descs.append(f'□V{sq[0]}V{sq[1]}V{sq[2]}V{sq[3]} ={xor_val}')

    face_set = set(face_xors)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  7 граней гептаэдра как XOR-глифы Q6')
    lines.append('  Грань {V_i,...} → XOR битов → глиф Q6')
    lines.append('  4 треугольника (△) + 3 квадрата (□)')
    lines.append('  Жирный = XOR-глиф грани,  Цвет = yang_count')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            is_face = h in face_set
            is_vert = h in _HEPT_VERT_SET
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                if is_face:
                    c = _YANG_BG[yc] + _BOLD
                elif is_vert:
                    c = _YANG_ANSI[yc]
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
            is_face = h in face_set
            is_vert = h in _HEPT_VERT_SET
            if is_face:
                fi = face_xors.index(h)
                tag = '△' if fi < 4 else '□'
                tag += str(fi if fi < 4 else fi - 4)
            elif is_vert:
                bi = next(b for b in range(6) if h == (1 << b))
                tag = f'V{bi}'
            else:
                tag = '  '
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc] if (is_face or is_vert) else _YANG_ANSI[0]
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Список граней
    lines.append('  7 граней гептаэдра:')
    for i, (desc, xor_val) in enumerate(zip(face_descs, face_xors)):
        yc = yang_count(xor_val)
        if color:
            c = _YANG_ANSI[yc]
            lines.append(f'  {c}  F{i}: {desc}  ({format(xor_val,"06b")})'
                         f'  yang={yc}{_RESET}')
        else:
            lines.append(f'    F{i}: {desc}  ({format(xor_val,"06b")})'
                         f'  yang={yc}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='hept_glyphs',
        description='Гептаэдр и модель RP² через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('heptahedron', help='6 вершин гептаэдра как глифы')
    sub.add_parser('rp2',        help='кандидаты RP² (V,F,E) с χ=1')
    sub.add_parser('euler',      help='характеристика Эйлера χ по Q6')
    sub.add_parser('faces',      help='7 граней гептаэдра как XOR-глифы')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'heptahedron':
        print(render_heptahedron(color))
    elif args.cmd == 'rp2':
        print(render_rp2(color))
    elif args.cmd == 'euler':
        print(render_euler(color))
    elif args.cmd == 'faces':
        print(render_faces(color))


if __name__ == '__main__':
    main()
