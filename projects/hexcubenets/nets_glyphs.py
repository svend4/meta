"""nets_glyphs — Развёртки куба и гиперкубов через глифы Q6.

Каждый глиф (0..63) — 6-битный вектор, биты которого кодируют грани куба:
  бит 0 = T (Top / Верх)
  бит 1 = B (Bottom / Низ)
  бит 2 = F (Front / Перед)
  бит 3 = K (Back / Зад)
  бит 4 = L (Left / Лево)
  бит 5 = R (Right / Право)

6 граней куба ↔ 6 битов Q6: yang_count(h) = число «активных» граней.

Развёртки куба (cube nets):
  Куб имеет ровно 11 различных развёрток (с точностью до вращений).
  Симметрия: 5 зеркальных, 5 без симметрии, 1 центральная.

Гиперкубы: Q_n имеет развёрток:
  n=1: 1,  n=2: 1,  n=3: 11,  n=4: 261,  n=5: 33 064, ...

Визуализация:
  faces    — 6 граней куба как глифы (биты 0..5)
  nets     — 11 развёрток: какие грани «раскрыты» (yang ≤ 2)
  sym      — симметрия развёртки: mirror / none / central
  hypercube — число развёрток n-мерного куба (n=1..5)

Команды CLI:
  faces
  nets
  sym
  hypercube
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexcubenets.hexcubenets import CubeNets, Net, FACES
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# 6 граней = 6 битов
_FACE_BITS = {name: i for i, name in enumerate(FACES)}
_FACE_NAMES = list(FACES)   # ['T', 'B', 'F', 'K', 'L', 'R']

_NET_COLORS = [
    '\033[38;5;27m',  '\033[38;5;82m',  '\033[38;5;196m',
    '\033[38;5;208m', '\033[38;5;201m', '\033[38;5;226m',
    '\033[38;5;39m',  '\033[38;5;238m', '\033[38;5;46m',
    '\033[38;5;51m',  '\033[38;5;160m',
]

_SYM_COLOR = {
    'mirror':  '\033[38;5;82m',
    'central': '\033[38;5;226m',
    'none':    '\033[38;5;196m',
}


# ---------------------------------------------------------------------------
# 1. Грани куба как глифы
# ---------------------------------------------------------------------------

def render_faces(color: bool = True) -> str:
    """
    8×8 сетка Q6: биты 0..5 = грани куба T,B,F,K,L,R.

    yang_count(h) = число активных граней гексаграммы h.
    Слои yang=1 → 6 одиночных граней (базисные глифы: 1,2,4,8,16,32).
    """
    face_bits = [2 ** i for i in range(6)]   # глифы отдельных граней

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Грани куба как биты Q6: T B F K L R → биты 0..5')
    lines.append('  Гексаграмма h: бит i = «грань i активна»')
    lines.append('  yang_count(h) = число активных граней')
    lines.append('  yang=0: нет граней (глиф 0)   yang=6: все грани (глиф 63)')
    lines.append('  yang=1 (6 однобитных): грани T=1, B=2, F=4, K=8, L=16, R=32')
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
            # Имена активных граней
            active = ''.join(_FACE_NAMES[i] for i in range(6) if (h >> i) & 1)
            if not active:
                active = '∅'
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}{active:6s}{_RESET}')
            else:
                lbl.append(f'{active:6s}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    # Показать 6 граней
    lines.append('  6 граней куба (yang=1 глифы):')
    for i, name in enumerate(_FACE_NAMES):
        h = 1 << i
        if color:
            c = _YANG_ANSI[1]
            lines.append(f'  {c}  {name}: глиф {h:2d} = {format(h,"06b")}{_RESET}')
        else:
            lines.append(f'    {name}: глиф {h:2d} = {format(h,"06b")}')
    lines.append('')
    lines.append('  Пары противоположных граней (yang=2, сумма битов = 0b000011 etc.):')
    opposite_pairs = [(0, 1), (2, 3), (4, 5)]
    for i, j in opposite_pairs:
        h = (1 << i) | (1 << j)
        n1, n2 = _FACE_NAMES[i], _FACE_NAMES[j]
        if color:
            c = _YANG_ANSI[2]
            lines.append(f'  {c}  {n1}↔{n2}: глиф {h:2d} = {format(h,"06b")}{_RESET}')
        else:
            lines.append(f'    {n1}↔{n2}: глиф {h:2d} = {format(h,"06b")}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Развёртки куба
# ---------------------------------------------------------------------------

def render_nets(color: bool = True) -> str:
    """
    Показать все 11 развёрток куба.

    Для каждой развёртки: ASCII-схема + глифы 6 граней в порядке развёртки.
    Цвет = индекс развёртки.
    """
    cn = CubeNets()
    nets = cn.enumerate_all()
    classify = cn.classify()
    prove = cn.prove_count()

    lines: list[str] = []
    lines.append('╔' + '═' * 64 + '╗')
    lines.append('║  11 развёрток (nets) куба — все различны с точн. до вращений  ║')
    lines.append(f'║  {prove["total_subsets"]} покрывающих деревьев,  {prove["valid"]} валидных,'
                 f'  {prove["nets"]} попарно различных' + ' ' * 4 + '║')
    lines.append(f'║  Симметрия: зеркальных={classify["mirror"]}'
                 f'  центральных={classify["central"]}'
                 f'  без симметрии={classify["none"]}' + ' ' * 13 + '║')
    lines.append('╚' + '═' * 64 + '╝')
    lines.append('')

    for idx, net in enumerate(nets):
        sym = net.symmetry()
        if color:
            c = _NET_COLORS[idx % len(_NET_COLORS)]
            sc = _SYM_COLOR.get(sym, _YANG_ANSI[0])
            lines.append(f'  {c}─── Развёртка {idx+1}  {sc}симметрия={sym}{c} ───{_RESET}')
        else:
            lines.append(f'  ─── Развёртка {idx+1}  симметрия={sym} ───')

        # ASCII-схема развёртки
        ascii_lines = net.to_ascii().split('\n')
        for al in ascii_lines:
            if color:
                c = _NET_COLORS[idx % len(_NET_COLORS)]
                lines.append(f'  {c}{al}{_RESET}')
            else:
                lines.append(f'  {al}')

        # Глифы граней в этой развёртке (по координатам)
        face_glyphs = []
        for fi, fname in enumerate(_FACE_NAMES):
            h = 1 << fi
            face_glyphs.append(h)

        glyph_renders = [render_glyph(h) for h in face_glyphs]
        if color:
            c = _NET_COLORS[idx % len(_NET_COLORS)]
            glyph_renders = [[c + r + _RESET for r in g] for g in glyph_renders]

        lines.append('  Грани: ' + '  '.join(_FACE_NAMES))
        for ri in range(3):
            lines.append('    ' + '  '.join(g[ri] for g in glyph_renders))
        lines.append('')

    lines.append(f'  Всего 11 развёрток = число способов «раскрыть» куб в плоскость.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Симметрия развёрток
# ---------------------------------------------------------------------------

def render_sym(color: bool = True) -> str:
    """
    8×8 сетка: раскраска глифов по «симметрии» связанной развёртки.

    Каждый глиф h (0..63) имеет yang_count(h) активных граней.
    - yang=0: пустой куб (нет граней)
    - yang=1..5: «частичные» развёртки (поднабор граней)
    - yang=6: все 6 граней = полный куб

    Связь с развёртками: yang=2,3,4 — различные конфигурации граней.
    Раскраска: yang_count(h) mod 3 → {0, 1, 2} = тип конфигурации.
    """
    cn = CubeNets()
    nets = cn.enumerate_all()
    classify = cn.classify()

    # Для каждого набора граней (подмножество {T,B,F,K,L,R}) → тип симметрии
    # Определяется yang_count(h)
    yang_sym_label = {
        0: 'пусто',
        1: '1 грань',
        2: '2 грани',
        3: '3 грани',
        4: '4 грани',
        5: '5 граней',
        6: 'полный куб',
    }

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Симметрия развёрток куба: mirror / central / none')
    lines.append(f'  11 развёрток: {classify["mirror"]} зеркальных  '
                 f'{classify["central"]} центральных  {classify["none"]} без симметрии')
    lines.append('  В Q6: цвет = yang_count = число активных граней (0..6)')
    lines.append('  Жирный = yang=6 (все грани, «полная развёртка»)')
    lines.append('═' * 66)
    lines.append('')

    # Показать 3 типа симметрии их развёртки
    for sym_name, sym_color in _SYM_COLOR.items():
        sym_nets = [n for n in nets if n.symmetry() == sym_name]
        if color:
            lines.append(f'  {sym_color}─── {sym_name.upper()} ({len(sym_nets)} развёрток) ───{_RESET}')
        else:
            lines.append(f'  ─── {sym_name.upper()} ({len(sym_nets)} развёрток) ───')
        for net in sym_nets[:2]:
            for al in net.to_ascii().split('\n')[:3]:
                if color:
                    lines.append(f'    {sym_color}{al}{_RESET}')
                else:
                    lines.append(f'    {al}')
        lines.append('')

    # 8×8 карта глифов Q6 по yang_count
    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            yc = yang_count(h)
            rows3 = render_glyph(h)
            if color:
                if yc == 6:
                    c = _YANG_BG[6] + _BOLD
                elif yc == 0:
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
            yc = yang_count(h)
            label = yang_sym_label[yc][:4]
            if color:
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}{label}{_RESET}')
            else:
                lbl.append(label)
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Гиперкубные развёртки
# ---------------------------------------------------------------------------

def render_hypercube(color: bool = True) -> str:
    """
    Число развёрток n-мерного куба для n=1..5.

    Визуализация: глифы Q6 соответствуют 6-кубу Q6,
    биты 0..5 кодируют 6 осей измерения.
    """
    cn = CubeNets()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Развёртки n-мерных гиперкубов Q_n')
    lines.append('  Qₙ: n осей, 2n гиперграней, n·2^(n-1) рёбер')
    lines.append('  Q6: 6 осей ↔ 6 битов каждого глифа')
    lines.append('═' * 66)
    lines.append('')

    net_counts = []
    for n in range(1, 6):
        count = cn.hypercube_nets(n)
        net_counts.append((n, count))
        count_str = f'{count:,}' if isinstance(count, int) else str(count)
        if color:
            c = _YANG_ANSI[n]
            lines.append(f'  {c}  Q{n}: {count_str} развёрток{_RESET}')
        else:
            lines.append(f'    Q{n}: {count_str} развёрток')

    lines.append('')
    lines.append('  Каждое измерение Q_n ↔ один бит гексаграммы 0..63:')
    for i in range(6):
        h_axis = 1 << i
        if color:
            c = _YANG_ANSI[1]
            lines.append(f'  {c}  Ось {i}: глиф {h_axis:2d} = {format(h_axis,"06b")}'
                         f'  ({_FACE_NAMES[i]}){_RESET}')
        else:
            lines.append(f'    Ось {i}: глиф {h_axis:2d} = {format(h_axis,"06b")}'
                         f'  ({_FACE_NAMES[i]})')

    lines.append('')
    lines.append('  Q6 = пространство всех активаций {T,B,F,K,L,R}:')
    lines.append('  8×8 карта: цвет = yang_count = число осей')

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
                lbl.append(f'{c}Q{yc}{_RESET}')
            else:
                lbl.append(f'Q{yc}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='nets_glyphs',
        description='Развёртки куба и гиперкубов через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('faces',      help='6 граней куба как биты Q6')
    sub.add_parser('nets',       help='все 11 развёрток куба')
    sub.add_parser('sym',        help='симметрия развёрток (mirror/central/none)')
    sub.add_parser('hypercube',  help='число развёрток n-мерных гиперкубов')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'faces':
        print(render_faces(color))
    elif args.cmd == 'nets':
        print(render_nets(color))
    elif args.cmd == 'sym':
        print(render_sym(color))
    elif args.cmd == 'hypercube':
        print(render_hypercube(color))


if __name__ == '__main__':
    main()
