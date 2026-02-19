"""hexnav/nav_glyphs.py — Q6 глифы через навигатор-консоль.

hexnav — интерактивный навигатор по Q6: каждый шаг = переворот бита (смена
ребра Q6). Гексаграмма h разделяется на две триграммы:
    нижняя триграмма = h & 7       (биты 0..2)
    верхняя триграмма = (h >> 3) & 7  (биты 3..5)

Восемь триграмм (символы Ба-гуа):
    000 ☷ Кунь  (Земля)    001 ☶ Гэн   (Гора)
    010 ☵ Кань  (Вода)     011 ☴ Сюй   (Ветер)
    100 ☳ Чжэнь (Гром)     101 ☲ Ли    (Огонь)
    110 ☱ Дуй   (Озеро)    111 ☰ Цянь  (Небо)

Пример: h=21 (010101) → нижн.=101=☲Ли, верхн.=010=☵Кань.

Визуализация (8×8, Gray-код Q6):
  trigrams [--upper]   — верхние (по умолч.) или нижние триграммы (0..7)
  layers   [--start s] — BFS-слои из s (= расстояние Хэмминга от s)
  antipode             — антиподальные пары {h, h⊕63}
  bits     [--bit b]   — выделить все h с установленным битом b

Команды CLI:
  trigrams [--upper]
  layers   [--start s]
  antipode
  bits     [--bit b]
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, hamming, antipode, neighbors, to_bits
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Триграммы (Ба-гуа)
# ---------------------------------------------------------------------------

_TRIG_SYM  = ['☷', '☶', '☵', '☴', '☳', '☲', '☱', '☰']
_TRIG_NAME = ['Кунь', 'Гэн', 'Кань', 'Сюй', 'Чжэнь', 'Ли', 'Дуй', 'Цянь']
_TRIG_ELEM = ['Земля', 'Гора', 'Вода', 'Ветер', 'Гром', 'Огонь', 'Озеро', 'Небо']

# ANSI-цвета для 8 триграмм (стихии)
_TRIG_COLOR = [
    '\033[38;5;94m',    # 0 ☷ Кунь — коричневый (Земля)
    '\033[38;5;240m',   # 1 ☶ Гэн  — серый (Гора)
    '\033[38;5;39m',    # 2 ☵ Кань — синий (Вода)
    '\033[38;5;82m',    # 3 ☴ Сюй  — зелёный (Ветер)
    '\033[38;5;214m',   # 4 ☳ Чжэнь — оранжевый (Гром)
    '\033[38;5;196m',   # 5 ☲ Ли   — красный (Огонь)
    '\033[38;5;75m',    # 6 ☱ Дуй  — голубой (Озеро)
    '\033[38;5;226m',   # 7 ☰ Цянь — жёлтый (Небо)
]

# Цвета для расстояний (BFS-слои 0..6)
_DIST_COLOR = _YANG_ANSI  # удобно: yang_count = hamming от 0, dist = hamming от s

_GRAY3 = [i ^ (i >> 1) for i in range(8)]


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
# 1. Триграммы (верхние или нижние)
# ---------------------------------------------------------------------------

def render_trigrams(upper: bool = True, color: bool = True) -> str:
    """8×8 сетка: триграмма каждого глифа.

    upper=True  → верхняя триграмма = (h >> 3) & 7 (биты 5,4,3)
    upper=False → нижняя  триграмма = h & 7         (биты 2,1,0)

    Ярлык — числовой код 0..7 триграммы; цвет = стихия.
    """
    part = 'верхняя' if upper else 'нижняя'
    lines = _header(
        f'Навигатор: {part} триграмма h',
        '0=☷Земля 1=☶Гора 2=☵Вода 3=☴Ветер 4=☳Гром 5=☲Огонь 6=☱Озеро 7=☰Небо',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            tg = (h >> 3) & 7 if upper else h & 7
            sym = str(tg)
            if color:
                c = _TRIG_COLOR[tg]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  {"Верхние" if upper else "Нижние"} триграммы ({"биты 5,4,3" if upper else "биты 2,1,0"}):')
    for t in range(8):
        sym = _TRIG_SYM[t]
        name = _TRIG_NAME[t]
        elem = _TRIG_ELEM[t]
        cnt  = sum(1 for h in range(64)
                   if ((h >> 3) & 7 if upper else h & 7) == t)
        c = _TRIG_COLOR[t] if color else ''
        r = _RESET if color else ''
        lines.append(f'  {c}{t} {sym} {name:6s} ({elem:6s}){r}: {cnt} глифов')

    lines.append('')
    lines.append('  Примеры (h, нижн., верхн.):')
    for h in [0, 1, 21, 42, 63]:
        lo = h & 7
        hi = (h >> 3) & 7
        lines.append(
            f'    h={h:2d} {format(h,"06b")}: '
            f'нижн.={lo} {_TRIG_SYM[lo]}{_TRIG_NAME[lo]:6s}'
            f'  верхн.={hi} {_TRIG_SYM[hi]}{_TRIG_NAME[hi]}'
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. BFS-слои из начальной точки
# ---------------------------------------------------------------------------

def render_layers(start: int = 0, color: bool = True) -> str:
    """8×8 сетка: расстояние Хэмминга от start до каждого глифа.

    dist(start, h) = hamming(start, h) = popcount(start XOR h) ∈ {0,...,6}.
    Цвет = расстояние (как ян-слои от нуля).
    """
    lines = _header(
        f'Навигатор: BFS-слои из h={start} ({format(start, "06b")})',
        'Цифра = расстояние Хэмминга (= число ребёр до start)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h   = (row_g << 3) | col_g
            d   = hamming(start, h)
            sym = str(d)
            if color:
                c = _YANG_ANSI[d]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Старт: h={start} = {format(start,"06b")}')
    lines.append(f'  Антипод: h={antipode(start)} = {format(antipode(start),"06b")}')
    lines.append('  BFS-слои (размеры = C(6,d)):')
    for d in range(7):
        from math import comb
        cnt = comb(6, d)
        c = _YANG_ANSI[d] if color else ''
        r = _RESET if color else ''
        lines.append(f'    d={d}: {c}{cnt:2d} вершин{r}  (C(6,{d})={cnt})')

    # Соседи start
    nb = sorted(neighbors(start))
    lines.append(f'  Соседи h={start}: {nb}  (6 рёбер Q6)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Антиподальные пары
# ---------------------------------------------------------------------------

def render_antipode(color: bool = True) -> str:
    """8×8 сетка: полуплоскость антиподальных пар {h, h⊕63}.

    Для каждого h ∈ 0..63:
      h < 32 → нижняя половина (✦ голубой)
      h ≥ 32 → верхняя половина (✦ оранжевый)
    Антипод(h) = h XOR 63: расстояние = 6.
    """
    lines = _header(
        'Навигатор: антиподальные пары {h, h⊕63}',
        'L=нижн.полов.(h<32, голубой)  U=верхн.полов.(h≥32, оранж.)  dist=6',
    )

    _LO_COLOR = '\033[38;5;39m'    # голубой = нижняя
    _HI_COLOR = '\033[38;5;208m'   # оранжевый = верхняя

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h   = (row_g << 3) | col_g
            sym = 'L' if h < 32 else 'U'
            if color:
                c = _LO_COLOR if h < 32 else _HI_COLOR
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Антипод(h) = h XOR 63 (инвертировать все 6 битов)')
    lines.append('  Расстояние Хэмминга между h и антиподом = 6 (диаметр Q6)')
    lines.append('  Нижняя половина (h < 32): 32 глифа')
    lines.append('  Верхняя половина (h ≥ 32): 32 глифа')
    lines.append('  Каждая пара пересекает ровно 6 рёбер Q6')
    lines.append('')
    lines.append('  Пары по ян-слоям:')
    for k in range(4):
        ak = 6 - k
        from math import comb
        cnt = comb(6, k)
        c  = _YANG_ANSI[k] if color else ''
        ca = _YANG_ANSI[ak] if color else ''
        r  = _RESET if color else ''
        lines.append(
            f'    ян={k} ↔ ян={ak}: {c}{cnt}{r} пар  '
            f'(C(6,{k})={cnt})'
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Маска бита
# ---------------------------------------------------------------------------

def render_bits(bit: int = 0, color: bool = True) -> str:
    """8×8 сетка: бит b установлен в h (Y) или нет (N).

    Бит b делит Q6 на два изоморфных 5-куба (Q5 × {0,1}).
    32 глифа имеют bit=1, 32 глифа имеют bit=0.
    """
    lines = _header(
        f'Навигатор: бит {bit}  (h & (1<<{bit}))',
        f'Y=бит {bit} установлен  N=бит {bit} сброшен  (разбивает Q6 на 2×Q5)',
    )

    _SET_COLOR = '\033[38;5;82m'    # зелёный = бит установлен
    _CLR_COLOR = '\033[38;5;238m'   # серый   = бит сброшен

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h    = (row_g << 3) | col_g
            is_set = bool((h >> bit) & 1)
            sym  = 'Y' if is_set else 'N'
            if color:
                c = _SET_COLOR if is_set else _CLR_COLOR
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    ones  = sum(1 for h in range(64) if (h >> bit) & 1)
    zeros = 64 - ones
    lines.append(f'  Бит {bit} = 1: {ones} глифов   Бит {bit} = 0: {zeros} глифов')
    lines.append(f'  Переворот бита {bit}: перемещение по ребру Q6 → Q5-подкуб')
    lines.append('  Каждое ребро Q6 соответствует ровно одному биту.')
    lines.append('')
    lines.append(f'  6 бит × 2 полукуба = 12 подграфов Q5, но смежные — совпадают.')
    lines.append('  bit 0 → столбцы ☷(000) ↔ ☶(001):')

    # Показать пути через бит bit
    sample_pairs = [(h, h ^ (1 << bit)) for h in range(0, 8)]
    for a, b in sample_pairs[:4]:
        lines.append(
            f'    h={a:2d} ({format(a,"06b")}) ↔ h={b:2d} ({format(b,"06b")})'
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='nav_glyphs',
        description='Q6 глифы через навигатор: триграммы, расстояния, антиподы.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')

    sp = sub.add_parser('trigrams', help='триграммы (нижние / верхние)')
    sp.add_argument('--upper', action='store_true',
                    help='верхняя триграмма (биты 5,4,3); по умолч. нижняя')

    sl = sub.add_parser('layers', help='BFS-слои из стартовой вершины')
    sl.add_argument('--start', type=int, default=0,
                    metavar='S', help='стартовая вершина (0..63)')

    sub.add_parser('antipode', help='антиподальные пары {h, h⊕63}')

    sb = sub.add_parser('bits', help='маска конкретного бита')
    sb.add_argument('--bit', type=int, default=0, metavar='B',
                    help='бит для маскирования (0..5)')

    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'trigrams':
        print(render_trigrams(upper=getattr(args, 'upper', False), color=color))
    elif args.cmd == 'layers':
        s = max(0, min(63, args.start))
        print(render_layers(start=s, color=color))
    elif args.cmd == 'antipode':
        print(render_antipode(color=color))
    elif args.cmd == 'bits':
        b = max(0, min(5, args.bit))
        print(render_bits(bit=b, color=color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
