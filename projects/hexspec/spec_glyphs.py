"""hexspec/spec_glyphs.py — Q6 глифы через верификацию конечных автоматов.

HexProtocol — встроенный КА на вершинах Q6, моделирующий жизненный
цикл соединения. Состояния — конкретные гексаграммы; переходы — перевороты
ровно одного бита (рёбра Q6).

Топология автомата (8 именованных состояний):
  INIT    =  0  (000000)  начальное
  CONNECT =  1  (000001)  flip bit 0
  AUTH    =  3  (000011)  flip bit 1
  READY   =  7  (000111)  flip bit 2
  ACTIVE  = 15  (001111)  flip bit 3
  PROCESS = 31  (011111)  flip bit 4
  DONE    = 63  (111111)  конечное
  RESET   = 32  (100000)  возврат к INIT

Переходы:
  INIT→CONNECT, CONNECT→AUTH, AUTH→READY, READY→ACTIVE,
  ACTIVE→PROCESS, PROCESS→DONE   (главная цепочка)
  ACTIVE→ERROR (15→14), ERROR→RECOVER (14→6)  (ветка ошибок)
  RESET→INIT (32→0)  (возврат)

Запрещённые: 14 (ERROR) и 6 (RECOVER)  — ошибочные состояния.

Визуализация (8×8, Gray-код Q6):
  reach   — достижимые состояния из INIT
  cover   — покрытие переходов (используемые/возможные рёбра)
  dead    — тупиковые состояния (нет исходящих переходов, не финальные)
  paths   — тестовые сценарии: покрытие всех переходов

Команды CLI:
  reach
  cover
  dead
  paths
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexspec.verifier import Spec
from projects.hexspec.generator import (
    all_states_paths,
    all_transitions_paths,
    format_path,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Встроенный автомат HexProtocol
# ---------------------------------------------------------------------------

_STATES: dict[str, int] = {
    'INIT':    0,
    'CONNECT': 1,
    'AUTH':    3,
    'READY':   7,
    'ACTIVE':  15,
    'PROCESS': 31,
    'DONE':    63,
    'RESET':   32,
}

_TRANSITIONS: list[tuple[int, int]] = [
    (0,  1),   # INIT → CONNECT      (flip bit 0)
    (1,  3),   # CONNECT → AUTH      (flip bit 1)
    (3,  7),   # AUTH → READY        (flip bit 2)
    (7,  15),  # READY → ACTIVE      (flip bit 3)
    (15, 31),  # ACTIVE → PROCESS    (flip bit 4)
    (31, 63),  # PROCESS → DONE      (flip bit 5)
    (32, 0),   # RESET → INIT        (flip bit 5)
    (15, 14),  # ACTIVE → ERROR      (flip bit 0)
    (14, 6),   # ERROR → RECOVER     (flip bit 3)
]

_BIT_NAMES = ['conn', 'auth', 'ready', 'active', 'busy', 'done']

_SPEC = Spec(
    name='HexProtocol',
    bit_names=_BIT_NAMES,
    states=_STATES,
    transitions=_TRANSITIONS,
    initial=0,
    final={63},
    forbidden={14, 6},
    description='Q6 protocol state machine',
)

# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

_GRAY3 = [i ^ (i >> 1) for i in range(8)]

# Цвета для ролей вершин
_REACH_COLOR   = '\033[38;5;82m'    # зелёный = достижимо
_UNREACH_COLOR = '\033[38;5;238m'   # серый   = недостижимо
_NAMED_COLOR   = '\033[38;5;226m'   # жёлтый  = именованное состояние
_FORB_COLOR    = '\033[38;5;196m'   # красный = запрещённое
_FINAL_COLOR   = '\033[38;5;208m'   # оранжевый = финальное
_DEAD_COLOR    = '\033[38;5;196m'   # красный = тупик

# Обратный словарь hexagram → state name
_H2NAME: dict[int, str] = {v: k for k, v in _STATES.items()}


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
# 1. Достижимые состояния (reach)
# ---------------------------------------------------------------------------

def render_reach(color: bool = True) -> str:
    """8×8 сетка: достижимость из INIT по переходам HexProtocol.

    R = достижимо  F = запрещено (forbidden)  · = недостижимо
    Именованные состояния выделены ярко.
    """
    reachable = _SPEC.reachable_states()
    forbidden = _SPEC.forbidden_reachable()

    lines = _header(
        'HexProtocol: достижимые состояния из INIT (h=0)',
        'R=reach  F=forbidden  ·=unreachable  (ян-цвет по q6)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            name = _H2NAME.get(h)
            if h in forbidden:
                sym = 'F'
                c   = _FORB_COLOR if color else ''
            elif h in reachable:
                sym = 'R'
                c   = _REACH_COLOR if color else ''
            else:
                sym = '·'
                c   = _UNREACH_COLOR if color else ''
            r = _RESET if color else ''
            if color and name:
                cell = f'{_BOLD}{c}{sym}{r}'
            else:
                cell = f'{c}{sym}{r}'
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Всего достижимо: {len(reachable)}/64 состояний')
    lines.append(f'  Запрещённых достижимых: {len(forbidden)}')
    lines.append('  Именованные состояния:')
    for name, h in sorted(_STATES.items(), key=lambda x: x[1]):
        tag = '★INIT' if h == _SPEC.initial else ('★FINAL' if h in _SPEC.final
               else ('✗FORB' if h in _SPEC.forbidden else ''))
        lines.append(f'    h={h:2d} ({format(h, "06b")}) {name:8s} {tag}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Покрытие переходов (cover)
# ---------------------------------------------------------------------------

def render_cover(color: bool = True) -> str:
    """8×8 сетка: используемые переходы vs. все возможные рёбра Q6.

    Цвет — ян-слой источника. Ярлык:
      T = есть переход из/в данную вершину в спецификации
      · = вершина есть, но переходы не заданы
    """
    cov = _SPEC.coverage()
    used_edges  = set(_TRANSITIONS)
    # вершины, участвующие хотя бы в одном переходе
    in_trans: set[int] = set()
    for a, b in _TRANSITIONS:
        in_trans.add(a)
        in_trans.add(b)

    lines = _header(
        f'HexProtocol: покрытие переходов  '
        f'({cov["used"]}/{cov["possible"]}  ratio={cov["ratio"]:.3f})',
        'T=задан переход  ·=нет переходов  (цвет = ян-слой)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            sym = 'T' if h in in_trans else '·'
            if color:
                c = _YANG_ANSI[k] if h in in_trans else _UNREACH_COLOR
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Переходов в спецификации : {cov["used"]}')
    lines.append(f'  Возможных рёбер Q6       : {cov["possible"]}  (64×6/2=192)')
    lines.append(f'  Покрытие                 : {cov["ratio"]*100:.1f}%')
    lines.append('  Переходы HexProtocol:')
    for a, b in _TRANSITIONS:
        bit = (a ^ b).bit_length() - 1
        an  = _H2NAME.get(a, f'h={a}')
        bn  = _H2NAME.get(b, f'h={b}')
        lines.append(f'    {a:2d}→{b:2d}  [{an}→{bn}]  bit{bit}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Тупиковые состояния (dead)
# ---------------------------------------------------------------------------

def render_dead(color: bool = True) -> str:
    """8×8 сетка: тупиковые состояния (нет исходящих, не финальные).

    D = deadlock  F = финальное  · = прочие
    """
    deadlocks = _SPEC.deadlocks()
    finals    = _SPEC.final

    lines = _header(
        'HexProtocol: тупиковые состояния (deadlocks)',
        'D=тупик  F=финальное  ·=прочие',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            if h in deadlocks:
                sym = 'D'
                c   = _DEAD_COLOR if color else ''
            elif h in finals:
                sym = 'F'
                c   = _FINAL_COLOR if color else ''
            else:
                sym = '·'
                c   = _UNREACH_COLOR if color else ''
            r = _RESET if color else ''
            cells.append(f'{c}{sym}{r}')
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Тупиковых состояний: {len(deadlocks)}')
    if deadlocks:
        for h in sorted(deadlocks):
            lines.append(
                f'    h={h:2d} ({format(h, "06b")})  '
                f'{_H2NAME.get(h, "unnamed")}  yang={yang_count(h)}'
            )
    else:
        lines.append('    (нет тупиков среди именованных состояний)')
    lines.append(f'  Финальных состояний: {len(finals)}  '
                 f'({", ".join(str(h) for h in sorted(finals))})')
    # unreachable named
    unr = _SPEC.unreachable()
    lines.append(f'  Недостижимых именованных: {len(unr)}  '
                 f'({", ".join(_H2NAME.get(h, str(h)) for h in sorted(unr))})')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Тестовые пути (paths)
# ---------------------------------------------------------------------------

def render_paths(color: bool = True) -> str:
    """8×8 сетка: покрытие переходами тестовых сценариев.

    Вершины, задействованные хотя бы в одном тестовом пути — выделены.
    Число сценариев = число путей для покрытия всех переходов.
    """
    # Сценарии: покрыть все переходы
    paths = all_transitions_paths(_SPEC)
    # Все вершины, задействованные в тестах
    covered: set[int] = set()
    for path in paths:
        covered.update(path)

    lines = _header(
        f'HexProtocol: тестовые пути ({len(paths)} сценариев)',
        'P=в тестовом пути  ·=не задействован  (ян-цвет)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            sym = 'P' if h in covered else '·'
            if color:
                c = _YANG_ANSI[k] if h in covered else _UNREACH_COLOR
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Сценариев (all_transitions): {len(paths)}')
    lines.append(f'  Уникальных вершин в путях  : {len(covered)}')
    lines.append('')
    for i, path in enumerate(paths, 1):
        fmt = format_path(_SPEC, path)
        lines.append(f'  Путь {i}: длина={len(path) - 1}')
        for ln in fmt.splitlines():
            lines.append(f'    {ln}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='spec_glyphs',
        description='Q6 глифы: верификация КА HexProtocol.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('reach', help='достижимые состояния из INIT')
    sub.add_parser('cover', help='покрытие переходов (used/possible)')
    sub.add_parser('dead',  help='тупиковые состояния')
    sub.add_parser('paths', help='тестовые сценарии (все переходы)')
    args = p.parse_args(argv)
    color = not args.no_color

    dispatch = {
        'reach': render_reach,
        'cover': render_cover,
        'dead':  render_dead,
        'paths': render_paths,
    }
    if args.cmd in dispatch:
        print(dispatch[args.cmd](color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
