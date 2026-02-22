"""solan_ca.py — Клеточный автомат на алфавите Q6/Solan.

Каждая клетка = вершина Q6-гиперкуба (h ∈ 0..63).
Состояние отображается символом шрифта Solan/Stargate.

Правила перехода:
  xor   — next[i] = left XOR right   (операция группы (Z₂)⁶, фрактальная)
  xor3  — next[i] = left XOR self XOR right
  and   — next[i] = left AND right    (разрушительное)
  or    — next[i] = left OR  right    (заполняющее)
  hamming — next[i] = h ближайшего соседа по расстоянию Хэмминга

Начальные условия:
  center   — ноль везде, h=63 (⊠) в центре
  edge     — ноль везде, h=63 по левому краю
  random   — случайные Q6-состояния
  phonetic WORD — фонетическая кодировка русского слова (16-буквенный алфавит)

Режимы рендеринга:
  char    — 1 Solan-символ на клетку (компактно, 1 линия/шаг)
  braille — Braille U+2800 (4 chars × 2 линии на клетку)
  quad    — Quadrant-block ▀▄▌▐ (4 chars × 4 линии на клетку)

Запуск:
    python3 -m projects.hexglyph.solan_ca
    python3 -m projects.hexglyph.solan_ca --rule xor3 --steps 30 --width 48
    python3 -m projects.hexglyph.solan_ca --mode braille --width 16
    python3 -m projects.hexglyph.solan_ca --mode quad   --width 12
    python3 -m projects.hexglyph.solan_ca --ic phonetic --word РАТОН
    python3 -m projects.hexglyph.solan_ca --ic random --seed 42
    python3 -m projects.hexglyph.solan_ca --no-color
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.hexglyph import (
    h_to_char,
    render_braille,
    render_quad,
    RUSSIAN_PHONETIC,
    PHONETIC_H_TO_RU,
)
from projects.hexglyph.solan_phonetic import (
    _solan_char,
    phonetic_h,
    _SOLAN_MAP,
)

# ── ANSI ───────────────────────────────────────────────────────────────────

_RST  = '\033[0m'
_BOLD = '\033[1m'
_DIM  = '\033[2m'

# Цвет по Хэмминг-весу клетки (количество активных сегментов)
_W_COLOR = {
    0: '\033[38;5;236m',   # h=0  — пустой, очень тёмный
    1: '\033[38;5;24m',    # w=1  — тёмно-синий
    2: '\033[38;5;27m',    # w=2
    3: '\033[38;5;39m',    # w=3
    4: '\033[38;5;75m',    # w=4  — голубой
    5: '\033[38;5;117m',   # w=5
    6: '\033[38;5;231m',   # w=6  — ярко-белый (все сегменты)
}

# Особый цвет для фонетических вершин (русская буква)
_RU_COLOR = '\033[38;5;220m'   # золотой

_H_SET_RU = set(PHONETIC_H_TO_RU.keys())


def _hamming_weight(h: int) -> int:
    return bin(h).count('1')


def _cell_color(h: int) -> str:
    if h in _H_SET_RU:
        return _RU_COLOR
    return _W_COLOR[_hamming_weight(h)]


# ── Правила перехода ────────────────────────────────────────────────────────

def step(cells: list[int], rule: str = 'xor') -> list[int]:
    """Один шаг CA: вернуть новый список состояний.

    Граничные условия: тороидальные (периодические).
    """
    n = len(cells)
    if rule == 'xor':
        return [cells[(i - 1) % n] ^ cells[(i + 1) % n] for i in range(n)]
    if rule == 'xor3':
        return [cells[(i - 1) % n] ^ cells[i] ^ cells[(i + 1) % n]
                for i in range(n)]
    if rule == 'and':
        return [cells[(i - 1) % n] & cells[(i + 1) % n] for i in range(n)]
    if rule == 'or':
        return [cells[(i - 1) % n] | cells[(i + 1) % n] for i in range(n)]
    if rule == 'hamming':
        # next[i] = сосед с бо́льшим Хэмминг-весом
        nxt = []
        for i in range(n):
            left  = cells[(i - 1) % n]
            right = cells[(i + 1) % n]
            nxt.append(left if _hamming_weight(left) >= _hamming_weight(right)
                        else right)
        return nxt
    raise ValueError(f"Неизвестное правило: {rule!r}")


# ── Начальные условия ───────────────────────────────────────────────────────

def make_initial(width: int, ic: str = 'center', *,
                 word: str = '', seed: int | None = None) -> list[int]:
    """Сформировать начальное состояние CA.

    ic: 'center'   — h=63 в центре, остальные 0
        'edge'     — h=63 слева, остальные 0
        'random'   — случайные Q6-состояния
        'phonetic' — кодировка ``word`` (русские буквы → h-значения),
                     циклически повторяется до ширины width
    """
    if ic == 'center':
        cells = [0] * width
        cells[width // 2] = 63
        return cells
    if ic == 'edge':
        cells = [0] * width
        cells[0] = 63
        return cells
    if ic == 'random':
        rng = random.Random(seed)
        return [rng.randrange(64) for _ in range(width)]
    if ic == 'phonetic':
        encoded = []
        for ch in (word or 'РАТОН').upper():
            h = phonetic_h(ch)
            if h is not None:
                encoded.append(h)
        if not encoded:
            encoded = [63]
        # Циклически заполнить до width
        cells = [(encoded[i % len(encoded)]) for i in range(width)]
        return cells
    raise ValueError(f"Неизвестные начальные условия: {ic!r}")


# ── Рендеринг строки ────────────────────────────────────────────────────────

def render_row_char(cells: list[int], color: bool = True) -> str:
    """1 строка: каждая клетка → 1 Solan-символ."""
    parts = []
    for h in cells:
        ch = _solan_char(h)
        if color:
            parts.append(_cell_color(h) + ch + _RST)
        else:
            parts.append(ch)
    return ''.join(parts)


def render_row_braille(cells: list[int], color: bool = True) -> list[str]:
    """2 строки: каждая клетка → 4 Braille-символа (2×4 pix/char)."""
    lines = ['', '']
    for h in cells:
        ch = _solan_char(h)
        try:
            brl = render_braille(ch)
        except KeyError:
            brl = ['⠀⠀⠀⠀', '⠀⠀⠀⠀']
        if color:
            col = _cell_color(h)
            lines[0] += col + brl[0] + _RST
            lines[1] += col + brl[1] + _RST
        else:
            lines[0] += brl[0]
            lines[1] += brl[1]
    return lines


def render_row_quad(cells: list[int], color: bool = True) -> list[str]:
    """4 строки: каждая клетка → 4 quadrant-block символа (4×4 pix/char)."""
    lines = ['', '', '', '']
    for h in cells:
        ch = _solan_char(h)
        try:
            qd = render_quad(ch)
        except KeyError:
            qd = ['    '] * 4
        if color:
            col = _cell_color(h)
            for i in range(4):
                lines[i] += col + qd[i] + _RST
        else:
            for i in range(4):
                lines[i] += qd[i]
    return lines


# ── Легенда ─────────────────────────────────────────────────────────────────

def _legend(color: bool = True) -> str:
    """Короткая легенда цветов."""
    if not color:
        return ''
    parts = [f"{_BOLD}Легенда:{_RST}  "]
    for w in range(7):
        col = _W_COLOR[w]
        ch  = _solan_char(w)   # w → первое h с таким весом
        parts.append(f"{col}w={w}{_RST}")
        if w < 6:
            parts.append('  ')
    parts.append(f"  {_RU_COLOR}●{_RST}=рус.буква")
    return ''.join(parts)


# ── Анализ орбит ────────────────────────────────────────────────────────────

def find_orbit(
    cells: list[int],
    rule:  str = 'xor',
    max_steps: int = 5000,
) -> tuple[int | None, int | None]:
    """Найти транзиент и период орбиты для данного начального состояния.

    Возвращает ``(transient, period)`` или ``(None, None)``,
    если цикл не обнаружен за ``max_steps`` шагов.

    Алгоритм: наивный hash-map — O(max_steps · width) по времени и памяти.
    Для линейных правил (xor, xor3) период не превышает 2^width − 1,
    поэтому max_steps=5000 достаточен для width ≤ 12.
    """
    seen: dict[tuple[int, ...], int] = {}
    cur = list(cells)
    for t in range(max_steps + 1):
        key = tuple(cur)
        if key in seen:
            return seen[key], t - seen[key]
        seen[key] = t
        cur = step(cur, rule)
    return None, None


def print_orbit(cells: list[int], rule: str = 'xor',
                color: bool = True) -> None:
    """Найти и красиво напечатать информацию об орбите."""
    transient, period = find_orbit(cells, rule)
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    if transient is None:
        print(f"{bold}Орбита не найдена{reset} (период > 5000)")
        return

    print(f"{bold}Транзиент:{reset} {transient}   "
          f"{bold}Период:{reset} {period}")

    # Показать цикл (не более 12 шагов)
    cur = list(cells)
    for _ in range(transient):
        cur = step(cur, rule)

    limit = min(period, 12)
    print(f"{dim}──── цикл ({period} {'шаг' if period == 1 else 'шага' if 2 <= period <= 4 else 'шагов'}) ────{reset}")
    for t in range(limit):
        prefix = f"t={transient + t:>4d} │ "
        print(prefix + render_row_char(cur, color))
        cur = step(cur, rule)
    if period > limit:
        print(f"  {dim}... ещё {period - limit} шагов{reset}")
    print(f"{dim}{'─' * 20}{reset}")


# ── Основной прогон CA ──────────────────────────────────────────────────────

def run_ca(
    width:  int  = 40,
    steps:  int  = 20,
    rule:   str  = 'xor',
    ic:     str  = 'center',
    mode:   str  = 'char',
    color:  bool = True,
    word:   str  = '',
    seed:   int | None = None,
) -> None:
    """Запустить CA и напечатать эволюцию в stdout."""

    cells = make_initial(width, ic, word=word, seed=seed)

    sep = '─' * (width if mode == 'char' else width * 4)

    title = (f"  Q6-автомат  rule={rule}  ic={ic}  "
             f"width={width}  steps={steps}  mode={mode}")
    if color:
        print(_BOLD + title + _RST)
    else:
        print(title)
    print(sep)
    if color:
        print(_legend(color))
        print()

    for t in range(steps + 1):
        prefix = f"{t:3d} │ " if mode == 'char' else ''
        if mode == 'char':
            print(prefix + render_row_char(cells, color))
        elif mode == 'braille':
            rows = render_row_braille(cells, color)
            for r in rows:
                print(r)
        elif mode == 'quad':
            rows = render_row_quad(cells, color)
            for r in rows:
                print(r)
        cells = step(cells, rule)

    print(sep)


# ── Сравнение правил ────────────────────────────────────────────────────────

_ALL_RULES  = ['xor', 'xor3', 'and', 'or']
_RULE_NAMES = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}
_RULE_COLOR = {
    'xor':  '\033[38;5;75m',
    'xor3': '\033[38;5;117m',
    'and':  '\033[38;5;196m',
    'or':   '\033[38;5;220m',
}


def run_compare(
    width:  int  = 32,
    steps:  int  = 15,
    ic:     str  = 'center',
    mode:   str  = 'char',
    color:  bool = True,
    word:   str  = '',
    seed:   int | None = None,
) -> None:
    """Запустить CA для всех 4 правил и вывести последовательно."""
    cells0 = make_initial(width, ic, word=word, seed=seed)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    sep   = '─' * (width if mode == 'char' else width * 4)

    title = (f"  Q6-CA сравнение правил  ic={ic}  "
             f"width={width}  steps={steps}  mode={mode}")
    print(bold + title + reset)

    for rule in _ALL_RULES:
        rule_col = (_RULE_COLOR[rule] if color else '')
        print(f"\n{rule_col}{bold}{'─'*6} {_RULE_NAMES[rule]} {'─'*6}{reset}")
        cells = list(cells0)

        for t in range(steps + 1):
            prefix = f"{t:3d} │ " if mode == 'char' else ''
            if mode == 'char':
                print(prefix + render_row_char(cells, color))
            elif mode == 'braille':
                for row in render_row_braille(cells, color):
                    print(row)
            elif mode == 'quad':
                for row in render_row_quad(cells, color):
                    print(row)
            cells = step(cells, rule)

    print('\n' + sep)


# ── Анимация ─────────────────────────────────────────────────────────────────

def run_animate(
    width:  int   = 40,
    rule:   str   = 'xor',
    ic:     str   = 'center',
    delay:  float = 0.12,
    rows:   int   = 16,
    color:  bool  = True,
    word:   str   = '',
    seed:   int | None = None,
) -> None:
    """Анимация CA: скользящее окно из ``rows`` последних шагов.

    Обновляет экран на месте с помощью ANSI escape-кодов.
    Прерывается по Ctrl+C.
    """
    import time

    cells = make_initial(width, ic, word=word, seed=seed)

    # Предзаполнение истории
    hist: list[list[int]] = [list(cells)]
    for _ in range(rows - 1):
        cells = step(cells, rule)
        hist.append(list(cells))

    _HIDE = '\033[?25l'   # скрыть курсор
    _SHOW = '\033[?25h'   # показать курсор

    def _up(n: int) -> str:
        return f'\033[{n}A'

    def _draw(t_base: int) -> None:
        for i, row in enumerate(hist):
            t = t_base + i
            print(f'\033[2K\r{t:4d} │ ' + render_row_char(row, color))
        rule_col = _RULE_COLOR.get(rule, '')
        rc = rule_col if color else ''
        rs = _RST if color else ''
        di = _DIM if color else ''
        status = f"rule={rc}{_RULE_NAMES.get(rule, rule)}{rs}  width={width}  delay={delay:.2f}s  (Ctrl+C — стоп)"
        print(f'\033[2K\r  {di}{status}{rs}', end='', flush=True)

    sys.stdout.write(_HIDE)
    sys.stdout.flush()

    t = rows - 1
    _draw(0)

    try:
        while True:
            time.sleep(delay)
            cells = step(hist[-1], rule)
            hist.append(list(cells))
            if len(hist) > rows:
                hist.pop(0)
            t += 1
            sys.stdout.write(_up(rows + 1))
            sys.stdout.flush()
            _draw(t - rows + 1)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(_SHOW)
        sys.stdout.flush()
        print()


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Q6 Клеточный автомат — алфавит Solan/Stargate')
    parser.add_argument('--rule', choices=['xor', 'xor3', 'and', 'or', 'hamming'],
                        default='xor',
                        help='правило перехода (default: xor)')
    parser.add_argument('--ic', choices=['center', 'edge', 'random', 'phonetic'],
                        default='center',
                        help='начальные условия (default: center)')
    parser.add_argument('--word', default='РАТОН',
                        help='русское слово для режима --ic phonetic')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed для --ic random')
    parser.add_argument('--width', type=int, default=40,
                        help='ширина CA (число клеток, default: 40)')
    parser.add_argument('--steps', type=int, default=20,
                        help='число шагов эволюции (default: 20)')
    parser.add_argument('--mode', choices=['char', 'braille', 'quad'],
                        default='char',
                        help='режим рендеринга (default: char)')
    parser.add_argument('--no-color', action='store_true',
                        help='без ANSI-цветов')
    parser.add_argument('--orbit', action='store_true',
                        help='найти транзиент и период орбиты, показать цикл')
    parser.add_argument('--compare', action='store_true',
                        help='показать эволюцию для всех 4 правил подряд')
    parser.add_argument('--animate', action='store_true',
                        help='анимация: скользящее окно в реальном времени')
    parser.add_argument('--delay', type=float, default=0.12,
                        help='задержка между шагами анимации, сек (default: 0.12)')
    parser.add_argument('--rows', type=int, default=16,
                        help='число отображаемых строк в анимации (default: 16)')
    args = parser.parse_args()

    _color = not args.no_color
    _cells = make_initial(args.width, args.ic, word=args.word, seed=args.seed)

    if args.orbit:
        print_orbit(_cells, rule=args.rule, color=_color)
    elif args.compare:
        run_compare(
            width=args.width,
            steps=args.steps,
            ic=args.ic,
            mode=args.mode,
            color=_color,
            word=args.word,
            seed=args.seed,
        )
    elif args.animate:
        run_animate(
            width=args.width,
            rule=args.rule,
            ic=args.ic,
            delay=args.delay,
            rows=args.rows,
            color=_color,
            word=args.word,
            seed=args.seed,
        )
    else:
        run_ca(
            width=args.width,
            steps=args.steps,
            rule=args.rule,
            ic=args.ic,
            mode=args.mode,
            color=_color,
            word=args.word,
            seed=args.seed,
        )
