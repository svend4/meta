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
    args = parser.parse_args()

    run_ca(
        width=args.width,
        steps=args.steps,
        rule=args.rule,
        ic=args.ic,
        mode=args.mode,
        color=not args.no_color,
        word=args.word,
        seed=args.seed,
    )
