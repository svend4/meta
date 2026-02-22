"""solan_matrix.py — Матрица орбитальных расстояний Q6-лексикона.

Вычисляет полную N×N матрицу попарных орбитальных расстояний
для слов лексикона (или произвольного набора слов).

Возможности:
  • distance_matrix() → dict[(w1, w2): float]   (симметрична, диагональ=0)
  • nearest_pairs()  / farthest_pairs()          (топ похожих / непохожих пар)
  • print_heatmap()  — цветная матрица в терминале (1 блок = 1 ячейка)
  • print_nearest_pairs()  — список ближайших пар с барами
  • export_csv()     → строка CSV

Запуск:
    python3 -m projects.hexglyph.solan_matrix
    python3 -m projects.hexglyph.solan_matrix --pairs
    python3 -m projects.hexglyph.solan_matrix --csv > matrix.csv
    python3 -m projects.hexglyph.solan_matrix --words ГОРА ВОДА НОРА РАТОН
    python3 -m projects.hexglyph.solan_matrix --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_lexicon import LEXICON, all_signatures
from projects.hexglyph.solan_word import sig_distance
from projects.hexglyph.solan_ca import _RST, _BOLD, _DIM

# ── Матрица ──────────────────────────────────────────────────────────────────

def distance_matrix(
    words: list[str] | None = None,
    width: int = 16,
) -> dict[tuple[str, str], float]:
    """Полная N×N матрица попарных орбитальных расстояний.

    Ключ (w1, w2) — упорядоченная пара; матрица симметрична.
    Диагональные элементы равны 0.0.
    Возвращает float ('nan' если орбита не определена).
    """
    if words is None:
        words = LEXICON
    sigs = all_signatures(words=words, width=width)
    mat: dict[tuple[str, str], float] = {}
    for w1 in words:
        for w2 in words:
            if (w1, w2) not in mat:
                d = 0.0 if w1 == w2 else sig_distance(sigs[w1], sigs[w2])
                mat[(w1, w2)] = d
                mat[(w2, w1)] = d
    return mat


def nearest_pairs(
    mat:   dict[tuple[str, str], float],
    n:     int = 10,
) -> list[tuple[str, str, float]]:
    """Топ-N ближайших пар (w1 ≠ w2, без дублей)."""
    seen:  set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, float]] = []
    for (w1, w2), d in mat.items():
        if w1 == w2:
            continue
        key = (min(w1, w2), max(w1, w2))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((w1, w2, d))
    pairs.sort(key=lambda x: (float('inf') if x[2] != x[2] else x[2]))
    return pairs[:n]


def farthest_pairs(
    mat: dict[tuple[str, str], float],
    n:   int = 5,
) -> list[tuple[str, str, float]]:
    """Топ-N наиболее удалённых пар (w1 ≠ w2, без NaN)."""
    seen:  set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, float]] = []
    for (w1, w2), d in mat.items():
        if w1 == w2 or d != d:  # skip diagonal and NaN
            continue
        key = (min(w1, w2), max(w1, w2))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((w1, w2, d))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:n]


# ── ANSI-цветовой градиент ────────────────────────────────────────────────────

def _dist_ansi(d: float) -> str:
    """ANSI 256-colour: зелёный(0) → жёлтый(0.5) → красный(1.0), серый(NaN)."""
    if d != d:                    # NaN
        return '\033[38;5;238m'
    d = max(0.0, min(1.0, d))
    if d < 0.5:
        t = d * 2               # 0..1  (green→yellow)
        r = round(t * 5)
        g = 5
    else:
        t = (d - 0.5) * 2       # 0..1  (yellow→red)
        r = 5
        g = round((1.0 - t) * 5)
    idx = 16 + 36 * r + 6 * g   # b=0
    return f'\033[38;5;{idx}m'


def _dist_block(d: float) -> str:
    """Один символ-ячейка по значению расстояния."""
    if d != d:
        return '?'
    if d < 0.001:
        return '■'
    if d < 0.25:
        return '▓'
    if d < 0.50:
        return '▒'
    if d < 0.75:
        return '░'
    return '·'


# ── Вывод ────────────────────────────────────────────────────────────────────

def print_heatmap(
    words: list[str] | None = None,
    width: int  = 16,
    color: bool = True,
) -> None:
    """Цветная матрица расстояний в терминале.

    Каждая ячейка — 1 цветной блок (■▓▒░·).
    Строки = слова, столбцы = слова (3-символьные сокращения).
    """
    if words is None:
        words = LEXICON
    mat   = distance_matrix(words, width=width)
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    n     = len(words)
    abbr  = [w[:4].ljust(4) for w in words]

    # Заголовок
    print(bold + f"  Матрица орбитальных расстояний Q6"
          f"  ({n}×{n})  width={width}" + reset)
    print()

    # Шапка столбцов (каждый 1 символ + пробел для выравнивания)
    # Выводим 4 строки шапки (буквы аббревиатур по вертикали)
    row_label_w = 8
    for li in range(4):
        head = ' ' * row_label_w
        for ab in abbr:
            head += (dim + ab[li] + reset) if color else ab[li]
        print('  ' + head)
    print()

    # Строки матрицы
    for i, w in enumerate(words):
        row_lbl = f"{w[:7]:7s} "
        line = (bold + row_lbl + reset) if color else row_lbl
        for j, w2 in enumerate(words):
            d   = mat[(w, w2)]
            blk = _dist_block(d)
            if i == j:
                line += (dim + '·' + reset) if color else '·'
            else:
                line += ((_dist_ansi(d) if color else '') + blk + reset)
        print('  ' + line)

    print()
    # Легенда
    legend_items = [
        (0.0,  '■', 'идентичные орбиты (d=0)'),
        (0.15, '▓', 'd<0.25'),
        (0.40, '▒', '0.25≤d<0.50'),
        (0.65, '░', '0.50≤d<0.75'),
        (0.90, '·', 'd≥0.75'),
    ]
    parts = []
    for dv, blk, lbl in legend_items:
        col = (_dist_ansi(dv) if color else '')
        parts.append(f"{col}{blk}{reset} {dim}{lbl}{reset}")
    print('  ' + '  '.join(parts))


def print_nearest_pairs(
    words: list[str] | None = None,
    n:     int  = 10,
    width: int  = 16,
    color: bool = True,
) -> None:
    """Список n ближайших и 5 наиболее удалённых пар."""
    if words is None:
        words = LEXICON
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    mat    = distance_matrix(words, width=width)
    near   = nearest_pairs(mat, n=n)
    far    = farthest_pairs(mat, n=5)

    max_d  = max((d for _, _, d in near + far if d == d), default=1.0) or 1.0

    print(bold + f"  {n} ближайших орбитальных пар  width={width}" + reset)
    print()
    for w1, w2, d in near:
        bar_len = 20
        filled  = int(d / max_d * bar_len) if d == d else bar_len
        bar  = '█' * filled + '░' * (bar_len - filled)
        col  = _dist_ansi(d) if color else ''
        d_s  = f"{d:.3f}" if d == d else "  NaN"
        eq   = ' ≡' if d < 0.001 else ''
        print(f"  {w1:10s} ↔ {w2:10s}  "
              f"{col}|{bar}|{reset} {col}{d_s}{reset}{dim}{eq}{reset}")

    print()
    print(bold + f"  5 наиболее удалённых пар:" + reset)
    print()
    for w1, w2, d in far:
        bar_len = 20
        filled  = int(d / max_d * bar_len) if d == d else 0
        bar  = '█' * filled + '░' * (bar_len - filled)
        col  = _dist_ansi(d) if color else ''
        d_s  = f"{d:.3f}"
        print(f"  {w1:10s} ↔ {w2:10s}  "
              f"{col}|{bar}|{reset} {col}{d_s}{reset}")


def export_csv(
    words: list[str] | None = None,
    width: int = 16,
) -> str:
    """Экспортировать матрицу расстояний в формат CSV."""
    if words is None:
        words = LEXICON
    mat = distance_matrix(words, width=width)

    header = ',' + ','.join(words)
    rows   = [header]
    for w1 in words:
        vals = []
        for w2 in words:
            d = mat[(w1, w2)]
            vals.append('' if d != d else f"{d:.4f}")
        rows.append(f"{w1}," + ','.join(vals))
    return '\n'.join(rows)



# ── Сводка ──────────────────────────────────────────────────────────────────

def matrix_summary(
    words: list[str] | None = None,
    n:     int = 10,
    width: int = 16,
) -> dict:
    """JSON-friendly summary: nearest pairs in the orbital distance matrix."""
    _words = list(words) if words else list(LEXICON)
    mat    = distance_matrix(words=_words, width=width)
    pairs  = nearest_pairs(mat, n=n)
    return {
        'words':         _words,
        'n':             n,
        'width':         width,
        'nearest_pairs': [[w1, w2, round(d, 6)] for w1, w2, d in pairs],
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Матрица орбитальных расстояний Q6-лексикона')
    parser.add_argument('--words', nargs='+', metavar='WORD',
                        help='список слов (по умолчанию: весь лексикон)')
    parser.add_argument('--pairs', action='store_true',
                        help='показать ближайшие/дальние пары')
    parser.add_argument('--n', type=int, default=10,
                        help='число ближайших пар (default: 10)')
    parser.add_argument('--csv', action='store_true',
                        help='экспорт в CSV (stdout)')
    parser.add_argument('--width', type=int, default=16,
                        help='ширина CA (default: 16)')
    parser.add_argument('--no-color', action='store_true',
                        help='без ANSI-цветов')
    parser.add_argument('--json',     action='store_true',
                        help='JSON output')
    args = parser.parse_args()

    _words = args.words if args.words else None
    _color = not args.no_color

    if args.json:
        import json as _json
        print(_json.dumps(matrix_summary(_words, n=args.n, width=args.width), ensure_ascii=False, indent=2))
        import sys; sys.exit(0)
    if args.csv:
        print(export_csv(words=_words, width=args.width))
    elif args.pairs:
        print_nearest_pairs(words=_words, n=args.n,
                            width=args.width, color=_color)
    else:
        print_heatmap(words=_words, width=args.width, color=_color)
