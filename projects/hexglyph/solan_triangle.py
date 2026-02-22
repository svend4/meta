"""solan_triangle.py — треугольник Хассе B₆ с символами шрифта Solan.

Объединяет два визуальных слоя Q6:
  • hexvis  — теоретические ASCII-глифы (render_glyph)
  • hexglyph — растровые символы шрифта Solan 8×8

Для каждой из 64 вершин Q6 (h = 0..63) выводит:
  — символ Solan (4×4 компактный растр) если детектирован,
  — иначе — hexvis ASCII-глиф 3×3.

Расположение — треугольник Хассе: уровень 0 (вес 0) сверху, уровень 6 снизу.
Каждая ячейка = 4 строки × 4 символа + 2 разделителя = 6 строк высоты.

Запуск:
    python3 -m projects.hexglyph.solan_triangle [--hexvis] [--side-by-side]

Флаги:
    --hexvis        выводить только hexvis (без Solan)
    --side-by-side  рядом: hexvis | Solan
"""

from __future__ import annotations

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import render_glyph as hv_render_glyph
from projects.hexglyph.hexglyph import (
    glyph_bitmap, detect_segments, font_data, _CHARSET_64, char_to_h, h_to_char,
)

# ── ANSI цвета ─────────────────────────────────────────────────────────────

_RST = '\033[0m'
_YANG_ANSI = ['\033[38;5;240m', '\033[38;5;30m',  '\033[38;5;34m',
              '\033[38;5;220m', '\033[38;5;208m', '\033[38;5;160m',
              '\033[38;5;129m']
# Полутоновые версии тех же цветов (для последовательно назначенных вершин)
_YANG_MID  = ['\033[38;5;238m', '\033[38;5;23m',  '\033[38;5;22m',
              '\033[38;5;214m', '\033[38;5;202m', '\033[38;5;124m',
              '\033[38;5;93m']
_YANG_DIM  = ['\033[38;5;236m'] * 7   # dim fallback (hexvis)

# ── Построение таблицы h → Solan-символ ────────────────────────────────────

def _build_solan_map() -> dict[int, str]:
    """Для каждой Q6-вершины найти лучший символ Solan через detect_segments."""
    order = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789"
    h2ch: dict[int, str] = {}
    for ch in order:
        try:
            seg = detect_segments(ch)
            h = seg['h']
            if h not in h2ch:
                h2ch[h] = ch
        except KeyError:
            pass
    return h2ch


_SOLAN_MAP: dict[int, str] = _build_solan_map()


def _build_full_map() -> dict[int, str]:
    """Полный маппинг h → Solan-символ для всех 64 вершин Q6.

    Использует последовательное назначение char_to_h/h_to_char
    (упорядочивание по весу Хэмминга).  Все 64 вершины покрыты.
    """
    return {h: h_to_char(h) for h in range(64) if h_to_char(h) is not None}


# Полный маппинг: все 64 вершины (последовательное назначение)
_FULL_MAP: dict[int, str] = _build_full_map()


# ── Компактный 4×4 растр из 8×8 глифа ─────────────────────────────────────

def _compact4(rows: list[int]) -> list[str]:
    """Сжать 8×8 растр до 4×4: взять строки 0,2,4,6; колонки 0,2,4,6."""
    result = []
    for i in [0, 2, 4, 6]:
        v = rows[i]
        bits = format(v, '08b')
        result.append(bits[0] + bits[2] + bits[4] + bits[6])
    return result


def _solan_cell(ch: str, color: str) -> list[str]:
    """4 строки для Solan-символа ch (4×4 пикселей) с цветом."""
    try:
        rows = glyph_bitmap(ch)
        c4 = _compact4(rows)
        return [color + r.replace('1', '█').replace('0', '·') + _RST
                for r in c4]
    except KeyError:
        return [color + '···· ' + _RST] * 4


def _hexvis_cell(h: int, color: str) -> list[str]:
    """3 строки hexvis-глифа (3×3) + пустая строка, выровнено до 4."""
    g = hv_render_glyph(h)          # ['_  ', '| |', ' _ '] — 3 строки
    padded = [color + line.ljust(4) + _RST for line in g]
    padded.append(color + '    ' + _RST)
    return padded


# ── Треугольник Хассе ──────────────────────────────────────────────────────

def _rank_elements() -> list[list[int]]:
    ranks: list[list[int]] = [[] for _ in range(7)]
    for h in range(64):
        ranks[yang_count(h)].append(h)
    return ranks


def render_triangle(
    mode: str = 'solan',   # 'solan' | 'hexvis' | 'side'
    color: bool = True,
) -> str:
    """Вернуть строку с треугольником Хассе B₆.

    mode:
        'solan'  — символы Solan (детектированные) или hexvis как запасной вариант
        'hexvis' — только hexvis
        'side'   — hexvis слева, Solan справа для каждой вершины
    """
    ranks = _rank_elements()
    max_n = 20       # C(6,3)
    cell_w = 4 if mode in ('solan', 'hexvis') else 9
    sep = 1
    total_w = max_n * cell_w + (max_n - 1) * sep

    lines: list[str] = []

    # заголовок
    title = {
        'solan':  '  ◈ Треугольник Хассе B₆ — Solan (Старгейт)  ',
        'hexvis': '  ◈ Треугольник Хассе B₆ — hexvis             ',
        'side':   '  ◈ Треугольник Хассе B₆ — hexvis | Solan     ',
    }[mode]
    lines.append('\033[1m' + title + _RST if color else title)
    lines.append('')

    for k, elems in enumerate(ranks):
        ansi = _YANG_ANSI[k] if color else ''
        n = len(elems)
        used_w = n * cell_w + (n - 1) * sep
        pad = (total_w - used_w) // 2

        # Для каждого уровня нужно cell_h строк
        if mode == 'side':
            cell_h = 4
        else:
            cell_h = 4

        cells: list[list[str]] = []
        for h in elems:
            if mode == 'hexvis':
                cells.append(_hexvis_cell(h, ansi))
            elif mode == 'solan':
                if h in _SOLAN_MAP:
                    # Пиксельно подтверждённый — полная яркость
                    cells.append(_solan_cell(_SOLAN_MAP[h], ansi))
                elif h in _FULL_MAP:
                    # Последовательное назначение — полутон
                    mid = _YANG_MID[k] if color else ''
                    cells.append(_solan_cell(_FULL_MAP[h], mid))
                else:
                    dim = _YANG_DIM[k] if color else ''
                    cells.append(_hexvis_cell(h, dim))
            else:  # side
                hv = _hexvis_cell(h, ansi)
                ch = _SOLAN_MAP.get(h) or _FULL_MAP.get(h)
                sl = (_solan_cell(ch, ansi)
                      if ch is not None
                      else [ansi + '····' + _RST] * 4)
                # side-by-side: "hv|sl" — 3+1+4 = 8 chars
                cells.append([hv[i][:3] + '│' + sl[i][:4]
                               for i in range(cell_h)])

        # Собрать строки
        for row_i in range(cell_h):
            row_parts = [' ' * pad]
            for ci, cell in enumerate(cells):
                row_parts.append(cell[row_i])
                if ci < len(cells) - 1:
                    row_parts.append(' ' * sep)
            lines.append(''.join(row_parts))

        # Разделитель уровней
        weight_label = f' w={k} ({n}) '
        lines.append(' ' * 2 + ('\033[38;5;236m' if color else '') +
                     weight_label + _RST if color else weight_label)

    return '\n'.join(lines)


def print_triangle(mode: str = 'solan', color: bool = True) -> None:
    """Напечатать треугольник в stdout."""
    print(render_triangle(mode=mode, color=color))


# ── Статистика детектирования ──────────────────────────────────────────────

def detection_stats() -> dict:
    """Сводка: сколько вершин Q6 детектировано / назначено из шрифта.

    'detected' — пиксельно подтверждённые (из _SOLAN_MAP, 24/64).
    'assigned'  — последовательно назначенные (из _FULL_MAP, 64/64).
    """
    ranks = _rank_elements()
    detected = set(_SOLAN_MAP.keys())
    assigned = set(_FULL_MAP.keys())
    stats = {
        'total':          64,
        'detected':       len(detected),
        'assigned':       len(assigned),
        'missing':        64 - len(detected),
        'detected_list':  sorted(detected),
        'by_rank':        {},
    }
    for k, elems in enumerate(ranks):
        d = [h for h in elems if h in detected]
        a = [h for h in elems if h in assigned and h not in detected]
        stats['by_rank'][k] = {
            'total':    len(elems),
            'detected': len(d),
            'assigned': len(a),
            'chars':    [_SOLAN_MAP[h] for h in d],
            'chars_seq': [_FULL_MAP[h] for h in a],
        }
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Треугольник Хассе B₆ с символами шрифта Solan / hexvis')
    parser.add_argument('--hexvis',       action='store_true',
                        help='только hexvis-глифы')
    parser.add_argument('--side-by-side', action='store_true',
                        help='hexvis | Solan рядом')
    parser.add_argument('--no-color',     action='store_true',
                        help='без ANSI-цветов')
    parser.add_argument('--stats',        action='store_true',
                        help='показать статистику детектирования')
    parser.add_argument('--json',         action='store_true',
                        help='JSON output')
    args = parser.parse_args()

    color = not args.no_color

    if args.json:
        import json as _json
        print(_json.dumps(detection_stats(), ensure_ascii=False, indent=2))
        import sys; sys.exit(0)
    if args.stats:
        st = detection_stats()
        print(f"Пиксельно подтверждено:  {st['detected']}/{st['total']}")
        print(f"Последовательно назначено: {st['assigned']}/{st['total']}")
        print()
        for k, info in st['by_rank'].items():
            bar = ('█' * info['detected'] +
                   '▒' * info['assigned'] +
                   '·' * (info['total'] - info['detected'] - info['assigned']))
            chars_conf = ''.join(info['chars']) or '—'
            chars_seq  = ''.join(info['chars_seq'])
            line = f"  Вес {k} ({info['total']:2d}): [{bar}]  ██={chars_conf}"
            if chars_seq:
                line += f"  ▒▒={chars_seq}"
            print(line)
    else:
        if args.hexvis:
            mode = 'hexvis'
        elif args.side_by_side:
            mode = 'side'
        else:
            mode = 'solan'
        print_triangle(mode=mode, color=color)
