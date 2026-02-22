"""solan_phonetic.py — Русская фонетическая таблица шрифта Solan.

«Алфавит 16 букв / 36 звуков» — авторская система Энгеля (~2000-е гг.):
каждой из 16 базовых букв русского языка сопоставлен Q6-символ (Solan/Старгейт).

Модуль предоставляет:
  • render_phonetic_table()  — цветная таблица в терминале с 4×4 глифами
  • transliterate(text)      — транслитерация рус. текста символами Solan
  • phonetic_h(letter)       — Q6-индекс для русской буквы

Запуск:
    python3 -m projects.hexglyph.solan_phonetic
    python3 -m projects.hexglyph.solan_phonetic --encode "Привет"
    python3 -m projects.hexglyph.solan_phonetic --no-color
"""

from __future__ import annotations

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.hexglyph import (
    glyph_bitmap, RUSSIAN_PHONETIC, PHONETIC_H_TO_RU,
    render_quad, render_braille,
)
from projects.hexglyph.solan_triangle import (
    _SOLAN_MAP, _FULL_MAP, _compact4,
)

# ── ANSI цвета ─────────────────────────────────────────────────────────────

_RST   = '\033[0m'
_BOLD  = '\033[1m'
_DIM   = '\033[2m'

# Уровни уверенности → цвет
_CONF_COLOR = {
    'high':   '\033[38;5;120m',   # ярко-зелёный
    'medium': '\033[38;5;220m',   # жёлтый
    'low':    '\033[38;5;208m',   # оранжевый
}
_CONF_MARK = {
    'high':   '●●●',
    'medium': '●●○',
    'low':    '●○○',
}

# Цвет символа Solan
_SOLAN_COLOR = '\033[38;5;75m'    # голубой


# ── Вспомогательные функции ─────────────────────────────────────────────────

def _solan_char(h: int) -> str:
    """Вернуть Solan-символ для вершины h (подтверждённый или последовательный)."""
    return _SOLAN_MAP.get(h) or _FULL_MAP.get(h, '?')


def _glyph4(h: int) -> list[str]:
    """4 строки 4×4 пиксельного глифа для вершины h (режим ascii)."""
    ch = _solan_char(h)
    try:
        rows = glyph_bitmap(ch)
        c4   = _compact4(rows)
        return [r.replace('1', '█').replace('0', '·') for r in c4]
    except KeyError:
        return ['····'] * 4


def _glyph_quad(h: int) -> list[str]:
    """4 строки 4×4 quadrant-block рендер глифа (техника TVCP из svend4/infon)."""
    ch = _solan_char(h)
    try:
        return render_quad(ch)
    except KeyError:
        return ['    '] * 4


def _glyph_braille(h: int) -> list[str]:
    """2 строки 4×1 Braille-рендер глифа (2×4 → 1 char, U+2800-U+28FF)."""
    ch = _solan_char(h)
    try:
        return render_braille(ch)
    except KeyError:
        return ['⠀⠀⠀⠀'] * 2


def phonetic_h(letter: str) -> int | None:
    """Q6-индекс для русской буквы (из RUSSIAN_PHONETIC). None если неизвестно."""
    entry = RUSSIAN_PHONETIC.get(letter)
    return entry['h'] if entry else None


def transliterate(text: str) -> str:
    """Транслитерировать текст: русские буквы → символы Solan.

    Буквы, не вошедшие в таблицу, сохраняются как есть.
    Регистр не учитывается (всё приводится к верхнему).
    """
    result = []
    for ch in text.upper():
        h = phonetic_h(ch)
        if h is not None:
            result.append(_solan_char(h))
        else:
            result.append(ch)
    return ''.join(result)


# ── Таблица в терминале ─────────────────────────────────────────────────────

def _render_row(
    ru: str,
    entry: dict,
    color: bool = True,
    render_mode: str = 'ascii',
) -> list[str]:
    """Отрендерить одну строку таблицы.

    render_mode: 'ascii'   — 4×4 пикс. (█/·), 4 строки высоты
                 'quad'    — quadrant-block TVCP (▀▄▌▐…), 4 строки × 4 chars
                 'braille' — Braille Unicode, 2 строки × 4 chars
    """
    h    = entry['h']
    ch   = _solan_char(h)
    conf = entry['confidence']
    segs = entry['segs']
    desc = entry['desc']
    confirmed = h in _SOLAN_MAP

    if render_mode == 'quad':
        g = _glyph_quad(h)
    elif render_mode == 'braille':
        g = _glyph_braille(h)
    else:
        g = _glyph4(h)

    if color:
        cc   = _CONF_COLOR[conf]
        sc   = _SOLAN_COLOR
        dim  = _DIM if not confirmed else ''
        rst  = _RST
        bold = _BOLD
    else:
        cc = sc = dim = rst = bold = ''

    src_marker = '●' if confirmed else '○'
    mark = _CONF_MARK[conf]

    if render_mode == 'braille':
        # 2 строки высоты
        line0 = (f"  {bold}{cc}{ru:2s}{rst}  "
                 f"{dim}{sc}{g[0]}{rst}"
                 f"  h={h:2d}  {cc}{segs:<18s}{rst}  {cc}{mark}{rst}")
        line1 = (f"      "
                 f"{dim}{sc}{g[1]}{rst}"
                 f"  chr={sc}{dim}{ch}{rst}  {src_marker}  "
                 f"{cc}{conf:<6s}{rst}  {cc}{desc}{rst}")
        return [line0, line1]

    # ascii / quad — 4 строки высоты
    line0 = (f"  {bold}{cc}{ru:2s}{rst}   "
             f"{dim}{sc}{g[0]}{rst}"
             f"   h={h:2d}  {cc}{segs:<18s}{rst}"
             f"  {cc}{mark}{rst}")
    line1 = (f"       "
             f"{dim}{sc}{g[1]}{rst}"
             f"   chr={sc}{dim}{ch}{rst}  {src_marker}")
    line2 = (f"       "
             f"{dim}{sc}{g[2]}{rst}"
             f"   {cc}{conf:<6s}{rst}  {cc}{desc}{rst}")
    line3 = (f"       "
             f"{dim}{sc}{g[3]}{rst}")

    return [line0, line1, line2, line3]


def render_phonetic_table(color: bool = True, render_mode: str = 'ascii') -> str:
    """Вернуть строку с полной фонетической таблицей Solan.

    render_mode: 'ascii'   — 4×4 пиксельных глифа (█/·)
                 'quad'    — quadrant-block TVCP (▀▄▌▐…)  [техника svend4/infon]
                 'braille' — Braille Unicode (U+2800-U+28FF), максимально компактно
    """
    lines: list[str] = []

    mode_label = {'ascii': '4×4 █/·', 'quad': '4×4 quadrant ▀▄', 'braille': '2×4 Braille ⠿'}
    title = (f'  ◈ Алфавит «Старгейт» — Русская фонетика (Энгель, ~2000-е)'
             f'  [{mode_label.get(render_mode, render_mode)}]')
    sep   = '  ' + '─' * 58

    if color:
        lines.append(_BOLD + title + _RST)
    else:
        lines.append(title)

    lines.append(sep)

    hdr = (f"  {'Бук':4s}  {'Глиф':4s}   {'h':>2s}  {'Сегменты':<18s}  "
           f"{'Уверен.':7s}  Описание")
    lines.append(('' if not color else '\033[38;5;244m') + hdr +
                 (_RST if color else ''))
    lines.append(sep)

    for ru, entry in RUSSIAN_PHONETIC.items():
        rows = _render_row(ru, entry, color, render_mode=render_mode)
        lines.extend(rows)
        lines.append('')   # пустая строка между буквами

    lines.append(sep)
    total = len(RUSSIAN_PHONETIC)
    high  = sum(1 for v in RUSSIAN_PHONETIC.values() if v['confidence'] == 'high')
    med   = sum(1 for v in RUSSIAN_PHONETIC.values() if v['confidence'] == 'medium')
    low_  = sum(1 for v in RUSSIAN_PHONETIC.values() if v['confidence'] == 'low')
    footer = (f"  Букв задано: {total}/16  "
              f"[high={high}  medium={med}  low={low_}]  "
              f"Оставшиеся буквы: {16-total}")
    lines.append(('\033[38;5;240m' if color else '') + footer + (_RST if color else ''))

    return '\n'.join(lines)


def print_phonetic_table(color: bool = True, render_mode: str = 'ascii') -> None:
    """Напечатать фонетическую таблицу в stdout."""
    print(render_phonetic_table(color=color, render_mode=render_mode))


# ── Кодировщик фонетики ─────────────────────────────────────────────────────

def encode_phonetic(text: str) -> list[tuple[str, int | None, str]]:
    """Покодировать текст посимвольно.

    Возвращает список (исходный_символ, h, solan_char):
      h = None и solan_char = исходный_символ — для неизвестных.
    """
    result = []
    for ch in text.upper():
        h = phonetic_h(ch)
        sc = _solan_char(h) if h is not None else ch
        result.append((ch, h, sc))
    return result



# ── Сводка ──────────────────────────────────────────────────────────────────

def phonetic_summary(text: str = 'ГОРА') -> dict:
    """JSON-friendly phonetic encoding summary for *text*."""
    enc = encode_phonetic(text)
    return {
        'text':    text,
        'encoded': [[ch, h, sc] for ch, h, sc in enc],
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Русская фонетическая таблица шрифта Solan (Старгейт)')
    parser.add_argument('--encode',   metavar='TEXT',
                        help='транслитерировать текст в символы Solan')
    parser.add_argument('--no-color', action='store_true',
                        help='без ANSI-цветов')
    parser.add_argument('--mode', choices=['ascii', 'quad', 'braille'],
                        default='ascii',
                        help='режим отображения глифов: ascii (4×4 █/·), '
                             'quad (quadrant-block ▀▄, TVCP), '
                             'braille (2×4 Braille ⠿)')
    parser.add_argument('--json',     action='store_true',
                        help='JSON output')
    args = parser.parse_args()

    color = not args.no_color

    if args.json:
        import json as _json, sys
        _text = args.encode if args.encode else 'ГОРА'
        print(_json.dumps(phonetic_summary(_text), ensure_ascii=False, indent=2))
        sys.exit(0)
    if args.encode:
        encoded = encode_phonetic(args.encode)
        solan_str = ''.join(sc for _, _, sc in encoded)
        if color:
            print(f"\nИсходный текст:    {args.encode}")
            print(f"Solan-транслит.:   {_SOLAN_COLOR}{solan_str}{_RST}")
            print()
            for orig, h, sc in encoded:
                h_str = f"h={h:2d}" if h is not None else "н/д "
                known = ' ✓' if h is not None else '  '
                print(f"  {orig} → {_SOLAN_COLOR}{sc}{_RST}  {h_str}{known}")
        else:
            print(f"Исходный текст:  {args.encode}")
            print(f"Solan-транслит.: {solan_str}")
            for orig, h, sc in encoded:
                h_str = f"h={h:2d}" if h is not None else "н/д"
                print(f"  {orig} → {sc}  {h_str}")
    else:
        print_phonetic_table(color=color, render_mode=args.mode)
