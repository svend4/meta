"""solan_word.py — Семантический анализ слов через Q6-автомат.

Русское слово → фонетическая кодировка Q6 → начальные условия CA →
орбитальная сигнатура (транзиент/период для 4 правил) → сравнение слов.

Конвейер:
    слово "РАТОН"
    → h-ценности [15, 63, 48, 47, 3]    (phonetic_h)
    → Solan-символы «РАТОН» в алфавите   (h_to_char)
    → CA width=16: циклически расширить  (pad_to)
    → orbit(xor) → (T=0, P=4)
    → сигнатура {xor:(0,4), xor3:(...), and:(...), or:(...)}

Сравнение двух слов: нормированное L₁-расстояние по периодам.

Запуск:
    python3 -m projects.hexglyph.solan_word --word РАТОН
    python3 -m projects.hexglyph.solan_word --word РАТОН --steps 20
    python3 -m projects.hexglyph.solan_word --word РАТОН --show-ca
    python3 -m projects.hexglyph.solan_word --compare РАТОН СТОЛ ГОРА ВОДА
    python3 -m projects.hexglyph.solan_word --no-color
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    step, find_orbit, render_row_char,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_phonetic import phonetic_h, _solan_char

_ALL_RULES = ['xor', 'xor3', 'and', 'or']


# ── Кодирование ─────────────────────────────────────────────────────────────

def encode_word(word: str) -> list[int]:
    """Закодировать русское слово как список h-значений Q6.

    Не-фонетические символы (не входящие в 16-буквенный алфавит) пропускаются.
    Возвращает пустой список, если ни одна буква не распознана.
    """
    result = []
    for ch in word.upper():
        h = phonetic_h(ch)
        if h is not None:
            result.append(h)
    return result


def pad_to(cells: list[int], width: int) -> list[int]:
    """Циклически растянуть (или обрезать) список клеток до ``width``."""
    if not cells:
        return [0] * width
    return [cells[i % len(cells)] for i in range(width)]


# ── Орбитальная сигнатура ────────────────────────────────────────────────────

Signature = dict[str, tuple[int | None, int | None]]


def word_signature(
    word:      str,
    width:     int = 16,
    max_steps: int = 2000,
) -> Signature:
    """Орбитальная сигнатура слова: {rule: (transient, period)}.

    Слово кодируется как Q6-последовательность, расширяется до ``width``
    и прогоняется через ``find_orbit`` для каждого из 4 правил.
    """
    raw = encode_word(word)
    if not raw:
        return {r: (None, None) for r in _ALL_RULES}
    cells = pad_to(raw, width)
    return {r: find_orbit(cells, r, max_steps) for r in _ALL_RULES}


def sig_distance(sig1: Signature, sig2: Signature) -> float:
    """Нормированное L₁-расстояние по периодам (0 = идентичные орбиты).

    Для каждого правила: |P₁ - P₂| / max(P₁, P₂, 1).
    Усредняется по правилам с известными периодами.
    Возвращает NaN, если ни у одного правила нет известного периода.
    """
    total, count = 0.0, 0
    for r in _ALL_RULES:
        _, p1 = sig1.get(r, (None, None))
        _, p2 = sig2.get(r, (None, None))
        if p1 is not None and p2 is not None:
            total += abs(p1 - p2) / max(p1, p2, 1)
            count += 1
    return total / count if count else float('nan')


def word_distance(word1: str, word2: str, width: int = 16) -> float:
    """Орбитальное расстояние между двумя словами (0..1, 0 = одинаковые)."""
    return sig_distance(
        word_signature(word1, width),
        word_signature(word2, width),
    )


# ── Рендеринг ────────────────────────────────────────────────────────────────

def _word_header(word: str, raw: list[int], width: int,
                 bold: str, reset: str, dim: str) -> None:
    encoded = ''.join(_solan_char(h) for h in raw)
    ru_seq  = ' '.join(
        ch for ch in word.upper() if phonetic_h(ch) is not None
    )
    h_seq   = '  '.join(str(h) for h in raw)
    print(bold + f"  Слово: {word.upper()}" + reset
          + f"  →  {encoded}  ({len(raw)} букв)")
    print(f"  {dim}{ru_seq}{reset}")
    print(f"  {dim}h: {h_seq}{reset}")


def print_word_ca(
    word:  str,
    width: int  = 24,
    steps: int  = 15,
    rule:  str  = 'xor',
    color: bool = True,
) -> None:
    """Пространственно-временная диаграмма CA для слова."""
    raw = encode_word(word)
    if not raw:
        print(f"  Слово '{word}' не содержит фонетических символов.")
        return

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    _word_header(word, raw, width, bold, reset, dim)
    rule_col = (_RULE_COLOR.get(rule, '') if color else '')
    print(f"\n  {rule_col}{bold}Правило: {_RULE_NAMES.get(rule, rule)}{reset}"
          f"  {dim}width={width}  steps={steps}{reset}\n")

    cells = pad_to(raw, width)
    sep   = '─' * (width + 7)
    print(sep)
    for t in range(steps + 1):
        print(f"{t:4d} │ " + render_row_char(cells, color))
        cells = step(cells, rule)
    print(sep)


def print_word_analysis(
    word:  str,
    width: int  = 16,
    color: bool = True,
) -> None:
    """Полный анализ: кодировка + орбиты для 4 правил."""
    raw = encode_word(word)
    if not raw:
        print(f"  Слово '{word}' не содержит фонетических символов.")
        return

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    _word_header(word, raw, width, bold, reset, dim)
    print()

    cells = pad_to(raw, width)
    print(f"  {'Правило':8s}  {'T (транзиент)':>14s}  {'P (период)':>11s}")
    print('  ' + '─' * 38)

    for r in _ALL_RULES:
        t, p = find_orbit(cells, r)
        col   = (_RULE_COLOR.get(r, '') if color else '')
        t_str = str(t) if t is not None else '—'
        p_str = str(p) if p is not None else '>2000'
        print(f"  {col}{r.upper():8s}{reset}"
              f"  {t_str:>14s}  {p_str:>11s}")
    print()


def print_comparison(
    words: list[str],
    width: int  = 16,
    color: bool = True,
) -> None:
    """Сравнить несколько слов по орбитальным сигнатурам."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    sigs = {w: word_signature(w, width) for w in words}

    print(bold + f"  Сравнение слов  width={width}" + reset)
    print()

    # Header
    rule_hdrs = '  '.join(
        (_RULE_COLOR.get(r, '') if color else '') + f"P({r.upper():4s})" + reset
        for r in _ALL_RULES
    )
    print(f"  {'Слово':14s}  {rule_hdrs}")
    print('  ' + '─' * (14 + 2 + len(_ALL_RULES) * 12))

    for w in words:
        sig   = sigs[w]
        parts = []
        for r in _ALL_RULES:
            _, p = sig[r]
            col  = (_RULE_COLOR.get(r, '') if color else '')
            p_str = str(p) if p is not None else '?'
            parts.append(f"{col}{p_str:>8s}{reset}")
        print(f"  {w.upper():14s}  {'  '.join(parts)}")

    print()
    if len(words) >= 2:
        print(bold + "  Орбитальные расстояния:" + reset)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                d = sig_distance(sigs[words[i]], sigs[words[j]])
                bar = '█' * int(d * 20) + '░' * (20 - int(d * 20))
                print(f"    {words[i].upper():10s} ↔ {words[j].upper():10s} "
                      f"|{bar}| {d:.3f}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Семантический анализ слов через Q6-автомат')
    parser.add_argument('--word', default='РАТОН',
                        help='русское слово для анализа')
    parser.add_argument('--compare', nargs='+', metavar='WORD',
                        help='сравнить список слов по орбитальным сигнатурам')
    parser.add_argument('--show-ca', action='store_true',
                        help='показать пространственно-временную диаграмму')
    parser.add_argument('--rule', choices=_ALL_RULES, default='xor',
                        help='правило для --show-ca (default: xor)')
    parser.add_argument('--width', type=int, default=16,
                        help='ширина CA (default: 16)')
    parser.add_argument('--steps', type=int, default=15,
                        help='число шагов для --show-ca (default: 15)')
    parser.add_argument('--no-color', action='store_true',
                        help='без АНSI-цветов')
    parser.add_argument('--json',     action='store_true',
                        help='JSON output')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        import json as _json
        sig = word_signature(args.word, args.width)
        print(_json.dumps({'word': args.word.upper(), 'signature': {r: list(v) for r, v in sig.items()}}, ensure_ascii=False, indent=2))
        import sys; sys.exit(0)
    if args.compare:
        words = list(dict.fromkeys([args.word] + args.compare))  # уникальные, порядок
        print_comparison(words, width=args.width, color=_color)
    elif args.show_ca:
        print_word_ca(args.word, width=args.width, steps=args.steps,
                      rule=args.rule, color=_color)
    else:
        print_word_analysis(args.word, width=args.width, color=_color)
