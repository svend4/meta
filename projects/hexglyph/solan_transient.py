"""solan_transient.py — Анализ транзиентов орбит Q6.

word_signature() возвращает (transient, period) для каждого правила.
Предыдущие модули использовали только период; транзиент содержит
дополнительную информацию:

  XOR  transient ∈ {2,8}  ≡ XOR3 period (идеальная корреляция!)
  XOR3 transient = 0       (мгновенный аттрактор для всех слов)
  AND  transient ∈ {1,2,3,4,7}  H≈1.78 bits (vs. 0.99 по периоду)
  OR   transient ∈ {1,2,3,4,6,7} H≈1.96 bits (vs. 0.99 по периоду)

Полных уникальных сигнатур: 13  (vs. 5 только по периодам)
Ключ: (xor_t, and_t, and_p, or_t, or_p)

Возможности:
  • build_transient_data()    — полный анализ; 13 классов
  • transient_classes()       — список классов + состав
  • transient_dist()          — расстояние с учётом транзиентов
  • print_transient_analysis()— таблица сравнения энтропий
  • print_transient_classes() — 13 классов с составом

Запуск:
    python3 -m projects.hexglyph.solan_transient
    python3 -m projects.hexglyph.solan_transient --classes
    python3 -m projects.hexglyph.solan_transient --json
    python3 -m projects.hexglyph.solan_transient --no-color
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_ca      import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES = tuple(_ALL_RULES)  # ('xor', 'xor3', 'and', 'or')


# ── Core types ────────────────────────────────────────────────────────────────

# Full signature key: (xor_t, and_t, and_p, or_t, or_p)
# (xor3_t is always 0; xor_p is always 1; xor_t ≡ xor3_p)
FullKey = tuple[int, int, int, int, int]


def full_key(word: str, width: int = 16) -> FullKey:
    """Полный ключ слова: (xor_t, and_t, and_p, or_t, or_p)."""
    from projects.hexglyph.solan_word import word_signature
    sig = word_signature(word, width=width)
    return (
        sig['xor'][0],   # xor_t  (≡ xor3_p)
        sig['and'][0],   # and_t
        sig['and'][1],   # and_p
        sig['or'][0],    # or_t
        sig['or'][1],    # or_p
    )


def _shannon(values: list[int | None]) -> float:
    valid = [v for v in values if v is not None]
    n = len(valid)
    if n == 0:
        return 0.0
    cnt: dict[int, int] = {}
    for v in valid:
        cnt[v] = cnt.get(v, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in cnt.values() if c > 0)


# ── Per-word full signatures ──────────────────────────────────────────────────

def all_full_signatures(
    words: list[str] | None = None,
    width: int = 16,
) -> dict[str, FullKey]:
    """dict[word → (xor_t, and_t, and_p, or_t, or_p)]."""
    if words is None:
        words = LEXICON
    return {w: full_key(w, width=width) for w in words}


# ── Transient distance ────────────────────────────────────────────────────────

def transient_dist(sig1: dict, sig2: dict) -> float:
    """Расстояние с учётом транзиентов и периодов.

    Для каждого правила: (|t1-t2|/max(t1,t2,1) + |p1-p2|/max(p1,p2,1)) / 2
    Усредняется по правилам.
    """
    total, count = 0.0, 0
    for r in RULES:
        t1, p1 = sig1.get(r, (None, None))
        t2, p2 = sig2.get(r, (None, None))
        if None in (t1, p1, t2, p2):
            continue
        td = abs(t1 - t2) / max(t1, t2, 1)
        pd = abs(p1 - p2) / max(p1, p2, 1)
        total += (td + pd) / 2
        count += 1
    return total / count if count else float('nan')


# ── Main analysis ─────────────────────────────────────────────────────────────

def build_transient_data(
    words: list[str] | None = None,
    width: int = 16,
) -> dict[str, Any]:
    """Полный транзиентный анализ.

    Возвращает dict:
      'words':       [...]
      'signatures':  dict[word → FullKey]
      'by_rule':     dict[rule → {'transients', 'periods', 'entropy_t',
                                   'entropy_p', 'unique_t', 'unique_tp'}]
      'classes':     list of {'key': FullKey, 'words': [...], 'count': int}
      'n_classes':   int
      'xor_t_isomorphic_xor3_p': bool
    """
    from projects.hexglyph.solan_word import word_signature

    if words is None:
        words = LEXICON
    n = len(words)

    all_sigs = {w: word_signature(w, width=width) for w in words}

    # Per-rule transient/period data
    by_rule: dict[str, dict] = {}
    for rule in RULES:
        transients = {w: all_sigs[w][rule][0] for w in words}
        periods    = {w: all_sigs[w][rule][1] for w in words}
        tp_pairs   = {w: all_sigs[w][rule]    for w in words}
        cnt_t: dict[int, int] = {}
        for t in transients.values():
            if t is not None:
                cnt_t[t] = cnt_t.get(t, 0) + 1
        by_rule[rule] = {
            'transients':  transients,
            'periods':     periods,
            'tp_pairs':    tp_pairs,
            'entropy_t':   _shannon(list(transients.values())),
            'entropy_p':   _shannon(list(periods.values())),
            'unique_t':    sorted({t for t in transients.values() if t is not None}),
            'hist_t':      dict(sorted(cnt_t.items())),
            'unique_tp':   sorted({v for v in tp_pairs.values() if None not in v}),
        }

    # Full 13-class signatures
    classes_dict: dict[FullKey, list[str]] = defaultdict(list)
    signatures: dict[str, FullKey] = {}
    for w in words:
        k = (
            all_sigs[w]['xor'][0],
            all_sigs[w]['and'][0],
            all_sigs[w]['and'][1],
            all_sigs[w]['or'][0],
            all_sigs[w]['or'][1],
        )
        signatures[w] = k
        classes_dict[k].append(w)

    classes = sorted(
        [{'key': k, 'words': v, 'count': len(v)}
         for k, v in classes_dict.items()],
        key=lambda x: -x['count'],
    )

    # Check structural property: XOR transient ≡ XOR3 period
    xor_iso = all(
        all_sigs[w]['xor'][0] == all_sigs[w]['xor3'][1]
        for w in words
        if all_sigs[w]['xor'][0] is not None
    )

    return {
        'words':                    list(words),
        'signatures':               signatures,
        'by_rule':                  by_rule,
        'classes':                  classes,
        'n_classes':                len(classes),
        'xor_t_isomorphic_xor3_p':  xor_iso,
    }


def transient_classes(
    words: list[str] | None = None,
    width: int = 16,
) -> list[dict[str, Any]]:
    """13 классов эквивалентности (ключ = full_key)."""
    data = build_transient_data(words, width=width)
    return data['classes']


def transient_dict(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Dict для viewer.html / JSON-экспорта."""
    return {
        'words':      data['words'],
        'signatures': {
            w: list(k) for w, k in data['signatures'].items()
        },
        'by_rule': {
            rule: {
                'transients':  {w: v for w, v in d['transients'].items()},
                'periods':     {w: v for w, v in d['periods'].items()},
                'entropy_t':   round(d['entropy_t'], 6),
                'entropy_p':   round(d['entropy_p'], 6),
                'unique_t':    d['unique_t'],
                'hist_t':      {str(k): v for k, v in d['hist_t'].items()},
            }
            for rule, d in data['by_rule'].items()
        },
        'classes': [
            {'key': list(c['key']), 'words': c['words'], 'count': c['count']}
            for c in data['classes']
        ],
        'n_classes':                 data['n_classes'],
        'xor_t_isomorphic_xor3_p':   data['xor_t_isomorphic_xor3_p'],
    }


# ── Terminal output ───────────────────────────────────────────────────────────

def print_transient_analysis(
    words:  list[str] | None = None,
    width:  int  = 16,
    color:  bool = True,
) -> None:
    """Таблица: транзиентные и периодные энтропии, уникальные значения."""
    if words is None:
        words = LEXICON

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    data = build_transient_data(words, width=width)
    n    = len(words)

    print(bold + f"  ◈ Транзиентный анализ Q6  n={n}" + reset)
    print()
    print(f"  {'Правило':8s}  {'H(транз)':>12}  {'H(пер)':>10}  "
          f"{'ΔH':>8}  {'Уник.Т':>7}  Распределение транзиентов")
    print('  ' + '─' * 78)

    for rule in RULES:
        d   = data['by_rule'][rule]
        col = (_RULE_COLOR.get(rule, '') if color else '')
        lbl = _RULE_NAMES.get(rule, rule.upper())

        hist  = '  '.join(f"t={t}:{c}" for t, c in sorted(d['hist_t'].items()))
        delta = d['entropy_t'] - d['entropy_p']
        d_s   = ('+' if delta >= 0 else '') + f"{delta:.4f}"

        note = ''
        if rule == 'xor':
            note = (dim + '  ← транз≡XOR3-период' + reset) if color \
                else '  ← транз≡XOR3-период'
        elif rule == 'xor3' and d['entropy_t'] == 0.0:
            note = (dim + '  ← мгновенный аттрактор' + reset) if color \
                else '  ← мгновенный аттрактор'

        print(f"  {col}{lbl:<8}{reset}  "
              f"{d['entropy_t']:>10.4f} bits  "
              f"{d['entropy_p']:>8.4f}  "
              f"{d_s:>8}  "
              f"{len(d['unique_t']):>7}  "
              f"{dim}{hist}{reset}{note}")

    print()
    print(f"  Полных уникальных сигнатур: "
          f"{bold}{data['n_classes']}{reset}  "
          f"(vs. 5 только по периодам)")
    print(f"  Ключ: (xor_t, and_t, and_p, or_t, or_p)")

    if data['xor_t_isomorphic_xor3_p']:
        col = (_RULE_COLOR.get('xor', '') if color else '')
        c3  = (_RULE_COLOR.get('xor3', '') if color else '')
        print(f"\n  {bold}Структурное свойство:{reset}  "
              f"{col}XOR транзиент{reset} ≡ {c3}XOR3 период{reset}  "
              f"(для всех {n} слов)")
    print()


def print_transient_classes(
    words:  list[str] | None = None,
    width:  int  = 16,
    color:  bool = True,
) -> None:
    """13 классов эквивалентности с полными ключами."""
    if words is None:
        words = LEXICON

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    data = build_transient_data(words, width=width)
    palette = [
        '\033[38;5;75m',  '\033[38;5;120m', '\033[38;5;220m',
        '\033[38;5;213m', '\033[38;5;208m', '\033[38;5;147m',
        '\033[38;5;159m', '\033[38;5;199m', '\033[38;5;154m',
        '\033[38;5;226m', '\033[38;5;171m', '\033[38;5;45m',
        '\033[38;5;82m',
    ]

    print(bold + f"  ◈ 13 классов эквивалентности Q6  n={len(words)}" + reset)
    print()
    print(f"  {'Кл':>3}  xor_t  and_t  and_p  or_t  or_p  "
          f"{'Слов':>5}  Состав")
    print('  ' + '─' * 76)

    for ci, cls in enumerate(data['classes']):
        xor_t, and_t, and_p, or_t, or_p = cls['key']
        col    = (palette[ci % len(palette)]) if color else ''
        wlist  = '  '.join(cls['words'])
        print(f"  {col}{ci+1:>3}    {xor_t:>3}    {and_t:>3}    "
              f"{and_p:>3}   {or_t:>3}    {or_p:>3}  "
              f"{cls['count']:>5}  {dim}{wlist}{reset}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Транзиентный анализ орбит Q6')
    parser.add_argument('--words',    nargs='+', metavar='WORD')
    parser.add_argument('--classes',  action='store_true')
    parser.add_argument('--json',     action='store_true')
    parser.add_argument('--width',    type=int, default=16)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _words = args.words if args.words else None
    _color = not args.no_color

    if args.json:
        ws   = _words or list(LEXICON)
        data = build_transient_data(ws, width=args.width)
        print(json.dumps(transient_dict(data), ensure_ascii=False, indent=2))
    elif args.classes:
        print_transient_classes(_words, width=args.width, color=_color)
    else:
        print_transient_analysis(_words, width=args.width, color=_color)
        print_transient_classes(_words, width=args.width, color=_color)
