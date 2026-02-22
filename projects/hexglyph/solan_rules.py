"""solan_rules.py — Покомпонентный анализ CA-правил Q6.

Для каждого из 4 правил (xor, xor3, and, or) вычисляет:
  • матрицу расстояний (только период этого правила)
  • MDS-вложение в ℝ²
  • распределение периодов и энтропию Шеннона

Ключевой вывод:
  • XOR — все 49 слов имеют период=1 → H=0, нулевая дискриминация
  • XOR3, AND, OR — каждое правило даёт ровно 2 значения периода (~1 бит)
  • Комбинация трёх информативных правил порождает ровно 5 классов
    эквивалентности (= 5 компонент орбитального графа)

Возможности:
  • build_all_rules()         — данные по всем 4 правилам
  • signature_classes()       — 5 классов эквивалентности (xor3,and,or)-ключ
  • print_rule_comparison()   — таблица энтропий и распределений
  • print_signature_classes() — 5 классов с составом
  • rules_dict()              — dict для viewer.html / JSON

Запуск:
    python3 -m projects.hexglyph.solan_rules
    python3 -m projects.hexglyph.solan_rules --classes
    python3 -m projects.hexglyph.solan_rules --json
    python3 -m projects.hexglyph.solan_rules --no-color
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_ca      import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)
from projects.hexglyph.solan_mds     import _gram, _power_iter, mds_stress

RULES = tuple(_ALL_RULES)  # ('xor', 'xor3', 'and', 'or')


# ── Per-rule helpers ──────────────────────────────────────────────────────────

def per_rule_dist(sig1: dict, sig2: dict, rule: str) -> float:
    """Расстояние по одному правилу: |P1−P2| / max(P1,P2,1)."""
    _, p1 = sig1.get(rule, (None, None))
    _, p2 = sig2.get(rule, (None, None))
    if p1 is None or p2 is None:
        return float('nan')
    return abs(p1 - p2) / max(p1, p2, 1)


def _shannon(values: list[int | None]) -> float:
    """Shannon entropy (bits) из списка значений (None пропускаются)."""
    valid = [v for v in values if v is not None]
    n = len(valid)
    if n == 0:
        return 0.0
    counts: dict[int, int] = {}
    for v in valid:
        counts[v] = counts.get(v, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


# ── Main analysis ─────────────────────────────────────────────────────────────

def build_all_rules(
    words: list[str] | None = None,
    width: int = 16,
) -> dict[str, dict[str, Any]]:
    """Вычислить данные для всех 4 правил.

    Возвращает dict[rule → {
        words, periods, dmat, coords, stress, entropy, unique_periods
    }]
    """
    from projects.hexglyph.solan_word import word_signature

    if words is None:
        words = LEXICON
    n = len(words)

    # Compute all signatures once
    all_sigs = {w: word_signature(w, width=width) for w in words}

    result: dict[str, dict[str, Any]] = {}
    for rule in RULES:
        periods = {w: all_sigs[w].get(rule, (None, None))[1] for w in words}

        # Per-rule n×n distance matrix
        dmat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    p1 = periods[words[i]]
                    p2 = periods[words[j]]
                    if p1 is None or p2 is None:
                        dmat[i][j] = 0.0
                    else:
                        dmat[i][j] = abs(p1 - p2) / max(p1, p2, 1)

        # MDS (handles degenerate all-zero matrix gracefully: coords → 0)
        B        = _gram(dmat)
        eigpairs = _power_iter(B, k=2)
        coords   = [[0.0, 0.0] for _ in range(n)]
        for d, (lam, vec) in enumerate(eigpairs):
            scale = math.sqrt(max(0.0, lam))
            for i in range(n):
                coords[i][d] = vec[i] * scale

        entropy = _shannon(list(periods.values()))
        unique  = sorted({v for v in periods.values() if v is not None})

        result[rule] = {
            'words':          list(words),
            'periods':        periods,
            'dmat':           dmat,
            'coords':         coords,
            'stress':         mds_stress(dmat, coords),
            'entropy':        entropy,
            'unique_periods': unique,
        }

    return result


def signature_classes(
    words: list[str] | None = None,
    width: int = 16,
) -> list[dict[str, Any]]:
    """5 классов эквивалентности по ключу (xor3, and, or)-периодов.

    Возвращает список dict{'key', 'words', 'count'}, отсортированный по убыванию.
    """
    from projects.hexglyph.solan_word import word_signature

    if words is None:
        words = LEXICON

    classes: dict[tuple, list[str]] = {}
    for w in words:
        sig = word_signature(w, width=width)
        key = tuple(sig[r][1] for r in ('xor3', 'and', 'or'))
        classes.setdefault(key, []).append(w)

    return sorted(
        [{'key': k, 'words': v, 'count': len(v)}
         for k, v in classes.items()],
        key=lambda x: -x['count'],
    )


def rules_dict(
    all_data: dict[str, dict[str, Any]],
    classes:  list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dict для viewer.html / JSON."""
    out: dict[str, Any] = {}
    for rule, d in all_data.items():
        out[rule] = {
            'words':          d['words'],
            'periods':        {w: d['periods'][w] for w in d['words']},
            'coords':         [[round(c[0], 6), round(c[1], 6)]
                               for c in d['coords']],
            'stress':         round(d['stress'], 6),
            'entropy':        round(d['entropy'], 6),
            'unique_periods': d['unique_periods'],
        }
    if classes is not None:
        out['classes'] = [
            {'key': list(c['key']), 'words': c['words'], 'count': c['count']}
            for c in classes
        ]
    return out


# ── Terminal output ───────────────────────────────────────────────────────────

def print_rule_comparison(
    words:  list[str] | None = None,
    width:  int  = 16,
    color:  bool = True,
) -> None:
    """Таблица сравнения правил: энтропия, уникальных периодов, гистограмма."""
    if words is None:
        words = LEXICON

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    data = build_all_rules(words, width=width)
    n    = len(words)

    print(bold + f"  ◈ Покомпонентный анализ CA-правил Q6  n={n}" + reset)
    print()
    print(f"  {'Правило':<8}  {'Энтропия':>10}  {'Уник.':>6}  "
          f"{'Стресс MDS':>10}  Распределение периодов")
    print('  ' + '─' * 72)

    for rule in RULES:
        d   = data[rule]
        col = (_RULE_COLOR.get(rule, '') if color else '')
        lbl = _RULE_NAMES.get(rule, rule.upper())

        # Period histogram
        cnt: dict[int, int] = {}
        for p in d['periods'].values():
            if p is not None:
                cnt[p] = cnt.get(p, 0) + 1
        hist = '  '.join(
            f"P={p}:{c}" for p, c in sorted(cnt.items(), key=lambda x: -x[1])
        )
        stress_s = f"{d['stress']:.4f}"
        h_s      = f"{d['entropy']:.4f} bits"

        note = ''
        if d['entropy'] == 0.0:
            note = (dim + '  ← нулевая дискриминация' + reset) if color \
                else '  ← нулевая дискриминация'

        print(f"  {col}{lbl:<8}{reset}  {h_s:>14}  "
              f"{len(d['unique_periods']):>6}  {stress_s:>10}  "
              f"{dim}{hist}{reset}{note}")

    print()

    # Best rule
    best = max(RULES, key=lambda r: data[r]['entropy'])
    bc   = (_RULE_COLOR.get(best, '') if color else '')
    bl   = _RULE_NAMES.get(best, best.upper())
    print(f"  Наибольшая дискриминация: {bc}{bl}{reset}  "
          f"(H={data[best]['entropy']:.4f} bits)")
    print()

    # Key insight
    print(bold + "  Ключевой вывод:" + reset)
    print(f"  XOR  даёт период=1 для всех {n} слов → нулевая дискриминация")
    print(f"  XOR3, AND, OR — по 2 значения периода, "
          f"3×1 бит = 5 уникальных классов")
    print()


def print_signature_classes(
    words:  list[str] | None = None,
    width:  int  = 16,
    color:  bool = True,
) -> None:
    """Вывести 5 классов эквивалентности (xor3,and,or)-ключу."""
    if words is None:
        words = LEXICON

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    classes = signature_classes(words, width=width)
    palette = [
        '\033[38;5;75m', '\033[38;5;120m', '\033[38;5;220m',
        '\033[38;5;213m', '\033[38;5;208m',
    ]

    print(bold + f"  ◈ Классы эквивалентности Q6  n={len(words)}" + reset)
    print(f"  Всего классов: {len(classes)}")
    print()
    print(f"  {'Кл':>3}  {'(xor3,and,or)':>15}  "
          f"{'Слов':>5}  Состав")
    print('  ' + '─' * 70)

    for ci, cls in enumerate(classes):
        xor3, and_, or_ = cls['key']
        col  = (palette[ci % len(palette)]) if color else ''
        key_s = f"({xor3},{and_},{or_})"
        words_s = '  '.join(cls['words'])
        print(f"  {col}{ci+1:>3}  {key_s:>15}  "
              f"{cls['count']:>5}  {dim}{words_s}{reset}")

    print()
    print(f"  {dim}Классы = компоненты орбитального графа Q6{reset}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Покомпонентный анализ CA-правил Q6')
    parser.add_argument('--words',   nargs='+', metavar='WORD')
    parser.add_argument('--classes', action='store_true',
                        help='показать классы эквивалентности')
    parser.add_argument('--json',    action='store_true')
    parser.add_argument('--width',   type=int, default=16)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _words = args.words if args.words else None
    _color = not args.no_color

    if args.json:
        ws    = _words or list(LEXICON)
        data  = build_all_rules(ws, width=args.width)
        cls   = signature_classes(ws, width=args.width)
        print(json.dumps(rules_dict(data, cls),
                         ensure_ascii=False, indent=2))
    elif args.classes:
        print_signature_classes(_words, width=args.width, color=_color)
    else:
        print_rule_comparison(_words, width=args.width, color=_color)
        print_signature_classes(_words, width=args.width, color=_color)
