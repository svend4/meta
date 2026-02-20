"""solan_entropy.py — Энтропийный анализ Q6-автомата.

Вычисляет энтропию Шеннона распределения клеточных состояний
на каждом шаге эволюции для 4 правил CA.  Динамика отображается
спарклайнами из блочных символов Unicode (▁▂▃▄▅▆▇█).

Физический смысл правил:
  XOR ⊕ — биекция на (Z₂)⁶, энтропия сохраняется
  XOR3  — аналогично, сдвиговый осциллятор
  AND & — разрушительное, клетки «гасят» друг друга → H→0
  OR  | — заполняющее, клетки «накапливают» биты → H→H(63)

Запуск:
    python3 -m projects.hexglyph.solan_entropy
    python3 -m projects.hexglyph.solan_entropy --ic random --seed 42 --width 40 --steps 30
    python3 -m projects.hexglyph.solan_entropy --ic phonetic --word РАТОН --steps 20
    python3 -m projects.hexglyph.solan_entropy --rule xor --width 20 --steps 50
    python3 -m projects.hexglyph.solan_entropy --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import step, make_initial

# ── Вычисление энтропии ─────────────────────────────────────────────────────

def entropy(cells: list[int]) -> float:
    """Энтропия Шеннона (бит) распределения состояний клеток.

    H = −Σ p_i · log₂(p_i),  где p_i = доля клеток в состоянии i.

    Минимум 0 (все клетки одинаковые),
    максимум log₂(64) ≈ 6 бит (равномерное распределение).
    """
    n = len(cells)
    if n == 0:
        return 0.0
    counts = Counter(cells)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def entropy_profile(
    cells: list[int],
    rule:  str,
    steps: int,
) -> list[float]:
    """Профиль H[t] для t = 0 .. steps (включительно)."""
    cur = list(cells)
    profile = [entropy(cur)]
    for _ in range(steps):
        cur = step(cur, rule)
        profile.append(entropy(cur))
    return profile


def entropy_profiles(
    cells: list[int],
    steps: int,
    rules: list[str] | None = None,
) -> dict[str, list[float]]:
    """Профили для нескольких правил сразу."""
    if rules is None:
        rules = ['xor', 'xor3', 'and', 'or']
    return {r: entropy_profile(cells, r, steps) for r in rules}


# ── Спарклайн ───────────────────────────────────────────────────────────────

_BLOCK = ' ▁▂▃▄▅▆▇█'


def sparkline(values: list[float], max_val: float) -> str:
    """Компактная визуализация числового ряда блочными символами."""
    if max_val <= 0:
        return ' ' * len(values)
    out = []
    for v in values:
        idx = int(8 * v / max_val)
        out.append(_BLOCK[max(0, min(8, idx))])
    return ''.join(out)


# ── ANSI ────────────────────────────────────────────────────────────────────

_RST  = '\033[0m'
_BOLD = '\033[1m'
_DIM  = '\033[2m'

_RULE_META: dict[str, tuple[str, str, str]] = {
    'xor':  ('XOR ⊕', '\033[38;5;75m',  'линейная (Z₂)⁶, паттерн Серпинского'),
    'xor3': ('XOR3 ', '\033[38;5;117m', 'линейная (Z₂)⁶, сдвиговый режим'),
    'and':  ('AND &', '\033[38;5;196m', 'разрушительное — H убывает к 0'),
    'or':   ('OR | ', '\033[38;5;220m', 'накапливающее  — зависит от IC'),
}

_TREND_COLOR = {
    '→': '\033[33m',   # жёлтый — нейтрально
    '↑': '\033[32m',   # зелёный — рост
    '↓': '\033[31m',   # красный — убывание
}

_RULES_ORDER = ['xor', 'xor3', 'and', 'or']


# ── Вывод ───────────────────────────────────────────────────────────────────

def print_entropy_chart(
    cells:    list[int],
    steps:    int  = 20,
    ic_label: str  = 'center',
    rules:    list[str] | None = None,
    color:    bool = True,
) -> None:
    """Спарклайн-таблица энтропии H(t) для заданных правил."""
    if rules is None:
        rules = _RULES_ORDER

    profiles = {r: entropy_profile(cells, r, steps) for r in rules}
    max_h    = max(h for prof in profiles.values() for h in prof)
    max_h    = max(max_h, 1e-9)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    width = len(cells)
    print(bold + f"  Энтропия Q6-CA"
          f"  ic={ic_label}  width={width}  steps={steps}" + reset)
    print(f"  Max H(t) = {max_h:.3f} бит"
          f"  (теор. max = {math.log2(64):.1f} бит для 64 состояний)")
    print()

    # Заголовок: ось t
    axis = f"t=0{'─' * (steps - 3)}t={steps}"
    print(f"  {'правило':7s}  {axis:{steps + 1}s}  "
          f"{'H₀':>5}  {'Hf':>5}  тренд  описание")
    print('  ' + '─' * max(60, steps + 30))

    for r in rules:
        label, col, desc = _RULE_META.get(r, (r, '', ''))
        prof  = profiles[r]
        spark = sparkline(prof, max_h)
        h0, hf = prof[0], prof[-1]

        if   hf > h0 + 0.02:  trend = '↑'
        elif hf < h0 - 0.02:  trend = '↓'
        else:                  trend = '→'

        rule_col  = (col                    if color else '')
        trend_col = (_TREND_COLOR[trend]    if color else '')

        print(f"  {rule_col}{label:7s}{reset}"
              f"  {spark}"
              f"  {rule_col}{h0:5.3f}{reset}"
              f"  {trend_col}{hf:5.3f}{reset}"
              f"  {trend_col}{trend:^5}{reset}"
              f"  {dim}{desc}{reset}")

    print()
    print(f"  {dim}Спарклайн: {_BLOCK[1]}=low → {_BLOCK[8]}=max ({max_h:.2f} бит){reset}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Энтропийный анализ Q6-автомата — спарклайны H(t)')
    parser.add_argument('--ic', choices=['center', 'edge', 'random', 'phonetic'],
                        default='center',
                        help='начальные условия (default: center)')
    parser.add_argument('--word', default='РАТОН',
                        help='слово для --ic phonetic')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed для --ic random')
    parser.add_argument('--width', type=int, default=40,
                        help='ширина CA (default: 40)')
    parser.add_argument('--steps', type=int, default=30,
                        help='число шагов (default: 30)')
    parser.add_argument('--rule', choices=['xor', 'xor3', 'and', 'or'],
                        default=None,
                        help='одно правило вместо всех 4')
    parser.add_argument('--no-color', action='store_true',
                        help='без ANSI-цветов')
    args = parser.parse_args()

    _cells = make_initial(args.width, args.ic, word=args.word, seed=args.seed)
    _rules = [args.rule] if args.rule else None

    print_entropy_chart(
        _cells,
        steps=args.steps,
        ic_label=args.ic,
        rules=_rules,
        color=not args.no_color,
    )
