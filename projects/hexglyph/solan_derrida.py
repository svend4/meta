"""solan_derrida.py — Диаграмма Деррида для Q6 CA.

Для пары состояний (s1, s2) на нормированном расстоянии x = d(s1,s2)/(width·6)
измеряем расстояние после одного шага: y = d(step(s1), step(s2))/(width·6).

Точка (x, y) — «диаграмма Деррида»:
  y < x  → упорядоченный CA (возмущения убывают)
  y > x  → хаотичный CA    (возмущения растут)
  y = x  → нейтральный / граница

Две выборки:
  «Лексикон» — все C(49,2) = 1176 пар слов (реальные Q6-состояния)
  «Случайные» — n_random пар случайных Q6-состояний (теоретическая кривая)

Аналитическая аппроксимация (отожжённое приближение):
  XOR  (left^right):      y = 2x(1−x)          → пересечение y=x при x=0.5
  XOR3 (l^c^r):           y = 3x − 6x² + 4x³   → наклон 3 при x→0 (хаос)
  AND  (left&right, q=½): y = x(1−x)/2 + ...   — числ. оценка
  OR   (left|right, q=½): симметрично AND

Функции:
    state_dist_norm(s1, s2, width) → float
    derrida_point(s1, s2, rule, width) → (float, float)
    lexicon_points(rule, width) → list[(x, y)]
    random_points(rule, width, n, seed) → list[(x, y)]
    derrida_curve(points, n_bins) → dict
    analytic_curve(rule, n_pts) → list[(x, y)]
    classify_rule(rule, width, n_random, seed) → str
    build_derrida_data(width, n_random, seed) → dict
    derrida_dict(width, n_random, seed) → dict
    print_derrida(rule, width, color)
    print_derrida_summary(width, color)

Запуск:
    python3 -m projects.hexglyph.solan_derrida
    python3 -m projects.hexglyph.solan_derrida --rule xor3 --no-color
    python3 -m projects.hexglyph.solan_derrida --summary
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    step,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_N_BITS        = 6           # bits per Q6 cell
_DEFAULT_WIDTH = 16
_DEFAULT_N     = 1000        # random pairs for theoretical curve
_DEFAULT_SEED  = 42
_DEFAULT_WORDS = list(LEXICON)

# Normalising constant for a width-16 CA: max distance = 16 × 6 = 96
_MAX_DIST_FN   = lambda w: w * _N_BITS


# ── Distance ──────────────────────────────────────────────────────────────────

def _q6h(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def state_dist_norm(s1: list[int], s2: list[int], width: int = _DEFAULT_WIDTH) -> float:
    """Нормированное Q6-Хэмминг-расстояние ∈ [0, 1]."""
    raw = sum(_q6h(a, b) for a, b in zip(s1, s2))
    return raw / _MAX_DIST_FN(width)


# ── Single Derrida point ──────────────────────────────────────────────────────

def derrida_point(
    s1:    list[int],
    s2:    list[int],
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> tuple[float, float]:
    """Одна точка (x_before, y_after) для пары состояний."""
    x = state_dist_norm(s1, s2, width)
    y = state_dist_norm(step(s1, rule), step(s2, rule), width)
    return x, y


# ── Lexicon-word pairs ────────────────────────────────────────────────────────

def lexicon_points(
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[tuple[float, float]]:
    """Все C(49,2) = 1176 пар слов лексикона → список (x, y)."""
    words = _DEFAULT_WORDS
    cells = [pad_to(encode_word(w), width) for w in words]
    pts: list[tuple[float, float]] = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            pts.append(derrida_point(cells[i], cells[j], rule, width))
    return pts


# ── Random Q6 state pairs ─────────────────────────────────────────────────────

def _random_q6_state(width: int, rng: random.Random) -> list[int]:
    """Случайное Q6-состояние ширины `width`."""
    return [rng.randint(0, 63) for _ in range(width)]


def random_points(
    rule:  str,
    width: int = _DEFAULT_WIDTH,
    n:     int = _DEFAULT_N,
    seed:  int = _DEFAULT_SEED,
) -> list[tuple[float, float]]:
    """n случайных пар Q6-состояний → список (x, y)."""
    rng = random.Random(seed)
    return [
        derrida_point(
            _random_q6_state(width, rng),
            _random_q6_state(width, rng),
            rule, width,
        )
        for _ in range(n)
    ]


# ── Binned average curve ──────────────────────────────────────────────────────

def derrida_curve(
    points:  list[tuple[float, float]],
    n_bins:  int = 20,
) -> dict:
    """Усреднённая кривая Деррида: бинируем x → среднее y.

    Возвращает dict:
        bins        : list[float]   — центры бинов (n_bins значений)
        mean_y      : list[float]   — среднее y в каждом бине
        count       : list[int]     — число точек в каждом бине
        above_diag  : int           — число точек выше y=x (хаотичных)
        below_diag  : int           — число точек ниже y=x (упорядоченных)
        on_diag     : int           — точек с y≈x (±0.01)
    """
    bins_sum  = [0.0] * n_bins
    bins_cnt  = [0]   * n_bins
    above = below = on = 0
    for x, y in points:
        bi = min(int(x * n_bins), n_bins - 1)
        bins_sum[bi] += y
        bins_cnt[bi] += 1
        if   y > x + 1e-9: above += 1
        elif y < x - 1e-9: below += 1
        else:               on    += 1
    mean_y = [
        (bins_sum[i] / bins_cnt[i] if bins_cnt[i] else float('nan'))
        for i in range(n_bins)
    ]
    centers = [(i + 0.5) / n_bins for i in range(n_bins)]
    return {
        'bins':       centers,
        'mean_y':     mean_y,
        'count':      bins_cnt,
        'above_diag': above,
        'below_diag': below,
        'on_diag':    on,
    }


# ── Analytic approximation (annealed, q = ½) ──────────────────────────────────

def analytic_curve(rule: str, n_pts: int = 50) -> list[tuple[float, float]]:
    """Аналитическое отожжённое приближение Деррида-кривой при q=½.

    Формулы (каждый бит, q = ½):
      XOR  (l^r):      E[y|x] = 2x(1−x)
      XOR3 (l^c^r):    E[y|x] = 3x − 6x² + 4x³
      AND  (l&r, q=½): E[y|x] = x/2 + x²/4   (approx)
      OR   (l|r, q=½): E[y|x] = x/2 + x²/4   (same by symmetry)

    Note: AND/OR formulas are approximate (exact depends on marginal q).
    """
    pts: list[tuple[float, float]] = []
    for i in range(n_pts + 1):
        x = i / n_pts
        if rule == 'xor':
            y = 2 * x * (1 - x)
        elif rule == 'xor3':
            y = 3*x - 6*x**2 + 4*x**3
        elif rule == 'and':
            # Exact for q=½: P(l&r=1) = 1/4; P(diff after one step)
            # P(l1&r1=1, l2&r2=0) + P(l1&r1=0, l2&r2=1)
            # ≈ x/2 · (1 - x/2) · 2  (rough)
            y = x * (1 - x / 2) * 0.5
        else:  # or
            # Symmetric to AND by De Morgan
            y = x * (1 - x / 2) * 0.5
        pts.append((x, min(1.0, max(0.0, y))))
    return pts


# ── Rule classification ───────────────────────────────────────────────────────

def classify_rule(
    rule:     str,
    width:    int = _DEFAULT_WIDTH,
    n_random: int = _DEFAULT_N,
    seed:     int = _DEFAULT_SEED,
) -> str:
    """Классифицировать правило: 'ordered' / 'chaotic' / 'complex'.

    Используем долю точек выше/ниже диагонали y=x на случайных парах.
      > 60% выше диагонали → 'chaotic'
      < 30% выше диагонали → 'ordered'
      иначе              → 'complex'
    """
    pts = random_points(rule, width, n_random, seed)
    n   = len(pts)
    above = sum(1 for x, y in pts if y > x + 1e-9)
    frac  = above / n if n else 0.0
    if   frac > 0.60: return 'chaotic'
    elif frac < 0.30: return 'ordered'
    else:             return 'complex'


# ── Full dataset ──────────────────────────────────────────────────────────────

def build_derrida_data(
    width:    int = _DEFAULT_WIDTH,
    n_random: int = _DEFAULT_N,
    seed:     int = _DEFAULT_SEED,
) -> dict:
    """Полный анализ диаграмм Деррида для всех 4 правил.

    Возвращает dict:
        width         : int
        n_random      : int
        rules         : {rule: {lex_curve, rnd_curve, analytic, classification}}
        lex_points    : {rule: list[(x,y)]}   — лексикон-пары
        rnd_points    : {rule: list[(x,y)]}   — случайные пары
    """
    result: dict = {'width': width, 'n_random': n_random, 'rules': {}}
    for rule in _ALL_RULES:
        lex_pts = lexicon_points(rule, width)
        rnd_pts = random_points(rule, width, n_random, seed)
        ana     = analytic_curve(rule)
        lex_cur = derrida_curve(lex_pts)
        rnd_cur = derrida_curve(rnd_pts)
        cls     = classify_rule(rule, width, n_random, seed)
        result['rules'][rule] = {
            'lex_curve':      lex_cur,
            'rnd_curve':      rnd_cur,
            'analytic':       [(round(x, 4), round(y, 4)) for x, y in ana],
            'classification': cls,
            'lex_n':          len(lex_pts),
            'rnd_n':          len(rnd_pts),
        }
    return result


# ── JSON export ───────────────────────────────────────────────────────────────

def derrida_dict(
    width:    int = _DEFAULT_WIDTH,
    n_random: int = _DEFAULT_N,
    seed:     int = _DEFAULT_SEED,
) -> dict:
    """JSON-совместимый словарь диаграмм Деррида (без сырых точек)."""
    d    = build_derrida_data(width, n_random, seed)
    rules_out: dict[str, dict] = {}
    for rule in _ALL_RULES:
        r = d['rules'][rule]
        rules_out[rule] = {
            'lex_curve':      {
                'bins':       r['lex_curve']['bins'],
                'mean_y':     [round(v, 4) if not math.isnan(v) else None
                               for v in r['lex_curve']['mean_y']],
                'above_diag': r['lex_curve']['above_diag'],
                'below_diag': r['lex_curve']['below_diag'],
            },
            'rnd_curve':      {
                'bins':       r['rnd_curve']['bins'],
                'mean_y':     [round(v, 4) if not math.isnan(v) else None
                               for v in r['rnd_curve']['mean_y']],
                'above_diag': r['rnd_curve']['above_diag'],
                'below_diag': r['rnd_curve']['below_diag'],
            },
            'analytic':       r['analytic'],
            'classification': r['classification'],
        }
    return {'width': width, 'n_random': n_random, 'rules': rules_out}


def derrida_summary(
    width:    int = _DEFAULT_WIDTH,
    n_random: int = _DEFAULT_N,
    seed:     int = _DEFAULT_SEED,
) -> dict:
    """Alias for build_derrida_data — standard *_summary convention."""
    return build_derrida_data(width, n_random, seed)


# ── ASCII display ─────────────────────────────────────────────────────────────

_CLASSIFY_LABEL = {
    'ordered': 'упорядоченный',
    'chaotic': 'хаотичный',
    'complex': 'сложный',
}
_CLASSIFY_COLOR = {
    'ordered': '\033[38;5;46m',   # green
    'chaotic': '\033[38;5;196m',  # red
    'complex': '\033[38;5;226m',  # yellow
}


def print_derrida(
    rule:     str = 'xor3',
    width:    int = _DEFAULT_WIDTH,
    n_random: int = _DEFAULT_N,
    color:    bool = True,
) -> None:
    """Текстовая диаграмма Деррида (гистограмма: x → y)."""
    lex_pts = lexicon_points(rule, width)
    rnd_pts = random_points(rule, width, n_random)
    lex_cur = derrida_curve(lex_pts)
    ana     = analytic_curve(rule)
    cls     = classify_rule(rule, width, n_random)

    rule_col  = _RULE_COLOR.get(rule, '') if color else ''
    rule_name = _RULE_NAMES.get(rule, rule.upper())
    cls_col   = _CLASSIFY_COLOR.get(cls, '') if color else ''
    cls_name  = _CLASSIFY_LABEL.get(cls, cls)
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    print(f"{bold}  ◈ Диаграмма Деррида  {rule_col}{rule_name}{rst}  "
          f"→ {cls_col}{cls_name}{rst}")
    print(f"  {'─' * 60}")
    print(f"  {'x':>4}  {'y̅_лекс':>7}  {'y̅_аналит':>10}  {'диаграмма'}")
    print(f"  {'─' * 60}")

    n_bins = len(lex_cur['bins'])
    ana_dict = {round(x, 2): y for x, y in ana}
    for i, (cx, my) in enumerate(zip(lex_cur['bins'], lex_cur['mean_y'])):
        if lex_cur['count'][i] == 0:
            continue
        # Analytic reference
        ana_key = round(round(cx * 20) / 20, 2)
        ay = ana_dict.get(ana_key, float('nan'))
        ay_str = f'{ay:.3f}' if not math.isnan(ay) else '  — '
        my_str = f'{my:.3f}' if not math.isnan(my) else '  — '
        # Bar: y value on x-axis (draw the (x,y) relationship)
        if not math.isnan(my):
            # Mark whether above/below diagonal
            arrow = '↑' if my > cx + 0.01 else ('↓' if my < cx - 0.01 else '=')
            bc = ('\033[38;5;196m' if my > cx + 0.01 else
                  ('\033[38;5;46m' if my < cx - 0.01 else
                   '\033[38;5;226m')) if color else ''
            bar_len = 30
            y_pos   = min(bar_len - 1, int(my * bar_len))
            x_pos   = min(bar_len - 1, int(cx * bar_len))
            bar     = [' '] * bar_len
            bar[x_pos] = '|'    # diagonal reference
            bar[y_pos] = '█'
            bar_str = ''.join(bar)
            print(f"  {cx:.2f}  {bc}{my_str}{rst}  {ay_str:>10}  "
                  f"{bc}{bar_str}{rst} {arrow}")
    print()
    above = lex_cur['above_diag']
    below = lex_cur['below_diag']
    total = above + below + lex_cur['on_diag']
    pct   = above / total * 100 if total else 0
    print(f"  Лексикон: {total} пар · {above} выше диагонали ({pct:.0f}%)")
    print()


def print_derrida_summary(
    width: int = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Сводка классификаций по всем правилам."""
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    print(f"\n{bold}  ◈ Классификация Деррида — Q6 CA{rst}")
    print(f"  {'─' * 50}")
    print(f"  {'Правило':12s}  {'Тип':12s}  {'% хаотичных пар (лекс.)'}")
    print(f"  {'─' * 50}")
    for rule in _ALL_RULES:
        lex_pts = lexicon_points(rule, width)
        lex_cur = derrida_curve(lex_pts)
        total   = (lex_cur['above_diag'] + lex_cur['below_diag']
                   + lex_cur['on_diag'])
        pct     = lex_cur['above_diag'] / total * 100 if total else 0
        cls     = classify_rule(rule, width)
        rule_col = _RULE_COLOR.get(rule, '') if color else ''
        cls_col  = _CLASSIFY_COLOR.get(cls, '') if color else ''
        name     = _RULE_NAMES.get(rule, rule)
        print(f"  {rule_col}{name:12s}{rst}  {cls_col}{_CLASSIFY_LABEL[cls]:12s}{rst}  {pct:.0f}%")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Диаграмма Деррида Q6 CA')
    parser.add_argument('--rule',    default='xor3', choices=_ALL_RULES)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--all',     action='store_true')
    parser.add_argument('--n',       type=int, default=_DEFAULT_N)
    parser.add_argument('--width',   type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--json',     action='store_true', help='JSON output')
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()
    color = not args.no_color
    if args.json:
        import json as _json
        print(_json.dumps(derrida_dict(args.width, args.n),
                          ensure_ascii=False, indent=2))
    elif args.summary:
        print_derrida_summary(args.width, color)
    elif args.all:
        for rule in _ALL_RULES:
            print_derrida(rule, args.width, args.n, color)
    else:
        print_derrida(args.rule, args.width, args.n, color)


if __name__ == '__main__':
    _main()
