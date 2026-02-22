"""solan_cross.py — Pairwise Cell Q6 Cross-Correlation Analysis.

For a period-P attractor of width N=16 each cell i traces a temporal
Q6-value sequence seq_i = [orbit[t][i] : t = 0…P−1].

This module computes the N×N Pearson cross-correlation matrix:

    r(i, j) = Cov(seq_i, seq_j) / (σ_i · σ_j)  ∈ [−1, 1]

where the covariance and standard deviations are taken over the P orbit
steps.  When a cell is temporally constant (σ = 0), r(i, ·) is undefined
(returned as None).

    cross_corr_matrix(word, rule, width)   → N×N matrix (with Nones)
    cross_summary(word, rule, width)       → full statistics dict

Key discoveries
──────────────────────────────────────────────────────────────────────────────
  МАТ XOR3  (P=8, N=16):
      Perfectly synchronised pairs (r = +1.000):
          (0, 15):  seq0 = [24,63,24,63,63,63,24,63]
                    seq15= [24,48,24,48,48,48,24,48]  — same up/down pattern
          (7,  8):  seq7 = [63,23,23,23,23,23,23,23]
                    seq8 = [48,23,23,23,23,23,23,23]  — both quasi-frozen,
                           same drop-then-flat pattern
      Pearson r=+1 does NOT require identical values; it requires the same
      relative fluctuation pattern (linear covariation).
      Pair (4, 11): r = 0.9999508 — near-sync but not exact.
      mean_abs_r = 0.42 (moderate global synchrony).

  ГОРА / РОТА XOR3  (P=2):
      With only 2 time steps every pair of active cells has |r| = 1:
      either they oscillate in phase (r=+1) or anti-phase (r=−1).
      This creates two large synchrony blocks.

  ТУМАН XOR3  (P=8):
      No perfect pairs; highest r = 0.971 between cells (1,15) —
      near-perfect but not exact, reflecting slight phase offset.
      mean_abs_r ≈ 0.50.

  XOR rule  (P=1, all zeros):
      Every cell is constant → all off-diagonal r undefined (None).
      n_defined = 0.

  AND / OR  (typically P=1 or P=2):
      P=1 → all undefined; P=2 → ±1 blocks as for ГОРА.

Terminology
──────────────────────────────────────────────────────────────────────────────
  sync pair    : r(i, j) ≥  1 − ε   (perfectly correlated, in-phase)
  antisync pair: r(i, j) ≤ −1 + ε   (perfectly anti-correlated, anti-phase)
  ε = 1e-9

Запуск:
    python3 -m projects.hexglyph.solan_cross --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_cross --word ТУМАН --rule xor3
    python3 -m projects.hexglyph.solan_cross --table --rule xor3
    python3 -m projects.hexglyph.solan_cross --json --word МАТ
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES          = tuple(_ALL_RULES)
_DEFAULT_WIDTH = 16
_SYNC_EPS      = 1e-9


# ── Pearson correlation ───────────────────────────────────────────────────────

def pearson(xs: list[int], ys: list[int]) -> float | None:
    """Pearson correlation between two equal-length sequences.

    Returns None when either sequence is constant (σ = 0).
    """
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num  = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x < 1e-20 or var_y < 1e-20:
        return None
    return num / math.sqrt(var_x * var_y)


# ── Core computation ──────────────────────────────────────────────────────────

def cross_corr_matrix(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[list[float | None]]:
    """N×N Pearson cross-correlation matrix between cells' temporal sequences.

    Returns a list of N lists, each of length N.
    Diagonal entries are 1.0 (or None if cell is constant).
    Off-diagonal entries are Pearson r ∈ [−1, 1] or None.
    """
    from projects.hexglyph.solan_transfer import get_orbit

    orbit = get_orbit(word, rule, width)
    P     = len(orbit)
    N     = width
    seqs  = [[int(orbit[t][i]) for t in range(P)] for i in range(N)]

    mat: list[list[float | None]] = []
    for i in range(N):
        row: list[float | None] = []
        for j in range(N):
            if i == j:
                # Diagonal: 1 if active, None if constant
                var_i = sum((v - sum(seqs[i]) / P) ** 2 for v in seqs[i])
                row.append(1.0 if var_i > 1e-20 else None)
            else:
                row.append(pearson(seqs[i], seqs[j]))
        mat.append(row)
    return mat


# ── Per-word summary ──────────────────────────────────────────────────────────

def cross_summary(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Pairwise cross-correlation statistics for one word/rule.

    Keys
    ────
    word, rule, period, n_cells
    matrix          : list[list[float|None]]  — N×N correlation matrix

    n_sync_pairs    : int    — pairs with r ≥ 1−ε
    n_antisync_pairs: int    — pairs with r ≤ −1+ε
    sync_pairs      : list[(i,j)]   — ordered, i < j
    antisync_pairs  : list[(i,j)]

    max_r           : float  — maximum off-diagonal r (or None if all undefined)
    max_r_pair      : tuple  — (i, j) achieving max_r
    min_r           : float  — minimum off-diagonal r
    min_r_pair      : tuple
    mean_abs_r      : float  — mean of |r| over all off-diagonal defined pairs
    n_defined       : int    — number of off-diagonal defined pairs

    spatial_decay   : list[float]   — mean |r| at circular lag d=1…N//2
    n_frozen_cells  : int           — cells with constant sequence (σ=0)
    frozen_cells    : list[int]
    """
    from projects.hexglyph.solan_transfer import get_orbit

    orbit = get_orbit(word, rule, width)
    P     = len(orbit)
    N     = width
    seqs  = [[int(orbit[t][i]) for t in range(P)] for i in range(N)]

    # Build matrix
    mat = cross_corr_matrix(word, rule, width)

    # Identify frozen cells
    frozen = [i for i in range(N) if mat[i][i] is None]

    # Off-diagonal statistics
    sync_pairs:     list[tuple[int, int]] = []
    antisync_pairs: list[tuple[int, int]] = []
    off_vals:       list[float]           = []
    max_r:   float | None = None
    min_r:   float | None = None
    max_pair: tuple[int, int] | None = None
    min_pair: tuple[int, int] | None = None

    for i in range(N):
        for j in range(i + 1, N):
            r = mat[i][j]
            if r is None:
                continue
            off_vals.append(r)
            if max_r is None or r > max_r:
                max_r = r;  max_pair = (i, j)
            if min_r is None or r < min_r:
                min_r = r;  min_pair = (i, j)
            if r >= 1.0 - _SYNC_EPS:
                sync_pairs.append((i, j))
            elif r <= -1.0 + _SYNC_EPS:
                antisync_pairs.append((i, j))

    n_defined  = len(off_vals)
    mean_abs_r = (sum(abs(v) for v in off_vals) / n_defined
                  if n_defined > 0 else 0.0)

    # Spatial decay: circular lag d = 1 … N//2
    from collections import defaultdict
    lag_vals: dict[int, list[float]] = defaultdict(list)
    for i in range(N):
        for j in range(i + 1, N):
            r = mat[i][j]
            if r is None:
                continue
            d = min(j - i, N - (j - i))
            lag_vals[d].append(abs(r))

    spatial_decay = []
    for d in range(1, N // 2 + 1):
        vals = lag_vals[d]
        spatial_decay.append(round(sum(vals) / len(vals), 6) if vals else 0.0)

    return {
        'word':             word,
        'rule':             rule,
        'period':           P,
        'n_cells':          N,
        'matrix':           mat,

        'n_sync_pairs':     len(sync_pairs),
        'n_antisync_pairs': len(antisync_pairs),
        'sync_pairs':       sync_pairs,
        'antisync_pairs':   antisync_pairs,

        'max_r':     round(max_r,  6) if max_r  is not None else None,
        'max_r_pair': max_pair,
        'min_r':     round(min_r,  6) if min_r  is not None else None,
        'min_r_pair': min_pair,
        'mean_abs_r': round(mean_abs_r, 6),
        'n_defined':  n_defined,

        'spatial_decay':  spatial_decay,
        'n_frozen_cells': len(frozen),
        'frozen_cells':   frozen,
    }


def all_cross(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict[str, Any]]:
    """cross_summary for all 4 CA rules."""
    return {r: cross_summary(word, r, width) for r in RULES}


def build_cross_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Full cross-correlation analysis for the lexicon."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)
    return {
        'words': list(words),
        'data':  {w: {r: cross_summary(w, r, width) for r in RULES}
                  for w in words},
    }


def cross_dict(s: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of cross_summary."""
    d = dict(s)
    # Replace None in matrix with a sentinel (float 'nan' is not JSON-legal)
    d['matrix'] = [
        [v if v is not None else 'null' for v in row]
        for row in s['matrix']
    ]
    d['sync_pairs']     = [list(p) for p in s['sync_pairs']]
    d['antisync_pairs'] = [list(p) for p in s['antisync_pairs']]
    d['max_r_pair']     = list(s['max_r_pair'])  if s['max_r_pair']  else None
    d['min_r_pair']     = list(s['min_r_pair'])  if s['min_r_pair']  else None
    return d


# ── Terminal output ───────────────────────────────────────────────────────────

# ANSI colour ramp: blue (−1) → dim white (0) → orange (+1)
_RAMP_NEG = ['\033[38;5;27m', '\033[38;5;33m', '\033[38;5;39m',
             '\033[38;5;45m', '\033[38;5;51m']
_RAMP_POS = ['\033[38;5;214m', '\033[38;5;208m', '\033[38;5;202m',
             '\033[38;5;196m', '\033[38;5;160m']


def _cell_char(r: float | None, color: bool) -> str:
    """Single-character cell representation of a correlation value."""
    if r is None:
        return '·' if not color else '\033[38;5;235m·\033[0m'
    if abs(r) < 0.05:
        return '0'
    idx = min(int(abs(r) / 0.2), 4)
    if r > 0:
        sym = ['░', '▒', '▓', '█', '█'][idx]
        col = _RAMP_POS[idx] if color else ''
    else:
        sym = ['░', '▒', '▓', '█', '█'][idx]
        col = _RAMP_NEG[idx] if color else ''
    return (col + sym + '\033[0m') if color else sym


def print_cross(
    word:  str,
    rule:  str,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print pairwise cross-correlation heatmap for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    s = cross_summary(word, rule, width)
    P = s['period']
    N = s['n_cells']

    print(bold + f"  ◈ Cell Cross-Correlation  {word.upper()}  "
          + col + lbl + reset + bold + f"  (P={P})" + reset)
    print()
    print(f"  N×N Pearson correlation matrix  "
          f"(orange=+1, blue=−1, ·=undefined/frozen):")
    print()

    # Header row
    print('  ' + ' ' * 3 + ''.join(f'{j:2d}' for j in range(N)))
    print('  ' + ' ' * 2 + '─' * (N * 2 + 1))

    for i in range(N):
        row_str = ''.join(_cell_char(s['matrix'][i][j], color) + ' '
                          for j in range(N))
        print(f"  {i:2d}│{row_str}")

    print()
    print(f"  n_defined      : {s['n_defined']} off-diagonal pairs")
    print(f"  mean |r|       : {s['mean_abs_r']:.4f}")
    if s['max_r'] is not None:
        print(f"  max r          : {s['max_r']:+.4f}  pair={s['max_r_pair']}")
        print(f"  min r          : {s['min_r']:+.4f}  pair={s['min_r_pair']}")
    print(f"  sync pairs     : {s['n_sync_pairs']}  {s['sync_pairs']}")
    print(f"  antisync pairs : {s['n_antisync_pairs']}  {s['antisync_pairs']}")
    if s['frozen_cells']:
        print(f"  frozen cells   : {s['frozen_cells']}")
    print()

    # Spatial decay
    if any(v > 0 for v in s['spatial_decay']):
        print(f"  Spatial decay (mean |r| vs circular lag):")
        bar_max = max(s['spatial_decay']) or 1.0
        for d, v in enumerate(s['spatial_decay'], start=1):
            bar = '█' * max(1, round(v / bar_max * 20))
            print(f"    d={d:2d}  {bar:<20s}  {v:.4f}")
        print()


def print_cross_table(
    words: list[str] | None = None,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Summary table: cross-correlation stats for all lexicon words."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Cell Cross-Correlation Summary ({lbl}, n={len(words)})" + reset)
    print()
    print(f"  {'Слово':12s}  {'P':>3}  {'n_sync':>6}  {'n_anti':>6}  "
          f"{'max_r':>6}  {'min_r':>6}  {'mean|r|':>7}  {'frozen':>6}")
    print('  ' + '─' * 70)

    for word in words:
        s = cross_summary(word, rule, width)
        mr = f"{s['max_r']:+.3f}" if s['max_r'] is not None else '  n/a'
        mn = f"{s['min_r']:+.3f}" if s['min_r'] is not None else '  n/a'
        print(f"  {word.upper():12s}  {s['period']:>3}  "
              f"{s['n_sync_pairs']:>6}  {s['n_antisync_pairs']:>6}  "
              f"{mr:>6}  {mn:>6}  {s['mean_abs_r']:>7.4f}  "
              f"{s['n_frozen_cells']:>6}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pairwise cell Q6 cross-correlation analysis')
    parser.add_argument('--word',   metavar='WORD', default='МАТ')
    parser.add_argument('--rule',   choices=list(RULES), default='xor3')
    parser.add_argument('--table',  action='store_true')
    parser.add_argument('--json',   action='store_true')
    parser.add_argument('--width',  type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = cross_summary(args.word.upper(), args.rule, args.width)
        print(json.dumps(cross_dict(s), ensure_ascii=False, indent=2))
    elif args.table:
        print_cross_table(rule=args.rule, width=args.width, color=_color)
    else:
        print_cross(args.word.upper(), args.rule, args.width, _color)
