"""solan_multistep.py — Multi-step Hamming Distance Matrix of Q6 Attractor Orbits.

For a periodic orbit of period P the N-cell states s_0, …, s_{P-1}
form a metric space under Q6 (cell-level) Hamming distance:

    H[t1][t2] = |{i : orbit[t1][i] ≠ orbit[t2][i]}|   ∈ {0, …, N}

This P×P symmetric matrix reveals the full geometry of the orbit state space.

Key metrics
───────────
  diameter   : max H[t1][t2]  — most different pair of orbit states
  radius     : min_t max_{t'} H[t][t']  — eccentricity of most central state
  girth      : min_{t1≠t2} H[t1][t2]   — nearest non-identical pair
  orbit_spread: mean off-diagonal distance
  is_regular : diameter == radius  (all states equidistant to one another)

Relation to solan_hamming.py
────────────────────────────
  consecutive_hamming[t] = H[t][(t+1)%P]   — the lag-1 super-diagonal.
  solan_multistep adds all P² pairs, revealing multi-scale orbit geometry.

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1)           : 1×1 matrix [0]; diameter=radius=girth=0.
  ГОРА  AND  (P=2, 47↔1)    : H[0][1]=16; diameter=radius=girth=16;
                               is_regular=True; orbit_spread=16.0.
  P=2 theorem                : ALL P=2 orbits satisfy H[0][1]=N=16
                               (both states differ in every cell).
  ТУМАН XOR3 (P=8)           : diameter=16, radius=16, girth=6,
                               orbit_spread≈13.93; is_regular=True.
                               30/56 off-diagonal pairs = 16 (max).
  МАТ XOR3   (P=8)           : girth=4, orbit_spread≈9.71 — most compact
                               orbit geometry among P=8 XOR3 words.
  General P≥2                : diameter always ≤ N = 16.

Запуск:
    python3 -m projects.hexglyph.solan_multistep --word ТУМАН --rule xor3
    python3 -m projects.hexglyph.solan_multistep --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_multistep --table --no-color
    python3 -m projects.hexglyph.solan_multistep --json --word ГОРА
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_ca      import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES = tuple(_ALL_RULES)   # ('xor', 'xor3', 'and', 'or')


# ── Core computation ──────────────────────────────────────────────────────────

def hamming_dist(a: list[int], b: list[int]) -> int:
    """Q6 cell-level Hamming distance: number of cells that differ."""
    return sum(1 for x, y in zip(a, b) if x != y)


def orbit_dist_matrix(orbit: list[tuple]) -> list[list[int]]:
    """P×P symmetric Hamming distance matrix for the orbit states.

    dist_matrix[t1][t2] = number of cells that differ between
    orbit[t1] and orbit[t2].  Diagonal is always 0.
    """
    P = len(orbit)
    mat: list[list[int]] = [[0] * P for _ in range(P)]
    for t1 in range(P):
        for t2 in range(t1 + 1, P):
            d = hamming_dist(orbit[t1], orbit[t2])
            mat[t1][t2] = d
            mat[t2][t1] = d
    return mat


def eccentricity(dist_mat: list[list[int]]) -> list[int]:
    """Per-state eccentricity: max distance from state t to any other state."""
    P = len(dist_mat)
    return [max(dist_mat[t]) for t in range(P)]


def center_steps(dist_mat: list[list[int]]) -> list[int]:
    """Indices of states with minimum eccentricity (the metric 'center')."""
    ecc = eccentricity(dist_mat)
    rad = min(ecc)
    return [t for t, e in enumerate(ecc) if e == rad]


def periphery_steps(dist_mat: list[list[int]]) -> list[int]:
    """Indices of states with maximum eccentricity (the metric 'periphery')."""
    ecc = eccentricity(dist_mat)
    diam = max(ecc)
    return [t for t, e in enumerate(ecc) if e == diam]


def dist_histogram(dist_mat: list[list[int]]) -> dict[int, int]:
    """Frequency distribution of off-diagonal Hamming distances."""
    P = len(dist_mat)
    hist: dict[int, int] = {}
    for t1 in range(P):
        for t2 in range(P):
            if t1 != t2:
                d = dist_mat[t1][t2]
                hist[d] = hist.get(d, 0) + 1
    return hist


# ── Per-word summary ──────────────────────────────────────────────────────────

def multistep_summary(
    word:  str,
    rule:  str,
    width: int = 16,
) -> dict[str, Any]:
    """Full multi-step Hamming analysis for one word/rule.

    Keys
    ────
    word, rule, period, n_cells
    dist_matrix      : list[list[int]]  — P×P
    eccentricity     : list[int]        — per-state max distance
    diameter         : int
    radius           : int
    center_steps     : list[int]
    periphery_steps  : list[int]
    girth            : int              — min off-diagonal distance (0 if P=1)
    orbit_spread     : float            — mean off-diagonal distance
    dist_histogram   : dict[int,int]   — distribution of off-diagonal distances
    is_regular       : bool             — diameter == radius
    """
    from projects.hexglyph.solan_perm import get_orbit

    orbit = get_orbit(word, rule, width)
    P     = len(orbit)
    N     = width

    mat  = orbit_dist_matrix(orbit)
    ecc  = eccentricity(mat)
    diam = max(ecc)
    rad  = min(ecc)
    ctrs = center_steps(mat)
    peri = periphery_steps(mat)

    off_diag = [mat[t1][t2] for t1 in range(P) for t2 in range(P) if t1 != t2]
    girth  = min(off_diag) if off_diag else 0
    spread = sum(off_diag) / len(off_diag) if off_diag else 0.0
    hist   = dist_histogram(mat)

    return {
        'word':          word,
        'rule':          rule,
        'period':        P,
        'n_cells':       N,
        'dist_matrix':   mat,
        'eccentricity':  ecc,
        'diameter':      diam,
        'radius':        rad,
        'center_steps':  ctrs,
        'periphery_steps': peri,
        'girth':         girth,
        'orbit_spread':  round(spread, 6),
        'dist_histogram': hist,
        'is_regular':    diam == rad,
    }


def all_multistep(
    word:  str,
    width: int = 16,
) -> dict[str, dict[str, Any]]:
    """Run multistep_summary for all 4 rules."""
    return {r: multistep_summary(word, r, width) for r in RULES}


def build_multistep_data(
    words: list[str] | None = None,
    width: int = 16,
) -> dict[str, Any]:
    """Full analysis for the entire lexicon.

    Returns dict:
      'words': [...],
      'data':  dict[word → {rule → summary}]
    """
    if words is None:
        words = list(LEXICON)
    return {
        'words': list(words),
        'data':  {w: all_multistep(w, width) for w in words},
    }


def multistep_dict(summary: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of multistep_summary."""
    h = summary['dist_histogram']
    return {
        'word':            summary['word'],
        'rule':            summary['rule'],
        'period':          summary['period'],
        'n_cells':         summary['n_cells'],
        'dist_matrix':     summary['dist_matrix'],
        'eccentricity':    summary['eccentricity'],
        'diameter':        summary['diameter'],
        'radius':          summary['radius'],
        'center_steps':    summary['center_steps'],
        'periphery_steps': summary['periphery_steps'],
        'girth':           summary['girth'],
        'orbit_spread':    summary['orbit_spread'],
        'dist_histogram':  {str(k): v for k, v in sorted(h.items())},
        'is_regular':      summary['is_regular'],
    }


# ── Terminal output ───────────────────────────────────────────────────────────

def print_multistep(
    word:  str,
    rule:  str,
    width: int  = 16,
    color: bool = True,
) -> None:
    """Print multi-step Hamming matrix and summary for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    s     = multistep_summary(word, rule, width)
    P     = s['period']
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Multi-step Hamming  {word.upper()}  {col}{lbl}{reset}"
          + bold + f"  (P={P}, N={s['n_cells']})" + reset)
    print()

    if P == 1:
        print(f"  {dim}P=1: fixed point — all distances trivially 0.{reset}")
        print()
        return

    # Distance matrix
    print(f"  {'':4s}" + ''.join(f"  t{t:1d}" for t in range(P)))
    print('  ' + '─' * (4 + P * 4))
    for t1 in range(P):
        row = f"  t{t1:1d} │"
        for t2 in range(P):
            d = s['dist_matrix'][t1][t2]
            if t1 == t2:
                row += f"  {dim}·{reset} " if color else '   · '
            elif d == s['diameter']:
                row += f" {bold}{d:2d}{reset} " if color else f' {d:2d} '
            else:
                row += f"  {d:2d} "
        print(row)
    print()

    # Metrics
    print(f"  diameter    = {bold}{s['diameter']}{reset}   "
          f"(most different pair of orbit states)")
    print(f"  radius      = {bold}{s['radius']}{reset}   "
          f"({'regular — all equidistant' if s['is_regular'] else 'center steps: ' + str(s['center_steps'])})")
    print(f"  girth       = {bold}{s['girth']}{reset}   "
          f"(nearest non-identical orbit-state pair)")
    print(f"  spread      = {bold}{s['orbit_spread']:.4f}{reset}   "
          f"(mean off-diagonal distance)")

    # Histogram
    hist = s['dist_histogram']
    if hist:
        print(f"\n  Distance distribution (off-diagonal {P*(P-1)} pairs):")
        for d, cnt in sorted(hist.items()):
            bar = '█' * min(cnt, 30)
            print(f"    d={d:2d} │{bar}│ {cnt}")
    print()


def print_multistep_table(
    words: list[str] | None = None,
    width: int  = 16,
    color: bool = True,
) -> None:
    """Table: key orbit geometry metrics for all words × rules."""
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    print(bold + f"  ◈ Multi-step Hamming — orbit geometry  n={len(words)}" + reset)
    print()

    # Header
    rule_hdrs = '  '.join(
        (_RULE_COLOR.get(r, '') if color else '') + f"{_RULE_NAMES.get(r,r):5s}" + reset
        for r in RULES
    )
    print(f"  {'Слово':12s}  {'P':>3}  girth  spread  "
          f"diam  {'':3s}  [{rule_hdrs}  spread]")
    print('  ' + '─' * 90)

    for word in words:
        data = all_multistep(word, width)
        # Show XOR3 details prominently
        s3   = data['xor3']
        P    = s3['period']
        col3 = (_RULE_COLOR.get('xor3', '') if color else '')
        parts = []
        for r in RULES:
            s = data[r]
            col = (_RULE_COLOR.get(r, '') if color else '')
            if s['period'] == 1:
                parts.append(f"{dim}  — {reset}")
            else:
                sp = s['orbit_spread']
                parts.append(f"{col}{sp:5.2f}{reset}")
        spreads = '  '.join(parts)
        g3   = s3['girth']
        sp3  = s3['orbit_spread']
        d3   = s3['diameter']
        reg  = ('✓' if s3['is_regular'] else '·') if color else \
               ('Y' if s3['is_regular'] else 'N')
        print(f"  {word.upper():12s}  {P:>3}  "
              f"{col3}{g3:5d}  {sp3:6.2f}  {d3:4d}  {reg:3s}{reset}  "
              f"[{spreads}]")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-step Hamming distance matrix for Q6 CA orbits')
    parser.add_argument('--word',  metavar='WORD', default='ТУМАН')
    parser.add_argument('--rule',  choices=list(RULES), default='xor3')
    parser.add_argument('--table', action='store_true',
                        help='print table for full lexicon')
    parser.add_argument('--json',  action='store_true')
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = multistep_summary(args.word.upper(), args.rule, args.width)
        print(json.dumps(multistep_dict(s), ensure_ascii=False, indent=2))
    elif args.table:
        print_multistep_table(width=args.width, color=_color)
    else:
        print_multistep(args.word.upper(), args.rule, args.width, color=_color)
