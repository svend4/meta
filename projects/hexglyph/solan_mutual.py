"""solan_mutual.py — Mutual Information Analysis of Q6 CA attractor orbits.

For each word and rule, analyses the information shared between pairs of CA cells
over the attractor cycle (periodic orbit).

    I(X_i ; X_j) = H(X_i) + H(X_j) − H(X_i, X_j)   [bits, base-2 log]

Probability distributions are computed from the exact orbit:
    P(X_i = v) = (number of attractor steps where cell i = v) / period

Typical results (width=16):
    XOR   → period=1: all cells constant → H=0, MI=0 everywhere
    XOR3  → period=2: H ≤ 1 bit per cell; MI=1 if both cells vary, else 0
    XOR3  → period=8 (ТУМАН): H ≤ 3 bits; rich off-diagonal MI structure
    AND   → similar short periods; partial structure depending on attractor
    OR    → similar to AND; often non-zero MI near boundaries

MI profile by distance:
    mi_profile(M, width) → list of length width//2 + 1
    element d = average MI between cells at circular distance d

Функции:
    attractor_states(word, rule, width)       → list[list[int]]
    cell_entropy(states, idx)                 → float   (bits)
    cell_mi(states, i, j)                     → float   (bits)
    entropy_profile(word, rule, width)        → list[float]
    mi_matrix(word, rule, width)              → list[list[float]]
    mi_profile(mi_mat, width)                 → list[float]
    trajectory_mutual(word, rule, width)      → dict
    all_mutual(word, width)                   → dict[str, dict]
    build_mutual_data(words, width)           → dict
    mutual_dict(word, width)                  → dict
    print_mutual(word, rule, width, color)    → None
    print_mutual_stats(words, width, color)   → None

Запуск:
    python3 -m projects.hexglyph.solan_mutual --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_mutual --word ТУМАН --all-rules --no-color
    python3 -m projects.hexglyph.solan_mutual --stats
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    step, find_orbit,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_N_BITS        = 6           # bits per Q6 cell
_MAX_MI        = _N_BITS     # upper bound: MI ≤ H(X) ≤ log2(64) = 6 bits
_DEFAULT_WIDTH = 16
_DEFAULT_WORDS = list(LEXICON)

_SHADE = [' ', '░', '▒', '▓', '█']    # 5 shade levels for ASCII heatmap


# ── Information primitives ────────────────────────────────────────────────────

def _entropy_from_counter(counts: Counter, n: int) -> float:
    """Shannon entropy (bits) from a frequency Counter."""
    if n == 0:
        return 0.0
    total = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            total -= p * math.log2(p)
    return total


def cell_entropy(states: list[list[int]], idx: int) -> float:
    """Shannon entropy in bits of cell *idx* over the attractor cycle.

    For a period-P attractor each step contributes 1/P probability to its value.
    Result lies in [0, log2(P)] (at most log2(P) if all P values are distinct).
    """
    vals = [s[idx] for s in states]
    return _entropy_from_counter(Counter(vals), len(vals))


def cell_mi(states: list[list[int]], i: int, j: int) -> float:
    """Mutual Information (bits) between cells *i* and *j* over the attractor.

    I(X_i ; X_j) = H(X_i) + H(X_j) − H(X_i, X_j)

    Note: cell_mi(states, i, i) == cell_entropy(states, i).
    """
    n = len(states)
    if n == 0:
        return 0.0
    vals_i  = [s[i] for s in states]
    vals_j  = [s[j] for s in states]
    pairs   = list(zip(vals_i, vals_j))
    h_i     = _entropy_from_counter(Counter(vals_i),  n)
    h_j     = _entropy_from_counter(Counter(vals_j),  n)
    h_ij    = _entropy_from_counter(Counter(pairs),   n)
    return max(0.0, h_i + h_j - h_ij)   # max(0,…) guards against float rounding


# ── Attractor orbit ───────────────────────────────────────────────────────────

def attractor_states(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[list[int]]:
    """States in the attractor cycle (length = period, ≥ 1)."""
    cells = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells, rule)
    period = max(period, 1)
    c = cells[:]
    for _ in range(transient):
        c = step(c, rule)
    states: list[list[int]] = []
    for _ in range(period):
        states.append(c[:])
        c = step(c, rule)
    return states


# ── Per-word analysis ─────────────────────────────────────────────────────────

def entropy_profile(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[float]:
    """Per-cell entropy (bits) for all *width* cells."""
    states = attractor_states(word, rule, width)
    return [cell_entropy(states, i) for i in range(width)]


def mi_matrix(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[list[float]]:
    """*width* × *width* matrix of pairwise MI values (bits).

    M[i][j] = I(X_i ; X_j).
    Diagonal M[i][i] = H(X_i) (entropy of cell i).
    Matrix is symmetric.
    """
    states = attractor_states(word, rule, width)
    M: list[list[float]] = []
    for i in range(width):
        row: list[float] = []
        for j in range(width):
            row.append(round(cell_mi(states, i, j), 6))
        M.append(row)
    return M


def mi_profile(
    mi_mat: list[list[float]],
    width:  int = _DEFAULT_WIDTH,
) -> list[float]:
    """Average MI between cells at each circular distance d = 0 … width//2.

    Element d = mean of M[i][j] for all pairs (i,j) with
    circular distance min(|i-j|, width-|i-j|) == d.
    """
    d_max  = width // 2
    totals = [0.0] * (d_max + 1)
    counts = [0  ] * (d_max + 1)
    for i in range(width):
        for j in range(width):
            diff = abs(i - j)
            d    = min(diff, width - diff)
            if d <= d_max:
                totals[d] += mi_mat[i][j]
                counts[d] += 1
    return [
        round(totals[d] / counts[d], 6) if counts[d] else 0.0
        for d in range(d_max + 1)
    ]


# ── Trajectory summary dict ───────────────────────────────────────────────────

def trajectory_mutual(
    word:  str,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
) -> dict:
    """Full MI analysis for one word + rule.

    Returns dict:
        word          : str
        rule          : str
        width         : int
        period        : int
        entropy       : list[float]   — per-cell entropy (length = width)
        M             : list[list[float]] — width × width MI matrix
        mi_by_dist    : list[float]   — avg MI at each circular distance
        mean_entropy  : float
        max_mi        : float         — max off-diagonal MI
        max_mi_pair   : tuple[int,int]
    """
    states  = attractor_states(word, rule, width)
    period  = len(states)
    M       = mi_matrix(word, rule, width)
    ent     = [M[i][i] for i in range(width)]
    mi_dist = mi_profile(M, width)

    max_mi   = 0.0
    best_ij  = (0, 0)
    for i in range(width):
        for j in range(width):
            if i != j and M[i][j] > max_mi:
                max_mi  = M[i][j]
                best_ij = (i, j)

    return {
        'word':         word.upper(),
        'rule':         rule,
        'width':        width,
        'period':       period,
        'entropy':      ent,
        'M':            M,
        'mi_by_dist':   mi_dist,
        'mean_entropy': round(sum(ent) / len(ent), 6) if ent else 0.0,
        'max_mi':       round(max_mi, 6),
        'max_mi_pair':  best_ij,
    }


def all_mutual(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict]:
    """trajectory_mutual for all 4 rules."""
    return {r: trajectory_mutual(word, r, width) for r in _ALL_RULES}


# ── Full dataset ───────────────────────────────────────────────────────────────

def build_mutual_data(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
) -> dict:
    """MI summary for the full lexicon × 4 rules.

    Returns dict:
        words        : list[str]
        width        : int
        per_rule     : {rule: {word: {period, mean_entropy, max_mi, max_mi_pair}}}
        ranking      : {rule: [(word, mean_entropy), …]} descending by mean_entropy
        max_h        : {rule: (word, mean_entropy)}
        min_h        : {rule: (word, mean_entropy)}
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            tr = trajectory_mutual(word, rule, width)
            per_rule[rule][word] = {
                'period':       tr['period'],
                'mean_entropy': tr['mean_entropy'],
                'max_mi':       tr['max_mi'],
                'max_mi_pair':  tr['max_mi_pair'],
            }

    ranking: dict[str, list] = {}
    max_h:   dict[str, tuple] = {}
    min_h:   dict[str, tuple] = {}
    for rule in _ALL_RULES:
        by_h = sorted(
            ((w, d['mean_entropy']) for w, d in per_rule[rule].items()),
            key=lambda x: -x[1],
        )
        ranking[rule] = by_h
        max_h[rule]   = by_h[0]
        min_h[rule]   = by_h[-1]

    return {
        'words':    words,
        'width':    width,
        'per_rule': per_rule,
        'ranking':  ranking,
        'max_h':    max_h,
        'min_h':    min_h,
    }


# ── JSON export ────────────────────────────────────────────────────────────────

def mutual_dict(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """JSON-serialisable MI summary (M matrix excluded)."""
    result: dict[str, object] = {
        'word':  word.upper(),
        'width': width,
        'rules': {},
    }
    for rule in _ALL_RULES:
        tr = trajectory_mutual(word, rule, width)
        result['rules'][rule] = {          # type: ignore[index]
            'period':       tr['period'],
            'mean_entropy': tr['mean_entropy'],
            'max_mi':       tr['max_mi'],
            'max_mi_pair':  list(tr['max_mi_pair']),
            'entropy':      tr['entropy'],
            'mi_by_dist':   tr['mi_by_dist'],
        }
    return result


# ── ASCII display ──────────────────────────────────────────────────────────────

def print_mutual(
    word:  str,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print MI matrix (ASCII heatmap) and per-cell entropy bars."""
    tr   = trajectory_mutual(word, rule, width)
    M    = tr['M']
    ent  = tr['entropy']
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    max_val = max(M[i][j] for i in range(width) for j in range(width))
    if max_val < 1e-9:
        max_val = 1.0

    print(f"{bold}  ◈ Взаимная информация Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  "
          f"(P={tr['period']}, H̄={tr['mean_entropy']:.3f} бит)")
    print(f"  {'─' * (width + 6)}")

    for i in range(width):
        parts: list[str] = []
        for j in range(width):
            v    = M[i][j]
            lvl  = min(int(v / max_val * (len(_SHADE) - 1) + 0.5), len(_SHADE) - 1)
            ch   = _SHADE[lvl]
            if i == j:
                parts.append(f"{dim}{ch}{rst}")
            elif v > 0:
                parts.append(f"{col}{ch}{rst}")
            else:
                parts.append(ch)
        # entropy bar at right
        bar_len = min(int(ent[i] / max_val * 10 + 0.5), 10)
        bar     = '▪' * bar_len
        print(f"  {i:2d} {''.join(parts)}  {col}{bar:<10s}{rst}  {ent[i]:.3f}b")

    print(f"  {'─' * (width + 6)}")
    dist = tr['mi_by_dist']
    dist_str = '  '.join(f"d{d}={v:.2f}" for d, v in enumerate(dist[:9]))
    print(f"  MI(d): {dist_str}")
    print(f"  max MI={tr['max_mi']:.3f}b  пара={tr['max_mi_pair']}")
    print()


def print_mutual_stats(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
    color: bool             = True,
) -> None:
    """Сводная таблица средней энтропии для лексикона × 4 правила."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Средняя энтропия H̄ Q6 аттракторов (бит){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            tr  = trajectory_mutual(word, rule, width)
            h   = tr['mean_entropy']
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{h:>8.3f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Mutual Information Analysis Q6 CA')
    parser.add_argument('--word',      default='ГОРА',  help='Русское слово')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_mutual_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_mutual(args.word, rule, args.width, color)
    else:
        print_mutual(args.word, args.rule, args.width, color)


if __name__ == '__main__':
    _main()
