"""solan_active.py — Active Information Storage (AIS) of Q6 CA.

Active Information Storage measures how much of each cell's past (k steps)
is predictive of its own next state — the cell's local "memory capacity".

Definition
──────────
For cell i with periodic binary sequence x_i(0) … x_i(P−1) (threshold=32):

    AIS_k(i) = MI( X_i(t+1) ; X_i(t), X_i(t−1), …, X_i(t−k+1) )
             = H_1 − ( H_{k+1} − H_k )

where H_n = Shannon entropy (bits) of the circular n-gram distribution
computed from the periodic attractor sequence.

AIS_k ∈ [0, H_1] ⊆ [0, 1 bit]  (binary alphabet).

Key results
───────────
  k = 2 is the default (2 past steps as context):

  • AND/OR fixed-point attractors : H_1 = 0  ⟹  AIS = 0 (trivial, no memory)
  • Period-2 alternating seqs     : H_1 = 1, H_2 = 1  ⟹  AIS_1 = 1 bit (perfect)
  • XOR3 ТУМАН P = 8             : H_1 ≈ 1 bit, AIS_2 ∈ (0, 1)
    (the 2-step past provides partial but not full prediction)
  • XOR  ТУМАН P = 1             : H_1 = 0, AIS = 0 (fixed point)

Complement to Transfer Entropy (solan_transfer.py):
  TE  : directed information BETWEEN cells (source → target)
  AIS : undirected information WITHIN each cell (past → future of same cell)

Functions
─────────
  cell_seqs(word, rule, width, threshold)   → list[list[int]]   N temporal seqs
  h_ngrams(seqs, n)                         → float             H_n in bits
  ais_k(seqs, k)                            → float             AIS at depth k
  ais_profile(word, rule, max_k, width)     → list[float]       AIS(k) k=1..max_k
  cell_ais_k(word, rule, k, width)          → list[float]       per-cell AIS
  ais_dict(word, rule, max_k, width)        → dict
  all_ais(word, max_k)                      → dict[str, dict]
  build_ais_data(words, max_k)              → dict
  print_ais(word, rule, color)              → None
  print_ais_stats(words, color)             → None

Запуск
──────
  python3 -m projects.hexglyph.solan_active --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_active --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_active --stats --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES      = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH  = 16
_DEFAULT_MAX_K  = 6
_DEFAULT_THRESH = 32


# ── Core building blocks ──────────────────────────────────────────────────────

def cell_seqs(
    word:      str,
    rule:      str,
    width:     int = _DEFAULT_WIDTH,
    threshold: int = _DEFAULT_THRESH,
) -> list[list[int]]:
    """Temporal binary sequences for all N cells on the periodic attractor.

    Returns a list of `width` sequences, each of length P (the orbit period).
    Sequences are circular — they represent one full attractor cycle.
    Binary coarsening: 0 if cell < threshold, 1 if cell ≥ threshold.
    """
    from projects.hexglyph.solan_symbolic import attractor_binary
    grid = attractor_binary(word, rule, width, threshold)   # P × N list
    return [[grid[t][i] for t in range(len(grid))] for i in range(width)]


def h_ngrams(seqs: list[list[int]], n: int) -> float:
    """Shannon entropy (bits) of the circular n-gram distribution.

    Pools n-grams from all sequences (with circular wrap).
    Returns max(0, -Σ p log₂ p).
    """
    counts: dict[tuple, int] = {}
    total = 0
    for seq in seqs:
        m = len(seq)
        for t in range(m):
            gram = tuple(seq[(t + i) % m] for i in range(n))
            counts[gram] = counts.get(gram, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    return max(0.0, -sum(
        (v / total) * math.log2(v / total) for v in counts.values() if v > 0
    ))


def ais_k(seqs: list[list[int]], k: int) -> float:
    """Active Information Storage at memory depth k.

    AIS_k = MI( X_{t+1} ; X_t, …, X_{t−k+1} )
           = H_1 − ( H_{k+1} − H_k )

    where H_n = h_ngrams(seqs, n).

    Clamped to [0, H_1] to avoid numerical negatives.
    Returns 0.0 for degenerate sequences (H_1 = 0).
    """
    h1  = h_ngrams(seqs, 1)
    hk  = h_ngrams(seqs, k)
    hk1 = h_ngrams(seqs, k + 1)
    cond = hk1 - hk          # H(X_{t+1} | X_{t−k+1},...,X_t)
    ais  = h1 - cond
    return max(0.0, min(round(ais, 6), round(h1, 6)))


# ── Higher-level analysis ─────────────────────────────────────────────────────

def ais_profile(
    word:   str,
    rule:   str,
    max_k:  int = _DEFAULT_MAX_K,
    width:  int = _DEFAULT_WIDTH,
) -> list[float]:
    """AIS(k) for k = 1 … max_k using all N temporal sequences pooled.

    Returns a list of max_k floats.
    Typically non-decreasing (more context → more information captured).
    Saturates at H_1 once k ≥ period (perfect predictor).
    """
    seqs = cell_seqs(word, rule, width)
    return [ais_k(seqs, k) for k in range(1, max_k + 1)]


def cell_ais_k(
    word:  str,
    rule:  str,
    k:     int = 2,
    width: int = _DEFAULT_WIDTH,
) -> list[float]:
    """Per-cell AIS(k) — computed from each cell's own temporal sequence.

    Returns a list of `width` floats ∈ [0, 1].
    High value: that cell's past is strongly predictive of its own future.
    Low value : cell dynamics are locally unpredictable from own history alone.
    """
    seqs = cell_seqs(word, rule, width)
    return [ais_k([s], k) for s in seqs]


def ais_dict(
    word:   str,
    rule:   str   = 'xor3',
    max_k:  int   = _DEFAULT_MAX_K,
    width:  int   = _DEFAULT_WIDTH,
) -> dict:
    """Full Active Information Storage analysis for one word × rule.

    Returns:
        word, rule, max_k
        h1              : float   H_1 = marginal binary entropy of all cells
        ais_profile     : list[float]  AIS(k) for k=1..max_k (pooled)
        ais_1           : float   AIS at depth 1
        ais_2           : float   AIS at depth 2
        cell_ais        : list[float]  per-cell AIS(k=2)
        total_ais       : float   sum of per-cell AIS(k=2)
        mean_ais        : float   total_ais / width
        max_cell_ais    : float   max per-cell value
        min_cell_ais    : float   min per-cell value
        ais_frac        : float   ais_2 / h1 ∈ [0,1]  (normalised global AIS)
    """
    word = word.upper()
    seqs = cell_seqs(word, rule, width)

    h1      = h_ngrams(seqs, 1)
    profile = [ais_k(seqs, k) for k in range(1, max_k + 1)]
    c_ais   = cell_ais_k(word, rule, k=2, width=width)

    total_ais = round(sum(c_ais), 6)
    mean_ais  = round(total_ais / width, 6) if width > 0 else 0.0
    ais2      = profile[1] if len(profile) >= 2 else profile[0] if profile else 0.0

    return {
        'word':         word,
        'rule':         rule,
        'max_k':        max_k,
        'h1':           round(h1, 6),
        'ais_profile':  profile,
        'ais_1':        profile[0] if profile else 0.0,
        'ais_2':        ais2,
        'cell_ais':     c_ais,
        'total_ais':    total_ais,
        'mean_ais':     mean_ais,
        'max_cell_ais': max(c_ais) if c_ais else 0.0,
        'min_cell_ais': min(c_ais) if c_ais else 0.0,
        'ais_frac':     round(ais2 / h1, 6) if h1 > 0 else 0.0,
    }


def all_ais(
    word:  str,
    max_k: int = _DEFAULT_MAX_K,
) -> dict[str, dict]:
    """ais_dict for all 4 rules."""
    return {r: ais_dict(word, r, max_k) for r in _ALL_RULES}


def build_ais_data(
    words: list[str] | None = None,
    max_k: int              = _DEFAULT_MAX_K,
) -> dict:
    """AIS summary across the lexicon × 4 rules.

    Returns:
        words, max_k,
        per_rule: {rule: {word: {h1, ais_2, ais_frac, mean_ais}}}
    """
    words = words if words is not None else list(LEXICON)
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = ais_dict(word, rule, max_k)
            per_rule[rule][word] = {
                'h1':       d['h1'],
                'ais_2':    d['ais_2'],
                'ais_frac': d['ais_frac'],
                'mean_ais': d['mean_ais'],
            }
    return {'words': words, 'max_k': max_k, 'per_rule': per_rule}


# ── ASCII / ANSI display ───────────────────────────────────────────────────────

_BAR_W = 20


def print_ais(
    word:  str  = 'ТУМАН',
    rule:  str  = 'xor3',
    color: bool = True,
) -> None:
    """Print AIS profile and per-cell values for one word × rule."""
    d    = ais_dict(word, rule)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    print(f"{bold}  ◈ AIS  {word.upper()}  |  "
          f"{col}{name}{rst}  H₁={d['h1']:.4f}  "
          f"AIS₂={d['ais_2']:.4f}  frac={d['ais_frac']:.3f}")
    print(f"  {'─' * 52}")
    print(f"  {'k':>3}  {'AIS(k)':>8}  profile")
    print(f"  {'─' * 52}")

    max_ais = max(d['ais_profile']) if d['ais_profile'] else 1.0
    for ki, v in enumerate(d['ais_profile'], start=1):
        bar_len = round(v / max_ais * _BAR_W) if max_ais > 0 else 0
        bar = (col if color else '') + '█' * bar_len + (rst if color else '') + \
              '░' * (_BAR_W - bar_len)
        print(f"  {ki:>3}  {v:>8.4f}  {bar}")

    print(f"\n  Per-cell AIS(k=2):")
    print(f"  ", end='')
    for ci, v in enumerate(d['cell_ais']):
        block = '█' if v > 0.5 else ('▓' if v > 0.25 else ('░' if v > 0 else '·'))
        print(f"{col if color else ''}{block}{rst if color else ''}", end='')
    print(f"  mean={d['mean_ais']:.4f}  total={d['total_ais']:.4f}")
    print()


def print_ais_stats(
    words: list[str] | None = None,
    color: bool             = True,
) -> None:
    """Table: mean AIS(k=2) per word × rule."""
    words = words if words is not None else list(LEXICON)
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Среднее AIS(k=2) по всему лексикону{rst}")
    print('  ' + '─' * (len(header) + 2))
    print('  ' + header)
    print('  ' + '─' * (len(header) + 2))
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = ais_dict(word, rule)
            v   = d['mean_ais']
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{v:>8.4f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Active Information Storage — Q6 CA')
    parser.add_argument('--word',      default='ТУМАН')
    parser.add_argument('--rule',      default='xor3', choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_ais_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_ais(args.word, rule, color)
    else:
        print_ais(args.word, args.rule, color)


if __name__ == '__main__':
    _main()
