"""solan_symbolic.py — Symbolic Dynamics of Q6 CA attractors.

Coarse-grains each Q6 cell value to a binary symbol:
    b = 1  if  v >= threshold  (default: 32 = half of 63)
    b = 0  otherwise

then analyses the binary sequences that appear on the periodic attractor.

Two dual perspectives:
  Temporal (per-cell): for each cell i, the binary sequence
      σ_i = [b_i(0), b_i(1), …, b_i(P-1)]   (periodic, length P)
  Spatial (per-step): for each attractor step t, the binary row
      ρ_t = [b_0(t), b_1(t), …, b_{N-1}(t)]  (length N)

Statistics extracted:
  ngram_dist(n)   : frequency table of all n-grams in all temporal sequences
  block_entropy(n): Shannon H of n-gram distribution (bits) for n = 1..max_n
  topological_h   : estimate of topological entropy = max_{n≥2} H_n / n
  transition_mat  : 2×2 symbol-to-symbol transition probability matrix
  forbidden(n)    : n-grams that never appear in any temporal sequence
  n_unique_temp   : number of distinct temporal patterns among N cells
  n_unique_spat   : number of distinct spatial patterns among P steps
  spatial_entropy : Shannon entropy of spatial pattern distribution
  temporal_entropy: Shannon entropy of temporal cell pattern distribution
  symbol_bias     : fraction of 1-symbols across entire attractor binary grid

Period-1 attractors (all-zeros or all-ones) → trivially ordered:
  H_n = 0 for all n, topological_h = 0, no forbidden patterns at n=1.

Функции:
    binarize(v, threshold)                               → int
    attractor_binary(word, rule, width, threshold)       → list[list[int]]
    ngrams(seq, n, circular)                             → list[tuple[int, …]]
    ngram_dist(seqs, n)                                  → dict[tuple, int]
    block_entropy_n(seqs, n)                             → float
    block_entropy_profile(seqs, max_n)                   → list[float]
    transition_matrix(seqs)                              → list[list[float]]
    symbolic_dict(word, rule, width, threshold, max_n)   → dict
    all_symbolic(word, threshold, max_n)                 → dict[str, dict]
    build_symbolic_data(words, threshold)                → dict
    print_symbolic(word, rule, threshold, color)         → None
    print_symbolic_stats(words, color)                   → None

Запуск:
    python3 -m projects.hexglyph.solan_symbolic --word ТУМАН --rule xor3
    python3 -m projects.hexglyph.solan_symbolic --word ГОРА --all-rules --no-color
    python3 -m projects.hexglyph.solan_symbolic --stats --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    step, find_orbit,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16
_DEFAULT_THR   = 32     # binarisation threshold
_DEFAULT_MAX_N = 6      # max n-gram length
_Q6_MAX        = 63


# ── Symbol coarse-graining ────────────────────────────────────────────────────

def binarize(v: int, threshold: int = _DEFAULT_THR) -> int:
    """Map Q6 value v to binary symbol: 1 if v >= threshold, else 0."""
    return 1 if v >= threshold else 0


def attractor_binary(
    word:      str,
    rule:      str,
    width:     int = _DEFAULT_WIDTH,
    threshold: int = _DEFAULT_THR,
) -> list[list[int]]:
    """P × N binary matrix of the periodic attractor.

    Returns a list of P rows (one per attractor step), each a list of N
    binary symbols.  P = period, N = width.
    For period-1 attractors the list has exactly 1 row.
    """
    cells             = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells[:], rule)
    period            = max(period, 1)
    # advance past the transient
    c = cells[:]
    for _ in range(transient):
        c = step(c, rule)
    # collect one full period
    grid: list[list[int]] = []
    for _ in range(period):
        grid.append([binarize(v, threshold) for v in c])
        c = step(c, rule)
    return grid


# ── n-gram helpers ─────────────────────────────────────────────────────────────

def ngrams(
    seq:      list[int],
    n:        int,
    circular: bool = True,
) -> list[tuple[int, ...]]:
    """All n-grams of a sequence (circular = wrap around at the end)."""
    L = len(seq)
    if circular:
        return [tuple(seq[(i + k) % L] for k in range(n))
                for i in range(L)]
    return [tuple(seq[i:i + n]) for i in range(L - n + 1)]


def ngram_dist(
    seqs: list[list[int]],
    n:    int,
) -> dict[tuple[int, ...], int]:
    """Aggregate circular n-gram frequency counts over a list of sequences."""
    counts: dict[tuple[int, ...], int] = {}
    for seq in seqs:
        for gram in ngrams(seq, n):
            counts[gram] = counts.get(gram, 0) + 1
    return counts


def _entropy_from_counts(counts: dict) -> float:
    """Shannon entropy (bits) from a raw count dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(0.0, -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    ))


def block_entropy_n(seqs: list[list[int]], n: int) -> float:
    """Shannon entropy of the n-gram distribution across all sequences."""
    return round(_entropy_from_counts(ngram_dist(seqs, n)), 8)


def block_entropy_profile(
    seqs:  list[list[int]],
    max_n: int = _DEFAULT_MAX_N,
) -> list[float]:
    """H_n for n = 1, 2, …, max_n."""
    return [block_entropy_n(seqs, n) for n in range(1, max_n + 1)]


def transition_matrix(seqs: list[list[int]]) -> list[list[float]]:
    """2×2 symbol-to-symbol transition probability matrix (circular).

    T[a][b] = P(next = b | cur = a)  averaged over all sequences.
    Returns [[P(0→0), P(0→1)], [P(1→0), P(1→1)]].
    """
    counts = [[0, 0], [0, 0]]
    for seq in seqs:
        L = len(seq)
        for i in range(L):
            a = seq[i]
            b = seq[(i + 1) % L]
            counts[a][b] += 1
    mat = [[0.0, 0.0], [0.0, 0.0]]
    for a in range(2):
        row_sum = counts[a][0] + counts[a][1]
        if row_sum > 0:
            mat[a][0] = round(counts[a][0] / row_sum, 8)
            mat[a][1] = round(counts[a][1] / row_sum, 8)
    return mat


def forbidden_ngrams(
    seqs: list[list[int]],
    n:    int,
) -> list[tuple[int, ...]]:
    """Binary n-grams that never appear in any sequence (sorted)."""
    observed = set(ngram_dist(seqs, n).keys())
    all_grams = [
        tuple((k >> (n - 1 - i)) & 1 for i in range(n))
        for k in range(1 << n)
    ]
    return sorted(g for g in all_grams if g not in observed)


# ── Full analysis ─────────────────────────────────────────────────────────────

def symbolic_dict(
    word:      str,
    rule:      str,
    width:     int = _DEFAULT_WIDTH,
    threshold: int = _DEFAULT_THR,
    max_n:     int = _DEFAULT_MAX_N,
) -> dict:
    """Full symbolic dynamics analysis for one word + rule.

    Returns dict:
        word, rule, width, threshold, max_n
        period             : int
        transient          : int
        binary_grid        : list[list[int]]   P × width binary matrix
        temporal_seqs      : list[list[int]]   width × P  (per-cell sequences)
        spatial_seqs       : list[list[int]]   P × width  (= binary_grid)
        symbol_bias        : float  fraction of 1-symbols in binary_grid
        n_unique_temp      : int    distinct temporal patterns among N cells
        n_unique_spat      : int    distinct spatial patterns among P steps
        temporal_entropy   : float  H of temporal-pattern distribution
        spatial_entropy    : float  H of spatial-pattern distribution
        block_entropy      : list[float]   H_n for n=1..max_n
        topological_h      : float  max(H_n/n) for n=2..max_n
        transition_mat     : list[list[float]]
        forbidden_2grams   : list[tuple]
        forbidden_3grams   : list[tuple]
        ngram_1            : dict[str, int]   1-gram counts
        ngram_2            : dict[str, int]   2-gram counts
    """
    cells     = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells[:], rule)
    period    = max(period, 1)

    bg    = attractor_binary(word, rule, width, threshold)  # P × N
    P     = len(bg)
    N     = width

    # temporal: per-cell sequence across attractor steps
    temp_seqs = [[bg[t][i] for t in range(P)] for i in range(N)]
    spat_seqs = bg  # P × N

    # symbol bias (fraction of 1s)
    total_syms = P * N
    n_ones     = sum(sum(row) for row in bg)
    bias       = round(n_ones / max(total_syms, 1), 8)

    # unique pattern counts
    temp_pats = [tuple(s) for s in temp_seqs]
    spat_pats = [tuple(s) for s in spat_seqs]
    n_ut      = len(set(temp_pats))
    n_us      = len(set(spat_pats))

    # pattern entropy
    def _pat_entropy(pats: list[tuple]) -> float:
        counts: dict[tuple, int] = {}
        for p in pats:
            counts[p] = counts.get(p, 0) + 1
        return round(_entropy_from_counts(counts), 8)

    te_h = _pat_entropy(temp_pats)
    sp_h = _pat_entropy(spat_pats)

    # block entropy profile (using temporal sequences)
    bep  = block_entropy_profile(temp_seqs, max_n)
    # topological entropy estimate
    topo = max((bep[n - 1] / n for n in range(2, max_n + 1) if n - 1 < len(bep)),
               default=0.0)
    topo = round(topo, 8)

    # transition matrix
    tm = transition_matrix(temp_seqs)

    # forbidden n-grams
    f2 = forbidden_ngrams(temp_seqs, 2)
    f3 = forbidden_ngrams(temp_seqs, 3)

    # n-gram count dicts (string keys for JSON)
    def _str_dict(d: dict[tuple, int]) -> dict[str, int]:
        return {''.join(map(str, k)): v for k, v in sorted(d.items())}

    ng1 = _str_dict(ngram_dist(temp_seqs, 1))
    ng2 = _str_dict(ngram_dist(temp_seqs, 2))

    return {
        'word':            word.upper(),
        'rule':            rule,
        'width':           width,
        'threshold':       threshold,
        'max_n':           max_n,
        'period':          P,
        'transient':       transient,
        'binary_grid':     bg,
        'temporal_seqs':   temp_seqs,
        'spatial_seqs':    spat_seqs,
        'symbol_bias':     bias,
        'n_unique_temp':   n_ut,
        'n_unique_spat':   n_us,
        'temporal_entropy':te_h,
        'spatial_entropy': sp_h,
        'block_entropy':   bep,
        'topological_h':   topo,
        'transition_mat':  tm,
        'forbidden_2grams':f2,
        'forbidden_3grams':f3,
        'ngram_1':         ng1,
        'ngram_2':         ng2,
    }


def all_symbolic(
    word:      str,
    threshold: int = _DEFAULT_THR,
    max_n:     int = _DEFAULT_MAX_N,
) -> dict[str, dict]:
    """symbolic_dict for all 4 rules."""
    return {r: symbolic_dict(word, r, threshold=threshold, max_n=max_n)
            for r in _ALL_RULES}


def build_symbolic_data(
    words:     list[str] | None = None,
    threshold: int              = _DEFAULT_THR,
) -> dict:
    """Symbolic summary for the full lexicon × 4 rules.

    Returns dict:
        words, per_rule: {rule: {word: {period, symbol_bias, topological_h,
                                         n_unique_temp, n_forbidden_2}}}
        ranking: {rule: [(word, topological_h), …] sorted descending}
    """
    words = words if words is not None else list(LEXICON)
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = symbolic_dict(word, rule, threshold=threshold, max_n=4)
            per_rule[rule][word] = {
                'period':        d['period'],
                'symbol_bias':   d['symbol_bias'],
                'topological_h': d['topological_h'],
                'n_unique_temp': d['n_unique_temp'],
                'n_forbidden_2': len(d['forbidden_2grams']),
            }
    ranking = {
        r: sorted(
            ((w, v['topological_h']) for w, v in per_rule[r].items()),
            key=lambda x: -x[1],
        )
        for r in _ALL_RULES
    }
    return {'words': words, 'per_rule': per_rule, 'ranking': ranking}


# ── ASCII / ANSI display ───────────────────────────────────────────────────────

def print_symbolic(
    word:      str  = 'ТУМАН',
    rule:      str  = 'xor3',
    threshold: int  = _DEFAULT_THR,
    color:     bool = True,
) -> None:
    """Print binary attractor grid + symbolic statistics."""
    d    = symbolic_dict(word, rule, threshold=threshold)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    print(f"{bold}  ◈ Символьная динамика Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  P={d['period']}  thr={threshold}")
    print(f"  {'─' * (d['width'] + 30)}")

    # binary grid: '█' = 1, '·' = 0
    for t, row in enumerate(d['binary_grid']):
        line = ''.join('█' if b else '·' for b in row)
        print(f"  {t:3d} {line}")

    print(f"  {'─' * (d['width'] + 30)}")
    bep = d['block_entropy']
    print(f"  bias={d['symbol_bias']:.3f}  "
          f"unique_temp={d['n_unique_temp']}  "
          f"unique_spat={d['n_unique_spat']}")
    print(f"  H_n = " + '  '.join(f"H{n+1}={bep[n]:.3f}" for n in range(len(bep))))
    print(f"  topol.h={d['topological_h']:.4f}  "
          f"forbidden_2={len(d['forbidden_2grams'])}  "
          f"forbidden_3={len(d['forbidden_3grams'])}")
    tm = d['transition_mat']
    print(f"  TM: 0→0={tm[0][0]:.3f}  0→1={tm[0][1]:.3f}  "
          f"1→0={tm[1][0]:.3f}  1→1={tm[1][1]:.3f}")
    print()


def print_symbolic_stats(
    words: list[str] | None = None,
    color: bool             = True,
) -> None:
    """Table: topological entropy per word × rule."""
    words = words if words is not None else list(LEXICON)
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Топологическая энтропия (h) по символьной динамике{rst}")
    print('  ' + '─' * (len(header) + 2))
    print('  ' + header)
    print('  ' + '─' * (len(header) + 2))
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = symbolic_dict(word, rule, max_n=4)
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{d['topological_h']:>8.4f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Symbolic Dynamics Q6 CA')
    parser.add_argument('--word',       default='ТУМАН')
    parser.add_argument('--rule',       default='xor3', choices=_ALL_RULES)
    parser.add_argument('--threshold',  type=int, default=_DEFAULT_THR)
    parser.add_argument('--all-rules',  action='store_true')
    parser.add_argument('--stats',      action='store_true')
    parser.add_argument('--no-color',   action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_symbolic_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_symbolic(args.word, rule, args.threshold, color)
    else:
        print_symbolic(args.word, args.rule, args.threshold, color)


if __name__ == '__main__':
    _main()
