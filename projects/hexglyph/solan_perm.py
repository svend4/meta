"""solan_perm.py — Permutation Entropy (PE) analysis for Q6 CA.

Permutation entropy (Bandt & Pompe 2002) characterises the ordinal complexity
of a time series.  For a cell's trajectory on the periodic attractor (period P,
extended cyclically), windows of length m are mapped to their rank-order
(ordinal) pattern; the Shannon entropy of the resulting pattern distribution is
normalised by log₂(m!) to give nPE ∈ [0, 1]:

    nPE_m = H({patterns}) / log₂(m!)

Ordinal pattern of a window (x_0, …, x_{m-1}):
    rank[i] = position of x_i in the stably-sorted window
    Stable sort: equal values are ordered by their index (ascending), so the
    result is unique even for repeated values.

Key properties:
    period=1 → all windows identical → only 1 pattern → nPE = 0
    period=2, m=2 → 2 distinct windows → nPE_2 = 1.0 (maximal for m=2)
    period=P, m=P, all-distinct → up to P patterns out of m! possible
    nPE is always ≤ 1; equals 1 only when all m! patterns are equally likely

Comparison with other complexity measures in this project:
    LZ76 (solan_complexity) — compression of the full space-time bitstring
    Shannon entropy (solan_entropy) — marginal distribution of cell values
    Permutation entropy (this) — temporal ordering structure of cell series

Функции:
    ordinal_pattern(window)                       → tuple[int, …]
    perm_entropy(series, m)                       → float   (normalised, 0–1)
    spatial_pe(word, rule, width, m)              → list[float]
    pe_dict(word, rule, width, m)                 → dict
    all_pe(word, width, m)                        → dict[str, dict]
    build_pe_data(words, width, m_vals, seed)     → dict
    print_pe(word, rule, width, m, color)         → None
    print_pe_stats(words, width, m, color)        → None

Запуск:
    python3 -m projects.hexglyph.solan_perm --word ТУМАН --rule xor3 --m 3
    python3 -m projects.hexglyph.solan_perm --word ГОРА --all-rules --m 4
    python3 -m projects.hexglyph.solan_perm --stats
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_transfer import get_orbit
from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16
_DEFAULT_M     = 3
_DEFAULT_M_VALS = [2, 3, 4]
_DEFAULT_WORDS  = list(LEXICON)


# ── Ordinal pattern ────────────────────────────────────────────────────────────

def ordinal_pattern(window: tuple | list) -> tuple:
    """Stable-sort ordinal pattern of *window*.

    Returns a tuple of length len(window) where element i is the rank of
    window[i] among all elements (ties broken by index — ascending).

    Example:
        ordinal_pattern([3, 1, 3]) → (1, 0, 2)
            # 1 is smallest (rank 0), then 3 at index 0 (rank 1), 3 at index 2 (rank 2)
    """
    m = len(window)
    # stable argsort: sort by (value, position)
    order = sorted(range(m), key=lambda i: (window[i], i))
    ranks = [0] * m
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return tuple(ranks)


# ── Permutation entropy ────────────────────────────────────────────────────────

def perm_entropy(series: list[int], m: int) -> float:
    """Normalised permutation entropy of order *m* for *series* (cyclic).

    Treats *series* as a cyclic time series of period P = len(series).
    Extracts P windows of length m (with cyclic wrap), computes ordinal
    patterns, and returns H(patterns) / log₂(m!) ∈ [0, 1].

    Returns 0.0 for period-1 series (all values identical) or m < 2.
    """
    P = len(series)
    if P < 1 or m < 2:
        return 0.0
    counts: Counter = Counter()
    for t in range(P):
        window = tuple(series[(t + k) % P] for k in range(m))
        counts[ordinal_pattern(window)] += 1
    total   = P
    pe      = -sum((c / total) * math.log2(c / total)
                   for c in counts.values() if c > 0)
    max_pe  = math.log2(math.factorial(m))
    return max(0.0, round(pe / max_pe, 8)) if max_pe > 0 else 0.0


# ── Spatial PE profile ────────────────────────────────────────────────────────

def spatial_pe(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
    m:     int = _DEFAULT_M,
) -> list[float]:
    """Normalised PE for each cell's time series on the attractor.

    Returns a list of length *width*.
    """
    orbit = get_orbit(word, rule, width)
    return [perm_entropy([s[i] for s in orbit], m) for i in range(width)]


# ── Per-word dict ─────────────────────────────────────────────────────────────

def pe_dict(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
    m:     int = _DEFAULT_M,
) -> dict:
    """Full PE analysis for one word + rule + order m.

    Returns dict:
        word         : str
        rule         : str
        width        : int
        m            : int
        period       : int
        max_patterns : int           = min(period, factorial(m))
        profile      : list[float]   = nPE per cell
        mean_pe      : float
        max_pe_val   : float         (max over cells)
        min_pe_val   : float
        spatial_var  : float         = variance of profile
        multi_m      : dict[int, float]   = mean nPE for m in {2,3,4}
    """
    orbit  = get_orbit(word, rule, width)
    period = len(orbit)
    profile = [perm_entropy([s[i] for s in orbit], m) for i in range(width)]
    mean_pe = round(sum(profile) / width, 8)

    # Theoretical max distinct patterns
    max_patterns = min(period, math.factorial(m))

    # Spatial variance
    var = round(
        sum((v - mean_pe) ** 2 for v in profile) / width, 8
    )

    # Mean PE for multiple m values
    multi_m: dict[int, float] = {}
    for mm in _DEFAULT_M_VALS:
        prof_mm = [perm_entropy([s[i] for s in orbit], mm) for i in range(width)]
        multi_m[mm] = round(sum(prof_mm) / width, 8)

    return {
        'word':         word.upper(),
        'rule':         rule,
        'width':        width,
        'm':            m,
        'period':       period,
        'max_patterns': max_patterns,
        'profile':      profile,
        'mean_pe':      mean_pe,
        'max_pe_val':   max(profile),
        'min_pe_val':   min(profile),
        'spatial_var':  var,
        'multi_m':      {str(k): v for k, v in multi_m.items()},
    }


def all_pe(
    word:  str,
    width: int = _DEFAULT_WIDTH,
    m:     int = _DEFAULT_M,
) -> dict[str, dict]:
    """pe_dict for all 4 rules."""
    return {r: pe_dict(word, r, width, m) for r in _ALL_RULES}


# ── Full dataset ─────────────────────────────────────────────────────────────

def build_pe_data(
    words:  list[str] | None = None,
    width:  int              = _DEFAULT_WIDTH,
    m_vals: list[int]        = _DEFAULT_M_VALS,
) -> dict:
    """PE summary for the full lexicon × 4 rules × m values.

    Returns dict:
        words    : list[str]
        m_vals   : list[int]
        per_rule : {rule: {word: {mean_pe, period, multi_m}}}
        ranking  : {rule: [(word, mean_pe), …]}  descending (for default m=3)
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = pe_dict(word, rule, width)
            per_rule[rule][word] = {
                'mean_pe': d['mean_pe'],
                'period':  d['period'],
                'multi_m': d['multi_m'],
            }
    ranking: dict[str, list] = {}
    for rule in _ALL_RULES:
        ranking[rule] = sorted(
            ((w, v['mean_pe']) for w, v in per_rule[rule].items()),
            key=lambda x: -x[1],
        )
    return {'words': words, 'm_vals': m_vals, 'per_rule': per_rule,
            'ranking': ranking}


# ── ASCII display ──────────────────────────────────────────────────────────────

_BAR_CHARS = ' ▁▂▃▄▅▆▇█'


def _bar_char(v: float) -> str:
    idx = min(int(v * (len(_BAR_CHARS) - 1) + 0.5), len(_BAR_CHARS) - 1)
    return _BAR_CHARS[idx]


def print_pe(
    word:  str,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    m:     int  = _DEFAULT_M,
    color: bool = True,
) -> None:
    """Print PE spatial profile and multi-m comparison."""
    d    = pe_dict(word, rule, width, m)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    print(f"{bold}  ◈ Перестановочная энтропия Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  P={d['period']}  m={m}  "
          f"mean={d['mean_pe']:.4f}  max_pat={d['max_patterns']}")
    print(f"  {'─' * 40}")
    # Spatial bar chart
    profile = d['profile']
    print(f"  {'Клетка':7s}  nPE (m={m})")
    for i, v in enumerate(profile):
        bar = _bar_char(v)
        marker = col if v >= 0.5 else dim
        print(f"  {i:2d}       {marker}{bar}{rst}  {v:.4f}")
    print(f"  {'─' * 40}")
    # Multi-m comparison
    multi_m = d['multi_m']
    print(f"  Ср.nPE по m: " + '  '.join(
        f"m={mm}: {float(mv):.4f}" for mm, mv in sorted(multi_m.items())
    ))
    print()


def print_pe_stats(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
    m:     int              = _DEFAULT_M,
    color: bool             = True,
) -> None:
    """Summary table: mean nPE for each word × rule."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Перестановочная энтропия Q6 (ср.nPE, m={m}){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = pe_dict(word, rule, width, m)
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{d['mean_pe']:>8.4f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Permutation Entropy Q6 CA')
    parser.add_argument('--word',      default='ТУМАН', help='Русское слово')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--m',         type=int, default=_DEFAULT_M,
                        choices=[2, 3, 4, 5])
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_pe_stats(color=color, width=args.width, m=args.m)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_pe(args.word, rule, args.width, args.m, color)
    else:
        print_pe(args.word, args.rule, args.width, args.m, color)


if __name__ == '__main__':
    _main()
