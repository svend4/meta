"""
solan_block.py — Block Entropy & Entropy Rate of Q6 CA Attractor.

For the binary attractor sequences (threshold=32) of each cell i, the
n-gram (block) entropy and its derived measures are computed:

    H_n  =  −Σ_{x∈{0,1}^n} P(x) log₂ P(x)        (bits)

where P(x) is estimated by counting circular n-grams pooled across all
N cells' period-P binary sequences.

Derived quantities
──────────────────
  Conditional entropy   h_n  = H_{n+1} − H_n  (n ≥ 1)
      Monotone non-increasing by sub-additivity of entropy.
      Represents uncertainty about the next symbol given n-step context.

  Entropy rate          h_∞  ≈ h_{max_n−1}
      For periodic sequences h_∞ = 0 (eventually deterministic), so
      h_n → 0 as n → ∞.

  Excess entropy        E  ≈ H_{max_n}
      For periodic sequences with h_∞ = 0:  E = lim_{n→∞} H_n.
      Measures the total complexity (bits) needed to understand the pattern.
      Related to AIS by:  E = Σ_{k=1}^{∞} AIS_k  (truncated at max_n).

  Saturation index      n* = smallest n where h_n < 0.01 bit
      The effective context length needed for (near-)deterministic prediction.

Key results
───────────
  XOR  ТУМАН  (P=1)   : H_n = 0 ∀n  →  E=0,  h_∞=0,  n*=1
  AND/OR fixed-point  : H_n = 0 ∀n  →  E=0
  ГОРА AND    (P=2)   : H_n = 1 ∀n≥1  →  E=1,  h_n=0 for n≥1,  n*=2
  ГОРА XOR3   (P=2)   : H_1=1, H_2=2, H_n=2 for n≥2  →  E=2,  n*=3
  ТУМАН XOR3  (P=8)   : H_n grows toward ~log₂(P·W)  →  E≈4.5–7,  n*>8

Functions
─────────
  cell_seqs(word, rule, width, threshold)     → list[list[int]]
  h_block(seqs, n)                            → float   H_n in bits
  block_profile(word, rule, max_n, width)     → list[float]  H_1..H_{max_n}
  h_rate_profile(profile)                     → list[float]  h_1..h_{max_n−1}
  saturation_index(profile, tol)              → int
  excess_entropy_estimate(profile)            → float
  block_dict(word, rule, max_n, width)        → dict
  all_block(word, max_n, width)               → dict[str, dict]
  build_block_data(words, max_n, width)       → dict
  print_block(word, rule, max_n, color)       → None
  print_block_stats(words, color)             → None

Запуск
──────
  python3 -m projects.hexglyph.solan_block --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_block --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_block --stats --no-color
"""

from __future__ import annotations
import math
import collections
import sys
import argparse

# ── Canonical rules ────────────────────────────────────────────────────────────
_RULES: list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH: int = 16
_DEFAULT_MAX_N: int = 8


# ── Binary attractor sequences ─────────────────────────────────────────────────

def cell_seqs(word: str, rule: str, width: int = _DEFAULT_WIDTH,
              threshold: int = 32) -> list[list[int]]:
    """Return N binary periodic sequences (one per cell) from the attractor."""
    from projects.hexglyph.solan_symbolic import attractor_binary
    grid = attractor_binary(word.upper(), rule, width, threshold)
    if not grid:
        return [[] for _ in range(width)]
    return [[grid[t][i] for t in range(len(grid))] for i in range(width)]


# ── Core computation ───────────────────────────────────────────────────────────

def h_block(seqs: list[list[int]], n: int) -> float:
    """Shannon entropy H_n (bits) of circular n-grams pooled across all seqs."""
    if n <= 0:
        return 0.0
    counts: dict[tuple[int, ...], int] = {}
    total = 0
    for seq in seqs:
        m = len(seq)
        if m == 0:
            continue
        for t in range(m):
            gram = tuple(seq[(t + j) % m] for j in range(n))
            counts[gram] = counts.get(gram, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    return max(0.0, -sum(
        (v / total) * math.log2(v / total)
        for v in counts.values() if v > 0
    ))


def block_profile(word: str, rule: str, max_n: int = _DEFAULT_MAX_N,
                  width: int = _DEFAULT_WIDTH) -> list[float]:
    """Return [H_1, H_2, ..., H_{max_n}] block entropy profile."""
    seqs = cell_seqs(word, rule, width)
    return [round(h_block(seqs, n), 6) for n in range(1, max_n + 1)]


def h_rate_profile(profile: list[float]) -> list[float]:
    """Return conditional entropy sequence h_n = H_{n+1} − H_n.

    Length = len(profile) − 1.  Non-negative and monotone non-increasing.
    """
    if len(profile) < 2:
        return []
    return [round(max(0.0, profile[i + 1] - profile[i]), 6)
            for i in range(len(profile) - 1)]


def saturation_index(profile: list[float], tol: float = 0.01) -> int:
    """Smallest n (1-based) where h_n < tol (context where prediction is near-certain).

    Returns max_n if never satisfied.
    """
    rates = h_rate_profile(profile)
    for i, h in enumerate(rates):
        if h < tol:
            return i + 2          # n = index+2 because rates[0] = h_2 = H_2 - H_1
    return len(profile)


def excess_entropy_estimate(profile: list[float]) -> float:
    """Excess entropy estimate E ≈ H_{max_n}.

    For periodic sequences h_∞ = 0, so E = lim H_n = H_{max_n} (lower bound).
    """
    return profile[-1] if profile else 0.0


# ── Block dictionary ───────────────────────────────────────────────────────────

def block_dict(word: str, rule: str, max_n: int = _DEFAULT_MAX_N,
               width: int = _DEFAULT_WIDTH) -> dict:
    """Full block entropy analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    profile = block_profile(word, rule, max_n, width)
    rates = h_rate_profile(profile)
    n_star = saturation_index(profile)
    exc = excess_entropy_estimate(profile)
    h1 = profile[0] if profile else 0.0
    h_inf = rates[-1] if rates else 0.0

    return {
        'word':             word.upper(),
        'rule':             rule,
        'period':           period,
        'max_n':            max_n,
        'h_profile':        profile,
        'h_rate':           rates,
        'h1':               h1,
        'h_inf_estimate':   h_inf,
        'excess_entropy':   exc,
        'saturation_n':     n_star,
        'normalised_E':     round(exc / math.log2(max(period, 1) * width), 6) if period > 0 else 0.0,
    }


def all_block(word: str, max_n: int = _DEFAULT_MAX_N,
              width: int = _DEFAULT_WIDTH) -> dict[str, dict]:
    """Run block_dict for all 4 rules."""
    return {rule: block_dict(word, rule, max_n, width) for rule in _RULES}


def build_block_data(words: list[str], max_n: int = _DEFAULT_MAX_N,
                     width: int = _DEFAULT_WIDTH) -> dict:
    """Build aggregated block entropy data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = block_dict(word, rule, max_n, width)
            per_rule[rule][word.upper()] = {k: d[k] for k in (
                'period', 'h1', 'h_inf_estimate', 'excess_entropy',
                'saturation_n', 'normalised_E', 'h_profile', 'h_rate')}
    return {'words': [w.upper() for w in words], 'width': width,
            'max_n': max_n, 'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'
_SHD  = '░'

def _bar(v: float, width: int = 20) -> str:
    filled = round(v * width)
    return _BAR * filled + _SHD * (width - filled)


def print_block(word: str = 'ТУМАН', rule: str = 'xor3',
                max_n: int = _DEFAULT_MAX_N, color: bool = True) -> None:
    d = block_dict(word, rule, max_n)
    c = _RCOL.get(rule, '') if color else ''
    r = _RST if color else ''
    RULE = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    print(f'  {c}◈ Block Entropy  {word.upper()}  |  {RULE}  '
          f'P={d["period"]}  E≈{d["excess_entropy"]:.4f}  n*={d["saturation_n"]}{r}')
    print('  ' + '─' * 62)
    prof = d['h_profile']
    rates = d['h_rate']
    h_max = max(prof) if prof else 1.0
    print(f'  {"n":>4}  {"H_n":>8}  {"h_n=ΔH":>8}  bar(H_n/H_max)')
    print('  ' + '─' * 62)
    for i, hn in enumerate(prof):
        hn_rate = rates[i - 1] if i > 0 else 0.0
        bar = _bar(hn / h_max if h_max > 0 else 0.0)
        print(f'  {i + 1:>4}  {hn:>8.4f}  {hn_rate:>8.4f}  {bar}')
    print()
    print(f'  H₁={d["h1"]:.4f}  h∞≈{d["h_inf_estimate"]:.4f}  '
          f'E≈{d["excess_entropy"]:.4f}  n*={d["saturation_n"]}  '
          f'E_norm={d["normalised_E"]:.4f}')
    print()


def print_block_stats(words: list[str] | None = None, max_n: int = _DEFAULT_MAX_N,
                      color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_block(word, rule, max_n, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Block Entropy & Entropy Rate (Q6 CA)')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--max-n',     type=int, default=_DEFAULT_MAX_N)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(block_dict(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.stats:
        print_block_stats(max_n=args.max_n, color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_block(args.word, rule, args.max_n, color)
    else:
        print_block(args.word, args.rule, args.max_n, color)


if __name__ == '__main__':
    _cli()
