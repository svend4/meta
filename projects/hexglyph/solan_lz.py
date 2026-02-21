"""
solan_lz.py — Lempel-Ziv 1976 Complexity of Q6 CA Attractors.

LZ76 (copy-parsing) counts the minimum number of distinct phrases needed
to exhaustively parse a binary string without copying from the future:

    At position i, find the shortest s[i:j] that does NOT appear as a
    contiguous substring of s[0:i].  Count it, advance to j.

    C_LZ76(s) = number of phrases in this greedy parsing.

Normalised complexity (Lempel & Ziv 1976):
    C_norm = C_LZ76(s) / (n / log₂n)   where n = len(s)
    C_norm → 1  for a truly random string of length n
    C_norm ≪ 1  for periodic / highly structured strings

Three granularities
───────────────────
  cell_lz      Per cell: cell's P temporal values → 6P-bit string
  spatial_lz   Per time step: N=16 cell values → 96-bit string
  full_lz      Full attractor: P steps × 16 cells → 96P-bit string
               (most reliable normalisation for short periods)

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1): full_norm ≈ 0.48   (96 zeros, trivially structured)
  ГОРА  AND  (P=2): full_norm ≈ 0.43   (alternating → lowest complexity)
  ГОРА  XOR3 (P=2): full_norm ≈ 0.59
  ТУМАН XOR3 (P=8): full_norm ≈ 0.60   (richest attractor → highest LZ)

LZ76 is asymptotically equivalent to the entropy rate H (for ergodic
processes) but works without probability estimation and detects structure
even in very short sequences — complementing the entropy-based measures
in solan_block.py, solan_perm.py and solan_entropy.py.

Functions
─────────
  lz76(s)                              → int
  to_binary(val, bits)                 → str
  lz_of_series(series)                 → dict
  lz_of_spatial(vals)                  → dict
  lz_dict(word, rule, width)           → dict
  all_lz(word, width)                  → dict[str, dict]
  build_lz_data(words, width)          → dict
  print_lz(word, rule, color)          → None
  print_lz_stats(words, color)         → None

Запуск
──────
  python3 -m projects.hexglyph.solan_lz --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_lz --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_lz --stats --no-color
"""

from __future__ import annotations
import math
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_BITS:      int = 6          # bits per Q6 symbol (0–63)
_DEFAULT_W: int = 16


# ── Core algorithm ─────────────────────────────────────────────────────────────

def lz76(s: str) -> int:
    """LZ76 copy-parsing complexity: number of distinct phrases.

    For each position i, finds the shortest s[i:j] that does not appear
    as a contiguous substring of s[0:i].  Time: O(n²) per phrase → O(n³)
    total, adequate for sequences up to ~1000 bits.
    """
    n = len(s)
    if n == 0:
        return 0
    i, c = 0, 0
    while i < n:
        j = i + 1
        while j <= n and s[i:j] in s[:i]:
            j += 1
        c += 1
        i = j
    return c


def to_binary(val: int, bits: int = _BITS) -> str:
    """Convert non-negative integer to zero-padded binary string."""
    return format(int(val) & ((1 << bits) - 1), f'0{bits}b')


def _normalise(c: int, n: int) -> float:
    """Normalised LZ76 complexity.  Returns 1.0 for n ≤ 1."""
    if n <= 1:
        return 1.0
    return round(c / (n / math.log2(n)), 8)


# ── Series / spatial helpers ──────────────────────────────────────────────────

def lz_of_series(series: list[int]) -> dict:
    """LZ76 of a single cell's temporal series (cell_lz)."""
    s = ''.join(to_binary(v) for v in series)
    n = len(s)
    c = lz76(s)
    return {'bits': n, 'lz': c, 'norm': _normalise(c, n)}


def lz_of_spatial(vals: list[int]) -> dict:
    """LZ76 of one time step's spatial pattern (spatial_lz)."""
    s = ''.join(to_binary(v) for v in vals)
    n = len(s)
    c = lz76(s)
    return {'bits': n, 'lz': c, 'norm': _normalise(c, n)}


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Full dictionary ────────────────────────────────────────────────────────────

def lz_dict(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full LZ76 analysis for one word × rule (all three granularities)."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']
    orbit  = _get_orbit(word, rule, width)

    # Cell-level
    cell_lz = [
        lz_of_series([orbit[t][i] for t in range(period)])
        for i in range(width)
    ]
    mean_cell_norm = round(
        sum(c['norm'] for c in cell_lz) / width, 8)

    # Spatial (per time step)
    spatial_lz = [
        lz_of_spatial([orbit[t][i] for i in range(width)])
        for t in range(period)
    ]
    mean_sp_norm = round(
        sum(s['norm'] for s in spatial_lz) / max(period, 1), 8)

    # Full attractor
    full_s = ''.join(
        to_binary(orbit[t][i])
        for t in range(period)
        for i in range(width)
    )
    full_n = len(full_s)
    full_c = lz76(full_s)
    full_lz = {'bits': full_n, 'lz': full_c, 'norm': _normalise(full_c, full_n)}

    return {
        'word':            word.upper(),
        'rule':            rule,
        'period':          period,
        'cell_lz':         cell_lz,
        'mean_cell_norm':  mean_cell_norm,
        'spatial_lz':      spatial_lz,
        'mean_sp_norm':    mean_sp_norm,
        'full_lz':         full_lz,
    }


def all_lz(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """lz_dict for all 4 rules."""
    return {rule: lz_dict(word, rule, width) for rule in _RULES}


def build_lz_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Aggregated LZ76 data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = lz_dict(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'mean_cell_norm', 'mean_sp_norm', 'full_lz')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'


def _bar(v: float, w: int = 24) -> str:
    filled = round(min(max(v, 0.0), 2.0) * w / 2)
    return _BAR * filled + '░' * (w - filled)


def print_lz(word: str = 'ТУМАН', rule: str = 'xor3',
             color: bool = True) -> None:
    d   = lz_dict(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    fl  = d['full_lz']
    print(f'  {c}◈ LZ76  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'full: bits={fl["bits"]} c={fl["lz"]} norm={fl["norm"]:.4f}  '
          f'mean_cell_norm={d["mean_cell_norm"]:.4f}{r}')
    print('  ' + '─' * 62)
    print(f'  {"cell":>4}  {"bits":>5}  {"lz":>4}  {"norm":>7}  bar (0 → 1 → 2)')
    print('  ' + '─' * 62)
    for i, cd in enumerate(d['cell_lz']):
        bar = _bar(cd['norm'])
        print(f'  {i:>4}  {cd["bits"]:>5}  {cd["lz"]:>4}  '
              f'{cd["norm"]:>7.4f}  {bar}')
    fl2 = d['full_lz']
    print(f'\n  full attractor: {fl2["bits"]} bits  '
          f'lz={fl2["lz"]}  norm={fl2["norm"]:.4f}  {_bar(fl2["norm"])}')
    print()


def print_lz_stats(words: list[str] | None = None,
                   color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_lz(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='LZ76 algorithmic complexity for Q6 CA attractors')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_lz_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_lz(args.word, rule, color)
    else:
        print_lz(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
