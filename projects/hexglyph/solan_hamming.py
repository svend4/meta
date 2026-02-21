"""
solan_hamming.py — Hamming Weight Dynamics of Q6 CA Attractor States.

The Hamming weight  w(v) = popcount(v & 63) ∈ {0, 1, …, 6}  counts the
number of 1-bits in a Q6 value.  It serves as the "energy" or "density"
of a single cell's state.

Key statistics
──────────────
  Per time step t:
    μ_t = mean(w(orbit[t][i]) for i = 0…N−1)   spatial mean weight
    σ_t = std of the same                         spatial heterogeneity

  Over the orbit:
    global_mean    = mean of all μ_t              (density = global_mean / 6)
    osc_std        = std of [μ_0 … μ_{P-1}]       temporal oscillation of mean
    spatial_mean_std = mean of [σ_0 … σ_{P-1}]   average spatial heterogeneity

  Weight histogram: fraction of all (cell, step) pairs with each weight 0…6.

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      All weights = 0. density=0.  osc_std=0.  spatial_std=0.
      Histogram: {0: 1.0} — vacuum state.

  ГОРА AND  (P=2, 47↔1)
      Group A (even cells):  w(47)=5 at step 0,  w(1)=1 at step 1.
      Group B (odd  cells):  w(1)=1  at step 0,  w(47)=5 at step 1.
      → μ_0 = μ_1 = 3.0 — MEAN IS CONSTANT despite oscillation!
        osc_std = 0 (temporal mean stable)
        spatial_mean_std = 2.0 (large spatial spread at every step)
      Histogram: {1: 0.5, 5: 0.5} — BIMODAL (two distinct energy levels).
      Anti-phase oscillation keeps the mean fixed while the variance is large.

  ГОРА XOR3  (P=2, 4 spatial clusters)
      Profile = [4.5, 2.5] → osc_std = 1.0 (genuine temporal oscillation).
      Histogram: diverse {1,2,3,4,5,6} — wide spread.

  ТУМАН XOR3  (P=8)
      Histogram: {0, 2, 4, 6} only — ALL EVEN WEIGHTS.
      XOR3 rule preserves Hamming-weight parity globally:
        w(a⊕b⊕c) has parity = parity(w(a)+w(b)+w(c)).
        If all initial cell weights are even, the orbit stays even.
      parity_even_fraction = 1.0  (structural conserved quantity).
      Profile oscillates: [3.12, 2.25, 3.75, 3.5, 2.88, 3.5, 3.5, 2.75]
      osc_std ≈ 0.47.  spatial_mean_std ≈ 1.20.

Interpretation
  density ≈ 0   : system mostly in low-energy states (few 1-bits)
  density ≈ 0.5 : balanced activation (≈ 3 bits on average)
  density ≈ 1   : system mostly in high-energy states (many 1-bits)
  osc_std > 0   : the mean density fluctuates over the orbit (temporal rhythm)
  osc_std = 0   : mean density is constant (may still have large spatial spread)
  bimodal hist  : system alternates between two energy levels
  even-only hist: XOR3 parity conservation (all initial weights even)

Functions
─────────
  hamming_weight(v)                          → int  ∈ {0, …, 6}
  orbit_weight_grid(word, rule, width)       → list[list[int]]  P×N grid
  state_mean_weight(weights)                 → float
  state_weight_std(weights)                  → float
  orbit_weight_profile(word, rule, width)    → list[dict]  (per-step)
  weight_histogram(word, rule, width)        → dict[int, float]
  parity_even_fraction(word, rule, width)    → float
  cell_weight_stats(w_series)                → dict
  all_cell_weight_stats(word, rule, width)   → list[dict]
  weight_summary(word, rule, width)          → dict
  all_weights(word, width)                   → dict[str, dict]
  build_weight_data(words, width)            → dict
  print_hamming(word, rule, color)           → None
  print_weight_stats(words, color)           → None

Запуск
──────
  python3 -m projects.hexglyph.solan_hamming --word ГОРА --rule and
  python3 -m projects.hexglyph.solan_hamming --word ТУМАН --all-rules --no-color
  python3 -m projects.hexglyph.solan_hamming --stats --no-color
"""

from __future__ import annotations
import sys
import argparse
import math

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_MAX_W:     int = 6      # maximum Hamming weight for Q6


# ── Core primitives ────────────────────────────────────────────────────────────

def hamming_weight(v: int) -> int:
    """Number of 1-bits in the Q6 value v (popcount(v & 63))."""
    return bin(v & 63).count('1')


def state_mean_weight(weights: list[int]) -> float:
    """Mean Hamming weight across a state vector."""
    if not weights:
        return 0.0
    return sum(weights) / len(weights)


def state_weight_std(weights: list[int]) -> float:
    """Population std of Hamming weights across a state vector."""
    n = len(weights)
    if n == 0:
        return 0.0
    mu = sum(weights) / n
    return math.sqrt(sum((w - mu) ** 2 for w in weights) / n)


# ── Orbit helpers ─────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


def orbit_weight_grid(word: str, rule: str,
                      width: int = _DEFAULT_W) -> list[list[int]]:
    """
    P×N grid of Hamming weights: grid[t][i] = w(orbit[t][i]).
    """
    orbit = _get_orbit(word, rule, width)
    return [[hamming_weight(orbit[t][i]) for i in range(width)]
            for t in range(len(orbit))]


# ── Profile and histogram ─────────────────────────────────────────────────────

def orbit_weight_profile(word: str, rule: str,
                         width: int = _DEFAULT_W) -> list[dict]:
    """
    Per-step weight statistics: list of P dicts, each with keys:
      step, mean, std, min, max, range.
    """
    grid = orbit_weight_grid(word, rule, width)
    profile = []
    for t, row in enumerate(grid):
        mu  = state_mean_weight(row)
        std = state_weight_std(row)
        profile.append({
            'step':  t,
            'mean':  round(mu, 6),
            'std':   round(std, 6),
            'min':   min(row),
            'max':   max(row),
            'range': max(row) - min(row),
        })
    return profile


def weight_histogram(word: str, rule: str,
                     width: int = _DEFAULT_W) -> dict[int, float]:
    """
    Fraction of all (cell, step) pairs with each Hamming weight 0…6.

    Keys are only weights that actually occur.
    """
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    total = P * width
    counts: dict[int, int] = {}
    for t in range(P):
        for i in range(width):
            w = hamming_weight(orbit[t][i])
            counts[w] = counts.get(w, 0) + 1
    return {k: round(v / total, 6) for k, v in sorted(counts.items())}


def parity_even_fraction(word: str, rule: str,
                         width: int = _DEFAULT_W) -> float:
    """
    Fraction of all (cell, step) pairs whose Hamming weight is even (0,2,4,6).

    = 1.0 for ТУМАН XOR3 (parity conservation).
    = 0.0 for ГОРА AND (both weights 1 and 5 are odd).
    """
    hist  = weight_histogram(word, rule, width)
    return round(sum(v for k, v in hist.items() if k % 2 == 0), 6)


# ── Per-cell statistics ────────────────────────────────────────────────────────

def cell_weight_stats(w_series: list[int]) -> dict:
    """Full Hamming-weight statistics for a single cell's weight series."""
    P = len(w_series)
    if P == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0,
                'range': 0, 'density': 0.0, 'n_distinct': 0}
    mu  = sum(w_series) / P
    std = math.sqrt(sum((w - mu) ** 2 for w in w_series) / P)
    return {
        'mean':     round(mu, 6),
        'std':      round(std, 6),
        'min':      min(w_series),
        'max':      max(w_series),
        'range':    max(w_series) - min(w_series),
        'density':  round(mu / _MAX_W, 6),
        'n_distinct': len(set(w_series)),
    }


def all_cell_weight_stats(word: str, rule: str,
                          width: int = _DEFAULT_W) -> list[dict]:
    """Per-cell Hamming-weight stats (list of length = width)."""
    grid = orbit_weight_grid(word, rule, width)
    P    = len(grid)
    out  = []
    for i in range(width):
        w_series = [grid[t][i] for t in range(P)]
        s        = cell_weight_stats(w_series)
        s['cell'] = i
        out.append(s)
    return out


# ── Summary ───────────────────────────────────────────────────────────────────

def weight_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Comprehensive Hamming-weight statistics for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj    = word_trajectory(word.upper(), rule, width)
    period  = traj['period']
    profile = orbit_weight_profile(word, rule, width)
    hist    = weight_histogram(word, rule, width)
    pef     = parity_even_fraction(word, rule, width)
    css     = all_cell_weight_stats(word, rule, width)

    # Global mean and density
    step_means = [p['mean'] for p in profile]
    P          = len(step_means)
    g_mean     = sum(step_means) / P if P else 0.0

    # Temporal oscillation of mean weight
    osc_std    = math.sqrt(sum((m - g_mean) ** 2 for m in step_means) / P) if P else 0.0

    # Spatial heterogeneity: mean of per-step std
    spatial_stds   = [p['std'] for p in profile]
    spatial_mean_std = sum(spatial_stds) / P if P else 0.0

    # Bimodal check: hist has exactly 2 distinct weights
    is_bimodal  = len(hist) == 2
    all_even    = all(k % 2 == 0 for k in hist.keys())
    all_odd     = all(k % 2 == 1 for k in hist.keys())

    # Weight range over entire orbit
    min_w = min(p['min'] for p in profile) if profile else 0
    max_w = max(p['max'] for p in profile) if profile else 0

    return {
        'word':            word.upper(),
        'rule':            rule,
        'period':          period,
        'global_mean':     round(g_mean, 6),
        'density':         round(g_mean / _MAX_W, 6),
        'osc_std':         round(osc_std, 6),
        'spatial_mean_std': round(spatial_mean_std, 6),
        'step_profile':    profile,
        'histogram':       hist,
        'parity_even_frac': pef,
        'is_bimodal':      is_bimodal,
        'all_even_weights': all_even,
        'all_odd_weights':  all_odd,
        'min_weight':      min_w,
        'max_weight':      max_w,
        'cell_stats':      css,
    }


def all_weights(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """weight_summary for all 4 rules."""
    return {rule: weight_summary(word, rule, width) for rule in _RULES}


def build_weight_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact weight data for a list of words (no cell_stats, no profile)."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = weight_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'global_mean', 'density', 'osc_std',
                 'spatial_mean_std', 'histogram', 'parity_even_frac',
                 'is_bimodal', 'all_even_weights', 'all_odd_weights',
                 'min_weight', 'max_weight')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'


def _hist_bar(fraction: float, w: int = 16) -> str:
    filled = round(fraction * w)
    return '█' * filled + '░' * (w - filled)


def print_hamming(word: str = 'ГОРА', rule: str = 'and',
                  color: bool = True) -> None:
    d   = weight_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    notes = []
    if d['is_bimodal']:      notes.append('BIMODAL')
    if d['all_even_weights']: notes.append('even-only')
    if d['all_odd_weights']:  notes.append('odd-only')
    note_str = '  [' + ', '.join(notes) + ']' if notes else ''

    print(f'  {c}◈ Hamming  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'density={d["density"]:.3f}  osc_std={d["osc_std"]:.3f}  '
          f'spatial_std={d["spatial_mean_std"]:.3f}{note_str}{r}')
    print('  ' + '─' * 62)

    # Step profile
    print(f'  Step profile (mean weight / step):')
    for p in d['step_profile']:
        bar = _hist_bar(p['mean'] / _MAX_W)
        print(f'    t={p["step"]}: μ={p["mean"]:5.2f}  σ={p["std"]:5.2f}  '
              f'[{p["min"]},{p["max"]}]  {bar}')

    # Weight histogram
    print(f'\n  Weight histogram (fraction by weight value):')
    for k in range(_MAX_W + 1):
        frac = d['histogram'].get(k, 0.0)
        if frac > 0:
            bar = _hist_bar(frac)
            print(f'    w={k}: {frac:.3f}  {bar}')

    # Summary metrics
    print(f'\n  parity_even_frac={d["parity_even_frac"]:.3f}  '
          f'weight_range=[{d["min_weight"]},{d["max_weight"]}]')
    print()


def print_weight_stats(words: list[str] | None = None,
                       color: bool = True) -> None:
    WORDS = words or ['ТУМАН', 'ГОРА']
    for word in WORDS:
        for rule in _RULES:
            print_hamming(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Hamming weight dynamics of Q6 CA')
    p.add_argument('--word',      default='ГОРА')
    p.add_argument('--rule',      default='and', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_weight_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_hamming(args.word, rule, color)
    else:
        print_hamming(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
