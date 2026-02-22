"""
solan_edge.py — Spatial Edge Density of Q6 CA Attractors.

A *spatial edge* occurs when two adjacent ring cells hold different Q6
values.  The edge density at step t is:

    E_t = |{ i : orbit[t][i] ≠ orbit[t][(i+1) mod N] }| / N   ∈ [0, 1]

E_t = 0   ⟺  all cells share one value  (zero boundary structure)
E_t = 1   ⟺  every adjacent pair differs  (maximum fragmentation)

We also resolve E_t to each of the 6 bit-planes:

    E_t^b = |{ i : bit_b(orbit[t][i]) ≠ bit_b(orbit[t][(i+1)%N]) }| / N

giving a 6-vector of bit-level edge densities per step.

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      E_t = 0.  bit_E = [0]*6.  Zero boundary structure.

  ГОРА AND  (P=2, anti-phase alternation {47, 1})
      E_t = 1.0 (maximum).
      mean bit_E = [0, 1, 1, 1, 0, 1]  — bits 0 and 4 are
      NEVER spatial boundaries; bits 1,2,3,5 are ALWAYS boundaries.
      ★  Spatial boundary pattern matches bitflip "always-flip" bits.

  ГОРА XOR3  (P=2, 4 clusters each repeating 4 cells)
      E_t = 1.0 (maximum, constant).
      bit 0 = 1 in ALL cells → mean bit_E[0] = 0.
      Other bits form boundaries in exactly 50% of adjacent pairs
      (mean bit_E[1..5] = 0.5) because 4 distinct cluster values
      create 4 domain walls in 16 positions.

  ТУМАН XOR3  (P=8)
      E_t ∈ {0.8125, 0.9375, 1.0}  — temporal oscillation.
      Bit-level edge vectors vary richly each step.

Functions
─────────
  edge_density(state)                             → float
  bit_edge_density(state, b)                      → float
  bit_edge_vector(state)                          → list[float]
  orbit_edge_profile(word, rule, width)           → list[float]
  orbit_bit_edge_profile(word, rule, width)       → list[list[float]]
  edge_stats(word, rule, width)                   → dict
  mean_bit_edge(word, rule, width)                → list[float]
  classify_bit_edge(val, eps)                     → str
  edge_summary(word, rule, width)                 → dict
  all_edges(word, width)                          → dict[str, dict]
  build_edge_data(words, width)                   → dict
  print_edge(word, rule, color)                   → None
  print_edge_stats(words, color)                  → None

Запуск
──────
  python3 -m projects.hexglyph.solan_edge --word ГОРА --rule and --no-color
  python3 -m projects.hexglyph.solan_edge --word ТУМАН --all-rules --no-color
  python3 -m projects.hexglyph.solan_edge --stats --no-color
"""

from __future__ import annotations
import sys
import argparse
import math

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


# ── Core computations ──────────────────────────────────────────────────────────

def edge_density(state: list[int]) -> float:
    """Fraction of adjacent ring pairs where Q6 values differ.

    E = |{i : state[i] ≠ state[(i+1)%N]}| / N ∈ [0, 1].
    Returns 0.0 for empty or single-cell state.
    """
    N = len(state)
    if N < 2:
        return 0.0
    diff = sum(1 for i in range(N) if state[i] != state[(i + 1) % N])
    return diff / N


def bit_edge_density(state: list[int], b: int) -> float:
    """Fraction of adjacent pairs where bit *b* of adjacent cells differs."""
    N = len(state)
    if N < 2:
        return 0.0
    diff = sum(1 for i in range(N)
               if ((state[i] >> b) & 1) != ((state[(i + 1) % N] >> b) & 1))
    return diff / N


def bit_edge_vector(state: list[int]) -> list[float]:
    """6-vector of bit-level edge densities E^0, E^1, …, E^5."""
    return [bit_edge_density(state, b) for b in range(6)]


# ── Orbit helpers ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int) -> list[list[int]]:
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


def orbit_edge_profile(word: str, rule: str,
                       width: int = _DEFAULT_W) -> list[float]:
    """Edge density at each attractor step: [E_0, …, E_{P−1}]."""
    orbit = _get_orbit(word, rule, width)
    return [round(edge_density(list(orbit[t])), 6) for t in range(len(orbit))]


def orbit_bit_edge_profile(word: str, rule: str,
                           width: int = _DEFAULT_W) -> list[list[float]]:
    """P×6 matrix of bit-level edge densities per step."""
    orbit = _get_orbit(word, rule, width)
    return [[round(bit_edge_density(list(orbit[t]), b), 6) for b in range(6)]
            for t in range(len(orbit))]


# ── Statistics ─────────────────────────────────────────────────────────────────

def edge_stats(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Summary statistics of the Q6-level edge density profile."""
    profile = orbit_edge_profile(word, rule, width)
    P = len(profile)
    if P == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'delta': 0.0, 'profile': []}
    mean_e = sum(profile) / P
    var_e  = sum((e - mean_e) ** 2 for e in profile) / P
    return {
        'mean':    round(mean_e, 6),
        'std':     round(math.sqrt(var_e) if var_e > 0 else 0.0, 6),
        'min':     round(min(profile), 6),
        'max':     round(max(profile), 6),
        'delta':   round(max(profile) - min(profile), 6),
        'profile': profile,
    }


def mean_bit_edge(word: str, rule: str,
                  width: int = _DEFAULT_W) -> list[float]:
    """Time-averaged bit-level edge vector (one float per bit 0..5)."""
    bep = orbit_bit_edge_profile(word, rule, width)
    P = len(bep)
    if P == 0:
        return [0.0] * 6
    return [round(sum(bep[t][b] for t in range(P)) / P, 6) for b in range(6)]


# ── Classification ─────────────────────────────────────────────────────────────

_EDGE_CLASSES = ('ZERO', 'FULL', 'HALF', 'INTERMEDIATE')


def classify_bit_edge(val: float, eps: float = 0.02) -> str:
    """Classify a mean bit-edge density into one of four categories.

    ZERO         val ≈ 0    — bit never forms a spatial boundary
    FULL         val ≈ 1    — bit always forms spatial boundaries
    HALF         val ≈ 0.5  — bit forms boundaries in half of adjacent pairs
    INTERMEDIATE otherwise
    """
    if val <= eps:
        return 'ZERO'
    if val >= 1.0 - eps:
        return 'FULL'
    if abs(val - 0.5) <= eps:
        return 'HALF'
    return 'INTERMEDIATE'


def _count_classes(classes: list[str]) -> dict[str, int]:
    cnt: dict[str, int] = {c: 0 for c in _EDGE_CLASSES}
    for c in classes:
        cnt[c] = cnt.get(c, 0) + 1
    return cnt


# ── Summary ────────────────────────────────────────────────────────────────────

def edge_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full spatial-edge summary for word × rule."""
    st      = edge_stats(word, rule, width)
    mbe     = mean_bit_edge(word, rule, width)
    classes = [classify_bit_edge(v) for v in mbe]
    counts  = _count_classes(classes)

    # Characterise temporal variability of edge density
    if st['delta'] < 1e-9:
        variability = 'constant'
    elif st['delta'] < 0.25:
        variability = 'low'
    elif st['delta'] < 0.5:
        variability = 'moderate'
    else:
        variability = 'high'

    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)

    return {
        'word':             word.upper(),
        'rule':             rule,
        'period':           traj['period'],
        'profile':          st['profile'],
        'mean_E':           st['mean'],
        'std_E':            st['std'],
        'min_E':            st['min'],
        'max_E':            st['max'],
        'delta_E':          st['delta'],
        'variability':      variability,
        'mean_bit_edge':    mbe,
        'bit_edge_classes': classes,
        'class_counts':     counts,
    }


def all_edges(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """edge_summary for all 4 rules."""
    return {rule: edge_summary(word, rule, width) for rule in _RULES}


def build_edge_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact edge data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = edge_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'profile', 'mean_E', 'std_E',
                                   'min_E', 'max_E', 'delta_E', 'variability',
                                   'mean_bit_edge', 'bit_edge_classes',
                                   'class_counts')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_DIM  = '\033[2m'


def print_edge(word: str = 'ГОРА', rule: str = 'and',
               color: bool = True) -> None:
    d   = edge_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    dim = _DIM if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    print(f'  {c}◈ Edge  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'variability={d["variability"]}{r}')
    print('  ' + '─' * 62)

    print(f'  Edge density profile:')
    for t, e in enumerate(d['profile']):
        bar_len = 20
        bar = '█' * int(round(e * bar_len)) + '░' * (bar_len - int(round(e * bar_len)))
        print(f'    t={t:2d}  E={e:.4f}  |{bar}|')

    print(f'\n  Statistics:')
    print(f'    mean_E = {d["mean_E"]:.4f}  delta_E = {d["delta_E"]:.4f}  '
          f'range = [{d["min_E"]:.4f}, {d["max_E"]:.4f}]')

    print(f'\n  Time-averaged bit-level edge density:')
    for b, (v, cls) in enumerate(zip(d['mean_bit_edge'], d['bit_edge_classes'])):
        bar_len = 12
        bar = '█' * int(round(v * bar_len)) + '░' * (bar_len - int(round(v * bar_len)))
        print(f'    bit {b}  E_b={v:.3f}  |{bar}|  {dim}{cls}{r}')

    print(f'\n  Bit-edge class counts: ', end='')
    parts = [f'{k}={v}' for k, v in d['class_counts'].items() if v > 0]
    print('  '.join(parts))
    print()


def print_edge_stats(words: list[str] | None = None,
                     color: bool = True) -> None:
    WORDS = words or ['ТУМАН', 'ГОРА']
    for word in WORDS:
        for rule in _RULES:
            print_edge(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Spatial Edge Density of Q6 CA')
    p.add_argument('--word',      default='ГОРА')
    p.add_argument('--rule',      default='and', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(edge_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.stats:
        print_edge_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_edge(args.word, rule, color)
    else:
        print_edge(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
