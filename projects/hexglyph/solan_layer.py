"""
solan_layer.py — Bit-Layer Decomposition of Q6 CA Attractors.

Because all Q6 rules (XOR, XOR3, AND, OR) are BITWISE operations, each of
the 6 bit planes (b=0..5) evolves INDEPENDENTLY:

    new_b(i) = rule_1bit(left_b(i), center_b(i), right_b(i))

where subscript b denotes the b-th bit of the cell value.

This independence yields a fundamental result:

    LCM THEOREM: lcm(p₀, p₁, …, p₅) = P

where pᵦ is the period of bit-b plane and P is the overall orbit period.

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1)
      All 6 planes frozen to 0.
      frozen_0={0,1,2,3,4,5}  frozen_1={}  active={}
      lcm(1,1,1,1,1,1) = 1 = P  ✓

  ГОРА AND  (P=2)
      Bit 0: frozen-1  (bit 0 of {47,1} is always 1).
      Bit 4: frozen-0  (bit 4 of {47,1} is always 0).
      Bits 1,2,3,5: period 2 (anti-phase 50/50 spatial pattern).
      frozen_0={4}  frozen_1={0}  active={1,2,3,5}
      lcm(1,2,2,2,1,2) = 2 = P  ✓

  ГОРА XOR3  (P=2)
      Bit 0: frozen-1.
      Bits 1,2,3,5: period 2, density oscillates 75%↔25%.
      Bit 4:        period 2, density oscillates 50%↔50%.
      frozen_0={}  frozen_1={0}  active={1,2,3,4,5}
      lcm(1,2,2,2,2,2) = 2 = P  ✓
      ★ Bit-1,2,3,5 density 75%↔25% vs bit-4 density 50%↔50%
        reveals asymmetric cluster structure from solan_symm.

  ГОРА OR  (P=1)
      All 6 planes frozen to 1 (OR saturates all bits to 1).
      frozen_0={}  frozen_1={0,1,2,3,4,5}  active={}
      lcm(1,1,1,1,1,1) = 1 = P  ✓

  ТУМАН XOR3  (P=8)
      All 6 planes have period 8 (no frozen planes).
      frozen_0={}  frozen_1={}  active={0,1,2,3,4,5}
      lcm(8,8,8,8,8,8) = 8 = P  ✓
      Bit densities vary non-uniformly across 8 steps.

Functions
─────────
  bit_plane(word, rule, bit, width)       → list[tuple[int,...]]
  plane_period(word, rule, bit, width)    → int
  plane_type(word, rule, bit, width)      → str
  plane_density(word, rule, bit, width)   → list[float]
  layer_periods(word, rule, width)        → list[int]
  active_bits(word, rule, width)          → list[int]
  frozen_bits(word, rule, width)          → tuple[list[int], list[int]]
  lcm_equals_period(word, rule, width)    → bool
  layer_summary(word, rule, width)        → dict
  all_layers(word, width)                 → dict[str, dict]
  build_layer_data(words, width)          → dict
  print_layer(word, rule, color)          → None
  print_layer_table(words, color)         → None

Запуск
──────
  python3 -m projects.hexglyph.solan_layer --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_layer --table --no-color
"""

from __future__ import annotations
import sys
import argparse
import math
from functools import reduce

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16

_PLANE_TYPES = ('frozen_0', 'frozen_1', 'uniform_alt',
                'uniform_irr', 'spatial', 'irregular')


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def _lcm(a: int, b: int) -> int:
    return a * b // _gcd(a, b)

def _lcm_list(lst: list[int]) -> int:
    return reduce(_lcm, lst, 1)


# ── Core functions ─────────────────────────────────────────────────────────────

def bit_plane(word: str, rule: str, bit: int,
              width: int = _DEFAULT_W) -> list[tuple[int, ...]]:
    """Periodic orbit of bit-b plane: list[tuple of 0/1 values per cell].

    Each element is one time step of the periodic attractor orbit with
    only the b-th bit extracted.  Verified bitwise-independent of other
    planes for all Q6 rules.
    """
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    return [tuple((v >> bit) & 1 for v in state) for state in orbit]


def plane_period(word: str, rule: str, bit: int,
                 width: int = _DEFAULT_W) -> int:
    """Minimal period of bit-b plane (divisor of overall orbit period P)."""
    plane = bit_plane(word, rule, bit, width)
    P = len(plane)
    for d in range(1, P + 1):
        if P % d == 0 and all(plane[t] == plane[t % d] for t in range(P)):
            return d
    return P  # pragma: no cover


def plane_type(word: str, rule: str, bit: int,
               width: int = _DEFAULT_W) -> str:
    """Classify the bit-b plane into one of 5 types.

    'frozen_0'   — all values 0 at all (t, i)
    'frozen_1'   — all values 1 at all (t, i)
    'uniform_alt'— all cells same at each step, and value alternates 0↔1
    'uniform_irr'— all cells same at each step, period ≥ 2 non-alternating
    'spatial'    — cells differ at some step (non-uniform spatial pattern)
    """
    plane = bit_plane(word, rule, bit, width)
    all_vals = {v for row in plane for v in row}
    spatially_uniform = all(len(set(row)) == 1 for row in plane)

    if all_vals == {0}:
        return 'frozen_0'
    if all_vals == {1}:
        return 'frozen_1'
    if spatially_uniform:
        step_vals = [row[0] for row in plane]
        # alternating: 0,1,0,1,... or 1,0,1,0,...
        if all(step_vals[t] != step_vals[t + 1] for t in range(len(plane) - 1)):
            return 'uniform_alt'
        return 'uniform_irr'
    return 'spatial'


def plane_density(word: str, rule: str, bit: int,
                  width: int = _DEFAULT_W) -> list[float]:
    """Per-step density (fraction of cells with bit b = 1) over the orbit."""
    plane = bit_plane(word, rule, bit, width)
    N = width
    return [round(sum(row) / N, 6) for row in plane]


# ── Aggregate layer functions ──────────────────────────────────────────────────

def layer_periods(word: str, rule: str,
                  width: int = _DEFAULT_W) -> list[int]:
    """List of per-plane minimal periods [p₀, p₁, p₂, p₃, p₄, p₅]."""
    return [plane_period(word, rule, b, width) for b in range(6)]


def active_bits(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """Bits whose plane has non-trivial dynamics (not frozen to 0 or 1)."""
    return [b for b in range(6)
            if plane_type(word, rule, b, width) not in ('frozen_0', 'frozen_1')]


def frozen_bits(word: str, rule: str,
                width: int = _DEFAULT_W) -> tuple[list[int], list[int]]:
    """(frozen_0_bits, frozen_1_bits): bits frozen to 0 or 1 respectively."""
    f0, f1 = [], []
    for b in range(6):
        pt = plane_type(word, rule, b, width)
        if pt == 'frozen_0':
            f0.append(b)
        elif pt == 'frozen_1':
            f1.append(b)
    return (f0, f1)


def lcm_equals_period(word: str, rule: str,
                      width: int = _DEFAULT_W) -> bool:
    """Verify the LCM Theorem: lcm(p₀,...,p₅) == P (overall orbit period).

    Should be True for all words and rules because the 6 bit planes
    evolve independently and the orbit period is exactly the LCM of
    the individual plane periods.
    """
    from projects.hexglyph.solan_traj import word_trajectory
    P   = word_trajectory(word.upper(), rule, width)['period']
    lps = layer_periods(word, rule, width)
    return _lcm_list(lps) == P


# ── Summary ────────────────────────────────────────────────────────────────────

def layer_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full bit-layer summary for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    P      = traj['period']
    lps    = layer_periods(word, rule, width)
    types  = [plane_type(word, rule, b, width) for b in range(6)]
    dens   = [plane_density(word, rule, b, width) for b in range(6)]
    act    = active_bits(word, rule, width)
    f0, f1 = frozen_bits(word, rule, width)
    lcm_ok = _lcm_list(lps) == P

    # density variance across time (for active planes): measure of oscillation
    dens_var = [
        round(sum((d - sum(ds) / len(ds)) ** 2 for d in ds) / len(ds), 6)
        if len(ds) > 1 else 0.0
        for ds in dens
    ]

    # mean density across all t for each bit
    mean_dens = [round(sum(d) / len(d), 6) for d in dens]

    return {
        'word':         word.upper(),
        'rule':         rule,
        'period':       P,
        'plane_periods': lps,
        'plane_types':  types,
        'plane_density': dens,
        'mean_density': mean_dens,
        'density_var':  dens_var,
        'active_bits':  act,
        'n_active':     len(act),
        'frozen_0_bits': f0,
        'frozen_1_bits': f1,
        'n_frozen':     len(f0) + len(f1),
        'lcm_period':   _lcm_list(lps),
        'lcm_equals_P': lcm_ok,
    }


def all_layers(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """layer_summary for all 4 rules."""
    return {rule: layer_summary(word, rule, width) for rule in _RULES}


def build_layer_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact layer data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = layer_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'plane_periods', 'plane_types',
                                   'mean_density', 'density_var',
                                   'active_bits', 'n_active',
                                   'frozen_0_bits', 'frozen_1_bits',
                                   'n_frozen', 'lcm_period', 'lcm_equals_P')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'
_TYPE_SYM = {
    'frozen_0':   '▪0',
    'frozen_1':   '▪1',
    'uniform_alt': '⇌ ',
    'uniform_irr': '≈ ',
    'spatial':    '⋯ ',
    'irregular':  '?? ',
}
_DENS_COLORS = [
    '\033[34m', '\033[36m', '\033[32m', '\033[33m', '\033[31m', '\033[35m',
]


def print_layer(word: str = 'ГОРА', rule: str = 'xor3',
                color: bool = True) -> None:
    d   = layer_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    print(f'  {c}◈ Layer  {word.upper()}  |  {lbl}  '
          f'P={d["period"]}  active={d["n_active"]}  frozen={d["n_frozen"]}{r}')
    print('  ' + '─' * 62)
    print(f'  LCM theorem: lcm({",".join(map(str,d["plane_periods"]))}) '
          f'= {d["lcm_period"]} = P → {d["lcm_equals_P"]}')
    print(f'  frozen_0={d["frozen_0_bits"]}  frozen_1={d["frozen_1_bits"]}  '
          f'active={d["active_bits"]}')
    print()
    for b in range(6):
        pt     = d['plane_types'][b]
        prd    = d['plane_periods'][b]
        sym    = _TYPE_SYM.get(pt, '?? ')
        bc     = _DENS_COLORS[b] if color else ''
        dens   = [f'{x:.2f}' for x in d['plane_density'][b]]
        dvar   = d['density_var'][b]
        print(f'  {bc}b{b}{r}  {sym}  p={prd}  '
              f'μ={d["mean_density"][b]:.3f}  σ²={dvar:.4f}  '
              f'density(t)=[{", ".join(dens)}]')
    print()


def print_layer_table(words: list[str] | None = None,
                      color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    head_parts = []
    for rl in _RULES:
        col = _RCOL.get(rl, '') if color else ''
        head_parts.append(f'{col}{rl.upper():>5s}  act frz{R}')
    print(f'  {"Слово":10s}  ' + '  '.join(head_parts))
    print('  ' + '─' * 60)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = _RCOL.get(rule, '') if color else ''
            act = len(active_bits(word, rule))
            frz = len(frozen_bits(word, rule)[0]) + len(frozen_bits(word, rule)[1])
            P   = layer_summary(word, rule)['period']
            parts.append(f'{col}P={P:<3d} {act:>2d}  {frz:>2d}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Bit-Layer Decomposition of Q6 CA')
    p.add_argument('--word',      default='ГОРА')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.table:
        print_layer_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_layer(args.word, rule, color)
    else:
        print_layer(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
