"""
solan_dist.py — Pairwise Orbit-Step Distance Map of Q6 CA Attractors.

For a P-step attractor orbit, we compute the P×P symmetric distance matrix:

    M_Q6[t][s]  = |{i : orbit[t][i] ≠ orbit[s][i]}|   ∈ {0..N}    (Q6-level)
    M_bit[t][s] = Σᵢ popcount(orbit[t][i] XOR orbit[s][i]) ∈ {0..6N} (bit-level)

Diagonal is always 0.  Key scalars derived from the matrix:

    diameter_q6  = max off-diagonal entry of M_Q6
    mean_dist    = mean of all off-diagonal entries
    near_returns = {t > 0 : M_Q6[0][t] < N/2}   (partial returns to start)
    packing_eff  = mean_dist / diameter_q6        (→ 1 = uniform spread)

Relation to existing modules
──────────────────────────────
  solan_recurrence.py : binary thresholded matrix  R[t,s] = (M_Q6[t,s] ≤ ε)
  solan_dist          : full continuous distance matrix (no threshold)

Key results  (N = width = 16)
──────────────────────────────
  ТУМАН XOR  (P=1)
      1×1 matrix [[0]].  diameter_q6=0, mean_dist=0.  Trivial.

  ГОРА AND  (P=2)
      M_Q6 = [[0,16],[16,0]].  All 16 cells differ between the two steps.
      47 XOR 1 = 0b101110 → 4 bit-diffs/cell → M_bit[0][1] = 64.
      diameter_q6=16 (maximum possible), mean_dist=16, packing_eff=1.0.

  ГОРА XOR3  (P=2)
      M_Q6 = [[0,16],[16,0]].  diameter_q6=16, mean_dist=16.
      M_bit[0][1] = 48  (cluster-dependent bit differences).

  ТУМАН XOR3  (P=8) — richest case
      distance_series = [0, 16, 16, 6, 16, 16, 12, 14]
      ★  NEAR-RETURN at t=3: only 6/16 cells differ from t=0.
          bit_distance(0,3) = 18  (18/(6·16) = 18.75% of possible bits).
      Matrix statistics:  mean_dist=13.93  packing_eff=0.87
      Many distinct P=8/XOR3 words share this t=3 near-return signature.

Functions
─────────
  step_distance_q6(a, b)                   → int
  step_distance_bits(a, b)                  → int
  distance_series_q6(word, rule, width)     → list[int]
  distance_series_bits(word, rule, width)   → list[int]
  distance_matrix_q6(word, rule, width)     → list[list[int]]
  distance_matrix_bits(word, rule, width)   → list[list[int]]
  orbit_diameter_q6(word, rule, width)      → int
  orbit_diameter_bits(word, rule, width)    → int
  mean_distance_q6(word, rule, width)       → float
  packing_efficiency(word, rule, width)     → float
  near_returns(word, rule, width, thresh)   → list[int]
  dist_summary(word, rule, width)           → dict
  all_dist(word, width)                     → dict[str, dict]
  build_dist_data(words, width)             → dict
  print_dist(word, rule, color)             → None
  print_dist_table(words, color)            → None

Запуск
──────
  python3 -m projects.hexglyph.solan_dist --word ТУМАН --rule xor3 --no-color
  python3 -m projects.hexglyph.solan_dist --table --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


# ── Core distance primitives ───────────────────────────────────────────────────

def step_distance_q6(a: list[int] | tuple[int, ...],
                     b: list[int] | tuple[int, ...]) -> int:
    """Q6-level distance: number of cells where the two states differ."""
    return sum(x != y for x, y in zip(a, b))


def step_distance_bits(a: list[int] | tuple[int, ...],
                       b: list[int] | tuple[int, ...]) -> int:
    """Bit-level distance: total popcount of XOR across all cell pairs."""
    return sum(bin(x ^ y).count('1') for x, y in zip(a, b))


# ── Series ─────────────────────────────────────────────────────────────────────

def distance_series_q6(word: str, rule: str,
                       width: int = _DEFAULT_W) -> list[int]:
    """Q6 distance from orbit step 0 for each step t = 0..P−1."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    ref   = orbit[0]
    return [step_distance_q6(ref, step) for step in orbit]


def distance_series_bits(word: str, rule: str,
                         width: int = _DEFAULT_W) -> list[int]:
    """Bit-level distance from orbit step 0 for each step t = 0..P−1."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    ref   = orbit[0]
    return [step_distance_bits(ref, step) for step in orbit]


# ── Matrices ───────────────────────────────────────────────────────────────────

def distance_matrix_q6(word: str, rule: str,
                       width: int = _DEFAULT_W) -> list[list[int]]:
    """Symmetric P×P Q6-distance matrix: M[t][s] = d_Q6(orbit[t], orbit[s])."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    return [[step_distance_q6(orbit[t], orbit[s]) for s in range(P)]
            for t in range(P)]


def distance_matrix_bits(word: str, rule: str,
                         width: int = _DEFAULT_W) -> list[list[int]]:
    """Symmetric P×P bit-distance matrix: M[t][s] = d_bit(orbit[t], orbit[s])."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    return [[step_distance_bits(orbit[t], orbit[s]) for s in range(P)]
            for t in range(P)]


# ── Aggregate statistics ───────────────────────────────────────────────────────

def orbit_diameter_q6(word: str, rule: str, width: int = _DEFAULT_W) -> int:
    """Maximum Q6-distance over all orbit step pairs (excluding diagonal)."""
    mat = distance_matrix_q6(word, rule, width)
    P   = len(mat)
    if P <= 1:
        return 0
    return max(mat[t][s] for t in range(P) for s in range(t + 1, P))


def orbit_diameter_bits(word: str, rule: str, width: int = _DEFAULT_W) -> int:
    """Maximum bit-distance over all orbit step pairs (excluding diagonal)."""
    mat = distance_matrix_bits(word, rule, width)
    P   = len(mat)
    if P <= 1:
        return 0
    return max(mat[t][s] for t in range(P) for s in range(t + 1, P))


def mean_distance_q6(word: str, rule: str, width: int = _DEFAULT_W) -> float:
    """Mean Q6-distance over all off-diagonal orbit step pairs."""
    mat = distance_matrix_q6(word, rule, width)
    P   = len(mat)
    if P <= 1:
        return 0.0
    vals = [mat[t][s] for t in range(P) for s in range(P) if t != s]
    return round(sum(vals) / len(vals), 6)


def packing_efficiency(word: str, rule: str, width: int = _DEFAULT_W) -> float:
    """mean_dist / diameter_q6: how uniformly orbit steps are spread in space.

    → 1.0: all pairs are at maximum distance (fully packed).
    → 0.0: most pairs are near each other (clustered).
    P = 1 (single-step orbit) returns 0.0 by convention.
    """
    diam = orbit_diameter_q6(word, rule, width)
    if diam == 0:
        return 0.0
    return round(mean_distance_q6(word, rule, width) / diam, 6)


def near_returns(word: str, rule: str, width: int = _DEFAULT_W,
                 threshold: int | None = None) -> list[int]:
    """Step indices t > 0 where d_Q6(orbit[0], orbit[t]) < threshold.

    Default threshold = N // 2 (fewer than half the cells differ).
    """
    N   = width
    thr = threshold if threshold is not None else N // 2
    ds  = distance_series_q6(word, rule, width)
    return [t for t, d in enumerate(ds) if t > 0 and d < thr]


# ── Summary ────────────────────────────────────────────────────────────────────

def dist_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full distance summary for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj    = word_trajectory(word.upper(), rule, width)
    P       = traj['period']
    N       = width
    ds_q6   = distance_series_q6(word, rule, width)
    ds_bit  = distance_series_bits(word, rule, width)
    mat     = distance_matrix_q6(word, rule, width)
    diam_q6 = orbit_diameter_q6(word, rule, width)
    diam_bt = orbit_diameter_bits(word, rule, width)
    mean_d  = mean_distance_q6(word, rule, width)
    pack    = packing_efficiency(word, rule, width)
    nr      = near_returns(word, rule, width)

    # Normalised distances
    ds_q6_norm = [round(d / N, 6) for d in ds_q6]
    ds_bit_norm = [round(d / (N * 6), 6) for d in ds_bit]

    # Closest pair (t ≠ s)
    if P > 1:
        closest_dist = min(mat[t][s] for t in range(P) for s in range(P) if t != s)
        closest_pair = next((t, s)
                             for t in range(P) for s in range(t + 1, P)
                             if mat[t][s] == closest_dist)
    else:
        closest_dist = 0
        closest_pair = (0, 0)

    return {
        'word':               word.upper(),
        'rule':               rule,
        'period':             P,
        'N':                  N,
        'distance_series_q6':  ds_q6,
        'distance_series_bits': ds_bit,
        'ds_q6_norm':         ds_q6_norm,
        'ds_bit_norm':        ds_bit_norm,
        'diameter_q6':        diam_q6,
        'diameter_bits':      diam_bt,
        'diam_q6_norm':       round(diam_q6 / N, 6),
        'mean_dist_q6':       mean_d,
        'mean_norm':          round(mean_d / N, 6),
        'packing_efficiency': pack,
        'near_returns':       nr,
        'closest_dist_q6':   closest_dist,
        'closest_pair':       closest_pair,
    }


def all_dist(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """dist_summary for all 4 rules."""
    return {rule: dist_summary(word, rule, width) for rule in _RULES}


def build_dist_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact distance data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = dist_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'distance_series_q6', 'distance_series_bits',
                                   'diameter_q6', 'diameter_bits',
                                   'mean_dist_q6', 'mean_norm',
                                   'packing_efficiency', 'near_returns',
                                   'closest_dist_q6', 'closest_pair')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'N_max_q6': width, 'N_max_bits': width * 6, 'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'
_HEAT = ['  ', '░ ', '▒ ', '▓ ', '██']


def _hcell(val: int, vmax: int) -> str:
    if vmax == 0 or val == 0:
        return f'  {val:2d}'
    frac = val / vmax
    sym  = '░' if frac < 0.25 else ('▒' if frac < 0.5 else ('▓' if frac < 0.75 else '█'))
    return f'{sym}{val:2d}'


def print_dist(word: str = 'ТУМАН', rule: str = 'xor3',
               color: bool = True) -> None:
    d   = dist_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',   'or':   'OR |'}.get(rule, rule)
    P   = d['period']
    N   = d['N']

    print(f'  {c}◈ Dist  {word.upper()}  |  {lbl}  P={P}  '
          f'diam={d["diameter_q6"]}/{N}  mean={d["mean_norm"]:.3f}  '
          f'pack={d["packing_efficiency"]:.3f}{r}')
    print('  ' + '─' * 62)
    print(f'  Q6  series : {d["distance_series_q6"]}')
    print(f'  Bit series : {d["distance_series_bits"]}')
    print(f'  Near-returns (thr={N//2}): {d["near_returns"]}')
    print(f'  Closest pair: t={d["closest_pair"]}  dist_q6={d["closest_dist_q6"]}')
    print(f'  Diameter Q6={d["diameter_q6"]}/{N}  '
          f'Bit={d["diameter_bits"]}/{N*6}={round(d["diameter_bits"]/(N*6),3):.3f}')

    if P > 1:
        mat  = distance_matrix_q6(word, rule)
        vmax = d['diameter_q6'] or 1
        print(f'\n  Q6 distance matrix ({P}×{P}):')
        print('      ' + '  '.join(f's={s}' for s in range(P)))
        for t in range(P):
            row_s = '  '.join(_hcell(mat[t][s], vmax) for s in range(P))
            print(f'  t={t}: {row_s}')
    print()


def print_dist_table(words: list[str] | None = None,
                     color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    print(f'  {"Слово":10s}  '
          + '  '.join(
              (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5s}  diam  pack' + R
              for rl in _RULES))
    print('  ' + '─' * 70)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col  = (_RCOL.get(rule, '') if color else '')
            d    = dist_summary(word, rule)
            diam = d['diameter_q6']
            pack = d['packing_efficiency']
            parts.append(f'{col}{diam:>8d}  {pack:.3f}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Pairwise Orbit Distance Map of Q6 CA')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(dist_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.table:
        print_dist_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_dist(args.word, rule, color)
    else:
        print_dist(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
