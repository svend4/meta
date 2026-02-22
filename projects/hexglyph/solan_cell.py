"""
solan_cell.py — Per-Cell Temporal Analysis of Q6 CA Attractor Orbits.

For each cell position i ∈ {0, …, N−1} on the ring, the attractor orbit
defines a temporal series:

    X_i = [orbit[0][i], orbit[1][i], …, orbit[P−1][i]]

This module characterises each cell's individual dynamics and aggregates
these into spatial statistics over all N cells.

Key per-cell metrics
─────────────────────
  cell_vocab_size  |V_i|   number of distinct Q6 values in X_i
  cell_is_frozen         True when |V_i| = 1 (cell never changes)
  cell_transitions       #{t ∈ {0..P−1} : X_i[t] ≠ X_i[(t+1) mod P]}
  cell_mean              mean Q6 value over the orbit
  cell_var               variance of Q6 values

Spatial statistics (aggregated across all N cells at each step t)
──────────────────────────────────────────────────────────────────
  spatial_variance[t]    variance of {orbit[t][i] : i=0..N−1}

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      All 16 cells frozen at 0.  n_frozen=16.
      spatial_var=[0.0]  (uniform zero state).
      mean_vocab_size=1.0  mean_transitions=0.0

  ГОРА AND  (P=2, anti-phase {47, 1})
      0 frozen cells.  Each cell alternates 47↔1 (even cells) or 1↔47 (odd).
      cell_vocab_size=2  for every cell;  cell_transitions=2 for every cell.
      spatial_var=[529.0, 529.0]  — constant because the anti-phase pattern
        has the same variance at both steps (8×47 + 8×1 both times).
      mean_vocab_size=2.0  mean_transitions=2.0

  ГОРА XOR3  (P=2, 4-cluster spatial pattern)
      0 frozen cells.  4-fold repeating pattern of pairs:
        cell 0: [49, 33]   cell 1: [47, 17]
        cell 2: [15, 31]   cell 3: [63,  1]   then repeats for cells 4..15.
      cell_vocab_size=2  for all cells.
      spatial_var=[308.75, 164.75]  — different at each step because the two
        attractor states have different spatial spreads.
      mean_vocab_size=2.0  mean_transitions=2.0

  ГОРА OR  (P=1, all=63)
      All 16 cells frozen at 63.  n_frozen=16.
      spatial_var=[0.0]  (uniform saturated state).

  ТУМАН XOR3  (P=8)
      0 frozen cells.  Strong spatial heterogeneity:
        vocab_sizes = [4,6,7,7,6,5,4,3,3,4,5,6,7,7,6,4]  (symmetric)
        transitions = [4,6,7,8,8,8,8,8,8,8,8,8,8,7,6,4]
        ★ Cells 3..12 make 8 transitions (maximum: change at every step).
        ★ Min vocab size = 3 (cells 7, 8).   Max vocab size = 7 (cells 2, 12).
      spatial_var min=195.90 (step 1)  max=440.12 (step 0)
      mean_vocab_size=5.25  mean_transitions=7.125

Functions
─────────
  cell_series(word, rule, cell_idx, width)      → list[int]
  cell_vocab(word, rule, cell_idx, width)       → list[int]
  cell_hist(word, rule, cell_idx, width)        → dict[int, int]
  cell_vocab_size(word, rule, cell_idx, width)  → int
  cell_is_frozen(word, rule, cell_idx, width)   → bool
  cell_transitions(word, rule, cell_idx, width) → int
  cell_mean(word, rule, cell_idx, width)        → float
  cell_var(word, rule, cell_idx, width)         → float
  frozen_cells(word, rule, width)               → list[int]
  spatial_variance(word, rule, width)           → list[float]
  cell_summary(word, rule, cell_idx, width)     → dict
  orbit_cell_matrix(word, rule, width)          → list[dict]
  cell_agg(word, rule, width)                   → dict
  all_cell(word, width)                         → dict[str, dict]
  build_cell_data(words, width)                 → dict
  print_cell(word, rule, color)                 → None
  print_cell_table(words, color)                → None

Запуск
──────
  python3 -m projects.hexglyph.solan_cell --word ТУМАН --rule xor3 --no-color
  python3 -m projects.hexglyph.solan_cell --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_cell --table --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


def _get_orbit(word: str, rule: str, width: int) -> list[tuple[int, ...]]:
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Core per-cell functions ────────────────────────────────────────────────────

def cell_series(word: str, rule: str, cell_idx: int,
                width: int = _DEFAULT_W) -> list[int]:
    """Temporal series of Q6 values for cell i over the orbit: [orbit[t][i]]."""
    orbit = _get_orbit(word, rule, width)
    return [orbit[t][cell_idx] for t in range(len(orbit))]


def cell_vocab(word: str, rule: str, cell_idx: int,
               width: int = _DEFAULT_W) -> list[int]:
    """Sorted list of distinct Q6 values taken by cell i over the orbit."""
    return sorted(set(cell_series(word, rule, cell_idx, width)))


def cell_hist(word: str, rule: str, cell_idx: int,
              width: int = _DEFAULT_W) -> dict[int, int]:
    """Frequency count of each Q6 value for cell i (sorted by descending count)."""
    from collections import Counter
    cnt = Counter(cell_series(word, rule, cell_idx, width))
    return dict(sorted(cnt.items(), key=lambda x: -x[1]))


def cell_vocab_size(word: str, rule: str, cell_idx: int,
                    width: int = _DEFAULT_W) -> int:
    """Number of distinct Q6 values in cell i's temporal series."""
    return len(cell_vocab(word, rule, cell_idx, width))


def cell_is_frozen(word: str, rule: str, cell_idx: int,
                   width: int = _DEFAULT_W) -> bool:
    """True when cell i takes the same value at every step (never changes)."""
    return cell_vocab_size(word, rule, cell_idx, width) == 1


def cell_transitions(word: str, rule: str, cell_idx: int,
                     width: int = _DEFAULT_W) -> int:
    """Number of time steps t where orbit[t][i] ≠ orbit[(t+1)%P][i].

    Ranges from 0 (fully frozen) to P (changes at every step).
    """
    s = cell_series(word, rule, cell_idx, width)
    P = len(s)
    return sum(1 for t in range(P) if s[t] != s[(t + 1) % P])


def cell_mean(word: str, rule: str, cell_idx: int,
              width: int = _DEFAULT_W) -> float:
    """Mean Q6 value of cell i over the orbit."""
    s = cell_series(word, rule, cell_idx, width)
    return round(sum(s) / len(s), 6)


def cell_var(word: str, rule: str, cell_idx: int,
             width: int = _DEFAULT_W) -> float:
    """Variance of Q6 values of cell i over the orbit."""
    s = cell_series(word, rule, cell_idx, width)
    mu = sum(s) / len(s)
    return round(sum((v - mu) ** 2 for v in s) / len(s), 6)


# ── Spatial aggregates ─────────────────────────────────────────────────────────

def frozen_cells(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """List of cell indices that are frozen (never change) over the orbit."""
    return [i for i in range(width)
            if cell_is_frozen(word, rule, i, width)]


def spatial_variance(word: str, rule: str,
                     width: int = _DEFAULT_W) -> list[float]:
    """Per-step spatial variance of Q6 values across the N cells.

    spatial_variance[t] = Var({orbit[t][i] : i = 0..N−1}).
    High values indicate spatially heterogeneous states;
    0.0 indicates a uniform (all cells equal) state.
    """
    orbit = _get_orbit(word, rule, width)
    N = width
    result = []
    for state in orbit:
        vals = list(state)
        mu = sum(vals) / N
        result.append(round(sum((v - mu) ** 2 for v in vals) / N, 6))
    return result


# ── Summary functions ──────────────────────────────────────────────────────────

def cell_summary(word: str, rule: str, cell_idx: int,
                 width: int = _DEFAULT_W) -> dict:
    """Full summary for a single cell."""
    s    = cell_series(word, rule, cell_idx, width)
    P    = len(s)
    voc  = sorted(set(s))
    mu   = sum(s) / P
    var  = sum((v - mu) ** 2 for v in s) / P
    tc   = sum(1 for t in range(P) if s[t] != s[(t + 1) % P])
    return {
        'word':       word.upper(),
        'rule':       rule,
        'cell_idx':   cell_idx,
        'period':     P,
        'series':     s,
        'vocab':      voc,
        'vocab_size': len(voc),
        'is_frozen':  len(voc) == 1,
        'frozen_val': voc[0] if len(voc) == 1 else None,
        'transitions': tc,
        'mean':       round(mu, 6),
        'var':        round(var, 6),
    }


def orbit_cell_matrix(word: str, rule: str,
                      width: int = _DEFAULT_W) -> list[dict]:
    """Per-cell summary for all N cells (list indexed by cell index 0..N−1)."""
    return [cell_summary(word, rule, i, width) for i in range(width)]


def cell_agg(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Aggregate cell statistics for word × rule.

    Returns a dict with population-level metrics across all N cells.
    """
    from projects.hexglyph.solan_traj import word_trajectory
    P         = word_trajectory(word.upper(), rule, width)['period']
    cells     = orbit_cell_matrix(word, rule, width)
    n_frz     = sum(1 for c in cells if c['is_frozen'])
    vsizes    = [c['vocab_size']  for c in cells]
    trans     = [c['transitions'] for c in cells]
    means_c   = [c['mean']        for c in cells]
    vars_c    = [c['var']         for c in cells]
    spvar     = spatial_variance(word, rule, width)
    frz_list  = [c['cell_idx']   for c in cells if c['is_frozen']]

    return {
        'word':              word.upper(),
        'rule':              rule,
        'period':            P,
        'n_cells':           width,
        'n_frozen':          n_frz,
        'frozen_cell_ids':   frz_list,
        'vocab_sizes':       vsizes,
        'mean_vocab_size':   round(sum(vsizes) / width, 6),
        'max_vocab_size':    max(vsizes),
        'min_vocab_size':    min(vsizes),
        'transitions':       trans,
        'mean_transitions':  round(sum(trans) / width, 6),
        'max_transitions':   max(trans),
        'min_transitions':   min(trans),
        'cell_means':        [round(m, 4) for m in means_c],
        'cell_vars':         [round(v, 4) for v in vars_c],
        'spatial_variance':  spvar,
        'mean_spatial_var':  round(sum(spvar) / len(spvar), 6),
        'max_spatial_var':   max(spvar),
        'min_spatial_var':   min(spvar),
        'uniform_spatial':   len(set(spvar)) == 1,
    }


def all_cell(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """cell_agg for all 4 rules."""
    return {rule: cell_agg(word, rule, width) for rule in _RULES}


def build_cell_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact cell data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = cell_agg(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'n_frozen', 'vocab_sizes',
                                   'mean_vocab_size', 'max_vocab_size',
                                   'transitions', 'mean_transitions',
                                   'max_transitions',
                                   'mean_spatial_var', 'max_spatial_var',
                                   'min_spatial_var', 'uniform_spatial')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'
_DIM  = '\033[2m'
_CELL_COLS = [
    '\033[34m', '\033[36m', '\033[32m', '\033[33m',
    '\033[31m', '\033[35m', '\033[37m', '\033[90m',
]


def print_cell(word: str = 'ТУМАН', rule: str = 'xor3',
               color: bool = True) -> None:
    d   = cell_agg(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',  'or':   'OR |'}.get(rule, rule)

    print(f'  {c}◈ Cell  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'frozen={d["n_frozen"]}/{d["n_cells"]}  '
          f'mean_vocab={d["mean_vocab_size"]:.2f}{r}')
    print('  ' + '─' * 62)
    print(f'  mean_transitions={d["mean_transitions"]:.3f}  '
          f'max_transitions={d["max_transitions"]}')
    print(f'  spatial_var: min={d["min_spatial_var"]:.2f}  '
          f'max={d["max_spatial_var"]:.2f}  '
          f'uniform={d["uniform_spatial"]}')
    print()
    bar_max = max(d['vocab_sizes']) if d['vocab_sizes'] else 1
    for i in range(d['n_cells']):
        bc  = _CELL_COLS[i % len(_CELL_COLS)] if color else ''
        vs  = d['vocab_sizes'][i]
        tc  = d['transitions'][i]
        bar = '█' * vs + '░' * (bar_max - vs)
        frz = ' ★frozen' if d['n_frozen'] > 0 and i in d['frozen_cell_ids'] else ''
        print(f'  {bc}c{i:2d}{r}  vocab={vs}  trans={tc:2d}  [{bar}]{frz}')
    print()


def print_cell_table(words: list[str] | None = None,
                     color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    head = '  '.join(
        (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5s} frz mvoc mtran' + R
        for rl in _RULES)
    print(f'  {"Слово":10s}  {head}')
    print('  ' + '─' * 72)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = _RCOL.get(rule, '') if color else ''
            d   = cell_agg(word, rule)
            parts.append(f'{col}{d["n_frozen"]:>3d} '
                         f'{d["mean_vocab_size"]:>4.1f} '
                         f'{d["mean_transitions"]:>5.2f}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Per-Cell Temporal Analysis of Q6 CA')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args  = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(cell_agg(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.table:
        print_cell_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_cell(args.word, rule, color)
    else:
        print_cell(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
