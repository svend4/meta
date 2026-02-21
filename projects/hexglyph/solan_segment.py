"""
solan_segment.py — Spatial Domain Segmentation of Q6 CA Attractor Orbits.

At each orbit step t the ring state is decomposed into *spatial segments*:
maximal circular arcs of consecutive cells carrying the same Q6 value.
This is the SPATIAL dual of solan_runs.py (which analyses temporal runs for
each individual cell).

Segment detection on a periodic ring
──────────────────────────────────────
For a state s = [s₀, s₁, …, s_{N−1}] on the ring the domain walls are:

    walls = {i ∈ {0..N−1} : s[i] ≠ s[(i+1) mod N]}

|walls| = 0   → one segment of length N (uniform state).
|walls| = k   → k segments; the i-th segment starts at (walls[i]+1)%N
               and ends at walls[(i+1)%N] (indices mod k).

Each segment is described by a (value, length) pair.

Relationship to solan_boundary
────────────────────────────────
  n_segments[t] = n_active_boundaries[t]   (when n_active > 0)
  n_segments[t] = 1                        (when n_active = 0)

solan_boundary gives WHICH cells are boundaries and WHAT values the XOR
takes; solan_segment gives the LENGTH distribution of the resulting domains.

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      One segment of length 16.  n_segs=[1]  max_len=[16]
      Ring is a single homogeneous domain.

  ГОРА AND  (P=2, anti-phase 47↔1)
      16 segments each of length 1 at every step.
      n_segs=[16, 16]  max_len=[1, 1]  mean_n_segs=16
      Ring is maximally fragmented: every cell is its own domain.
      ★ Despite the boundary orbit being constant (solan_boundary: b_period=1),
        the segment structure is also constant (n_segs=16 at both steps).

  ГОРА XOR3  (P=2, 4-periodic pattern)
      16 segments each of length 1 at every step.
      n_segs=[16, 16]  max_len=[1, 1]
      Like ГОРА AND: fully fragmented, no two adjacent cells share a value.

  ГОРА OR  (P=1, all=63)
      One segment of length 16.  n_segs=[1]  max_len=[16]

  ТУМАН XOR3  (P=8)
      n_segs=[15, 16, 16, 16, 16, 16, 16, 13]  mean_n_segs=15.5
      max_len=[2, 1, 1, 1, 1, 1, 1, 4]
      ★ Step 7: a wrap-around segment of length 4 appears — cells 14, 15,
        0, 1 all carry value 48, forming a segment that crosses the ring
        boundary.  (orbit[7] = [48,48,51,…,48,48].)
      Steps 1..6 are fully fragmented (n_segs=N, max_len=1).

Functions
─────────
  spatial_segments(state)                     → list[tuple[int,int]]
  seg_lengths_per_cell(state)                 → list[int]
  n_segments_step(word, rule, t, width)       → int
  n_segments(word, rule, width)               → list[int]
  max_seg_length(word, rule, width)           → list[int]
  min_seg_length(word, rule, width)           → list[int]
  mean_seg_length(word, rule, width)          → list[float]
  seg_lengths(word, rule, width)              → list[list[int]]
  global_max_seg_length(word, rule, width)    → int
  segment_summary(word, rule, width)          → dict
  all_segment(word, width)                    → dict[str, dict]
  build_segment_data(words, width)            → dict
  print_segment(word, rule, color)            → None
  print_segment_table(words, color)           → None

Запуск
──────
  python3 -m projects.hexglyph.solan_segment --word ТУМАН --rule xor3 --no-color
  python3 -m projects.hexglyph.solan_segment --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_segment --table --no-color
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


# ── Core spatial-segment functions ────────────────────────────────────────────

def spatial_segments(state: tuple[int, ...] | list[int]) -> list[tuple[int, int]]:
    """Return the list of (value, length) pairs for spatial segments on the ring.

    A spatial segment is a maximal circular arc of consecutive cells with the
    same Q6 value.  The ring is periodic: cell N−1 is adjacent to cell 0.

    If the ring is uniform (all cells equal), returns [(value, N)].
    Otherwise, the returned list has length equal to n_active_boundaries
    (solan_boundary) and the sum of lengths equals N.
    """
    N = len(state)
    walls = [i for i in range(N) if state[i] != state[(i + 1) % N]]
    if not walls:
        return [(int(state[0]), N)]
    segs: list[tuple[int, int]] = []
    nw = len(walls)
    for k in range(nw):
        start  = (walls[k] + 1) % N
        end    = walls[(k + 1) % nw]
        length = (end - start + 1) if end >= start else (N - start + end + 1)
        segs.append((int(state[start]), length))
    return segs


def seg_lengths_per_cell(state: tuple[int, ...] | list[int]) -> list[int]:
    """For each cell position i, return the length of the segment it belongs to.

    Useful for colour-coding each cell by segment size in a heatmap.
    """
    N    = len(state)
    out  = [0] * N
    segs = spatial_segments(state)
    # Reconstruct cell-to-segment mapping
    walls = [i for i in range(N) if state[i] != state[(i + 1) % N]]
    if not walls:
        return [N] * N
    nw = len(walls)
    for k, (val, length) in enumerate(segs):
        start = (walls[k] + 1) % N
        for j in range(length):
            out[(start + j) % N] = length
    return out


# ── Per-orbit functions ────────────────────────────────────────────────────────

def n_segments_step(word: str, rule: str, t: int,
                    width: int = _DEFAULT_W) -> int:
    """Number of spatial segments at a single orbit step t."""
    orbit = _get_orbit(word, rule, width)
    return len(spatial_segments(orbit[t]))


def n_segments(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """Number of spatial segments at each orbit step."""
    orbit = _get_orbit(word, rule, width)
    return [len(spatial_segments(s)) for s in orbit]


def max_seg_length(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """Maximum segment length at each orbit step."""
    orbit = _get_orbit(word, rule, width)
    return [max(l for _, l in spatial_segments(s)) for s in orbit]


def min_seg_length(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """Minimum segment length at each orbit step."""
    orbit = _get_orbit(word, rule, width)
    return [min(l for _, l in spatial_segments(s)) for s in orbit]


def mean_seg_length(word: str, rule: str,
                    width: int = _DEFAULT_W) -> list[float]:
    """Mean segment length at each orbit step (= width / n_segments[t])."""
    orbit = _get_orbit(word, rule, width)
    N = width
    return [round(N / len(spatial_segments(s)), 6) for s in orbit]


def seg_lengths(word: str, rule: str, width: int = _DEFAULT_W) -> list[list[int]]:
    """Sorted list of all segment lengths at each orbit step."""
    orbit = _get_orbit(word, rule, width)
    return [sorted([l for _, l in spatial_segments(s)], reverse=True)
            for s in orbit]


def global_max_seg_length(word: str, rule: str,
                           width: int = _DEFAULT_W) -> int:
    """Maximum segment length observed across all orbit steps."""
    return max(max_seg_length(word, rule, width))


# ── Summary function ───────────────────────────────────────────────────────────

def segment_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full spatial-segment summary for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    orbit    = _get_orbit(word, rule, width)
    P        = len(orbit)
    n_segs   = [len(spatial_segments(s)) for s in orbit]
    max_lens = [max(l for _, l in spatial_segments(s)) for s in orbit]
    min_lens = [min(l for _, l in spatial_segments(s)) for s in orbit]
    all_lens = [sorted([l for _, l in spatial_segments(s)], reverse=True)
                for s in orbit]
    return {
        'word':              word.upper(),
        'rule':              rule,
        'period':            P,
        'n_cells':           width,
        'n_segments':        n_segs,
        'mean_n_segments':   round(sum(n_segs) / P, 6),
        'max_n_segments':    max(n_segs),
        'min_n_segments':    min(n_segs),
        'max_seg_length':    max_lens,
        'min_seg_length':    min_lens,
        'global_max_len':    max(max_lens),
        'global_min_len':    min(min_lens),
        'seg_lengths':       all_lens,
        'fully_fragmented':  all(ns == width for ns in n_segs),
        'always_uniform':    all(ns == 1 for ns in n_segs),
    }


def all_segment(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """segment_summary for all 4 rules."""
    return {rule: segment_summary(word, rule, width) for rule in _RULES}


def build_segment_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact segment data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = segment_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'n_segments', 'mean_n_segments',
                                   'max_n_segments', 'min_n_segments',
                                   'max_seg_length', 'global_max_len',
                                   'global_min_len', 'seg_lengths',
                                   'fully_fragmented', 'always_uniform')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'


def print_segment(word: str = 'ТУМАН', rule: str = 'xor3',
                  color: bool = True) -> None:
    d   = segment_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',  'or':   'OR |'}.get(rule, rule)

    ff  = ' ★fully-fragmented' if d['fully_fragmented'] else ''
    uni = ' ★always-uniform'   if d['always_uniform']   else ''
    print(f'  {c}◈ Segment  {word.upper()}  |  {lbl}  P={d["period"]}'
          f'  mean_n_segs={d["mean_n_segments"]:.2f}'
          f'  global_max={d["global_max_len"]}{ff}{uni}{r}')
    print('  ' + '─' * 66)
    bar_max = max(d['max_seg_length'])
    for t in range(d['period']):
        ns     = d['n_segments'][t]
        ml     = d['max_seg_length'][t]
        ml_min = d['min_seg_length'][t]
        bar    = '█' * ml + '░' * (bar_max - ml)
        lens   = d['seg_lengths'][t][:6]
        print(f'  t={t}: n_segs={ns:2d}  max={ml}  min={ml_min}  '
              f'[{bar}]  {lens}')
    print()


def print_segment_table(words: list[str] | None = None,
                         color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    head = '  '.join(
        (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5s} ns gmax ff' + R
        for rl in _RULES)
    print(f'  {"Слово":10s}  {head}')
    print('  ' + '─' * 72)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = _RCOL.get(rule, '') if color else ''
            d   = segment_summary(word, rule)
            ff  = '★' if d['fully_fragmented'] else ' '
            parts.append(f'{col}{d["mean_n_segments"]:>4.1f}'
                         f' {d["global_max_len"]:>3d} {ff}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Spatial Domain Segmentation of Q6 CA')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args  = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.table:
        print_segment_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_segment(args.word, rule, color)
    else:
        print_segment(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
