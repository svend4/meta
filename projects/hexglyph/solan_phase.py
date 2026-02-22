"""
solan_phase.py — Phase Offset Analysis of Q6 CA Attractor Cells.

Two cells i and j in a period-P attractor are *phase-synchronized with
offset k* if their temporal series satisfy

    series_j[t] = series_i[(t + k) mod P]   for all t ∈ 0 … P−1.

The N×N *phase matrix* M[i][j] = k  (or None if no such k exists) encodes
the full temporal structure of the attractor.

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, const 0)
      All 16 cells have the same series [0].
      Phase matrix: all entries = 0 (trivial, 100% sync).
      1 cluster (all 16 cells), no inter-cluster structure.

  ГОРА AND  (P=2, 47↔1)
      2 distinct series (8 cells each):
        Group A (cells 0,2,4,…,14): [47, 1]
        Group B (cells 1,3,5,…,15): [1, 47]
      Phase matrix: all 256 entries ≠ None → 100% synchronized.
        Within-group (A×A, B×B): offset 0.
        Between-group (A×B, B×A): offset 1 = P/2 (ANTI-PHASE).
      → Perfect anti-phase synchronization across even/odd cells.

  ГОРА XOR3  (P=2, 4 distinct series)
      4 groups of 4 cells each (spatial period 4):
        G0 {0,4,8,12}: [49,33]  G1 {1,5,9,13}: [47,17]
        G2 {2,6,10,14}: [15,31] G3 {3,7,11,15}: [63,1]
      Inter-group: None (series are not mutual circular shifts) → 25% sync.
      → Purely spatial clustering, no temporal phase relation.

  ТУМАН XOR3  (P=8)
      16 distinct series — all cells have unique orbits.
      Phase matrix: only diagonal = 0, rest = None → 6.25% sync.
      → Maximal phase diversity; no cross-cell synchronization.

Interpretation of sync_fraction
  1.0  : all pairs are phase-related (pure spatial/temporal structure)
  1/N  : only self-synchronization (all series unique, no phase relation)
  other: partial clustering (some spatial structure survives)

Functions
─────────
  series_match(si, sj, offset)              → bool
  phase_offset(si, sj)                      → int | None
  sync_matrix(word, rule, width)            → list[list[int|None]]  N×N
  sync_clusters(word, rule, width)          → list[list[int]]  (offset-0 groups)
  inter_cluster_offsets(word, rule, width)  → dict[(int,int), int|None]
  sync_fraction(word, rule, width)          → float
  phase_summary(word, rule, width)          → dict
  all_phase(word, width)                    → dict[str, dict]
  build_phase_data(words, width)            → dict
  print_phase(word, rule, color)            → None
  print_phase_stats(words, color)           → None

Запуск
──────
  python3 -m projects.hexglyph.solan_phase --word ГОРА --rule and
  python3 -m projects.hexglyph.solan_phase --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_phase --stats --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_NONE_VAL:  int = -1   # sentinel for "not synchronized" in matrix repr


# ── Core primitives ────────────────────────────────────────────────────────────

def series_match(si: list[int], sj: list[int], offset: int) -> bool:
    """True iff sj[t] == si[(t + offset) % P] for all t."""
    P = len(si)
    if P == 0 or len(sj) != P:
        return False
    return all(sj[t] == si[(t + offset) % P] for t in range(P))


def phase_offset(si: list[int], sj: list[int]) -> int | None:
    """
    Minimum non-negative offset k such that sj is a k-circular-shift of si.
    Returns None if no such k exists (series not phase-synchronized).
    """
    P = len(si)
    if P == 0 or len(sj) != P:
        return None
    for k in range(P):
        if series_match(si, sj, k):
            return k
    return None


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_series(word: str, rule: str, width: int) -> list[list[int]]:
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    return [[orbit[t][i] for t in range(P)] for i in range(width)]


# ── Phase matrix ──────────────────────────────────────────────────────────────

def sync_matrix(word: str, rule: str,
                width: int = _DEFAULT_W) -> list[list[int | None]]:
    """
    N×N phase matrix: M[i][j] = phase_offset(series_i, series_j).

    M[i][j] = 0      : cells i and j have identical series
    M[i][j] = k > 0  : series_j is a k-step forward shift of series_i
    M[i][j] = None   : cells not phase-synchronized
    Diagonal always = 0 (self-synchronization).
    """
    series = _get_series(word, rule, width)
    N = len(series)
    return [[phase_offset(series[i], series[j]) for j in range(N)]
            for i in range(N)]


# ── Cluster analysis ──────────────────────────────────────────────────────────

def sync_clusters(word: str, rule: str,
                  width: int = _DEFAULT_W) -> list[list[int]]:
    """
    Partition cells into offset-0 synchronization clusters (identical series).

    Returns list of groups; each group is a list of cell indices.
    """
    series = _get_series(word, rule, width)
    seen: dict[tuple[int, ...], list[int]] = {}
    for i, s in enumerate(series):
        key = tuple(s)
        if key not in seen:
            seen[key] = []
        seen[key].append(i)
    return list(seen.values())


def inter_cluster_offsets(
        word: str, rule: str,
        width: int = _DEFAULT_W) -> dict[tuple[int, int], int | None]:
    """
    Pairwise phase offsets between distinct synchronization clusters.

    Returns dict  (g1_idx, g2_idx) → phase_offset(rep_g1, rep_g2).
    """
    series  = _get_series(word, rule, width)
    clusters = sync_clusters(word, rule, width)
    reps = [series[g[0]] for g in clusters]
    result: dict[tuple[int, int], int | None] = {}
    for a in range(len(clusters)):
        for b in range(len(clusters)):
            if a != b:
                result[(a, b)] = phase_offset(reps[a], reps[b])
    return result


def sync_fraction(word: str, rule: str, width: int = _DEFAULT_W) -> float:
    """
    Fraction of cell pairs (i, j) — including i==j — that are phase-synchronized
    for some offset k ∈ {0, …, P−1}.
    """
    mat = sync_matrix(word, rule, width)
    N   = len(mat)
    synced = sum(1 for i in range(N) for j in range(N)
                 if mat[i][j] is not None)
    return synced / (N * N) if N > 0 else 0.0


# ── Summary ───────────────────────────────────────────────────────────────────

def phase_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Comprehensive phase-offset statistics for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj    = word_trajectory(word.upper(), rule, width)
    period  = traj['period']
    series  = _get_series(word, rule, width)
    N       = len(series)
    mat     = [[phase_offset(series[i], series[j]) for j in range(N)]
               for i in range(N)]

    # Sync fraction
    synced = sum(1 for i in range(N) for j in range(N)
                 if mat[i][j] is not None)
    sf = synced / (N * N) if N else 0.0

    # Offset histogram (for synchronized pairs only)
    offset_hist: dict[int, int] = {}
    for i in range(N):
        for j in range(N):
            v = mat[i][j]
            if v is not None:
                offset_hist[v] = offset_hist.get(v, 0) + 1

    # Clusters and inter-cluster structure
    clusters  = sync_clusters(word, rule, width)
    n_clusters = len(clusters)
    cluster_sizes = [len(g) for g in clusters]
    reps       = [series[g[0]] for g in clusters]
    ic_offsets = {}
    for a in range(n_clusters):
        for b in range(n_clusters):
            if a != b:
                ic_offsets[(a, b)] = phase_offset(reps[a], reps[b])

    # Are any inter-cluster offsets non-None?
    any_antiphase = any(v is not None for v in ic_offsets.values())

    # Distinct series count
    n_distinct = len({tuple(s) for s in series})

    # Dominant offset (most frequent offset among synced pairs)
    dominant_offset = (max(offset_hist, key=lambda k: offset_hist[k])
                       if offset_hist else 0)

    return {
        'word':            word.upper(),
        'rule':            rule,
        'period':          period,
        'n_distinct':      n_distinct,
        'sync_fraction':   round(sf, 6),
        'offset_hist':     {str(k): v for k, v in sorted(offset_hist.items())},
        'n_clusters':      n_clusters,
        'cluster_sizes':   cluster_sizes,
        'ic_offsets':      {str(k): v for k, v in ic_offsets.items()},
        'any_antiphase':   any_antiphase,
        'dominant_offset': dominant_offset,
        'matrix':          mat,
    }


def all_phase(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """phase_summary for all 4 rules."""
    return {rule: phase_summary(word, rule, width) for rule in _RULES}


def build_phase_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact phase data for a list of words (no matrix)."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = phase_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'n_distinct', 'sync_fraction', 'offset_hist',
                 'n_clusters', 'cluster_sizes', 'any_antiphase',
                 'dominant_offset')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'


def print_phase(word: str = 'ГОРА', rule: str = 'and',
                color: bool = True) -> None:
    d   = phase_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    mat = d['matrix']
    N   = len(mat)

    print(f'  {c}◈ Phase  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'distinct={d["n_distinct"]}  sync={d["sync_fraction"]:.3f}  '
          f'clusters={d["n_clusters"]}{r}')
    print('  ' + '─' * 62)

    # Matrix (compact)
    print(f'  Phase matrix  (. = None, digit = offset, * = >9):')
    header = '    ' + ''.join(f'{j:>3}' for j in range(N))
    print(header)
    for i in range(N):
        row_str = ''.join(
            '  .' if v is None else (f'{v:3d}' if v <= 9 else '  *')
            for v in mat[i]
        )
        print(f'    {i:>2}: {row_str}')

    # Clusters
    clusters  = [g for g in [
        [j for j in range(N) if mat[i][j] == 0] for i in range(N)
        if i == min(j for j in range(N) if mat[i][j] == 0)
    ]]
    # Simpler: from phase_summary
    print(f'\n  Offset-0 clusters: {d["n_clusters"]}')
    # Recompute clusters properly
    from projects.hexglyph.solan_perm import get_orbit
    orbit  = get_orbit(word.upper(), rule)
    P = len(orbit)
    series = [[orbit[t][i] for t in range(P)] for i in range(N)]
    seen: dict[tuple, list] = {}
    for i, s in enumerate(series):
        k = tuple(s)
        if k not in seen: seen[k] = []
        seen[k].append(i)
    for gi, (key, cells) in enumerate(seen.items()):
        print(f'    G{gi} {cells}')

    # Inter-cluster offsets
    if d['ic_offsets']:
        print(f'\n  Inter-cluster offsets:')
        for pair_str, off in d['ic_offsets'].items():
            print(f'    {pair_str}: {off}')
    if d['any_antiphase']:
        print(f'\n  ★ Anti-phase synchronization detected (inter-cluster offset > 0)')

    # Offset histogram
    print(f'\n  Offset histogram (synced pairs):')
    for k, cnt in sorted(d['offset_hist'].items(), key=lambda x: int(x[0])):
        print(f'    k={k}: {cnt}')
    print()


def print_phase_stats(words: list[str] | None = None,
                      color: bool = True) -> None:
    WORDS = words or ['ТУМАН', 'ГОРА', 'ЛЕБЕДЬ']
    for word in WORDS:
        for rule in _RULES:
            print_phase(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Phase offset analysis of Q6 CA attractor')
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
        print(_json.dumps(phase_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.stats:
        print_phase_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_phase(args.word, rule, color)
    else:
        print_phase(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
