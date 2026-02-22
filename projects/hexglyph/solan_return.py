"""
solan_return.py — First-Return Map of Q6 CA Attractor Cells.

For each cell's temporal series of length P (exactly one attractor period),
the first-return map is the set of consecutive-value pairs:

    RM(i) = {(x_t, x_{t+1 mod P}) : t = 0, …, P−1}   (circular)

This plots the next-state function x(t) → x(t+1) on a 64×64 grid.
For a period-P attractor, RM(i) contains exactly P distinct points
(no repetition in a generic series; fewer if consecutive values repeat).

Aggregate map (all N=16 cells):
    M[a][b] = number of cells whose return map contains (a, b)
    M ∈ {0, …, 16}^{64×64}

Jump distribution: |x_{t+1} − x_t| across all cells × all steps
    Reveals whether attractor dynamics are "smooth" (small steps) or
    "jumpy" (large state changes per CA step)

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all 0) : 1 point   {(0,0)}, on diagonal, mean_jump=0
  ГОРА  AND  (P=2, ±46)   : 2 points  {(47,1),(1,47)}, off-diagonal,
                            mean_jump=46, max_jump=46
  ТУМАН XOR3 (P=8)        : 46 distinct pairs, mean_jump≈20, max_jump=60,
                            3 diagonal points (14 cell-steps are self-loops)

Functions
─────────
  return_map(series)                   → list[tuple[int,int]]
  map_stats(pairs)                     → dict
  aggregate_map(word, rule, width)     → dict[tuple,int]
  jump_histogram(word, rule, width)    → list[int]  (len 64, index=jump)
  return_dict(word, rule, width)       → dict
  all_return(word, width)              → dict[str, dict]
  build_return_data(words, width)      → dict
  print_return(word, rule, color)      → None
  print_return_stats(words, color)     → None

Запуск
──────
  python3 -m projects.hexglyph.solan_return --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_return --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_return --stats --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_Q6:        int = 64          # Q6 alphabet size (values 0–63)
_DEFAULT_W: int = 16


# ── Core computation ──────────────────────────────────────────────────────────

def return_map(series: list[int]) -> list[tuple[int, int]]:
    """Circular first-return map: [(x_t, x_{t+1 mod P}) for t in 0..P-1]."""
    P = len(series)
    if P == 0:
        return []
    return [(series[t], series[(t + 1) % P]) for t in range(P)]


def map_stats(pairs: list[tuple[int, int]]) -> dict:
    """Statistics of a first-return map."""
    if not pairs:
        return {'n_pairs': 0, 'n_distinct': 0, 'diagonal_count': 0,
                'mean_jump': 0.0, 'max_jump': 0}
    distinct   = set(pairs)
    diagonal   = sum(1 for x, y in distinct if x == y)
    jumps      = [abs(y - x) for x, y in pairs]
    return {
        'n_pairs':       len(pairs),
        'n_distinct':    len(distinct),
        'diagonal_count': diagonal,
        'mean_jump':     round(sum(jumps) / len(jumps), 6),
        'max_jump':      max(jumps),
    }


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Aggregate map ─────────────────────────────────────────────────────────────

def aggregate_map(word: str, rule: str,
                  width: int = _DEFAULT_W) -> dict[tuple[int, int], int]:
    """Aggregate return map over all cells: (a,b) → count of cells."""
    orbit  = _get_orbit(word, rule, width)
    P      = len(orbit)
    counts: dict[tuple[int, int], int] = {}
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        for pair in return_map(series):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def jump_histogram(word: str, rule: str,
                   width: int = _DEFAULT_W) -> list[int]:
    """Jump-size histogram: hist[j] = number of (cell, step) pairs with |Δx|=j."""
    orbit  = _get_orbit(word, rule, width)
    P      = len(orbit)
    hist   = [0] * _Q6
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        for (x, y) in return_map(series):
            j = abs(y - x)
            if j < _Q6:
                hist[j] += 1
    return hist


# ── Full dictionary ────────────────────────────────────────────────────────────

def return_dict(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full first-return map analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']
    orbit  = _get_orbit(word, rule, width)

    # Per-cell return maps and stats
    cell_maps: list[dict] = []
    for i in range(width):
        series = [orbit[t][i] for t in range(period)]
        pairs  = return_map(series)
        stats  = map_stats(pairs)
        cell_maps.append({'cell': i, 'pairs': pairs, **stats})

    agg   = aggregate_map(word, rule, width)
    jhist = jump_histogram(word, rule, width)

    total_pairs = width * period
    all_jumps   = [abs(y - x) * cnt for (x, y), cnt in agg.items()]
    # Recompute from histogram for correctness
    tot_j       = sum(j * hist for j, hist in enumerate(jhist))
    mean_jump   = round(tot_j / total_pairs, 6) if total_pairs else 0.0
    max_jump    = max((j for j, h in enumerate(jhist) if h > 0), default=0)
    diag_pairs  = sum(cnt for (x, y), cnt in agg.items() if x == y)

    return {
        'word':           word.upper(),
        'rule':           rule,
        'period':         period,
        'cell_maps':      cell_maps,
        'agg_map':        agg,
        'jump_hist':      jhist,
        'n_distinct':     len(agg),
        'mean_jump':      mean_jump,
        'max_jump':       max_jump,
        'diag_pairs':     diag_pairs,
        'total_pairs':    total_pairs,
    }


def all_return(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """return_dict for all 4 rules."""
    return {rule: return_dict(word, rule, width) for rule in _RULES}


def build_return_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Aggregated return map data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = return_dict(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'n_distinct', 'mean_jump', 'max_jump',
                 'diag_pairs', 'total_pairs')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'
_SHD  = '░'


def _jump_bar(h: list[int], w: int = 48) -> str:
    """Compact ASCII representation of jump histogram."""
    max_h = max(h) if any(h) else 1
    out = []
    for j, cnt in enumerate(h):
        if cnt == 0:
            continue
        filled = round(cnt / max_h * 8)
        out.append(f'{j}:{"█"*filled}({cnt})')
    return '  '.join(out[:8]) + ('  …' if len(out) > 8 else '')


def print_return(word: str = 'ТУМАН', rule: str = 'xor3',
                 color: bool = True) -> None:
    d   = return_dict(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    diag_pct = round(100 * d['diag_pairs'] / max(d['total_pairs'], 1), 1)
    print(f'  {c}◈ Return Map  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'distinct={d["n_distinct"]}  mean_Δx={d["mean_jump"]:.2f}  '
          f'max_Δx={d["max_jump"]}  diag={d["diag_pairs"]}({diag_pct}%){r}')
    print('  ' + '─' * 62)
    # Top-10 most frequent pairs
    top = sorted(d['agg_map'].items(), key=lambda kv: -kv[1])[:10]
    print(f'  {"(x,y)":>8}  {"cells":>5}')
    for (x, y), cnt in top:
        tag = ' ←diag' if x == y else ''
        print(f'  ({x:>2},{y:>2})    {cnt:>3}{tag}')
    # Jump histogram summary
    print(f'\n  Jump |Δx| distribution:')
    print(f'  {_jump_bar(d["jump_hist"])}')
    print()


def print_return_stats(words: list[str] | None = None,
                       color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_return(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='First-return map for Q6 CA attractor cells')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        d = return_dict(args.word, args.rule)
        print(_json.dumps(
            {k: v for k, v in d.items() if k not in ('cell_maps', 'agg_map')},
            ensure_ascii=False, indent=2))
    elif args.stats:
        print_return_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_return(args.word, rule, color)
    else:
        print_return(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
