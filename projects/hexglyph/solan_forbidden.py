"""
solan_forbidden.py — Forbidden Ordinal Patterns of Q6 CA Attractors.

An ordinal pattern is "forbidden" if it never appears in the time series
despite having a non-zero probability in a random process.  The set of
forbidden patterns is a direct fingerprint of deterministic structure:
random processes eventually use all m! ordinal patterns; deterministic
periodic attractors systematically exclude some.

Definition
──────────
For embedding dimension m, there are M = m! possible ordinal patterns.
A pattern π is *forbidden* for a time series x if it never appears in any
length-m window of x (after repeating the attractor period sufficiently).

Forbidden count:    F_m  =  M  −  |{observed patterns}|
Forbidden fraction: f_m  =  F_m / M  ∈  [0, 1]
Observed fraction:  o_m  =  1 − f_m  ∈  [0, 1]

Key results  (m = 3, M = 6, width = 16, ≥ 24 samples)
─────────────────────────────────────────────────────────
  XOR/AND/OR fixed-pt (P=1) : obs=1/6   F=5   f=0.833  ← only one window type
  ГОРА AND/XOR3   (P=2)     : obs=2/6   F=4   f=0.667
  ТУМАН XOR3      (P=8)     : pooled obs=6/6  F=0  f=0.000  ← all patterns seen
                              per-cell: 3–6 patterns (heterogeneous)

Multi-scale: for m=4 (M=24), ТУМАН XOR3 also uses all 24 patterns pooled.

Complexity ordering (by f_m, lower = more ergodic):
  fully random → f≈0 | ТУМАН XOR3 → f=0 | ГОРА periods → f=0.667 | fixed-pt → f=0.833

Functions
─────────
  ordinal_pattern(window)                             → tuple[int,…]
  all_patterns(m)                                     → frozenset  all m! patterns
  observed_cell(series, m, min_reps)                  → frozenset  per-cell
  observed_pooled(word, rule, width, m, min_reps)     → frozenset  across cells
  forbidden_cell_profile(word, rule, width, m)        → list[dict]
  forbidden_dict(word, rule, width, m)                → dict
  all_forbidden(word, width, m)                       → dict[str, dict]
  build_forbidden_data(words, width, m)               → dict
  print_forbidden(word, rule, m, color)               → None
  print_forbidden_stats(words, color)                 → None

Запуск
──────
  python3 -m projects.hexglyph.solan_forbidden --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_forbidden --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_forbidden --stats --no-color
"""

from __future__ import annotations
import math
import itertools
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_DEFAULT_M: int = 3
_MIN_REPS:  int = 8   # minimum period repetitions for series extension


# ── Ordinal pattern ────────────────────────────────────────────────────────────

def ordinal_pattern(window: list | tuple) -> tuple[int, ...]:
    """Rank-order permutation of *window* (stable sort, ties by index)."""
    m = len(window)
    indexed = sorted(range(m), key=lambda i: (window[i], i))
    ranks = [0] * m
    for rank, idx in enumerate(indexed):
        ranks[idx] = rank
    return tuple(ranks)


def all_patterns(m: int) -> frozenset[tuple[int, ...]]:
    """All M = m! ordinal patterns as a frozenset."""
    return frozenset(itertools.permutations(range(m)))


# ── Observed / forbidden patterns ─────────────────────────────────────────────

def observed_cell(series: list[int | float], m: int = _DEFAULT_M,
                  min_reps: int = _MIN_REPS) -> frozenset[tuple[int, ...]]:
    """Set of ordinal patterns observed in *series* (treated as periodic)."""
    P = len(series)
    if P == 0:
        return frozenset()
    repeat = max(min_reps, math.ceil(24 / max(P, 1)))
    n = P * repeat
    obs: set[tuple[int, ...]] = set()
    for j in range(n - m + 1):
        win = [series[(j + k) % P] for k in range(m)]
        obs.add(ordinal_pattern(win))
    return frozenset(obs)


def observed_pooled(word: str, rule: str, width: int = _DEFAULT_W,
                    m: int = _DEFAULT_M,
                    min_reps: int = _MIN_REPS) -> frozenset[tuple[int, ...]]:
    """Union of observed patterns across all cells' temporal series."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    if P == 0:
        return frozenset()
    pooled: set[tuple[int, ...]] = set()
    for i in range(width):
        s = [orbit[t][i] for t in range(P)]
        pooled |= observed_cell(s, m, min_reps)
    return frozenset(pooled)


# ── Per-cell profile ───────────────────────────────────────────────────────────

def forbidden_cell_profile(word: str, rule: str, width: int = _DEFAULT_W,
                            m: int = _DEFAULT_M,
                            min_reps: int = _MIN_REPS) -> list[dict]:
    """Per-cell forbidden pattern analysis."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    M = math.factorial(m)
    result = []
    for i in range(width):
        s = [orbit[t][i] for t in range(P)]
        obs = observed_cell(s, m, min_reps)
        F = M - len(obs)
        result.append({
            'cell':        i,
            'n_observed':  len(obs),
            'n_forbidden': F,
            'f_m':         round(F / M, 8),
            'o_m':         round(len(obs) / M, 8),
            'patterns':    sorted(obs),
        })
    return result


# ── Full dictionary ────────────────────────────────────────────────────────────

def forbidden_dict(word: str, rule: str, width: int = _DEFAULT_W,
                   m: int = _DEFAULT_M) -> dict:
    """Full forbidden-pattern analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    M      = math.factorial(m)
    cells  = forbidden_cell_profile(word, rule, width, m)
    pooled = observed_pooled(word, rule, width, m)

    F_pooled = M - len(pooled)
    f_pooled = round(F_pooled / M, 8)

    mean_obs  = round(sum(c['n_observed']  for c in cells) / width, 6)
    mean_f    = round(sum(c['f_m']         for c in cells) / width, 6)

    var_f  = sum((c['f_m'] - mean_f) ** 2 for c in cells) / width
    std_f  = round(math.sqrt(var_f), 6)

    forbidden_set = all_patterns(m) - pooled

    return {
        'word':           word.upper(),
        'rule':           rule,
        'period':         period,
        'm':              m,
        'M':              M,
        'n_observed':     len(pooled),
        'n_forbidden':    F_pooled,
        'f_m':            f_pooled,
        'o_m':            round(len(pooled) / M, 8),
        'observed_set':   sorted(pooled),
        'forbidden_set':  sorted(forbidden_set),
        'cell_profile':   cells,
        'mean_cell_obs':  mean_obs,
        'mean_cell_f':    mean_f,
        'std_cell_f':     std_f,
    }


def all_forbidden(word: str, width: int = _DEFAULT_W,
                  m: int = _DEFAULT_M) -> dict[str, dict]:
    """forbidden_dict for all 4 rules."""
    return {rule: forbidden_dict(word, rule, width, m) for rule in _RULES}


def build_forbidden_data(words: list[str], width: int = _DEFAULT_W,
                         m: int = _DEFAULT_M) -> dict:
    """Aggregated forbidden-pattern data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = forbidden_dict(word, rule, width, m)
            per_rule[rule][word.upper()] = {k: d[k] for k in (
                'period', 'M', 'n_observed', 'n_forbidden',
                'f_m', 'o_m', 'mean_cell_obs', 'mean_cell_f', 'std_cell_f')}
    return {'words': [w.upper() for w in words], 'width': width,
            'm': m, 'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'
_SHD  = '░'


def _bar(v: float, w: int = 20) -> str:
    filled = round(min(max(v, 0.0), 1.0) * w)
    return _BAR * filled + _SHD * (w - filled)


def _pat_str(pat: tuple[int, ...]) -> str:
    return '(' + ','.join(str(x) for x in pat) + ')'


def print_forbidden(word: str = 'ТУМАН', rule: str = 'xor3',
                    m: int = _DEFAULT_M, color: bool = True) -> None:
    d = forbidden_dict(word, rule, m=m)
    c = _RCOL.get(rule, '') if color else ''
    r = _RST if color else ''
    RULE = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    M = d['M']
    print(f'  {c}◈ Forbidden Patterns  {word.upper()}  |  {RULE}  '
          f'P={d["period"]}  m={m}  M={M}  '
          f'obs={d["n_observed"]}/{M}  F={d["n_forbidden"]}  '
          f'f={d["f_m"]:.4f}{r}')
    print('  ' + '─' * 64)
    print(f'  Observed:  {" ".join(_pat_str(p) for p in d["observed_set"])}')
    print(f'  Forbidden: {" ".join(_pat_str(p) for p in d["forbidden_set"]) or "—"}')
    print()
    print(f'  {"cell":>4}  {"obs":>4}  {"F":>3}  {"f_m":>6}  bar(obs/M)')
    print('  ' + '─' * 64)
    for cell in d['cell_profile']:
        bar = _bar(cell['o_m'])
        print(f'  {cell["cell"]:>4}  {cell["n_observed"]:>4}  '
              f'{cell["n_forbidden"]:>3}  {cell["f_m"]:>6.4f}  {bar}')
    print(f'\n  Pooled: obs={d["n_observed"]}/{M}  F={d["n_forbidden"]}  '
          f'f={d["f_m"]:.4f}  mean_cell_f={d["mean_cell_f"]:.4f}  '
          f'std={d["std_cell_f"]:.4f}')
    print()


def print_forbidden_stats(words: list[str] | None = None, m: int = _DEFAULT_M,
                          color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_forbidden(word, rule, m, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='Forbidden Ordinal Patterns for Q6 CA attractors')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--m',         type=int, default=_DEFAULT_M)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(forbidden_dict(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.stats:
        print_forbidden_stats(m=args.m, color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_forbidden(args.word, rule, args.m, color)
    else:
        print_forbidden(args.word, args.rule, args.m, color)


if __name__ == '__main__':
    _cli()
