"""
solan_ch_plane.py — Complexity-Entropy (C-H) Causality Plane for Q6 CA.

Places each word × rule attractor on the 2-dimensional C-H plane
(Rosso et al. 2007, López-Ruiz, Mancini & Calbet 1995):

    x-axis:  H_S  =  H_PE / log₂(m!)  ∈ [0, 1]
             Normalised permutation entropy (Bandt & Pompe 2002).

    y-axis:  C    =  H_S · D_JS(P ‖ U)  ∈ [0, 1]
             Statistical complexity — product of H_S and the
             Jensen-Shannon divergence between the ordinal-pattern
             distribution P and the uniform distribution U over all m!
             possible patterns.  D_JS ∈ [0, 1] is normalised by log₂(2).

Interpretation
──────────────
  Bottom-left  (H≈0, C≈0)  : deterministic, constant or fixed-point.
                              P concentrates on 1 pattern → H=0 → C=0.
  Bottom-right (H≈1, C≈0)  : fully random / maximally complex.
                              P ≈ U → D_JS ≈ 0 → C ≈ 0.
  Interior arc (H mid, C>0): structured, non-trivial dynamics.
                              P deviates from U in a regular way.

Key results  (m = 3)
────────────────────
  XOR  ТУМАН (P=1)   : (H=0.00, C=0.00)  — fixed-point, 1 ordinal pattern
  AND/OR fixed-point  : same
  ГОРА AND   (P=2)   : (H=0.39, C=0.18)  — 2 patterns (asc/desc), medium C
  ТУМАН XOR3 (P=8)   : (H=0.98, C=0.01)  — near-uniform, 6 of 6 patterns used

D_JS diagonal
─────────────
  D_JS is also computed independently per-cell: mean and std across cells
  reveal whether cells share the same complexity level.

Functions
─────────
  ordinal_pattern(window)                         → tuple
  pattern_dist(word, rule, m, width, min_len)     → dict[tuple, float]
  h_norm(dist, m)                                 → float  H_S ∈ [0,1]
  jsd_uniform(dist, m)                            → float  D_JS ∈ [0,1]
  statistical_complexity(dist, m)                 → float  C = H_S · D_JS
  ch_point(word, rule, m, width, min_len)         → tuple  (H_S, D_JS, C)
  cell_ch(word, rule, m, width, min_len)          → list[dict]  per-cell
  ch_dict(word, rule, m, width, min_len)          → dict
  all_ch(word, m, width, min_len)                 → dict[str, dict]
  build_ch_data(words, m, width, min_len)         → dict
  print_ch(word, rule, m, color)                  → None
  print_ch_stats(words, color)                    → None

Запуск
──────
  python3 -m projects.hexglyph.solan_ch_plane --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_ch_plane --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_ch_plane --stats --no-color
"""

from __future__ import annotations
import math
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:      list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W:  int = 16
_DEFAULT_M:  int = 3       # embedding dimension (m! = 6 ordinal patterns)
_DEFAULT_MIN_LEN: int = 64  # minimum temporal series length (repeat period)


# ── Ordinal pattern ────────────────────────────────────────────────────────────

def ordinal_pattern(window: list | tuple) -> tuple[int, ...]:
    """Rank-order (ordinal) pattern of *window* using stable sort.

    Returns a tuple of ranks: rank[i] = position of window[i] among all
    elements sorted stably (ties broken by index).
    """
    m = len(window)
    indexed = sorted(range(m), key=lambda i: (window[i], i))
    ranks = [0] * m
    for rank, idx in enumerate(indexed):
        ranks[idx] = rank
    return tuple(ranks)


# ── Pattern distribution ───────────────────────────────────────────────────────

def pattern_dist(word: str, rule: str, m: int = _DEFAULT_M,
                 width: int = _DEFAULT_W,
                 min_len: int = _DEFAULT_MIN_LEN) -> dict[tuple, float]:
    """Ordinal-pattern probability distribution over all cells' temporal series.

    Each cell's period-P attractor sequence is repeated to reach *min_len*
    samples; windows of length *m* are extracted and pooled across all cells.
    """
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    if P == 0:
        return {}

    repeat = max(1, math.ceil(min_len / P))
    counts: dict[tuple, int] = {}
    total = 0
    for i in range(width):
        seq = [orbit[t % P][i] for t in range(P * repeat)]
        n = len(seq)
        for j in range(n - m + 1):
            pat = ordinal_pattern(seq[j:j + m])
            counts[pat] = counts.get(pat, 0) + 1
            total += 1

    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


# ── H_S — normalised permutation entropy ──────────────────────────────────────

def h_norm(dist: dict[tuple, float], m: int = _DEFAULT_M) -> float:
    """Normalised permutation entropy H_S = H_PE / log₂(m!) ∈ [0, 1]."""
    M = math.factorial(m)
    if M <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in dist.values() if p > 0)
    return max(0.0, min(round(h / math.log2(M), 8), 1.0))


# ── D_JS — Jensen-Shannon divergence vs uniform ────────────────────────────────

def jsd_uniform(dist: dict[tuple, float], m: int = _DEFAULT_M) -> float:
    """Normalised Jensen-Shannon divergence D_JS(P ‖ U) ∈ [0, 1].

    U = uniform distribution over all m! ordinal patterns.
    Normalised by log₂(2) = 1 so that D_JS ∈ [0, 1].
    """
    M = math.factorial(m)
    if M == 0:
        return 0.0
    u = 1.0 / M

    # Entropy of mixture M = (P + U) / 2
    h_mix = 0.0
    for p in dist.values():
        q = (p + u) / 2
        if q > 0:
            h_mix -= q * math.log2(q)
    # Patterns not in dist have P[x] = 0 → mixture = u/2
    n_missing = M - len(dist)
    if n_missing > 0:
        q_miss = u / 2
        h_mix -= n_missing * q_miss * math.log2(q_miss)

    h_p = -sum(p * math.log2(p) for p in dist.values() if p > 0)
    h_u = math.log2(M)

    jsd = h_mix - (h_p + h_u) / 2   # in bits; maximum = log₂(2) = 1 bit
    return max(0.0, min(round(jsd, 8), 1.0))


# ── Statistical complexity ─────────────────────────────────────────────────────

def statistical_complexity(dist: dict[tuple, float],
                            m: int = _DEFAULT_M) -> float:
    """Statistical complexity C = H_S · D_JS ∈ [0, 1]."""
    return round(h_norm(dist, m) * jsd_uniform(dist, m), 8)


# ── C-H coordinates ────────────────────────────────────────────────────────────

def ch_point(word: str, rule: str, m: int = _DEFAULT_M,
             width: int = _DEFAULT_W,
             min_len: int = _DEFAULT_MIN_LEN) -> tuple[float, float, float]:
    """Return (H_S, D_JS, C) for the word × rule attractor."""
    dist = pattern_dist(word, rule, m, width, min_len)
    hs   = h_norm(dist, m)
    djs  = jsd_uniform(dist, m)
    c    = round(hs * djs, 8)
    return hs, djs, c


# ── Per-cell C-H ───────────────────────────────────────────────────────────────

def cell_ch(word: str, rule: str, m: int = _DEFAULT_M,
            width: int = _DEFAULT_W,
            min_len: int = _DEFAULT_MIN_LEN) -> list[dict]:
    """Per-cell C-H analysis: each cell's temporal series separately."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    repeat = max(1, math.ceil(min_len / max(P, 1)))
    result = []
    for i in range(width):
        seq = [orbit[t % P][i] for t in range(P * repeat)]
        n = len(seq)
        counts: dict[tuple, int] = {}
        total = 0
        for j in range(n - m + 1):
            pat = ordinal_pattern(seq[j:j + m])
            counts[pat] = counts.get(pat, 0) + 1
            total += 1
        dist_i = {k: v / total for k, v in counts.items()} if total else {}
        hs_i  = h_norm(dist_i, m)
        djs_i = jsd_uniform(dist_i, m)
        result.append({'cell': i, 'h_s': hs_i, 'd_js': djs_i,
                        'c': round(hs_i * djs_i, 8),
                        'n_patterns': len(dist_i)})
    return result


# ── Full dictionary ────────────────────────────────────────────────────────────

def ch_dict(word: str, rule: str, m: int = _DEFAULT_M,
            width: int = _DEFAULT_W,
            min_len: int = _DEFAULT_MIN_LEN) -> dict:
    """Full C-H plane analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    hs, djs, c = ch_point(word, rule, m, width, min_len)
    cells = cell_ch(word, rule, m, width, min_len)
    M     = math.factorial(m)

    mean_hs  = round(sum(cell['h_s'] for cell in cells) / width, 6)
    mean_djs = round(sum(cell['d_js'] for cell in cells) / width, 6)
    mean_c   = round(sum(cell['c']   for cell in cells) / width, 6)

    var_c    = sum((cell['c'] - mean_c) ** 2 for cell in cells) / width
    std_c    = round(math.sqrt(var_c), 6)

    return {
        'word':        word.upper(),
        'rule':        rule,
        'period':      period,
        'm':           m,
        'M':           M,
        'h_s':         hs,
        'd_js':        djs,
        'c':           c,
        'cell_ch':     cells,
        'mean_h_s':    mean_hs,
        'mean_d_js':   mean_djs,
        'mean_c':      mean_c,
        'std_c':       std_c,
        'n_patterns':  sum(1 for cell in cells if cell['n_patterns'] > 0),
    }


def all_ch(word: str, m: int = _DEFAULT_M,
           width: int = _DEFAULT_W,
           min_len: int = _DEFAULT_MIN_LEN) -> dict[str, dict]:
    """ch_dict for all 4 rules."""
    return {rule: ch_dict(word, rule, m, width, min_len) for rule in _RULES}


def build_ch_data(words: list[str], m: int = _DEFAULT_M,
                  width: int = _DEFAULT_W,
                  min_len: int = _DEFAULT_MIN_LEN) -> dict:
    """Aggregated C-H data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = ch_dict(word, rule, m, width, min_len)
            per_rule[rule][word.upper()] = {k: d[k] for k in (
                'period', 'h_s', 'd_js', 'c',
                'mean_h_s', 'mean_d_js', 'mean_c', 'std_c')}
    return {'words': [w.upper() for w in words], 'width': width,
            'm': m, 'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'
_SHD  = '░'


def _bar(v: float, width: int = 20) -> str:
    filled = round(min(max(v, 0.0), 1.0) * width)
    return _BAR * filled + _SHD * (width - filled)


def print_ch(word: str = 'ТУМАН', rule: str = 'xor3',
             m: int = _DEFAULT_M, color: bool = True) -> None:
    d = ch_dict(word, rule, m)
    c_col = _RCOL.get(rule, '') if color else ''
    r = _RST if color else ''
    RULE = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    M = d['M']
    print(f'  {c_col}◈ C-H Plane  {word.upper()}  |  {RULE}  P={d["period"]}  '
          f'm={m}  M={M} patterns{r}')
    print('  ' + '─' * 60)
    print(f'  {"Pooled":>10}  H_S={d["h_s"]:.4f}  D_JS={d["d_js"]:.4f}  '
          f'C={d["c"]:.4f}')
    print(f'  {"Mean/cell":>10}  H_S={d["mean_h_s"]:.4f}  D_JS={d["mean_d_js"]:.4f}  '
          f'C={d["mean_c"]:.4f}  std_C={d["std_c"]:.4f}')
    print()
    print(f'  {"cell":>5}  {"H_S":>6}  {"D_JS":>6}  {"C":>6}  '
          f'{"n_pat":>5}  bar(C)')
    print('  ' + '─' * 60)
    for cell in d['cell_ch']:
        bar = _bar(cell['c'], 20)
        print(f'  {cell["cell"]:>5}  {cell["h_s"]:>6.4f}  '
              f'{cell["d_js"]:>6.4f}  {cell["c"]:>6.4f}  '
              f'{cell["n_patterns"]:>5}  {bar}')
    print()


def print_ch_stats(words: list[str] | None = None, m: int = _DEFAULT_M,
                   color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_ch(word, rule, m, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='Complexity-Entropy (C-H) Causality Plane for Q6 CA')
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
        print(_json.dumps(ch_dict(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.stats:
        print_ch_stats(m=args.m, color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_ch(args.word, rule, args.m, color)
    else:
        print_ch(args.word, args.rule, args.m, color)


if __name__ == '__main__':
    _cli()
