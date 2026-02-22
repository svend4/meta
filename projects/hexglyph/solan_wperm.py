"""
solan_wperm.py — Weighted Permutation Entropy (WPE) of Q6 CA.

Extends Permutation Entropy (solan_perm.py) by weighting each ordinal
pattern's contribution by the *amplitude variance* of the embedding window
(Fadlallah et al. 2013).

Definition
──────────
For a time series x(t) and embedding dimension m, each length-m window
(x_j, x_{j+1}, …, x_{j+m−1}) contributes:

    ordinal pattern  π_j  (rank-order permutation, stable sort)
    variance weight  w_j  =  (1/(m−1)) Σ_{k=0}^{m−1} (x_{j+k} − x̄_j)²

The weighted pattern distribution:

    W_π  =  Σ_{j : π_j = π}  w_j          (summed variance per pattern)
    P̃_π  =  W_π / Σ_π W_π                 (normalised weight)

Weighted Permutation Entropy:

    WPE(m)  =  −Σ_π P̃_π log₂ P̃_π   (bits)

Normalised:

    nWPE(m)  =  WPE(m) / log₂(m!)  ∈  [0, 1]

Comparison with PE
──────────────────
When all windows have *equal variance*, P̃_π = count_π / total,
so WPE = PE (amplitude information is irrelevant).

When high-variance windows concentrate on certain patterns:
  nWPE < nPE  →  amplitude structure regularises the distribution
                 (dominant patterns carry more weight)
When high-variance windows spread across many patterns:
  nWPE > nPE  →  amplitude diversity amplifies complexity

Excess amplitude complexity:  ΔWPE = nWPE − nPE  ∈ [−1, 1]

Key results  (m = 3, period repeated to ≥ 24 samples)
──────────────────────────────────────────────────────
  XOR  ТУМАН (P=1)  : WPE=0, nWPE=0      — fixed point, zero variance
  AND/OR fixed-pt   : same
  ГОРА AND  (P=2)   : nWPE = nPE ≈ 0.387 — equal variance both patterns
  ТУМАН XOR3 (P=8)  : nWPE ≈ 0.664 < nPE ≈ 0.775 — concentrated amplitude

Functions
─────────
  ordinal_pattern(window)                       → tuple[int, …]
  window_weight(window)                         → float  (sample variance)
  wpe(series, m)                                → float  WPE in bits
  nwpe(series, m)                               → float  normalised WPE ∈ [0,1]
  spatial_wpe(word, rule, width, m)             → list[float]  per-cell nWPE
  wpe_dict(word, rule, width, m)                → dict
  all_wpe(word, width, m)                       → dict[str, dict]
  build_wpe_data(words, width, m)               → dict
  print_wpe(word, rule, m, color)               → None
  print_wpe_stats(words, color)                 → None

Запуск
──────
  python3 -m projects.hexglyph.solan_wperm --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_wperm --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_wperm --stats --no-color
"""

from __future__ import annotations
import math
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:      list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W:  int = 16
_DEFAULT_M:  int = 3       # embedding dimension
_MIN_REPS:   int = 8       # minimum period repetitions for series extension


# ── Ordinal pattern ────────────────────────────────────────────────────────────

def ordinal_pattern(window: list | tuple) -> tuple[int, ...]:
    """Rank-order permutation of *window* (stable sort, ties by index)."""
    m = len(window)
    indexed = sorted(range(m), key=lambda i: (window[i], i))
    ranks = [0] * m
    for rank, idx in enumerate(indexed):
        ranks[idx] = rank
    return tuple(ranks)


# ── Variance weight ────────────────────────────────────────────────────────────

def window_weight(window: list | tuple) -> float:
    """Sample variance of a window: (1/(m−1)) Σ (x_i − mean)²."""
    m = len(window)
    if m <= 1:
        return 0.0
    mean = sum(window) / m
    return sum((x - mean) ** 2 for x in window) / (m - 1)


# ── WPE computation ────────────────────────────────────────────────────────────

def wpe(series: list[int | float], m: int = _DEFAULT_M) -> float:
    """Weighted Permutation Entropy in bits.

    Returns 0.0 if total weighted variance is zero (constant series).
    """
    n = len(series)
    if n < m:
        return 0.0

    weighted: dict[tuple, float] = {}
    total_w = 0.0
    for j in range(n - m + 1):
        window = series[j:j + m]
        w = window_weight(window)
        pat = ordinal_pattern(window)
        weighted[pat] = weighted.get(pat, 0.0) + w
        total_w += w

    if total_w == 0.0:
        return 0.0

    return max(0.0, -sum(
        (v / total_w) * math.log2(v / total_w)
        for v in weighted.values() if v > 0
    ))


def nwpe(series: list[int | float], m: int = _DEFAULT_M) -> float:
    """Normalised WPE: WPE / log₂(m!) ∈ [0, 1]."""
    M = math.factorial(m)
    if M <= 1:
        return 0.0
    return round(min(wpe(series, m) / math.log2(M), 1.0), 8)


# ── Orbit helper ───────────────────────────────────────────────────────────────

def _get_series(word: str, rule: str, width: int,
                min_reps: int = _MIN_REPS) -> tuple[list[list[float]], int]:
    """Return (cell_series_list, period).  Each cell series has length ≥ P*min_reps."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    if P == 0:
        return [[0.0] * min_reps] * width, 0
    repeat = max(min_reps, math.ceil(24 / max(P, 1)))
    series = [
        [float(orbit[t % P][i]) for t in range(P * repeat)]
        for i in range(width)
    ]
    return series, P


# ── Spatial WPE (per-cell) ─────────────────────────────────────────────────────

def spatial_wpe(word: str, rule: str, width: int = _DEFAULT_W,
                m: int = _DEFAULT_M) -> list[float]:
    """Normalised WPE for each cell's temporal series on the attractor."""
    cell_series, _ = _get_series(word, rule, width)
    return [nwpe(s, m) for s in cell_series]


def spatial_pe(word: str, rule: str, width: int = _DEFAULT_W,
               m: int = _DEFAULT_M) -> list[float]:
    """Normalised PE (Bandt-Pompe) for each cell — reference for ΔWPE."""
    from projects.hexglyph.solan_perm import perm_entropy
    cell_series, _ = _get_series(word, rule, width)
    return [perm_entropy(s, m) for s in cell_series]


# ── WPE dictionary ─────────────────────────────────────────────────────────────

def wpe_dict(word: str, rule: str, width: int = _DEFAULT_W,
             m: int = _DEFAULT_M) -> dict:
    """Full WPE analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    wpe_profile  = spatial_wpe(word, rule, width, m)
    pe_profile   = spatial_pe(word, rule, width, m)
    delta        = [round(w - p, 8) for w, p in zip(wpe_profile, pe_profile)]

    mean_nwpe = round(sum(wpe_profile) / width, 8)
    mean_npe  = round(sum(pe_profile)  / width, 8)
    mean_delta = round(mean_nwpe - mean_npe, 8)

    var_wpe   = sum((v - mean_nwpe) ** 2 for v in wpe_profile) / width
    std_wpe   = round(math.sqrt(var_wpe), 8)

    return {
        'word':       word.upper(),
        'rule':       rule,
        'period':     period,
        'm':          m,
        'wpe_profile': wpe_profile,
        'pe_profile':  pe_profile,
        'delta':       delta,
        'mean_nwpe':  mean_nwpe,
        'mean_npe':   mean_npe,
        'mean_delta': mean_delta,
        'std_wpe':    std_wpe,
        'max_nwpe':   max(wpe_profile),
        'min_nwpe':   min(wpe_profile),
    }


def all_wpe(word: str, width: int = _DEFAULT_W,
            m: int = _DEFAULT_M) -> dict[str, dict]:
    """wpe_dict for all 4 rules."""
    return {rule: wpe_dict(word, rule, width, m) for rule in _RULES}


def build_wpe_data(words: list[str], width: int = _DEFAULT_W,
                   m: int = _DEFAULT_M) -> dict:
    """Aggregated WPE data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = wpe_dict(word, rule, width, m)
            per_rule[rule][word.upper()] = {k: d[k] for k in (
                'period', 'mean_nwpe', 'mean_npe', 'mean_delta',
                'std_wpe', 'max_nwpe', 'min_nwpe')}
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


def _delta_tag(d: float) -> str:
    if abs(d) < 0.005:
        return '  ≈ '
    return ' ↑ ' if d > 0 else ' ↓ '


def print_wpe(word: str = 'ТУМАН', rule: str = 'xor3',
              m: int = _DEFAULT_M, color: bool = True) -> None:
    d = wpe_dict(word, rule, m=m)
    c = _RCOL.get(rule, '') if color else ''
    r = _RST if color else ''
    RULE = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    print(f'  {c}◈ WPE  {word.upper()}  |  {RULE}  P={d["period"]}  '
          f'm={m}  mean_nWPE={d["mean_nwpe"]:.4f}  '
          f'ΔWPE={d["mean_delta"]:+.4f}{r}')
    print('  ' + '─' * 62)
    print(f'  {"cell":>4}  {"nWPE":>6}  {"nPE":>6}  {"ΔWPE":>7}  bar(nWPE)')
    print('  ' + '─' * 62)
    for i, (w_val, p_val, dv) in enumerate(
            zip(d['wpe_profile'], d['pe_profile'], d['delta'])):
        bar = _bar(w_val)
        tag = _delta_tag(dv)
        print(f'  {i:>4}  {w_val:>6.4f}  {p_val:>6.4f}  {dv:>+7.4f}  {bar}')
    print(f'\n  mean_nWPE={d["mean_nwpe"]:.4f}  mean_nPE={d["mean_npe"]:.4f}  '
          f'mean_ΔWPE={d["mean_delta"]:+.4f}  std={d["std_wpe"]:.4f}')
    print()


def print_wpe_stats(words: list[str] | None = None, m: int = _DEFAULT_M,
                    color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_wpe(word, rule, m, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='Weighted Permutation Entropy (WPE) for Q6 CA')
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
        print(_json.dumps(wpe_dict(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.stats:
        print_wpe_stats(m=args.m, color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_wpe(args.word, rule, args.m, color)
    else:
        print_wpe(args.word, args.rule, args.m, color)


if __name__ == '__main__':
    _cli()
