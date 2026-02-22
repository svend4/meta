"""
solan_moments.py — Statistical Moments of Per-Cell Temporal Series.

For each cell i, the temporal series {x_t : t=0…P-1} (one attractor orbit)
is treated as a sample; we compute:

    mean   μ  = (1/P) Σ x_t
    var    σ² = (1/P) Σ (x_t − μ)²          (population, not Bessel-corrected)
    std    σ  = √σ²
    skew   γ₁ = μ₃/σ³    (Fisher's g₁)      — signed asymmetry
    kurt   γ₂ = μ₄/σ⁴−3  (excess kurtosis)  — tailedness vs. normal (=0)

Special cases
  P=1 (fixed point): var=0, skew/kurt undefined → None
  P=2 two-value uniform (Bernoulli ½): skew=0, kurt=−2  (exact)

Key results (width=16)
──────────────────────
  ТУМАН XOR  (P=1, x=0)    : var=0, skew=None, kurt=None for all cells
  ГОРА  AND  (P=2, 47↔1)   : μ=24, var=529, std=23, skew=0.0, kurt=−2.0
                              (each cell is a symmetric two-point Bernoulli)
  ТУМАН XOR3 (P=8)          : per-cell var ∈ [12, 447], skew ∈ [−1.1,+1.0],
                              kurt ∈ [−1.73, −0.21] — strong spatial heterogeneity

Interpretation of excess kurtosis
  γ₂ < 0  platykurtic  — flatter than Gaussian (uniform-like, few outliers)
  γ₂ = 0  mesokurtic   — Gaussian
  γ₂ > 0  leptokurtic  — heavier tails (sharp peak)
  γ₂ = −2 minimum possible for any distribution (two-point, i.e. Bernoulli)

Functions
─────────
  temporal_moments(series)                     → dict
  cell_moments_list(word, rule, width)         → list[dict]  (one per cell)
  moments_summary(word, rule, width)           → dict
  all_moments(word, width)                     → dict[str, dict]
  build_moments_data(words, width)             → dict
  print_moments(word, rule, color)             → None
  print_moments_stats(words, color)            → None

Запуск
──────
  python3 -m projects.hexglyph.solan_moments --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_moments --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_moments --stats --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_Q6_MAX:    int = 63


# ── Core computation ──────────────────────────────────────────────────────────

def temporal_moments(series: list[int]) -> dict:
    """
    Population mean, variance, std, skewness (g1), excess kurtosis (g2)
    for a temporal series of Q6 values.

    Returns dict with keys: mean, var, std, skew, kurt.
    skew / kurt are None when var == 0 (constant series).
    """
    P = len(series)
    if P == 0:
        return {'mean': 0.0, 'var': 0.0, 'std': 0.0, 'skew': None, 'kurt': None}

    mu  = sum(series) / P
    var = sum((x - mu) ** 2 for x in series) / P

    if var < 1e-12:
        return {
            'mean': round(mu, 6),
            'var':  0.0,
            'std':  0.0,
            'skew': None,
            'kurt': None,
        }

    std  = var ** 0.5
    skew = sum((x - mu) ** 3 for x in series) / P / (std ** 3)
    kurt = sum((x - mu) ** 4 for x in series) / P / (std ** 4) - 3.0

    return {
        'mean': round(mu,   6),
        'var':  round(var,  6),
        'std':  round(std,  6),
        'skew': round(skew, 6),
        'kurt': round(kurt, 6),
    }


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Per-cell list ─────────────────────────────────────────────────────────────

def cell_moments_list(word: str, rule: str,
                      width: int = _DEFAULT_W) -> list[dict]:
    """One moments dict per cell (length = width)."""
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    out   = []
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        m      = temporal_moments(series)
        m['cell'] = i
        out.append(m)
    return out


# ── Summary ───────────────────────────────────────────────────────────────────

def _safe_stats(vals: list[float]) -> dict:
    """Mean, std, min, max for a list; None if empty."""
    if not vals:
        return {'mean': None, 'std': None, 'min': None, 'max': None}
    n = len(vals)
    mu  = sum(vals) / n
    std = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5
    return {
        'mean': round(mu,  6),
        'std':  round(std, 6),
        'min':  round(min(vals), 6),
        'max':  round(max(vals), 6),
    }


def moments_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Aggregate summary of per-cell moments."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']
    cms    = cell_moments_list(word, rule, width)

    means = [c['mean'] for c in cms]
    vars_ = [c['var']  for c in cms]
    skews = [c['skew'] for c in cms if c['skew'] is not None]
    kurts = [c['kurt'] for c in cms if c['kurt'] is not None]

    return {
        'word':   word.upper(),
        'rule':   rule,
        'period': period,
        'cell_moments': cms,
        'var_stats':    _safe_stats(vars_),
        'mean_stats':   _safe_stats(means),
        'skew_stats':   _safe_stats(skews),
        'kurt_stats':   _safe_stats(kurts),
        'n_constant':   width - len(skews),    # cells where var==0
        'n_defined':    len(skews),
    }


def all_moments(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """moments_summary for all 4 rules."""
    return {rule: moments_summary(word, rule, width) for rule in _RULES}


def build_moments_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact aggregated moments data for a list of words."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = moments_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'n_constant', 'n_defined',
                 'var_stats', 'skew_stats', 'kurt_stats')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return '—'
    return f'{v:.{digits}f}'


def _bar(v: float, lo: float, hi: float, w: int = 12) -> str:
    """ASCII bar in range [lo, hi]."""
    if hi <= lo:
        return '·' * w
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    filled = round(t * w)
    return '█' * filled + '░' * (w - filled)


def print_moments(word: str = 'ТУМАН', rule: str = 'xor3',
                  color: bool = True) -> None:
    d   = moments_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    vs  = d['var_stats']
    ss  = d['skew_stats']
    ks  = d['kurt_stats']
    header = (f'{c}◈ Moments  {word.upper()}  |  {lbl}  P={d["period"]}  '
              f'const={d["n_constant"]}  defined={d["n_defined"]}{r}')
    print(f'  {header}')
    print('  ' + '─' * 62)

    # Per-cell table header
    var_lo = vs['min'] if vs['min'] is not None else 0
    var_hi = vs['max'] if vs['max'] is not None else 1
    print(f'  {"cell":>4}  {"mean":>7}  {"var":>8}  {"std":>7}  {"skew":>7}  {"kurt":>7}')
    for m in d['cell_moments']:
        bar = _bar(m['var'], var_lo, var_hi) if var_hi > 0 else '─' * 12
        print(f'  {m["cell"]:>4}  {_fmt(m["mean"]):>7}  {_fmt(m["var"]):>8}'
              f'  {_fmt(m["std"]):>7}  {_fmt(m["skew"]):>7}  {_fmt(m["kurt"]):>7}  {bar}')

    # Aggregate
    print(f'\n  Aggregate over {d["n_defined"]} non-constant cells:')
    print(f'  var:  mean={_fmt(vs["mean"])}  std={_fmt(vs["std"])}  '
          f'range=[{_fmt(vs["min"])},{_fmt(vs["max"])}]')
    if ss['mean'] is not None:
        print(f'  skew: mean={_fmt(ss["mean"])}  std={_fmt(ss["std"])}  '
              f'range=[{_fmt(ss["min"])},{_fmt(ss["max"])}]')
    if ks['mean'] is not None:
        print(f'  kurt: mean={_fmt(ks["mean"])}  std={_fmt(ks["std"])}  '
              f'range=[{_fmt(ks["min"])},{_fmt(ks["max"])}]')
    print()


def print_moments_stats(words: list[str] | None = None,
                        color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_moments(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='Statistical moments of Q6 CA attractor cell series')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_moments_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_moments(args.word, rule, color)
    else:
        print_moments(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
