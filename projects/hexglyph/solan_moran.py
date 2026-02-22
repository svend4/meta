"""
solan_moran.py — Moran's I Spatial Autocorrelation of Q6 CA Attractors.

Moran's I measures the spatial autocorrelation of cell values on the
1-D circular lattice at each time step t during the attractor period.
Using nearest-neighbour weights (w_{ij}=1 if |i−j|=1 mod N):

    I(t) = (Σ_{i} (x_i−μ)(x_{i+1 mod N}−μ)) / (Σ_i (x_i−μ)²)

    I ∈ [−1, +1]  (approximately)
    I = +1   perfect positive spatial autocorrelation  (large blocks)
    I ≈  0   spatially random arrangement
    I = −1   perfect negative spatial autocorrelation  (checkerboard)
    NaN      constant spatial field (zero variance) → undefined

    Time-averaged: Ī = (1/P) Σ_{t=0}^{P−1} I(t)  (NaN steps excluded)

Derivation note
───────────────
For symmetric nearest-neighbour weights in 1-D ring:
    W = Σ_{i,j} w_{ij} = 2N  (each cell has left and right neighbour)
    Standard: I = (N/W)·(Σ_{sym pairs} w_{ij}(x_i−μ)(x_j−μ)) / Σ(x_i−μ)²
    With directed sum over N pairs (i → i+1 mod N):
        Σ_{sym} = 2·Σ_{directed}
    ⇒  I = (N/2N)·2·Σ_{i}(x_i−μ)(x_{i+1}−μ) / Σ(x_i−μ)²
          = Σ_{i}(x_i−μ)(x_{i+1}−μ) / Σ_i(x_i−μ)²

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all 0)     : I = NaN  (zero spatial variance)
  ГОРА  OR   (P=1, all 63)    : I = NaN
  ГОРА  AND  (P=2, alternating): I = −1.0  all steps  (perfect checkerboard)
  ТУМАН XOR3 (P=8)            : I varies −0.68 … +0.29 (rich spatial dynamics)

Functions
─────────
  morans_i(spatial)                     → float
  morans_i_series(word, rule, width)    → list[float]
  spatial_classification(i_val)         → str
  morans_i_dict(word, rule, width)      → dict
  all_morans_i(word, width)             → dict[str, dict]
  build_moran_data(words, width)        → dict
  print_moran(word, rule, color)        → None
  print_moran_stats(words, color)       → None

Запуск
──────
  python3 -m projects.hexglyph.solan_moran --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_moran --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_moran --stats --no-color
"""

from __future__ import annotations
import math
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


# ── Core computation ──────────────────────────────────────────────────────────

def morans_i(spatial: list[int | float]) -> float:
    """Moran's I on a circular 1-D lattice with nearest-neighbour weights.

    Returns NaN for constant (zero-variance) spatial fields.
    """
    n = len(spatial)
    if n < 2:
        return float('nan')
    mu    = sum(spatial) / n
    denom = sum((x - mu) ** 2 for x in spatial)
    if denom < 1e-12:
        return float('nan')
    numer = sum(
        (spatial[i] - mu) * (spatial[(i + 1) % n] - mu)
        for i in range(n)
    )
    return round(numer / denom, 8)


def spatial_classification(i_val: float) -> str:
    """Qualitative label for a Moran's I value."""
    if math.isnan(i_val):
        return 'constant'
    if i_val > 0.5:
        return 'strongly clustered'
    if i_val > 0.1:
        return 'clustered'
    if i_val >= -0.1:
        return 'random'
    if i_val >= -0.5:
        return 'dispersed'
    return 'strongly dispersed'


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Per-step Moran's I series ─────────────────────────────────────────────────

def morans_i_series(word: str, rule: str, width: int = _DEFAULT_W) -> list[float]:
    """Moran's I at each time step of the attractor period."""
    orbit = _get_orbit(word, rule, width)
    return [morans_i([float(orbit[t][i]) for i in range(width)])
            for t in range(len(orbit))]


# ── Summary dict ──────────────────────────────────────────────────────────────

def morans_i_dict(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full Moran's I analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']
    series = morans_i_series(word, rule, width)

    valid  = [v for v in series if not math.isnan(v)]
    mean_i = round(sum(valid) / len(valid), 8) if valid else float('nan')
    min_i  = round(min(valid), 8)              if valid else float('nan')
    max_i  = round(max(valid), 8)              if valid else float('nan')
    var_i  = (round(sum((v - mean_i) ** 2 for v in valid) / len(valid), 8)
              if valid else float('nan'))

    return {
        'word':           word.upper(),
        'rule':           rule,
        'period':         period,
        'series':         series,
        'mean_i':         mean_i,
        'min_i':          min_i,
        'max_i':          max_i,
        'var_i':          var_i,
        'classification': spatial_classification(mean_i),
        'n_valid':        len(valid),
    }


def all_morans_i(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """morans_i_dict for all 4 rules."""
    return {rule: morans_i_dict(word, rule, width) for rule in _RULES}


def build_moran_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Aggregated Moran's I data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = morans_i_dict(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'series', 'mean_i', 'min_i', 'max_i',
                 'var_i', 'classification', 'n_valid')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'
_NEG  = '▒'


def _moran_bar(v: float, w: int = 22) -> str:
    if math.isnan(v):
        return '─' * w
    half   = w // 2
    filled = round(min(abs(v), 1.0) * half)
    if v >= 0:
        return ' ' * (half - filled) + _BAR * filled + '│' + ' ' * (half - 1)
    else:
        return ' ' * (half - 1) + '│' + _NEG * filled + ' ' * (half - filled)


def print_moran(word: str = 'ТУМАН', rule: str = 'xor3',
                color: bool = True) -> None:
    d   = morans_i_dict(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    nan_str = lambda v: f'{v:+.4f}' if not math.isnan(v) else '  NaN  '
    print(f'  {c}◈ Moran\'s I  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'Ī={nan_str(d["mean_i"])}  [{nan_str(d["min_i"])}, {nan_str(d["max_i"])}]'
          f'  {d["classification"]}{r}')
    print('  ' + '─' * 60)
    print(f'  {"t":>3}  {"I(t)":>9}  bar  (← dispersed | clustered →)')
    print('  ' + '─' * 60)
    for t, v in enumerate(d['series']):
        tag = nan_str(v)
        bar = _moran_bar(v)
        print(f'  {t:>3}  {tag:>9}  {bar}')
    print()


def print_moran_stats(words: list[str] | None = None,
                      color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_moran(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Moran's I spatial autocorrelation for Q6 CA attractors")
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_moran_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_moran(args.word, rule, color)
    else:
        print_moran(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
