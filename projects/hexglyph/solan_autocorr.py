"""
solan_autocorr.py — Autocorrelation Function (ACF) of Q6 CA Attractors.

For each cell's temporal series (exactly one attractor period P), the
circular autocorrelation function reveals the lag-structure of temporal
dependencies:

    ACF(k) = (1/(N·σ²)) · Σ_{t=0}^{N−1} (x_t − μ)(x_{(t+k) mod N} − μ)

    ACF(0) = 1  always.
    Circular boundary: lags wrap around the period.

Derived quantities
──────────────────
  Decorrelation lag  τ₀  = smallest k ≥ 1 where ACF(k) ≤ 0
                     (first zero crossing; ∞ if ACF stays positive)
  Mean ACF power     Ā   = (1/max_lag) · Σ_{k=1}^{max_lag} ACF(k)²
                     (how much autocorrelation is present on average)
  ACF periodicity     = ACF is P-periodic by the circular boundary condition;
                     ACF(k) = ACF(P−k) (symmetry around P/2)

Key results  (width = 16, max_lag = P−1)
──────────────────────────────────────────
  XOR/AND/OR fixed-pt (P=1) : ACF = [1.0] only;  τ₀ = ∞ (no lags)
  ГОРА AND   (P=2)          : ACF = [1, −1];  perfect anti-persistence
  ГОРА XOR3  (P=2)          : same (all period-2 → ACF = [1, −1])
  ТУМАН XOR3 (P=8)          : rich cell-specific ACF; τ₀ = 1 or 2;
                              ACF symmetric around lag 4

Functions
─────────
  acf(series, max_lag)                   → list[float]  circular ACF
  decorrelation_lag(acf_vals)            → int | None
  mean_acf_power(acf_vals)               → float
  cell_acf_profile(word, rule, width, max_lag)  → list[dict]
  acf_dict(word, rule, width, max_lag)   → dict
  all_acf(word, width, max_lag)          → dict[str, dict]
  build_acf_data(words, width, max_lag)  → dict
  print_acf(word, rule, color)           → None
  print_acf_stats(words, color)          → None

Запуск
──────
  python3 -m projects.hexglyph.solan_autocorr --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_autocorr --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_autocorr --stats --no-color
"""

from __future__ import annotations
import math
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:      list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W:  int = 16
_DEFAULT_LAG: int = 8   # maximum lag (capped at P−1 per cell)


# ── ACF computation ────────────────────────────────────────────────────────────

def acf(series: list[int | float], max_lag: int | None = None) -> list[float]:
    """Circular autocorrelation function for a periodic series.

    Returns ACF values for lags k = 0, 1, …, min(max_lag, N−1).
    If the series is constant (σ² = 0), returns [1.0] + [nan]*max_lag.
    """
    n = len(series)
    if n == 0:
        return []
    if max_lag is None:
        max_lag = n - 1
    max_lag = min(max_lag, n - 1)

    mu  = sum(series) / n
    var = sum((x - mu) ** 2 for x in series) / n
    if var < 1e-12:
        return [1.0] + [float('nan')] * max_lag

    return [
        round(sum((series[t] - mu) * (series[(t + k) % n] - mu)
                  for t in range(n)) / (n * var), 8)
        for k in range(max_lag + 1)
    ]


def decorrelation_lag(acf_vals: list[float]) -> int | None:
    """First lag k ≥ 1 where ACF(k) ≤ 0.  Returns None if never."""
    for k in range(1, len(acf_vals)):
        v = acf_vals[k]
        if not math.isnan(v) and v <= 0:
            return k
    return None


def mean_acf_power(acf_vals: list[float]) -> float:
    """Mean squared ACF over lags k = 1, …, max_lag (excludes lag 0)."""
    vals = [v ** 2 for v in acf_vals[1:] if not math.isnan(v)]
    return round(sum(vals) / len(vals), 8) if vals else 0.0


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_attractor_series(word: str, rule: str,
                          width: int) -> tuple[list[list[float]], int]:
    """Return (cell_series_list, period).  Each series = exactly one period."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    if P == 0:
        return [[0.0]] * width, 0
    return [[float(orbit[t][i]) for t in range(P)] for i in range(width)], P


# ── Per-cell ACF profile ───────────────────────────────────────────────────────

def cell_acf_profile(word: str, rule: str, width: int = _DEFAULT_W,
                     max_lag: int = _DEFAULT_LAG) -> list[dict]:
    """Per-cell circular ACF analysis."""
    cell_series, P = _get_attractor_series(word, rule, width)
    result = []
    for i, s in enumerate(cell_series):
        lag = min(max_lag, P - 1)
        vals = acf(s, lag)
        tau0 = decorrelation_lag(vals)
        mpower = mean_acf_power(vals)
        result.append({
            'cell':      i,
            'acf':       vals,
            'max_lag':   lag,
            'tau0':      tau0,
            'mpower':    mpower,
        })
    return result


# ── Full dictionary ────────────────────────────────────────────────────────────

def acf_dict(word: str, rule: str, width: int = _DEFAULT_W,
             max_lag: int = _DEFAULT_LAG) -> dict:
    """Full ACF analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    cells  = cell_acf_profile(word, rule, width, max_lag)
    lag    = min(max_lag, period - 1)

    # Mean ACF across cells (average ignoring NaN)
    mean_acf_vals: list[float] = []
    for k in range(lag + 1):
        vs = [c['acf'][k] for c in cells
              if k < len(c['acf']) and not math.isnan(c['acf'][k])]
        mean_acf_vals.append(round(sum(vs) / len(vs), 8) if vs else float('nan'))

    tau0_vals = [c['tau0'] for c in cells if c['tau0'] is not None]
    mean_tau0 = round(sum(tau0_vals) / len(tau0_vals), 4) if tau0_vals else None

    mean_mp = round(sum(c['mpower'] for c in cells) / width, 8)

    return {
        'word':          word.upper(),
        'rule':          rule,
        'period':        period,
        'max_lag':       lag,
        'cell_profile':  cells,
        'mean_acf':      mean_acf_vals,
        'mean_tau0':     mean_tau0,
        'mean_mpower':   mean_mp,
    }


def all_acf(word: str, width: int = _DEFAULT_W,
            max_lag: int = _DEFAULT_LAG) -> dict[str, dict]:
    """acf_dict for all 4 rules."""
    return {rule: acf_dict(word, rule, width, max_lag) for rule in _RULES}


def build_acf_data(words: list[str], width: int = _DEFAULT_W,
                   max_lag: int = _DEFAULT_LAG) -> dict:
    """Aggregated ACF data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = acf_dict(word, rule, width, max_lag)
            per_rule[rule][word.upper()] = {k: d[k] for k in (
                'period', 'max_lag', 'mean_acf', 'mean_tau0', 'mean_mpower')}
    return {'words': [w.upper() for w in words], 'width': width,
            'max_lag': max_lag, 'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'
_SHD  = '░'
_NEG  = '▒'


def _bar_signed(v: float, w: int = 20) -> str:
    """Bar for signed ACF: right half = positive, left half = negative."""
    if math.isnan(v):
        return '─' * w
    half = w // 2
    mid  = half
    filled = round(abs(v) * half)
    if v >= 0:
        return ' ' * (w - mid - filled) + _BAR * filled + '│' + ' ' * (mid - 1)
    else:
        return ' ' * (w - mid - 1) + '│' + _NEG * filled + ' ' * (half - filled)


def print_acf(word: str = 'ТУМАН', rule: str = 'xor3',
              color: bool = True) -> None:
    d = acf_dict(word, rule)
    c = _RCOL.get(rule, '') if color else ''
    r = _RST if color else ''
    RULE = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    tau0_str = str(d['mean_tau0']) if d['mean_tau0'] is not None else '∞'
    print(f'  {c}◈ ACF  {word.upper()}  |  {RULE}  P={d["period"]}  '
          f'max_lag={d["max_lag"]}  τ₀≈{tau0_str}  '
          f'mean_power={d["mean_mpower"]:.4f}{r}')
    print('  ' + '─' * 60)
    print(f'  {"lag":>4}  {"mean_ACF":>9}  bar  (− ←|→ +)')
    print('  ' + '─' * 60)
    for k, v in enumerate(d['mean_acf']):
        tag = f'{v:+.4f}' if not math.isnan(v) else '  NaN  '
        bar = _bar_signed(v)
        print(f'  {k:>4}  {tag:>9}  {bar}')
    print(f'\n  τ₀≈{tau0_str}  mean_ACF²={d["mean_mpower"]:.4f}')
    print()


def print_acf_stats(words: list[str] | None = None,
                    color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_acf(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='Autocorrelation Function (ACF) for Q6 CA attractors')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--max-lag',   type=int, default=_DEFAULT_LAG)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_acf_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_acf(args.word, rule, color)
    else:
        print_acf(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
