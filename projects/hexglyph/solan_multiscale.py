"""
solan_multiscale.py — Multi-Scale Entropy (MSE) of Q6 CA Attractor Sequences.

Multi-Scale Entropy (Costa, Goldberger & Peng 2002) characterises complexity
across different temporal scales by:

  1. Coarse-graining the time series at scale τ:
         y_j^(τ)  =  (1/τ) Σ_{i=(j-1)τ+1}^{jτ}  x_i

  2. Computing Sample Entropy (SampEn) of the coarse-grained series:
         SampEn(m, r, N)  =  − ln(A / B)
     where B = # template matches of length m, A = of length m+1
     within Chebyshev tolerance r = 0.15 · σ(original_series)

  3. Plotting SampEn vs τ → the complexity profile.

Interpretation
──────────────
  Monotone ↓  : structured signal — complexity lost as detail averaged out
  Monotone ↑  : 1/f-like — complexity emerges at coarser scales
  Non-monotone: multi-scale structure (e.g. nested periodicities)
  Flat near 0 : fully predictable (periodic / fixed point)

The series for each cell is built by repeating the attractor period to give
at least MIN_LEN = 200 samples, ensuring reliable SampEn estimation.

Key results
───────────
  XOR/AND/OR fixed-point (P=1)   : SampEn ≈ 0 at all τ (constant)
  ГОРА AND (P=2, alternating)    : low SampEn that rises with τ (coarsening
                                   mixes 0/1 → variance decreases → r shrinks)
  ТУМАН XOR3 (P=8)               : non-monotone peak at τ=1 or τ=2, then drop
  ГОРА XOR3 (P=2, half const.)   : mixed profile between cells

Functions
─────────
  get_cell_series(word, rule, width, min_len)  → list[list[float]]  (N series)
  coarse_grain(series, tau)                    → list[float]
  sample_entropy(series, m, r)                 → float  (NaN if too short)
  mse_cell(series, max_tau, m)                 → list[float]  SampEn per τ
  mse_profile(word, rule, max_tau, width, m)   → list[list[float]]  N × max_tau
  mean_mse_profile(word, rule, max_tau, width, m) → list[float]
  mse_dict(word, rule, max_tau, width, m)      → dict
  all_mse(word, max_tau, width, m)             → dict[str, dict]
  build_mse_data(words, max_tau, width, m)     → dict
  print_mse(word, rule, max_tau, color)        → None
  print_mse_stats(words, color)                → None

Запуск
──────
  python3 -m projects.hexglyph.solan_multiscale --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_multiscale --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_multiscale --stats --no-color
"""

from __future__ import annotations
import math
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:       list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W:   int = 16
_DEFAULT_TAU: int = 8
_DEFAULT_M:   int = 2
_MIN_LEN:     int = 200     # minimum series length for SampEn estimation
_SAMPEN_R:    float = 0.15  # r = R_FACTOR * σ(series)
_MIN_CG_LEN:  int = 10      # minimum coarse-grained length for SampEn


# ── Series construction ────────────────────────────────────────────────────────

def get_cell_series(word: str, rule: str, width: int = _DEFAULT_W,
                    min_len: int = _MIN_LEN) -> list[list[float]]:
    """Return N raw-Q6 temporal series (one per cell) of length ≥ min_len.

    The attractor period is repeated as many times as needed.
    """
    from projects.hexglyph.solan_traj import word_trajectory
    traj = word_trajectory(word.upper(), rule, width)
    transient = traj['transient']
    period = max(traj['period'], 1)
    rows = traj['rows']
    attractor = rows[transient:transient + period]
    if not attractor:
        attractor = [rows[-1]] if rows else [[0] * width]
        period = 1

    repeat = max(1, math.ceil(min_len / period))
    total = period * repeat

    series: list[list[float]] = []
    for i in range(width):
        series.append([float(attractor[t % period][i]) for t in range(total)])
    return series


# ── Core computation ───────────────────────────────────────────────────────────

def coarse_grain(series: list[float], tau: int) -> list[float]:
    """Average non-overlapping blocks of length τ."""
    n = len(series)
    if tau <= 0:
        return list(series)
    return [sum(series[j * tau:(j + 1) * tau]) / tau
            for j in range(n // tau)]


def sample_entropy(series: list[float], m: int = _DEFAULT_M,
                   r: float | None = None) -> float:
    """Sample Entropy SampEn(m, r) of *series*.

    Returns NaN if the series is too short (< m + 2) or B = 0.
    """
    n = len(series)
    if n < m + 2:
        return float('nan')

    if r is None:
        mu = sum(series) / n
        sigma = math.sqrt(sum((x - mu) ** 2 for x in series) / n)
        r = _SAMPEN_R * sigma

    def _count(length: int) -> int:
        limit = n - length
        if limit < 2:
            return 0
        count = 0
        for i in range(limit):
            for j in range(limit):
                if i == j:
                    continue
                if all(abs(series[i + k] - series[j + k]) <= r
                       for k in range(length)):
                    count += 1
        return count

    B = _count(m)
    if B == 0:
        return float('nan')
    A = _count(m + 1)
    if A == 0:
        return float('inf')
    return max(0.0, -math.log(A / B))


def mse_cell(series: list[float], max_tau: int = _DEFAULT_TAU,
             m: int = _DEFAULT_M) -> list[float]:
    """MSE profile for a single cell series: [SampEn(τ=1), ..., SampEn(τ=max_τ)].

    Compute r once from the original series; apply same r to all scales.
    """
    n = len(series)
    if n == 0:
        return [float('nan')] * max_tau

    mu = sum(series) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in series) / n)
    r = _SAMPEN_R * sigma

    profile: list[float] = []
    for tau in range(1, max_tau + 1):
        cg = coarse_grain(series, tau)
        if len(cg) < _MIN_CG_LEN:
            profile.append(float('nan'))
        else:
            profile.append(sample_entropy(cg, m, r))
    return profile


def mse_profile(word: str, rule: str, max_tau: int = _DEFAULT_TAU,
                width: int = _DEFAULT_W, m: int = _DEFAULT_M) -> list[list[float]]:
    """MSE profiles for all N cells: shape (N, max_tau)."""
    series = get_cell_series(word, rule, width)
    return [mse_cell(s, max_tau, m) for s in series]


def mean_mse_profile(word: str, rule: str, max_tau: int = _DEFAULT_TAU,
                     width: int = _DEFAULT_W, m: int = _DEFAULT_M) -> list[float]:
    """Mean MSE profile over cells (ignoring NaN values)."""
    profiles = mse_profile(word, rule, max_tau, width, m)
    result: list[float] = []
    for tau_idx in range(max_tau):
        vals = [p[tau_idx] for p in profiles
                if not math.isnan(p[tau_idx]) and not math.isinf(p[tau_idx])]
        result.append(round(sum(vals) / len(vals), 6) if vals else float('nan'))
    return result


# ── MSE dictionary ─────────────────────────────────────────────────────────────

def mse_dict(word: str, rule: str, max_tau: int = _DEFAULT_TAU,
             width: int = _DEFAULT_W, m: int = _DEFAULT_M) -> dict:
    """Full MSE analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    profiles = mse_profile(word, rule, max_tau, width, m)
    mean_prof = mean_mse_profile(word, rule, max_tau, width, m)

    # Complexity index: area under mean MSE curve (trapezoidal, skip NaN)
    valid = [(i + 1, v) for i, v in enumerate(mean_prof)
             if not math.isnan(v) and not math.isinf(v)]
    complexity_index = round(sum(v for _, v in valid) / len(valid), 6) if valid else 0.0

    # Peak scale: τ with highest mean SampEn
    valid2 = [(i + 1, v) for i, v in enumerate(mean_prof)
              if not math.isnan(v) and not math.isinf(v)]
    peak_tau = max(valid2, key=lambda x: x[1])[0] if valid2 else 1
    peak_se  = max(v for _, v in valid2) if valid2 else 0.0

    return {
        'word':             word.upper(),
        'rule':             rule,
        'period':           period,
        'max_tau':          max_tau,
        'm':                m,
        'cell_profiles':    profiles,
        'mean_profile':     mean_prof,
        'complexity_index': complexity_index,
        'peak_tau':         peak_tau,
        'peak_se':          round(peak_se, 6),
    }


def all_mse(word: str, max_tau: int = _DEFAULT_TAU,
            width: int = _DEFAULT_W, m: int = _DEFAULT_M) -> dict[str, dict]:
    """mse_dict for all 4 rules."""
    return {rule: mse_dict(word, rule, max_tau, width, m) for rule in _RULES}


def build_mse_data(words: list[str], max_tau: int = _DEFAULT_TAU,
                   width: int = _DEFAULT_W, m: int = _DEFAULT_M) -> dict:
    """Build aggregated MSE data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = mse_dict(word, rule, max_tau, width, m)
            per_rule[rule][word.upper()] = {k: d[k] for k in (
                'period', 'mean_profile', 'complexity_index', 'peak_tau', 'peak_se')}
    return {'words': [w.upper() for w in words], 'width': width,
            'max_tau': max_tau, 'm': m, 'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'
_SHD  = '░'


def _bar(v: float, width: int = 20, vmax: float = 2.0) -> str:
    filled = round(min(v, vmax) / vmax * width) if vmax > 0 else 0
    return _BAR * filled + _SHD * (width - filled)


def print_mse(word: str = 'ТУМАН', rule: str = 'xor3',
              max_tau: int = _DEFAULT_TAU, color: bool = True) -> None:
    d = mse_dict(word, rule, max_tau)
    c = _RCOL.get(rule, '') if color else ''
    r = _RST if color else ''
    RULE = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    mean = d['mean_profile']
    vmax = max((v for v in mean if not math.isnan(v) and not math.isinf(v)), default=1.0)
    vmax = max(vmax, 0.001)
    print(f'  {c}◈ MSE  {word.upper()}  |  {RULE}  P={d["period"]}  '
          f'CI={d["complexity_index"]:.4f}  τ*={d["peak_tau"]}{r}')
    print('  ' + '─' * 60)
    print(f'  {"τ":>3}  {"SampEn":>8}  bar')
    print('  ' + '─' * 60)
    for tau, se in enumerate(mean, 1):
        bar = _bar(se, vmax=vmax) if not (math.isnan(se) or math.isinf(se)) else '─'*20
        tag = f'{se:.4f}' if not (math.isnan(se) or math.isinf(se)) else 'NaN  '
        print(f'  {tau:>3}  {tag:>8}  {bar}')
    print(f'\n  CI={d["complexity_index"]:.4f}  peak_SE={d["peak_se"]:.4f}  τ*={d["peak_tau"]}')
    print()


def print_mse_stats(words: list[str] | None = None, max_tau: int = _DEFAULT_TAU,
                    color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_mse(word, rule, max_tau, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Multi-Scale Entropy (MSE) of Q6 CA')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--max-tau',   type=int, default=_DEFAULT_TAU)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_mse_stats(max_tau=args.max_tau, color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_mse(args.word, rule, args.max_tau, color)
    else:
        print_mse(args.word, args.rule, args.max_tau, color)


if __name__ == '__main__':
    _cli()
