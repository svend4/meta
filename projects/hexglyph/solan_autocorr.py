"""
solan_autocorr.py — Temporal & Spatial Autocorrelation of CA Orbits.

Temporal autocorrelation captures the rhythmic structure of each cell's
time series within an attractor orbit.  Spatial autocorrelation captures
how values at one cell relate to values at nearby cells in a single orbit state.

═══════════════════════════════════════════════════════════════════════
Temporal autocorrelation
═══════════════════════════════════════════════════════════════════════

  For cell i with time series  xᵢ(t) = orbit[t][i]  (t = 0 … P−1):

    μᵢ   = (1/P) Σ_t xᵢ(t)
    σᵢ²  = (1/P) Σ_t (xᵢ(t) − μᵢ)²
    ρᵢ(τ) = (1/P·σᵢ²) Σ_t (xᵢ(t)−μᵢ) · (xᵢ((t+τ)%P)−μᵢ)   ∈ [−1, 1]

  ρᵢ(0) = 1  (self-correlation)
  ρᵢ(τ) is SYMMETRIC about the half-period: ρᵢ(τ) = ρᵢ(P−τ)
  For constant-series cells (σᵢ² = 0): ρᵢ(τ) := 1.0 by convention.

Key results  (width = 16)
─────────────────────────
  ТУМАН XOR / ГОРА OR  (P=1, fixed points)
      ρᵢ(τ) = 1.0 for all i, τ.  ★ Trivially constant (no dynamics).

  ГОРА AND / ГОРА XOR3  (P=2)
      ★ ρᵢ(1) = −1.0 for EVERY cell i, regardless of values!
      Proof: for P=2, (xᵢ(0)−μ) = −(xᵢ(1)−μ),
             so Σ_t (xᵢ(t)−μ)(xᵢ(t+1)−μ) = −Σ_t (xᵢ(t)−μ)² = −P·σᵢ²,
             giving ρ(1) = −1.  ANY period-2 orbit has perfect anti-correlation.

  ТУМАН XOR3  (P=8)
      Mean AC profile (averaged over 16 cells):
          [1.0, −0.256, −0.252, +0.141, −0.267, +0.141, −0.252, −0.256]
      ★ The profile is PALINDROMIC: mean_ac[τ] = mean_ac[P−τ] (lag-flip symmetry).
      ★ Inner cells (7, 8) — nearly period-3 time series:
          cell 7: [24, 20, 43, 24, 20, 43, 24, 20]  ρ(3) ≈ +0.34
          cell 8: [63, 36, 60, 63, 36, 60, 63, 36]  ρ(3) ≈ +0.48
          These near-period-3 cells have the most negative lag-1 correlation
          (ρᵢ(1) ≈ −0.62, vs −0.04 … +0.23 for edge cells).
      ★ Edge cells (0, 15) are anomalous: ρ(1) = +0.23 (POSITIVE at lag 1!),
          contrasting with all other cells which have ρ(1) < 0.
          Their series (e.g. [48,51,43,43,43,43,40,48]) has a long plateau.

═══════════════════════════════════════════════════════════════════════
Spatial autocorrelation
═══════════════════════════════════════════════════════════════════════

  For a fixed orbit step t and spatial lag d:

    ρₛ(d, t) = corr of (orbit[t][i], orbit[t][(i+d)%N])  over i=0..N−1
    ρₛ(d)    = mean over t = 0..P−1

  ρₛ(0) = 1.0  (each cell correlates with itself).
  For ТУМАН XOR3: peak negative spatial correlation at d=3 (−0.164).

═══════════════════════════════════════════════════════════════════════
Cross-correlation
═══════════════════════════════════════════════════════════════════════

  C(i,j) = Pearson correlation of time series of cell i and cell j
           (zero temporal lag; synchronous correlation).

Functions
─────────
  cell_series(orbit, i)                          → list[int]
  temporal_ac(series, lag)                       → float
  temporal_ac_profile(series)                    → list[float]
  mean_temporal_ac(orbit)                        → list[float]
  cell_ac_all(orbit)                             → list[list[float]]
  cell_crosscorr(s1, s2)                         → float
  crosscorr_matrix(orbit)                        → list[list[float]]
  spatial_ac(orbit, d)                           → float
  spatial_ac_profile(orbit)                      → list[float]
  autocorr_summary(word, rule, width)            → dict
  all_autocorr(word, width)                      → dict[str, dict]
  build_autocorr_data(words, width)              → dict
  print_autocorr(word, rule, color)              → None
  print_autocorr_table(words, color)             → None

Запуск
──────
  python3 -m projects.hexglyph.solan_autocorr --word ТУМАН --rule xor3 --no-color
  python3 -m projects.hexglyph.solan_autocorr --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_autocorr --table --no-color
"""

from __future__ import annotations
import math
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


def _get_orbit(word: str, rule: str, width: int) -> list[tuple[int, ...]]:
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Core statistics ────────────────────────────────────────────────────────────

def cell_series(orbit: list[tuple[int, ...]], i: int) -> list[int]:
    """Time series of cell i over the orbit: [orbit[0][i], …, orbit[P-1][i]]."""
    return [orbit[t][i] for t in range(len(orbit))]


def temporal_ac(series: list[int] | list[float], lag: int) -> float:
    """Pearson autocorrelation of `series` at given lag (orbit is periodic).

    ρ(τ) = [(1/P) Σ_t (x_t − μ)(x_{(t+τ)%P} − μ)] / σ²

    Returns 1.0 for constant series (σ²=0) by convention.
    """
    P  = len(series)
    mu = sum(series) / P
    var = sum((x - mu) ** 2 for x in series) / P
    if var < 1e-12:
        return 1.0
    cov = sum((series[t] - mu) * (series[(t + lag) % P] - mu)
              for t in range(P)) / P
    return round(cov / var, 8)


def temporal_ac_profile(series: list[int] | list[float]) -> list[float]:
    """Full temporal autocorrelation profile: [ρ(0), ρ(1), …, ρ(P-1)].

    Always starts at 1.0 (ρ(0) = 1).
    The profile is palindromic: ρ(τ) = ρ(P−τ).
    """
    P = len(series)
    return [temporal_ac(series, lag) for lag in range(P)]


def mean_temporal_ac(orbit: list[tuple[int, ...]]) -> list[float]:
    """Mean temporal AC profile, averaged over all N cells.

    Returns list of P floats (one per lag τ = 0..P-1).
    """
    P  = len(orbit)
    N  = len(orbit[0])
    profiles = [temporal_ac_profile(cell_series(orbit, i)) for i in range(N)]
    return [round(sum(profiles[i][lag] for i in range(N)) / N, 8)
            for lag in range(P)]


def cell_ac_all(orbit: list[tuple[int, ...]]) -> list[list[float]]:
    """Per-cell temporal AC profiles.  Shape: N × P."""
    N = len(orbit[0])
    return [temporal_ac_profile(cell_series(orbit, i)) for i in range(N)]


# ── Cross-correlation ──────────────────────────────────────────────────────────

def cell_crosscorr(s1: list[int] | list[float],
                   s2: list[int] | list[float]) -> float:
    """Synchronous Pearson cross-correlation between two cell time series.

    Returns corr(xᵢ, xⱼ) at zero lag.
    """
    P = len(s1)
    m1, m2 = sum(s1) / P, sum(s2) / P
    v1 = sum((x - m1) ** 2 for x in s1) / P
    v2 = sum((x - m2) ** 2 for x in s2) / P
    if v1 < 1e-12 or v2 < 1e-12:
        return 0.0
    cov = sum((s1[t] - m1) * (s2[t] - m2) for t in range(P)) / P
    return round(cov / math.sqrt(v1 * v2), 8)


def crosscorr_matrix(orbit: list[tuple[int, ...]]) -> list[list[float]]:
    """N × N symmetric cross-correlation matrix between all cell pairs."""
    N       = len(orbit[0])
    series  = [cell_series(orbit, i) for i in range(N)]
    mat: list[list[float]] = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            c = cell_crosscorr(series[i], series[j])
            mat[i][j] = c
            mat[j][i] = c
    return mat


# ── Spatial autocorrelation ────────────────────────────────────────────────────

def spatial_ac(orbit: list[tuple[int, ...]], d: int) -> float:
    """Mean spatial autocorrelation at distance d, averaged over orbit steps.

    For each step t, computes Pearson correlation between
    {orbit[t][i]} and {orbit[t][(i+d)%N]} over i, then averages over t.
    """
    P  = len(orbit)
    N  = len(orbit[0])
    vals: list[float] = []
    for t in range(P):
        s    = list(orbit[t])
        mu   = sum(s) / N
        var  = sum((x - mu) ** 2 for x in s) / N
        if var < 1e-12:
            continue
        cov  = sum((s[i] - mu) * (s[(i + d) % N] - mu)
                   for i in range(N)) / N
        vals.append(cov / var)
    # All steps had zero spatial variance → constant pattern → perfect correlation
    return round(sum(vals) / len(vals), 8) if vals else 1.0


def spatial_ac_profile(orbit: list[tuple[int, ...]],
                       max_d: int | None = None) -> list[float]:
    """Spatial autocorrelation profile for d = 0, 1, …, N//2.

    Returns list of floats.  ρₛ(0) = 1.0 always.
    """
    N = len(orbit[0])
    D = max_d if max_d is not None else N // 2
    return [spatial_ac(orbit, d) for d in range(D + 1)]


# ── Summary ────────────────────────────────────────────────────────────────────

def autocorr_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full autocorrelation summary for word × rule."""
    orbit    = _get_orbit(word, rule, width)
    P        = len(orbit)
    N        = width
    all_cell_ac = cell_ac_all(orbit)   # N × P
    mean_ac  = mean_temporal_ac(orbit) # P
    cc_mat   = crosscorr_matrix(orbit) # N × N

    # Lag-1 stats (skip for P=1)
    lag1_vals = [all_cell_ac[i][1] for i in range(N)] if P > 1 else [1.0] * N
    # Palindrome check: mean_ac[τ] == mean_ac[P-τ]
    is_palindrome = all(
        abs(mean_ac[tau] - mean_ac[P - tau]) < 1e-5
        for tau in range(1, P // 2 + 1)
    ) if P > 1 else True

    # Dominant lag: lag τ > 0 with maximum |mean_ac[τ]|
    if P > 1:
        dom_lag = max(range(1, P), key=lambda τ: abs(mean_ac[τ]))
    else:
        dom_lag = 0

    # Off-diagonal cross-correlation stats
    off_diag = [cc_mat[i][j] for i in range(N) for j in range(N) if i != j]
    mean_cc  = round(sum(off_diag) / len(off_diag), 6) if off_diag else 0.0

    # Spatial AC profile
    sp_ac    = spatial_ac_profile(orbit)

    return {
        'word':             word.upper(),
        'rule':             rule,
        'period':           P,
        'n_cells':          N,
        # Temporal AC
        'mean_ac':          mean_ac,
        'cell_ac':          all_cell_ac,
        'mean_ac_lag1':     round(mean_ac[1] if P > 1 else 1.0, 6),
        'max_ac_lag1':      round(max(lag1_vals), 6),
        'min_ac_lag1':      round(min(lag1_vals), 6),
        'dominant_lag':     dom_lag,
        'is_palindrome':    is_palindrome,
        'all_p2_anti':      all(abs(v + 1.0) < 1e-5 for v in lag1_vals) if P > 1 else False,
        # Cross-correlation
        'crosscorr_matrix': cc_mat,
        'mean_crosscorr':   mean_cc,
        # Spatial AC
        'spatial_ac':       sp_ac,
        'max_spatial_ac':   round(max(sp_ac[1:]), 6) if len(sp_ac) > 1 else 0.0,
        'min_spatial_ac':   round(min(sp_ac[1:]), 6) if len(sp_ac) > 1 else 0.0,
    }


def all_autocorr(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """autocorr_summary for all 4 rules."""
    return {rule: autocorr_summary(word, rule, width) for rule in _RULES}


def build_autocorr_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact autocorrelation data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = autocorr_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'mean_ac', 'mean_ac_lag1',
                                   'max_ac_lag1', 'min_ac_lag1',
                                   'dominant_lag', 'is_palindrome',
                                   'all_p2_anti', 'mean_crosscorr',
                                   'spatial_ac', 'max_spatial_ac',
                                   'min_spatial_ac')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'
_BAR_W = 16


def _ac_bar(v: float, w: int = _BAR_W) -> str:
    """Horizontal bar: negative=left (░), zero=centre, positive=right (█)."""
    half    = w // 2
    filled  = round(abs(v) * half)
    filled  = min(filled, half)
    if v >= 0:
        return '░' * half + '█' * filled + '░' * (half - filled)
    else:
        return '░' * (half - filled) + '█' * filled + '░' * half


def print_autocorr(word: str = 'ТУМАН', rule: str = 'xor3',
                   color: bool = True) -> None:
    d   = autocorr_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',  'or':   'OR |'}.get(rule, rule)
    pal = ' ★palindrome' if d['is_palindrome'] else ''
    p2a = ' ★all_p2_anti' if d['all_p2_anti'] else ''
    print(f'  {c}◈ Autocorr  {word.upper()}  |  {lbl}  P={d["period"]}'
          f'  dom_lag={d["dominant_lag"]}{pal}{p2a}{r}')
    print(f'  mean_ac_lag1={d["mean_ac_lag1"]:+.4f}'
          f'  [{d["min_ac_lag1"]:+.3f}, {d["max_ac_lag1"]:+.3f}]'
          f'  mean_cc={d["mean_crosscorr"]:+.4f}')
    print('  ' + '─' * 60)
    # Mean AC profile
    print(f'  mean temporal AC  (← neg | pos →)')
    for lag, v in enumerate(d['mean_ac']):
        bar = _ac_bar(v)
        print(f'  lag={lag:>2}  {v:>+6.3f}  [{bar}]')
    print()
    # Per-cell lag-1 AC
    print(f'  Per-cell lag-1 AC  (← neg | pos →)')
    for ci, ac_prof in enumerate(d['cell_ac']):
        v1 = ac_prof[1] if d['period'] > 1 else 1.0
        bar = _ac_bar(v1)
        print(f'  cell {ci:>2}  {v1:>+6.3f}  [{bar}]')
    print()
    # Spatial AC profile
    print(f'  Spatial AC profile:')
    for lag, v in enumerate(d['spatial_ac']):
        bar = _ac_bar(v)
        print(f'  d={lag:>2}  {v:>+6.3f}  [{bar}]')
    print()


def print_autocorr_table(words: list[str] | None = None,
                          color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    head = '  '.join(
        (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5} mac1 pal' + R
        for rl in _RULES)
    print(f'  {"Слово":10s}  {head}')
    print('  ' + '─' * 68)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = _RCOL.get(rule, '') if color else ''
            d   = autocorr_summary(word, rule)
            pal = '★' if d['is_palindrome'] else ' '
            parts.append(f'{col}{d["mean_ac_lag1"]:>+5.2f} {pal}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='Temporal & Spatial Autocorrelation of CA Orbits')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    p.add_argument('--no-color',  action='store_true')
    args  = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        d = autocorr_summary(args.word, args.rule)
        # crosscorr_matrix and cell_ac contain nested lists — JSON-safe
        print(_json.dumps(
            {k: v for k, v in d.items()
             if k not in ('crosscorr_matrix', 'cell_ac')},
            ensure_ascii=False, indent=2))
    elif args.table:
        print_autocorr_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_autocorr(args.word, rule, color)
    else:
        print_autocorr(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
