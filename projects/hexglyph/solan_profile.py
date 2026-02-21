"""solan_profile.py — Statistical Moment Profile of Q6 CA Orbit States.

At each orbit step t the N=16 cell values form a distribution over {0..63}.
The four standard moments characterise the "shape" of this distribution:

    mean(t)     = Σ_i orbit[t][i] / N           — mean Q6 level
    var(t)      = Σ_i (orbit[t][i] − mean)² / N — spread of values
    skewness(t) = (1/N) Σ_i ((orbit[t][i]−mean)/std)³  — asymmetry
    kurtosis(t) = (1/N) Σ_i ((orbit[t][i]−mean)/std)⁴ − 3  — tailedness

Additional shape statistics per step:
    range(t)         = max − min     — total spread
    mode(t)          = most common cell value
    mode_fraction(t) = mode count / N   — degree of value clustering
    n_distinct(t)    = # distinct Q6 values among the 16 cells

Extreme findings (width = 16, XOR3)
──────────────────────────────────────
  МАТ   t=1  skew= 2.52  kurt= 4.76  mode=23  count=14/16 (87.5%)
    State: [23,23,23,23,23,23,23,23,23,23,23,23,23,23,48,63]
    → Near-total collapse: 14 of 16 cells converge to value 23 at this orbit step.
    Mode 23 persists across all 8 МАТ XOR3 steps with declining counts:
    t=1:14/16  t=2:12/16  t=3,5:10/16  t=4:8/16  …

  ТУНДРА t=2  skew=−3.06  kurt= 8.41  mode=60  count=6/16
    State: [3,48,50,51,60,60,60,60,60,60,62,62,62,62,62,62]
    → Single outlier cell at value 3; 12 of 16 cells cluster near 60-62.
    Highest kurtosis in the full lexicon under XOR3.

  ДУГА  t=0,1  var≈32.2  range=14  — smallest non-zero variance in XOR3.
    All 16 cells within a 14-unit window — most spatially uniform orbit.

Temporal perspective (per-cell):
    temporal_mean(i)  = Σ_t orbit[t][i] / P  — cell i's orbit-average Q6 value
    temporal_var(i)   = Σ_t (orbit[t][i] − temporal_mean)² / P
    Cell with max temporal_var is the "most mobile" cell.

Запуск:
    python3 -m projects.hexglyph.solan_profile --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_profile --word ТУНДРА --rule xor3
    python3 -m projects.hexglyph.solan_profile --word ТУМАН --rule xor3
    python3 -m projects.hexglyph.solan_profile --table --no-color
    python3 -m projects.hexglyph.solan_profile --json --word МАТ
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES          = tuple(_ALL_RULES)
_DEFAULT_WIDTH = 16


# ── Moment computation ────────────────────────────────────────────────────────

def _step_moments(vals: list[int]) -> dict[str, Any]:
    """Compute all shape statistics for a single orbit step."""
    N   = len(vals)
    mu  = sum(vals) / N
    var = sum((v - mu) ** 2 for v in vals) / N
    std = var ** 0.5

    if std > 1e-9:
        skew = sum(((v - mu) / std) ** 3 for v in vals) / N
        kurt = sum(((v - mu) / std) ** 4 for v in vals) / N - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    cnt           = Counter(vals)
    mode_v, mode_c = cnt.most_common(1)[0]
    min_v         = min(vals)
    max_v         = max(vals)

    return {
        'mean':          round(mu, 6),
        'var':           round(var, 6),
        'std':           round(std, 6),
        'skewness':      round(skew, 6),
        'kurtosis':      round(kurt, 6),
        'range':         max_v - min_v,
        'min_val':       min_v,
        'max_val':       max_v,
        'mode':          mode_v,
        'mode_count':    mode_c,
        'mode_fraction': round(mode_c / N, 6),
        'n_distinct':    len(cnt),
    }


# ── Per-word summary ──────────────────────────────────────────────────────────

def profile_summary(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Statistical moment profile for one word/rule.

    Keys
    ────
    word, rule, period, n_cells

    spatial_profiles   : list[dict] — per-step moment dicts (P entries)
    spatial_mean       : list[float]   — mean at each step
    spatial_var        : list[float]   — variance at each step
    spatial_skewness   : list[float]   — skewness at each step
    spatial_kurtosis   : list[float]   — kurtosis at each step
    spatial_range      : list[int]     — range at each step
    mode_vals          : list[int]     — mode value at each step
    mode_counts        : list[int]     — mode count at each step
    mode_fractions     : list[float]   — mode fraction at each step
    n_distinct         : list[int]     — distinct values at each step

    max_mode_fraction       : float   — peak mode fraction across orbit
    max_mode_fraction_step  : int
    max_skew_abs            : float   — max |skewness| across orbit
    max_skew_abs_step       : int
    max_kurtosis            : float   — max kurtosis across orbit
    max_kurtosis_step       : int
    min_var                 : float   — minimum non-zero variance (or 0)
    min_var_step            : int
    max_range               : int     — maximum range across orbit
    max_range_step          : int

    mean_spatial_mean  : float  — orbit-average of mean(t)
    mean_spatial_var   : float  — orbit-average of var(t)

    dominant_mode_val  : int    — Q6 value appearing as mode most steps
    dominant_mode_n    : int    — how many orbit steps it is mode

    temporal_mean      : list[float]  — per-cell orbit average (N entries)
    temporal_var       : list[float]  — per-cell orbit variance (N entries)
    max_temporal_var_cell : int       — cell index with highest temporal var
    max_temporal_var      : float
    """
    from projects.hexglyph.solan_perm import get_orbit

    orbit = get_orbit(word, rule, width)
    P     = len(orbit)
    N     = width

    # ── Spatial (per-step) profiles ─────────────────────────────────────────
    profiles = [_step_moments(list(state)) for state in orbit]

    s_mean  = [p['mean']          for p in profiles]
    s_var   = [p['var']           for p in profiles]
    s_skew  = [p['skewness']      for p in profiles]
    s_kurt  = [p['kurtosis']      for p in profiles]
    s_range = [p['range']         for p in profiles]
    modes   = [p['mode']          for p in profiles]
    mcnts   = [p['mode_count']    for p in profiles]
    mfracs  = [p['mode_fraction'] for p in profiles]
    ndist   = [p['n_distinct']    for p in profiles]

    # Peak statistics
    mf_step = max(range(P), key=lambda t: mfracs[t])
    sk_step = max(range(P), key=lambda t: abs(s_skew[t]))
    ku_step = max(range(P), key=lambda t: s_kurt[t])
    rg_step = max(range(P), key=lambda t: s_range[t])

    nonzero_var = [(s_var[t], t) for t in range(P) if s_var[t] > 1e-9]
    if nonzero_var:
        min_var_v, min_var_t = min(nonzero_var)
    else:
        min_var_v, min_var_t = 0.0, 0

    # Dominant mode value (most steps it is the mode)
    mode_cnt_across = Counter(modes)
    dom_mode_val, dom_mode_n = mode_cnt_across.most_common(1)[0]

    mean_s_mean = sum(s_mean) / P
    mean_s_var  = sum(s_var) / P

    # ── Temporal (per-cell) profiles ─────────────────────────────────────────
    t_mean = [sum(orbit[t][i] for t in range(P)) / P for i in range(N)]
    t_var  = [
        sum((orbit[t][i] - t_mean[i]) ** 2 for t in range(P)) / P
        for i in range(N)
    ]
    max_tv_cell = max(range(N), key=lambda i: t_var[i])

    return {
        'word':   word,
        'rule':   rule,
        'period': P,
        'n_cells': N,

        'spatial_profiles':   profiles,
        'spatial_mean':       [round(v, 4) for v in s_mean],
        'spatial_var':        [round(v, 4) for v in s_var],
        'spatial_skewness':   [round(v, 4) for v in s_skew],
        'spatial_kurtosis':   [round(v, 4) for v in s_kurt],
        'spatial_range':      s_range,
        'mode_vals':          modes,
        'mode_counts':        mcnts,
        'mode_fractions':     [round(v, 4) for v in mfracs],
        'n_distinct':         ndist,

        'max_mode_fraction':       round(mfracs[mf_step], 6),
        'max_mode_fraction_step':  mf_step,
        'max_skew_abs':            round(abs(s_skew[sk_step]), 6),
        'max_skew_abs_step':       sk_step,
        'max_kurtosis':            round(s_kurt[ku_step], 6),
        'max_kurtosis_step':       ku_step,
        'min_var':                 round(min_var_v, 6),
        'min_var_step':            min_var_t,
        'max_range':               s_range[rg_step],
        'max_range_step':          rg_step,

        'mean_spatial_mean':   round(mean_s_mean, 4),
        'mean_spatial_var':    round(mean_s_var, 4),

        'dominant_mode_val':   dom_mode_val,
        'dominant_mode_n':     dom_mode_n,

        'temporal_mean':          [round(v, 4) for v in t_mean],
        'temporal_var':           [round(v, 4) for v in t_var],
        'max_temporal_var_cell':  max_tv_cell,
        'max_temporal_var':       round(t_var[max_tv_cell], 4),
    }


def all_profile(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict[str, Any]]:
    """Profile summary for all 4 CA rules."""
    return {r: profile_summary(word, r, width) for r in RULES}


def build_profile_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Full profile analysis for the lexicon."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)
    return {
        'words': list(words),
        'data':  {w: {r: profile_summary(w, r, width) for r in RULES}
                  for w in words},
    }


def profile_dict(s: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of profile_summary."""
    d = dict(s)
    # spatial_profiles contains nested dicts — already serialisable
    return d


# ── Terminal output ───────────────────────────────────────────────────────────

def print_profile(
    word:  str,
    rule:  str,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print statistical moment profile for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    s = profile_summary(word, rule, width)
    P = s['period']

    print(bold + f"  ◈ Moment Profile  {word.upper()}  "
          + col + lbl + reset + bold + f"  (P={P})" + reset)
    print()

    # Per-step table
    hi_col  = '\033[38;5;220m' if color else ''   # skewed/clustered
    lo_col  = '\033[38;5;120m' if color else ''   # uniform
    oth_col = '\033[38;5;117m' if color else ''   # other

    print(f"  {'t':2s}  {'mean':>7s}  {'var':>8s}  {'skew':>7s}  {'kurt':>7s}  "
          f"{'range':>5s}  {'mode':>5s}  {'frac':>5s}  ndist")
    print('  ' + '─' * 72)

    for t in range(P):
        p    = s['spatial_profiles'][t]
        mf   = p['mode_fraction']
        sk   = abs(p['skewness'])
        if mf >= 0.75 or sk >= 2.0:
            tc = hi_col
        elif p['var'] < 100:
            tc = lo_col
        else:
            tc = oth_col

        print(f"  t{t}  {p['mean']:>7.2f}  {p['var']:>8.2f}  "
              f"{p['skewness']:>7.3f}  {p['kurtosis']:>7.3f}  "
              f"{p['range']:>5d}  {tc}{p['mode']:>5d}{reset}  "
              f"{tc}{mf:>5.3f}{reset}  {p['n_distinct']}")

    print()
    print(f"  Peak statistics:")
    mf_t  = s['max_mode_fraction_step']
    print(f"    max mode_fraction : {s['max_mode_fraction']:.4f} "
          f"at t={mf_t}  "
          f"(mode={s['mode_vals'][mf_t]}  "
          f"count={s['mode_counts'][mf_t]}/{s['n_cells']} cells)")
    print(f"    max |skewness|    : {s['max_skew_abs']:.4f} "
          f"at t={s['max_skew_abs_step']}")
    print(f"    max kurtosis      : {s['max_kurtosis']:.4f} "
          f"at t={s['max_kurtosis_step']}")
    print(f"    max range         : {s['max_range']} "
          f"at t={s['max_range_step']}")
    print(f"    min var (nz)      : {s['min_var']:.4f} "
          f"at t={s['min_var_step']}")
    print(f"  Orbit averages:")
    print(f"    mean(mean)        : {s['mean_spatial_mean']:.4f}")
    print(f"    mean(var)         : {s['mean_spatial_var']:.4f}")
    print(f"  Dominant mode value : {s['dominant_mode_val']} "
          f"(mode at {s['dominant_mode_n']}/{P} steps)")
    print(f"  Max temporal var    : cell {s['max_temporal_var_cell']} "
          f"({s['max_temporal_var']:.2f})")
    print()


def print_profile_table(
    words: list[str] | None = None,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Summary table: moment profile stats for all lexicon words."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Moment Profile Summary ({lbl}, n={len(words)})" + reset)
    print()
    print(f"  {'Слово':12s}  {'P':>3}  "
          f"{'max_mf':>6}  {'max_sk':>6}  {'max_ku':>6}  "
          f"{'minvar':>7}  {'dom_mode':>8}  max_tv_cell")
    print('  ' + '─' * 78)

    for word in words:
        s = profile_summary(word, rule, width)
        P = s['period']
        print(f"  {word.upper():12s}  {P:>3}  "
              f"{s['max_mode_fraction']:>6.3f}  "
              f"{s['max_skew_abs']:>6.3f}  "
              f"{s['max_kurtosis']:>6.3f}  "
              f"{s['min_var']:>7.2f}  "
              f"{s['dominant_mode_val']:>6}({s['dominant_mode_n']:d}/{P})  "
              f"{dim}cell {s['max_temporal_var_cell']} "
              f"({s['max_temporal_var']:.1f}){reset}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Statistical moment profile of Q6 CA orbit states')
    parser.add_argument('--word',  metavar='WORD', default='МАТ')
    parser.add_argument('--rule',  choices=list(RULES), default='xor3')
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--json',  action='store_true')
    parser.add_argument('--width', type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = profile_summary(args.word.upper(), args.rule, args.width)
        print(json.dumps(profile_dict(s), ensure_ascii=False, indent=2))
    elif args.table:
        print_profile_table(rule=args.rule, width=args.width, color=_color)
    else:
        print_profile(args.word.upper(), args.rule, args.width, color=_color)
