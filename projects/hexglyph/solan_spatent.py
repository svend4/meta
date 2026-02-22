"""
solan_spatent.py — Spatial Entropy Profile of Q6 CA Attractors.

At each step t of the attractor orbit the N=16 cells hold Q6 values
v_0, …, v_{N−1} ∈ {0..63}.  The *spatial entropy* at step t is the
Shannon entropy of their distribution:

    H_t = − Σ_v  p_v · log2(p_v)     p_v = (# cells with value v) / N

H_t = 0   ⟺  all cells have the same value at step t  (zero diversity)
H_t = log2(N) = 4  ⟺  all N cells hold distinct values  (maximal diversity)

Over one period [t=0 .. P−1] we get the *spatial entropy profile*
[H_0, …, H_{P−1}].  Key statistics:

  mean_H  — time-averaged spatial diversity
  delta_H — max − min  (oscillation amplitude)
  std_H   — temporal variability of spatial diversity
  min_H / max_H

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      All cells equal 0 → H_t = 0.  Zero spatial diversity.

  ГОРА AND  (P=2, anti-phase clusters)
      8 even-indexed cells ↔ 8 odd-indexed cells, values {47, 1}.
      Each step: exactly 8 cells = 47, 8 cells = 1 → H_t = 1.0 bit.
      Profile constant → delta_H = 0.

  ГОРА XOR3  (P=2, 4 spatial clusters)
      4 distinct values, each in 4 cells → H_t = 2.0 bits (constant).
      delta_H = 0.

  ТУМАН XOR3  (P=8)
      Profile varies: H_t ∈ [2.17, 3.375].
      mean_H ≈ 2.85,  delta_H ≈ 1.20  (rich temporal modulation).
      ★  Most complex spatial diversity structure in the lexicon.

Functions
─────────
  spatial_entropy(state)                          → float
  orbit_spatial_entropy(word, rule, width)        → list[float]
  spatial_entropy_stats(word, rule, width)        → dict
  spatent_summary(word, rule, width)              → dict
  all_spatent(word, width)                        → dict[str, dict]
  build_spatent_data(words, width)                → dict
  print_spatent(word, rule, color)                → None
  print_spatent_stats(words, color)               → None

Запуск
──────
  python3 -m projects.hexglyph.solan_spatent --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_spatent --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_spatent --stats --no-color
"""

from __future__ import annotations
import sys
import argparse
import math

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_LOG2_W:    float = math.log2(_DEFAULT_W)   # 4.0  — theoretical max H


# ── Core computation ───────────────────────────────────────────────────────────

def spatial_entropy(state: list[int]) -> float:
    """Shannon entropy of the Q6-value distribution across N cells (in bits).

    H = 0   when all cells share one value (zero spatial diversity).
    H = log2(N) when all cells hold distinct values (max diversity).
    """
    n = len(state)
    if n == 0:
        return 0.0
    cnt: dict[int, int] = {}
    for v in state:
        cnt[v] = cnt.get(v, 0) + 1
    h = -sum((c / n) * math.log2(c / n) for c in cnt.values() if c > 0)
    return h if h > 0.0 else 0.0   # suppress −0.0


# ── Orbit helper ───────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int) -> list[list[int]]:
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Profile ────────────────────────────────────────────────────────────────────

def orbit_spatial_entropy(word: str, rule: str,
                          width: int = _DEFAULT_W) -> list[float]:
    """Spatial entropy profile [H_0, …, H_{P−1}] over one attractor period."""
    orbit = _get_orbit(word, rule, width)
    return [round(spatial_entropy(orbit[t]), 6) for t in range(len(orbit))]


def spatial_entropy_stats(word: str, rule: str,
                          width: int = _DEFAULT_W) -> dict:
    """Descriptive statistics of the spatial entropy profile."""
    profile = orbit_spatial_entropy(word, rule, width)
    P = len(profile)
    if P == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'delta': 0.0, 'profile': []}
    mean_h = sum(profile) / P
    var_h  = sum((h - mean_h) ** 2 for h in profile) / P
    std_h  = math.sqrt(var_h) if var_h > 0 else 0.0
    return {
        'mean':    round(mean_h, 6),
        'std':     round(std_h, 6),
        'min':     round(min(profile), 6),
        'max':     round(max(profile), 6),
        'delta':   round(max(profile) - min(profile), 6),
        'profile': profile,
    }


# ── Summary ────────────────────────────────────────────────────────────────────

def spatent_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full spatial-entropy summary for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    stats  = spatial_entropy_stats(word, rule, width)
    max_h  = math.log2(width) if width > 1 else 0.0

    # Normalised mean: 0 = all cells same, 1 = all cells distinct
    norm_mean = stats['mean'] / max_h if max_h > 0 else 0.0

    # Classify temporal variability of diversity
    if stats['delta'] < 1e-9:
        variability = 'constant'
    elif stats['delta'] < 0.5:
        variability = 'low'
    elif stats['delta'] < 1.0:
        variability = 'moderate'
    else:
        variability = 'high'

    return {
        'word':        word.upper(),
        'rule':        rule,
        'period':      period,
        'profile':     stats['profile'],
        'mean_H':      stats['mean'],
        'std_H':       stats['std'],
        'min_H':       stats['min'],
        'max_H':       stats['max'],
        'delta_H':     stats['delta'],
        'max_possible_H': round(max_h, 6),
        'norm_mean_H': round(norm_mean, 6),
        'variability': variability,
    }


def all_spatent(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """spatent_summary for all 4 rules."""
    return {rule: spatent_summary(word, rule, width) for rule in _RULES}


def build_spatent_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact spatial-entropy data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = spatent_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'profile', 'mean_H', 'std_H', 'min_H',
                 'max_H', 'delta_H', 'norm_mean_H', 'variability')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'max_possible_H': round(math.log2(width), 6) if width > 1 else 0.0,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'

_BAR_CHARS = ' ▁▂▃▄▅▆▇█'


def _h_bar(h: float, max_h: float = 4.0, width: int = 20) -> str:
    """ASCII bar proportional to h / max_h."""
    frac = min(h / max_h, 1.0) if max_h > 0 else 0.0
    filled = int(round(frac * width))
    return '█' * filled + '░' * (width - filled)


def print_spatent(word: str = 'ТУМАН', rule: str = 'xor3',
                  color: bool = True) -> None:
    d   = spatent_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    max_h = d['max_possible_H']

    print(f'  {c}◈ SpatEnt  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'variability={d["variability"]}{r}')
    print('  ' + '─' * 62)

    print(f'  Spatial entropy profile  (max possible H = {max_h:.2f} bits):')
    for t, ht in enumerate(d['profile']):
        bar = _h_bar(ht, max_h)
        print(f'    t={t:2d}  H={ht:5.3f} bits  |{bar}|')

    print(f'\n  Statistics:')
    print(f'    mean_H  = {d["mean_H"]:.4f}  ({d["norm_mean_H"]:.1%} of max)')
    print(f'    delta_H = {d["delta_H"]:.4f}  (max−min, temporal oscillation)')
    print(f'    std_H   = {d["std_H"]:.4f}')
    print(f'    range   = [{d["min_H"]:.4f}, {d["max_H"]:.4f}]')
    print()


def print_spatent_stats(words: list[str] | None = None,
                        color: bool = True) -> None:
    WORDS = words or ['ТУМАН', 'ГОРА']
    for word in WORDS:
        for rule in _RULES:
            print_spatent(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Spatial Entropy Profile of Q6 CA')
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
        print(_json.dumps(spatent_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.stats:
        print_spatent_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_spatent(args.word, rule, color)
    else:
        print_spatent(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
