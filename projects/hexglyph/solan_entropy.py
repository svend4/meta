"""
solan_entropy.py — Shannon Entropy Analysis of Q6 CA Attractor Orbits.

Two complementary entropy measures characterise the information content of
an attractor orbit:

  Spatial entropy H_s(t)   — entropy of the Q6 distribution *across cells*
                              at a single orbit step t:
                              H_s(t) = −Σ p_v · log₂(p_v)
                              where p_v = (# cells with value v) / N

  Temporal entropy H_c(i)  — entropy of the Q6 distribution *across orbit
                              steps* for a single cell i:
                              H_c(i) = −Σ p_v · log₂(p_v)
                              where p_v = (# steps with cell i = v) / P

The theoretical maximum for spatial entropy is log₂(min(N, 64)) bits.
For width=16: max spatial H_s = log₂(16) = 4 bits (all cells distinct values);
For width=16 and P=8:  max temporal H_c = log₂(8) = 3 bits (cell visits 8 distinct
values uniformly), or up to log₂(64)=6 bits for longer periods.

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, uniform zeros)
      H_s = [0.0]        max_H_s = 0.0  (single value → zero entropy)
      H_c = [0.0, …]     max_H_c = 0.0
      ★ Minimum entropy in both dimensions: perfectly ordered.

  ГОРА OR   (P=1, uniform 63)
      H_s = [0.0]        max_H_s = 0.0
      H_c = [0.0, …]     max_H_c = 0.0
      ★ Same as ТУМАН XOR: trivial fixed-point, zero entropy.

  ГОРА AND  (P=2, anti-phase 47↔1)
      H_s = [1.0, 1.0]   mean_H_s = 1.0  (two equally likely values)
      H_c = [1.0, …]     mean_H_c = 1.0  (each cell visits 2 values, equally)
      ★ H_s = H_c = log₂(2): bit-flip orbit, minimal nonzero entropy.

  ГОРА XOR3  (P=2, 4-periodic [49,47,15,63,…])
      H_s = [2.0, 2.0]   mean_H_s = 2.0  (4 distinct values, 4 cells each)
      H_c = [1.0, …]     mean_H_c = 1.0  (each cell oscillates between 2 values)
      ★ Spatial entropy DOUBLES compared to AND (4 vs 2 distinct values),
        but temporal entropy stays the same (each cell still has period 2).

  ТУМАН XOR3  (P=8, complex orbit)
      H_s = [2.311, 2.686, 2.174, 2.781, 3.250, 3.375, 3.125, 3.125]
      mean_H_s = 2.853   max_H_s = 3.375  (step 5: 11 distinct values)
      H_c per cell (first 4): [1.750, 2.406, 2.750, 2.750, …]
      mean_H_c = 2.234   max_H_c = 2.750  (cells 2,3,12,13)
                          min_H_c = 1.561  (cells 7,8, near centre)

      Step 5 analysis (H_s = 3.375 = 27/8):
        11 distinct values — 5 appear twice, 6 appear once
        H = −5·(2/16)·log₂(2/16) − 6·(1/16)·log₂(1/16)
          = 5·(1/8)·3 + 6·(1/16)·4 = 15/8 + 24/16 = 27/8 = 3.375 bits ✓

      ★ Symmetric H_c profile: H_c[i] = H_c[N−1−i] for all i
        (cells 0,15: 1.75; cells 1,14: 2.406; cells 2,13: 2.75;
         cells 3,12: 2.75; cells 4,11: 2.5; cells 5,10: 2.25;
         cells 6,9: 1.906; cells 7,8: 1.561)
      ★ Outer cells (0,15) and inner cells (7,8) have LOWER temporal
        entropy than mid-range cells (2,3,12,13) — the word interior
        drives the most temporal diversity.

Relationship to other modules
───────────────────────────────
  solan_vocab: orbit_vocab_size ≥ 2^H_c  (vocab size ≥ 2^entropy)
  solan_segment: long segments (always_uniform) → H_s = 0;
                 fully_fragmented does NOT imply high H_s (ГОРА AND: H_s=1)
  solan_balance: bit-level balance relates to marginal bit entropies

Functions
─────────
  spatial_entropy(state)                    → float
  temporal_entropy_cell(orbit, cell_idx)    → float
  spatial_entropy_orbit(word, rule, width)  → list[float]
  temporal_entropy_all(word, rule, width)   → list[float]
  mean_spatial_entropy(word, rule, width)   → float
  mean_temporal_entropy(word, rule, width)  → float
  max_spatial_entropy(word, rule, width)    → float
  min_spatial_entropy(word, rule, width)    → float
  max_temporal_entropy(word, rule, width)   → float
  min_temporal_entropy(word, rule, width)   → float
  entropy_summary(word, rule, width)        → dict
  all_entropy(word, width)                  → dict[str, dict]
  build_entropy_data(words, width)          → dict
  print_entropy(word, rule, color)          → None
  print_entropy_table(words, color)         → None

Запуск
──────
  python3 -m projects.hexglyph.solan_entropy --word ТУМАН --rule xor3 --no-color
  python3 -m projects.hexglyph.solan_entropy --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_entropy --table --no-color
"""

from __future__ import annotations
import sys
import math
import argparse
from collections import Counter

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_LOG2_N16:  float = 4.0   # log2(16) — max spatial entropy for width=16


def _get_orbit(word: str, rule: str, width: int) -> list[tuple[int, ...]]:
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Core entropy functions ─────────────────────────────────────────────────────

def spatial_entropy(state: tuple[int, ...] | list[int]) -> float:
    """Shannon entropy of the Q6 value distribution across cells at one step.

    H_s = −Σ p_v · log₂(p_v)  where p_v = count(v) / N.

    Returns 0.0 for a uniform state (all cells identical).
    Maximum value is log₂(min(N, 64)) bits.
    """
    N = len(state)
    counts = Counter(state)
    return max(0.0, -sum((c / N) * math.log2(c / N) for c in counts.values()))


def temporal_entropy_cell(orbit: list[tuple[int, ...]], cell_idx: int) -> float:
    """Shannon entropy of the Q6 value distribution for one cell over orbit steps.

    H_c(i) = −Σ p_v · log₂(p_v)  where p_v = count(v) / P.

    Returns 0.0 for a constant cell (orbit period 1 or cell visits same value).
    Maximum value is log₂(min(P, 64)) bits.
    """
    P = len(orbit)
    vals = [int(orbit[t][cell_idx]) for t in range(P)]
    counts = Counter(vals)
    return max(0.0, -sum((c / P) * math.log2(c / P) for c in counts.values()))


# ── Per-orbit functions ────────────────────────────────────────────────────────

def spatial_entropy_orbit(word: str, rule: str,
                           width: int = _DEFAULT_W) -> list[float]:
    """Spatial entropy H_s(t) at each orbit step (rounded to 6 dp)."""
    orbit = _get_orbit(word, rule, width)
    return [round(spatial_entropy(s), 6) for s in orbit]


def temporal_entropy_all(word: str, rule: str,
                          width: int = _DEFAULT_W) -> list[float]:
    """Temporal entropy H_c(i) for each cell (rounded to 6 dp)."""
    orbit = _get_orbit(word, rule, width)
    return [round(temporal_entropy_cell(orbit, i), 6) for i in range(width)]


def mean_spatial_entropy(word: str, rule: str,
                          width: int = _DEFAULT_W) -> float:
    """Mean spatial entropy over all orbit steps (rounded to 6 dp)."""
    hs = spatial_entropy_orbit(word, rule, width)
    return round(sum(hs) / len(hs), 6)


def mean_temporal_entropy(word: str, rule: str,
                           width: int = _DEFAULT_W) -> float:
    """Mean temporal entropy over all cells (rounded to 6 dp)."""
    hc = temporal_entropy_all(word, rule, width)
    return round(sum(hc) / len(hc), 6)


def max_spatial_entropy(word: str, rule: str,
                         width: int = _DEFAULT_W) -> float:
    """Maximum spatial entropy observed across orbit steps."""
    return max(spatial_entropy_orbit(word, rule, width))


def min_spatial_entropy(word: str, rule: str,
                         width: int = _DEFAULT_W) -> float:
    """Minimum spatial entropy observed across orbit steps."""
    return min(spatial_entropy_orbit(word, rule, width))


def max_temporal_entropy(word: str, rule: str,
                          width: int = _DEFAULT_W) -> float:
    """Maximum temporal entropy observed across cells."""
    return max(temporal_entropy_all(word, rule, width))


def min_temporal_entropy(word: str, rule: str,
                          width: int = _DEFAULT_W) -> float:
    """Minimum temporal entropy observed across cells."""
    return min(temporal_entropy_all(word, rule, width))


# ── Summary ───────────────────────────────────────────────────────────────────

def entropy_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full entropy summary for word × rule."""
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    hs    = [round(spatial_entropy(s), 6) for s in orbit]
    hc    = [round(temporal_entropy_cell(orbit, i), 6) for i in range(width)]
    max_th = math.log2(P) if P > 1 else 0.0
    hc_sym = all(abs(hc[i] - hc[width - 1 - i]) < 1e-9 for i in range(width // 2))
    return {
        'word':                word.upper(),
        'rule':                rule,
        'period':              P,
        'n_cells':             width,
        'max_possible_Hs':     round(math.log2(min(width, 64)), 6),
        'max_possible_Hc':     round(max_th, 6),
        # spatial
        'spatial_entropy':     hs,
        'mean_spatial_H':      round(sum(hs) / P, 6),
        'max_spatial_H':       max(hs),
        'min_spatial_H':       min(hs),
        # temporal
        'temporal_entropy':    hc,
        'mean_temporal_H':     round(sum(hc) / width, 6),
        'max_temporal_H':      max(hc),
        'min_temporal_H':      min(hc),
        # flags
        'zero_entropy':        all(h == 0.0 for h in hs),
        'constant_spatial':    len(set(hs)) == 1,
        'constant_temporal':   len(set(hc)) == 1,
        'symmetric_temporal':  hc_sym,
    }


def all_entropy(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """entropy_summary for all 4 rules."""
    return {rule: entropy_summary(word, rule, width) for rule in _RULES}


def build_entropy_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact entropy data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = entropy_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'spatial_entropy', 'mean_spatial_H',
                                   'max_spatial_H', 'min_spatial_H',
                                   'temporal_entropy', 'mean_temporal_H',
                                   'max_temporal_H', 'min_temporal_H',
                                   'zero_entropy', 'constant_spatial',
                                   'constant_temporal', 'symmetric_temporal')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'
_BAR_W = 30  # width in chars for entropy bar


def _entropy_bar(h: float, max_h: float = 4.0) -> str:
    frac = h / max_h if max_h > 0 else 0.0
    filled = round(frac * _BAR_W)
    return '█' * filled + '░' * (_BAR_W - filled)


def print_entropy(word: str = 'ТУМАН', rule: str = 'xor3',
                  color: bool = True) -> None:
    d   = entropy_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',  'or':   'OR |'}.get(rule, rule)
    ze  = ' ★zero_entropy'   if d['zero_entropy']   else ''
    sym = ' ★H_c_symmetric'  if d['symmetric_temporal'] else ''
    print(f'  {c}◈ Entropy  {word.upper()}  |  {lbl}  P={d["period"]}'
          f'  mean_H_s={d["mean_spatial_H"]:.3f}'
          f'  mean_H_c={d["mean_temporal_H"]:.3f}{ze}{sym}{r}')
    print('  ' + '─' * 68)
    max_h = d['max_possible_Hs']
    print(f'  {"step":>4}  {"H_s":>7}  bar ({max_h:.1f} bits max)')
    for t, hs in enumerate(d['spatial_entropy']):
        bar = _entropy_bar(hs, max_h)
        print(f'  t={t:>2}  {hs:>7.4f}  [{bar}]')
    print()
    print(f'  H_c per cell  (max_possible={d["max_possible_Hc"]:.3f} bits)')
    hc = d['temporal_entropy']
    max_hc = d['max_temporal_H'] if d['max_temporal_H'] > 0 else 1.0
    for i, h in enumerate(hc):
        bar = _entropy_bar(h, max_hc)
        print(f'  cell {i:>2}  {h:>7.4f}  [{bar}]')
    print()


def print_entropy_table(words: list[str] | None = None,
                         color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    head = '  '.join(
        (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5} mHs mHc ze' + R
        for rl in _RULES)
    print(f'  {"Слово":10s}  {head}')
    print('  ' + '─' * 76)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = _RCOL.get(rule, '') if color else ''
            d   = entropy_summary(word, rule)
            ze  = '★' if d['zero_entropy'] else ' '
            parts.append(f'{col}{d["mean_spatial_H"]:>4.2f}'
                         f' {d["mean_temporal_H"]:>4.2f} {ze}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

# ── Legacy compatibility (original solan_entropy API) ─────────────────────────
# These functions are kept for backward compatibility with solan_portrait and
# other modules that depend on the original solan_entropy interface.

def entropy(cells: list[int]) -> float:
    """Энтропия Шеннона (бит) распределения состояний клеток.

    H = −Σ p_i · log₂(p_i),  где p_i = доля клеток в состоянии i.
    Alias: equivalent to spatial_entropy() for list input.

    Минимум 0 (все клетки одинаковые),
    максимум log₂(64) ≈ 6 бит (равномерное распределение).
    """
    n = len(cells)
    if n == 0:
        return 0.0
    counts = Counter(cells)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def entropy_profile(cells: list[int], rule: str, steps: int) -> list[float]:
    """Профиль H[t] для t = 0 .. steps (включительно)."""
    from projects.hexglyph.solan_ca import step as ca_step
    cur = list(cells)
    profile = [entropy(cur)]
    for _ in range(steps):
        cur = ca_step(cur, rule)
        profile.append(entropy(cur))
    return profile


def entropy_profiles(cells: list[int], steps: int,
                     rules: list[str] | None = None) -> dict[str, list[float]]:
    """Профили для нескольких правил сразу."""
    if rules is None:
        rules = ['xor', 'xor3', 'and', 'or']
    return {r: entropy_profile(cells, r, steps) for r in rules}


_BLOCK = ' ▁▂▃▄▅▆▇█'


def sparkline(values: list[float], max_val: float) -> str:
    """Компактная визуализация числового ряда блочными символами."""
    if max_val <= 0:
        return ' ' * len(values)
    out = []
    for v in values:
        idx = int(8 * v / max_val)
        out.append(_BLOCK[max(0, min(8, idx))])
    return ''.join(out)


_RULE_META_L: dict[str, tuple[str, str, str]] = {
    'xor':  ('XOR ⊕', '\033[38;5;75m',  'линейная (Z₂)⁶, паттерн Серпинского'),
    'xor3': ('XOR3 ', '\033[38;5;117m', 'линейная (Z₂)⁶, сдвиговый режим'),
    'and':  ('AND &', '\033[38;5;196m', 'разрушительное — H убывает к 0'),
    'or':   ('OR | ', '\033[38;5;220m', 'накапливающее  — зависит от IC'),
}

_TREND_COLOR = {
    '→': '\033[33m',
    '↑': '\033[32m',
    '↓': '\033[31m',
}


def print_entropy_chart(cells: list[int], steps: int = 20,
                        ic_label: str = 'center',
                        rules: list[str] | None = None,
                        color: bool = True) -> None:
    """Спарклайн-таблица энтропии H(t) для заданных правил."""
    if rules is None:
        rules = ['xor', 'xor3', 'and', 'or']
    profiles = {r: entropy_profile(cells, r, steps) for r in rules}
    max_h    = max(h for prof in profiles.values() for h in prof)
    max_h    = max(max_h, 1e-9)
    bold  = '\033[1m' if color else ''
    reset = _RST      if color else ''
    dim   = '\033[2m' if color else ''
    width = len(cells)
    print(bold + f"  Энтропия Q6-CA"
          f"  ic={ic_label}  width={width}  steps={steps}" + reset)
    print(f"  Max H(t) = {max_h:.3f} бит"
          f"  (теор. max = {math.log2(64):.1f} бит для 64 состояний)")
    print()
    axis = f"t=0{'─' * (steps - 3)}t={steps}"
    print(f"  {'правило':7s}  {axis:{steps + 1}s}  "
          f"{'H₀':>5}  {'Hf':>5}  тренд  описание")
    print('  ' + '─' * max(60, steps + 30))
    for r in rules:
        label, col, desc = _RULE_META_L.get(r, (r, '', ''))
        prof  = profiles[r]
        spark = sparkline(prof, max_h)
        h0, hf = prof[0], prof[-1]
        if   hf > h0 + 0.02:  trend = '↑'
        elif hf < h0 - 0.02:  trend = '↓'
        else:                  trend = '→'
        rule_col  = (col                    if color else '')
        trend_col = (_TREND_COLOR[trend]    if color else '')
        print(f"  {rule_col}{label:7s}{reset}"
              f"  {spark}"
              f"  {rule_col}{h0:5.3f}{reset}"
              f"  {trend_col}{hf:5.3f}{reset}"
              f"  {trend_col}{trend:^5}{reset}"
              f"  {dim}{desc}{reset}")
    print()
    print(f"  {dim}Спарклайн: {_BLOCK[1]}=low → {_BLOCK[8]}=max ({max_h:.2f} бит){reset}")


def _cli() -> None:
    p = argparse.ArgumentParser(description='Shannon Entropy of Q6 CA Orbits')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args  = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(entropy_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.table:
        print_entropy_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_entropy(args.word, rule, color)
    else:
        print_entropy(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
