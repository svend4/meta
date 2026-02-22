"""
solan_balance.py — Bit Balance × Flip Frequency Analysis of Q6 CA Attractors.

For a cell with temporal series [v_0, …, v_{P−1}]:

    balance_b  = (# steps where bit b = 1) / P   ∈ [0, 1]
    flip_b     = (# steps where bit b changes)/ P ∈ [0, 1]

The 2D (balance, flip) pair fully characterises each bit's temporal behaviour.
Plotting all N×6 points on the *balance plane* reveals attractor structure.

Classification of (balance, flip) pairs
────────────────────────────────────────
  FROZEN_OFF  : balance ≈ 0,   flip ≈ 0   — bit permanently 0
  FROZEN_ON   : balance ≈ 1,   flip ≈ 0   — bit permanently 1
  STRICT_ALT  : balance ≈ 0.5, flip ≈ 1   — strict 0↔1 alternation
  OSCILLATING : balance ≈ 0.5, flip < 1   — near-balanced oscillation
  DC_BIAS     : balance ≠ 0.5, flip > 0   — biased oscillation

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      All bits: FROZEN_OFF.  Balance plane: 1 point at (0, 0).

  ГОРА AND  (P=2, 47↔1)
      b0 : balance=1.0  flip=0.0  FROZEN_ON
      b4 : balance=0.0  flip=0.0  FROZEN_OFF
      b1,2,3,5: balance=0.5  flip=1.0  STRICT_ALT
      Balance plane: 3 tight clusters at (1,0), (0,0), (0.5,1).

  ГОРА XOR3  (P=2, 4 spatial clusters)
      b0 always FROZEN_ON (balance=1.0) for all cells.
      b4 always STRICT_ALT for all cells.
      b1,2,3,5 vary by cluster: FROZEN_ON, FROZEN_OFF, or STRICT_ALT.
      → Rich per-cell structure; 4 distinct balance profiles.

  ТУМАН XOR3  (P=8)
      Aggregate ≈ uniform (balance≈0.5, flip≈0.47–0.50) → OSCILLATING.
      BUT individual cells have hidden frozen bits:
        cell 0 : b2=FROZEN_OFF(0.0), b5=FROZEN_ON(1.0)
        cell 1 : b4=FROZEN_ON(1.0)
        cell 8 : b2=FROZEN_ON(1.0),  b5=FROZEN_ON(1.0)
        cell 9 : b4=FROZEN_OFF(0.0)
      → Apparent "chaos" masks individual frozen channels.

Functions
─────────
  bit_balance(series)                         → list[float]  len=6
  bit_flip_freq(series)                       → list[float]  len=6
  classify_bit(balance, flip, eps)            → str
  bit_profile(series)                         → list[dict]   len=6
  cell_balance_stats(series)                  → dict
  all_cell_balance_stats(word, rule, width)   → list[dict]
  aggregate_balance(word, rule, width)        → dict
  balance_plane_points(word, rule, width)     → list[dict]
  count_classes(word, rule, width)            → dict[str, int]
  balance_summary(word, rule, width)          → dict
  all_balance(word, width)                    → dict[str, dict]
  build_balance_data(words, width)            → dict
  print_balance(word, rule, color)            → None
  print_balance_stats(words, color)           → None

Запуск
──────
  python3 -m projects.hexglyph.solan_balance --word ГОРА --rule and
  python3 -m projects.hexglyph.solan_balance --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_balance --stats --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_NBITS:     int = 6

_CLASSES = ('FROZEN_OFF', 'FROZEN_ON', 'STRICT_ALT', 'OSCILLATING', 'DC_BIAS')
_EPS     = 0.02     # tolerance for classification


# ── Core primitives ────────────────────────────────────────────────────────────

def bit_balance(series: list[int]) -> list[float]:
    """
    Per-bit 1-fraction: b_b = (# steps where bit b = 1) / P.

    Returns list of 6 floats in [0, 1].
    b_b = 0 : bit always 0 throughout the series
    b_b = 1 : bit always 1 throughout the series
    b_b = 0.5 : bit is 1 exactly half the time
    """
    P = len(series)
    if P == 0:
        return [0.0] * _NBITS
    return [sum(1 for v in series if v & (1 << b)) / P for b in range(_NBITS)]


def bit_flip_freq(series: list[int]) -> list[float]:
    """
    Per-bit flip frequency (circular): f_b = (# steps where bit b changes) / P.

    Recomputed locally to keep this module self-contained.
    """
    P = len(series)
    if P == 0:
        return [0.0] * _NBITS
    counts = [0] * _NBITS
    for t in range(P):
        mask = series[t] ^ series[(t + 1) % P]
        for b in range(_NBITS):
            if mask & (1 << b):
                counts[b] += 1
    return [c / P for c in counts]


def classify_bit(balance: float, flip: float,
                 eps: float = _EPS) -> str:
    """
    Classify a (balance, flip) pair into one of 5 categories:

    FROZEN_OFF  : balance ≈ 0, flip ≈ 0
    FROZEN_ON   : balance ≈ 1, flip ≈ 0
    STRICT_ALT  : balance ≈ 0.5, flip ≈ 1
    OSCILLATING : balance ≈ 0.5, flip ∈ (eps, 1-eps)
    DC_BIAS     : balance ≠ 0.5, flip > eps (biased oscillation)
    """
    if flip < eps:
        if balance < eps:
            return 'FROZEN_OFF'
        if balance > 1.0 - eps:
            return 'FROZEN_ON'
        return 'DC_BIAS'      # unusual: not flipping but intermediate balance
    if abs(balance - 0.5) < eps and flip > 1.0 - eps:
        return 'STRICT_ALT'
    if abs(balance - 0.5) < eps:
        return 'OSCILLATING'
    return 'DC_BIAS'


# ── Per-cell analysis ──────────────────────────────────────────────────────────

def bit_profile(series: list[int]) -> list[dict]:
    """
    Full per-bit profile for a single cell's series.

    Returns list of 6 dicts, one per bit b:
      {'bit': b, 'balance': float, 'flip': float, 'class': str}
    """
    bal  = bit_balance(series)
    flip = bit_flip_freq(series)
    return [
        {'bit': b, 'balance': round(bal[b], 6),
         'flip': round(flip[b], 6),
         'class': classify_bit(bal[b], flip[b])}
        for b in range(_NBITS)
    ]


def cell_balance_stats(series: list[int]) -> dict:
    """Full balance statistics for a single cell's temporal series."""
    profile = bit_profile(series)
    classes = [p['class'] for p in profile]
    return {
        'profile':      profile,
        'n_frozen_off': classes.count('FROZEN_OFF'),
        'n_frozen_on':  classes.count('FROZEN_ON'),
        'n_frozen':     classes.count('FROZEN_OFF') + classes.count('FROZEN_ON'),
        'n_strict_alt': classes.count('STRICT_ALT'),
        'n_oscillating': classes.count('OSCILLATING'),
        'n_dc_bias':    classes.count('DC_BIAS'),
    }


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Aggregate analysis ─────────────────────────────────────────────────────────

def all_cell_balance_stats(word: str, rule: str,
                           width: int = _DEFAULT_W) -> list[dict]:
    """Per-cell balance stats (list of length = width)."""
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    out   = []
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        s      = cell_balance_stats(series)
        s['cell'] = i
        out.append(s)
    return out


def aggregate_balance(word: str, rule: str,
                      width: int = _DEFAULT_W) -> dict:
    """
    Aggregate (mean over cells) balance and flip frequency per bit.

    Returns dict with keys:
      'balance'  : list[float]  len=6
      'flip'     : list[float]  len=6
      'class'    : list[str]    len=6   (classification of aggregate pair)
    """
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    agg_bal  = [0.0] * _NBITS
    agg_flip = [0.0] * _NBITS
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        for b, v in enumerate(bit_balance(series)):   agg_bal[b]  += v
        for b, v in enumerate(bit_flip_freq(series)): agg_flip[b] += v
    bal  = [v / width for v in agg_bal]
    flip = [v / width for v in agg_flip]
    return {
        'balance': [round(v, 6) for v in bal],
        'flip':    [round(v, 6) for v in flip],
        'class':   [classify_bit(bal[b], flip[b]) for b in range(_NBITS)],
    }


def balance_plane_points(word: str, rule: str,
                         width: int = _DEFAULT_W) -> list[dict]:
    """
    All N×6 (balance, flip) points for the balance plane scatter plot.

    Returns list of dicts: {'cell': i, 'bit': b, 'balance': f, 'flip': f, 'class': s}.
    """
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    pts   = []
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        for p in bit_profile(series):
            pts.append({**p, 'cell': i})
    return pts


def count_classes(word: str, rule: str,
                  width: int = _DEFAULT_W) -> dict[str, int]:
    """Count of each classification across all N×6 (cell, bit) pairs."""
    pts    = balance_plane_points(word, rule, width)
    result = {c: 0 for c in _CLASSES}
    for p in pts:
        result[p['class']] = result.get(p['class'], 0) + 1
    return result


# ── Summary ───────────────────────────────────────────────────────────────────

def balance_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Comprehensive balance statistics for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']
    agg    = aggregate_balance(word, rule, width)
    css    = all_cell_balance_stats(word, rule, width)
    pts    = balance_plane_points(word, rule, width)
    cls    = count_classes(word, rule, width)

    # Per-cell frozen bit counts
    total_frozen = sum(cs['n_frozen'] for cs in css)
    max_frozen   = max(cs['n_frozen'] for cs in css)

    # Most active bit (highest flip aggregate)
    flips = agg['flip']
    most_active_bit  = max(range(_NBITS), key=lambda b: flips[b])
    least_active_bit = min(range(_NBITS), key=lambda b: flips[b])

    return {
        'word':            word.upper(),
        'rule':            rule,
        'period':          period,
        'agg_balance':     agg['balance'],
        'agg_flip':        agg['flip'],
        'agg_class':       agg['class'],
        'class_counts':    cls,
        'total_frozen_bits': total_frozen,
        'max_frozen_per_cell': max_frozen,
        'most_active_bit': most_active_bit,
        'least_active_bit': least_active_bit,
        'cell_stats':      css,
        'n_points':        len(pts),
    }


def all_balance(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """balance_summary for all 4 rules."""
    return {rule: balance_summary(word, rule, width) for rule in _RULES}


def build_balance_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact balance data for a list of words (no cell_stats, no points)."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = balance_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'agg_balance', 'agg_flip', 'agg_class',
                 'class_counts', 'total_frozen_bits', 'max_frozen_per_cell',
                 'most_active_bit', 'least_active_bit')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'

_CLASS_SYM = {
    'FROZEN_OFF': '▪',    # dark
    'FROZEN_ON':  '▪',    # bright
    'STRICT_ALT': '⇅',
    'OSCILLATING': '〜',
    'DC_BIAS':    '⤢',
}

_CLASS_COL = {
    'FROZEN_OFF':  '\033[90m',
    'FROZEN_ON':   '\033[97m',
    'STRICT_ALT':  '\033[93m',
    'OSCILLATING': '\033[92m',
    'DC_BIAS':     '\033[35m',
}


def print_balance(word: str = 'ГОРА', rule: str = 'and',
                  color: bool = True) -> None:
    d   = balance_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    print(f'  {c}◈ Balance  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'frozen_bits={d["total_frozen_bits"]}  '
          f'most_active=b{d["most_active_bit"]}{r}')
    print('  ' + '─' * 62)

    # Aggregate per-bit summary
    print(f'  Aggregate (mean over cells):')
    print(f'    {"bit":>3}  {"balance":>7}  {"flip":>7}  class')
    for b in range(_NBITS):
        bal  = d['agg_balance'][b]
        flp  = d['agg_flip'][b]
        cls  = d['agg_class'][b]
        cc   = _CLASS_COL.get(cls, '') if color else ''
        sym  = _CLASS_SYM.get(cls, '?')
        print(f'    b{b}   {bal:7.3f}  {flp:7.3f}  {cc}{sym} {cls}{r}')

    # Class counts
    print(f'\n  Class counts (all {d["n_points"]} cell×bit pairs):')
    for cls in _CLASSES:
        cnt = d['class_counts'].get(cls, 0)
        if cnt:
            cc = _CLASS_COL.get(cls, '') if color else ''
            print(f'    {cc}{_CLASS_SYM.get(cls,"?")}{r} {cls}: {cnt}')

    # Per-cell frozen bits
    if d['total_frozen_bits'] > 0:
        print(f'\n  Per-cell frozen bits:')
        for cs in d['cell_stats']:
            frozen = [(p['bit'], p['class'], round(p['balance'],1))
                      for p in cs['profile']
                      if p['class'] in ('FROZEN_OFF', 'FROZEN_ON')]
            if frozen:
                print(f'    cell {cs["cell"]:>2}: {frozen}')
    print()


def print_balance_stats(words: list[str] | None = None,
                        color: bool = True) -> None:
    WORDS = words or ['ТУМАН', 'ГОРА']
    for word in WORDS:
        for rule in _RULES:
            print_balance(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Bit balance × flip frequency analysis')
    p.add_argument('--word',      default='ГОРА')
    p.add_argument('--rule',      default='and', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(balance_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.stats:
        print_balance_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_balance(args.word, rule, color)
    else:
        print_balance(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
