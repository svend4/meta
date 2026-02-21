"""
solan_bitflip.py — Bit-Flip Dynamics of Q6 CA Attractor Cells.

For each CA step x_t → x_{t+1} the 6-bit XOR mask

    flip[t] = x_t ⊕ x_{t+1 mod P}   ∈ {0, …, 63}

encodes exactly which bits changed.  Analysing these masks reveals the
"digital signature" of the attractor's update dynamics.

Per-cell statistics
───────────────────
  bit_flip_freqs   : f_b = fraction of steps where bit b flips  (b = 0…5)
  flip_entropy     : H of the distribution over flip masks  (bits)
  flip_cooccurrence: 6×6 matrix  M[b][b'] = P(b and b' flip at same step)
  dominant_mask    : most frequent flip mask (mode of flip distribution)

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, const 0) : flip mask always 0 → all freqs = 0, H = 0
  ГОРА  AND  (P=2, 47↔1)   : flip mask always 0b101110 = 46
                              freqs = [0, 1, 1, 1, 0, 1]  (bits 1,2,3,5)
                              H = 0  (perfectly deterministic flip pattern)
                              co-occurrence: identity block for {1,2,3,5}
  ТУМАН XOR3 (P=8)          : variable masks per step, non-trivial entropy
                              cell 0 masks: {0:4, 3:2, 24:2} → H = 1.5 bits
                              aggregate freqs ≈ uniform [0.44–0.50 per bit]
                              bit 3 most active (0.500), bit 5 least (0.438)

Interpretation
  f_b = 0   : bit b is FROZEN throughout the attractor (never changes)
  f_b = 1   : bit b ALWAYS flips at every step (strict alternation)
  f_b = 0.5 : bit b flips exactly half the time
  H = 0     : the same bits always flip in the same combination
  H > 0     : diverse flip patterns across the orbit

Functions
─────────
  flip_masks(series)                        → list[int]  (circular XOR masks)
  bit_flip_freqs(series)                    → list[float]  len=6
  flip_entropy(series)                      → float  (bits)
  flip_cooccurrence(series)                 → list[list[float]]  6×6
  cell_flip_stats(series)                   → dict
  all_cell_flip_stats(word, rule, width)    → list[dict]
  aggregate_flip_freqs(word, rule, width)   → list[float]  len=6
  flip_summary(word, rule, width)           → dict
  all_flips(word, width)                    → dict[str, dict]
  build_flip_data(words, width)             → dict
  print_bitflip(word, rule, color)          → None
  print_flip_stats(words, color)            → None

Запуск
──────
  python3 -m projects.hexglyph.solan_bitflip --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_bitflip --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_bitflip --stats --no-color
"""

from __future__ import annotations
import sys
import argparse
import math

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_NBITS:     int = 6      # Q6 = 6 bits


# ── Core computation ──────────────────────────────────────────────────────────

def flip_masks(series: list[int]) -> list[int]:
    """
    Circular bit-flip masks: flip[t] = series[t] XOR series[(t+1) mod P].

    Returns P masks in {0, …, 63}.  A zero mask indicates a plateau step
    (x_t == x_{t+1}).
    """
    P = len(series)
    if P == 0:
        return []
    return [series[t] ^ series[(t + 1) % P] for t in range(P)]


def bit_flip_freqs(series: list[int]) -> list[float]:
    """
    Per-bit flip frequency: f_b = fraction of steps where bit b flips.

    Returns list of 6 floats in [0, 1].
    """
    P = len(series)
    if P == 0:
        return [0.0] * _NBITS
    counts = [0] * _NBITS
    for mask in flip_masks(series):
        for b in range(_NBITS):
            if mask & (1 << b):
                counts[b] += 1
    return [c / P for c in counts]


def flip_entropy(series: list[int]) -> float:
    """
    Shannon entropy (bits) of the distribution over flip masks.

    H = −Σ_m p_m log₂ p_m   where p_m = fraction of steps with mask = m.
    H = 0 : always the same flip pattern (e.g. ГОРА AND)
    H > 0 : diverse flip patterns
    """
    P = len(series)
    if P == 0:
        return 0.0
    masks = flip_masks(series)
    counts: dict[int, int] = {}
    for m in masks:
        counts[m] = counts.get(m, 0) + 1
    return max(0.0, -sum((c / P) * math.log2(c / P)
                         for c in counts.values() if c > 0))


def flip_cooccurrence(series: list[int]) -> list[list[float]]:
    """
    6×6 co-occurrence matrix: M[b][b'] = P(bit b AND bit b' both flip at same step).

    Diagonal: M[b][b] = f_b (single-bit flip frequency).
    Off-diagonal: M[b][b'] = P(both b and b' flip simultaneously).
    """
    P = len(series)
    mat = [[0.0] * _NBITS for _ in range(_NBITS)]
    if P == 0:
        return mat
    masks = flip_masks(series)
    for mask in masks:
        for b in range(_NBITS):
            if mask & (1 << b):
                for b2 in range(_NBITS):
                    if mask & (1 << b2):
                        mat[b][b2] += 1
    return [[mat[b][b2] / P for b2 in range(_NBITS)] for b in range(_NBITS)]


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Per-cell stats ────────────────────────────────────────────────────────────

def cell_flip_stats(series: list[int]) -> dict:
    """Full bit-flip statistics for a single cell's temporal series."""
    P = len(series)
    if P == 0:
        return {
            'freqs': [0.0] * _NBITS, 'entropy': 0.0,
            'dominant_mask': 0, 'n_distinct_masks': 0,
            'cooccurrence': [[0.0] * _NBITS for _ in range(_NBITS)],
        }
    masks  = flip_masks(series)
    freqs  = bit_flip_freqs(series)
    ent    = flip_entropy(series)
    coocc  = flip_cooccurrence(series)
    counts: dict[int, int] = {}
    for m in masks:
        counts[m] = counts.get(m, 0) + 1
    dominant = max(counts, key=lambda m: counts[m])
    return {
        'freqs':            freqs,
        'entropy':          round(ent, 6),
        'dominant_mask':    dominant,
        'n_distinct_masks': len(counts),
        'cooccurrence':     coocc,
    }


def all_cell_flip_stats(word: str, rule: str,
                        width: int = _DEFAULT_W) -> list[dict]:
    """Per-cell flip stats (list of length = width)."""
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    out   = []
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        s      = cell_flip_stats(series)
        s['cell'] = i
        out.append(s)
    return out


def aggregate_flip_freqs(word: str, rule: str,
                         width: int = _DEFAULT_W) -> list[float]:
    """Mean per-bit flip frequency across all cells."""
    css  = all_cell_flip_stats(word, rule, width)
    agg  = [0.0] * _NBITS
    for cs in css:
        for b in range(_NBITS):
            agg[b] += cs['freqs'][b]
    return [v / width for v in agg]


# ── Summary ───────────────────────────────────────────────────────────────────

def flip_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Aggregate bit-flip statistics for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']
    css    = all_cell_flip_stats(word, rule, width)

    agg_freqs = aggregate_flip_freqs(word, rule, width)
    entropies = [cs['entropy'] for cs in css]
    n   = len(entropies)
    mu  = sum(entropies) / n if n else 0.0
    std = (sum((e - mu) ** 2 for e in entropies) / n) ** 0.5 if n else 0.0

    # Aggregate co-occurrence (mean over cells)
    agg_coocc = [[0.0] * _NBITS for _ in range(_NBITS)]
    for cs in css:
        for b in range(_NBITS):
            for b2 in range(_NBITS):
                agg_coocc[b][b2] += cs['cooccurrence'][b][b2]
    agg_coocc = [[v / width for v in row] for row in agg_coocc]

    most_active  = max(range(_NBITS), key=lambda b: agg_freqs[b])
    least_active = min(range(_NBITS), key=lambda b: agg_freqs[b])

    return {
        'word':          word.upper(),
        'rule':          rule,
        'period':        period,
        'cell_stats':    css,
        'agg_freqs':     [round(f, 6) for f in agg_freqs],
        'agg_coocc':     [[round(v, 6) for v in row] for row in agg_coocc],
        'entropy_mean':  round(mu, 6),
        'entropy_std':   round(std, 6),
        'entropy_max':   round(max(entropies), 6),
        'most_active_bit':   most_active,
        'least_active_bit':  least_active,
    }


def all_flips(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """flip_summary for all 4 rules."""
    return {rule: flip_summary(word, rule, width) for rule in _RULES}


def build_flip_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact flip data for a list of words."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = flip_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'agg_freqs', 'entropy_mean', 'entropy_std',
                 'entropy_max', 'most_active_bit', 'least_active_bit')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'


def _freq_bar(f: float, w: int = 8) -> str:
    filled = round(f * w)
    return '█' * filled + '░' * (w - filled)


def print_bitflip(word: str = 'ТУМАН', rule: str = 'xor3',
                  color: bool = True) -> None:
    d   = flip_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    print(f'  {c}◈ Bit-Flip  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'H_mean={d["entropy_mean"]:.3f}  '
          f'most_active=b{d["most_active_bit"]}  '
          f'least_active=b{d["least_active_bit"]}{r}')
    print('  ' + '─' * 62)

    # Aggregate bit flip frequencies
    print(f'  Aggregate flip freq per bit:')
    for b in range(_NBITS):
        f = d['agg_freqs'][b]
        bar = _freq_bar(f)
        print(f'    b{b}: {f:.3f}  {bar}')

    # Per-cell entropy and dominant mask
    print(f'\n  Per-cell entropy and dominant flip mask:')
    for cs in d['cell_stats']:
        freqs_str = '[' + ','.join(f'{f:.2f}' for f in cs['freqs']) + ']'
        print(f'    cell {cs["cell"]:>2}: H={cs["entropy"]:.3f}  '
              f'dom=0b{cs["dominant_mask"]:06b}={cs["dominant_mask"]:>2}  '
              f'n_masks={cs["n_distinct_masks"]}  freqs={freqs_str}')

    # Co-occurrence (aggregate)
    print(f'\n  Co-occurrence (aggregate, rows=b0..b5, cols=b0..b5):')
    for b in range(_NBITS):
        row = ' '.join(f'{v:.2f}' for v in d['agg_coocc'][b])
        print(f'    b{b}: {row}')
    print()


def print_flip_stats(words: list[str] | None = None,
                     color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_bitflip(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Bit-flip dynamics of Q6 CA attractor')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_flip_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_bitflip(args.word, rule, color)
    else:
        print_bitflip(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
