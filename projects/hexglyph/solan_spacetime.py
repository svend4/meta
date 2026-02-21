"""solan_spacetime.py — Space-time Evolution Diagram for Q6 CA.

Computes the full space-time grid (time × cells) from the initial condition
through the transient and into the periodic attractor.

Layout of the grid:
    row  t = 0        : initial condition (IC)
    rows t = 1 … T-1  : transient
    rows t = T … T+P-1: first attractor period
    rows t = T+P … end: repeated copies (extra_periods)

Each cell value is a Q6 integer in [0, 63].

For visualisation the value is mapped to a hue via HSV(v/63, 0.85, 0.80),
giving a rainbow-like colour scale.  The attractor boundary (row T) is marked
with a separator in the terminal display.

Statistics extracted from the grid:
    spatial_entropy(t)  : Shannon entropy of the width cell values at step t
    temporal_entropy(i) : Shannon entropy of cell i values over all steps
    mean_activity       : mean absolute step-to-step change across the grid
    transient_activity  : mean |s(t) - s(t-1)| during transient
    attractor_activity  : mean |s(t) - s(t-1)| on the periodic orbit

Функции:
    spacetime(word, rule, width, extra_periods)       → dict
    st_spatial_entropy(grid)                          → list[float]
    st_temporal_entropy(grid, width)                  → list[float]
    st_activity(grid)                                 → list[float]
    st_dict(word, rule, width, extra_periods)         → dict
    all_st(word, width, extra_periods)                → dict[str, dict]
    build_st_data(words, width)                       → dict
    print_spacetime(word, rule, width, extra_periods, color) → None
    print_st_stats(words, width, color)               → None

Запуск:
    python3 -m projects.hexglyph.solan_spacetime --word ТУМАН --rule xor3
    python3 -m projects.hexglyph.solan_spacetime --word ГОРА --all-rules
    python3 -m projects.hexglyph.solan_spacetime --stats --no-color
"""
from __future__ import annotations

import argparse
import colorsys
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    step, find_orbit,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES      = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH  = 16
_DEFAULT_EXTRA  = 2          # extra attractor periods to append
_DEFAULT_WORDS  = list(LEXICON)
_Q6_MAX         = 63


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _value_to_rgb(v: int) -> tuple[int, int, int]:
    """Map Q6 value 0–63 to an (R, G, B) colour via HSV hue cycling."""
    h = v / _Q6_MAX          # hue 0.0 … 1.0 (0 and 63 map to same hue — OK)
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.78)
    return int(r * 255), int(g * 255), int(b * 255)


def value_to_rgb(v: int) -> tuple[int, int, int]:
    """Public wrapper for _value_to_rgb (used by viewer export)."""
    return _value_to_rgb(v)


# ── Core space-time computation ────────────────────────────────────────────────

def spacetime(
    word:          str,
    rule:          str,
    width:         int = _DEFAULT_WIDTH,
    extra_periods: int = _DEFAULT_EXTRA,
) -> dict:
    """Full space-time evolution grid from IC to attractor.

    Returns dict:
        word          : str
        rule          : str
        width         : int
        transient     : int
        period        : int
        extra_periods : int
        n_steps       : int   = transient + period * (1 + extra_periods)
        grid          : list[list[int]]   (n_steps × width)
    """
    cells     = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells[:], rule)
    period    = max(period, 1)
    n_steps   = transient + period * (1 + extra_periods)

    grid: list[list[int]] = []
    c = cells[:]
    for _ in range(n_steps):
        grid.append(c[:])
        c = step(c, rule)

    return {
        'word':          word.upper(),
        'rule':          rule,
        'width':         width,
        'transient':     transient,
        'period':        period,
        'extra_periods': extra_periods,
        'n_steps':       n_steps,
        'grid':          grid,
    }


# ── Statistics on the grid ────────────────────────────────────────────────────

def _shannon(values: list[int]) -> float:
    """Shannon entropy in bits of a list of integer values."""
    if not values:
        return 0.0
    total = len(values)
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(0.0, -sum((c / total) * math.log2(c / total)
                        for c in counts.values() if c > 0))


def st_spatial_entropy(grid: list[list[int]]) -> list[float]:
    """Shannon entropy across cells at each time step.

    Returns list of length n_steps.
    """
    return [round(_shannon(row), 8) for row in grid]


def st_temporal_entropy(
    grid:  list[list[int]],
    width: int = _DEFAULT_WIDTH,
) -> list[float]:
    """Shannon entropy of each cell's time series over all steps.

    Returns list of length width.
    """
    T = len(grid)
    return [round(_shannon([grid[t][i] for t in range(T)]), 8)
            for i in range(width)]


def st_activity(grid: list[list[int]]) -> list[float]:
    """Mean absolute step-to-step change at each time t → t+1.

    Returns list of length n_steps − 1.
    """
    result = []
    for t in range(len(grid) - 1):
        a = grid[t]
        b = grid[t + 1]
        result.append(round(sum(abs(b[i] - a[i]) for i in range(len(a))) / len(a), 4))
    return result


# ── Full analysis dict ────────────────────────────────────────────────────────

def st_dict(
    word:          str,
    rule:          str,
    width:         int = _DEFAULT_WIDTH,
    extra_periods: int = _DEFAULT_EXTRA,
) -> dict:
    """JSON-serialisable space-time analysis for one word + rule.

    Returns dict with all fields from spacetime() plus:
        spatial_entropy    : list[float]   (one per time step)
        temporal_entropy   : list[float]   (one per cell)
        activity           : list[float]   (one per step transition)
        mean_spatial_h     : float
        mean_temporal_h    : float
        transient_activity : float   (mean activity during transient)
        attractor_activity : float   (mean activity on one cycle)
        ic_entropy         : float   (spatial entropy at t=0)
        attractor_entropy  : float   (mean spatial entropy on attractor)
    """
    st   = spacetime(word, rule, width, extra_periods)
    grid = st['grid']
    T    = st['transient']
    P    = st['period']

    sp_h = st_spatial_entropy(grid)
    te_h = st_temporal_entropy(grid, width)
    act  = st_activity(grid)

    mean_sp_h = round(sum(sp_h) / len(sp_h), 8) if sp_h else 0.0
    mean_te_h = round(sum(te_h) / len(te_h), 8) if te_h else 0.0

    trans_act = round(sum(act[:max(T, 0)]) / max(T, 1), 4) if T > 0 and act else 0.0
    attr_act  = round(sum(act[T:T + P - 1]) / max(P - 1, 1), 4) if P > 1 else 0.0

    ic_h   = sp_h[0] if sp_h else 0.0
    # mean over one attractor period
    if T < len(sp_h):
        attr_h = round(sum(sp_h[T:T + P]) / P, 8)
    else:
        attr_h = sp_h[-1] if sp_h else 0.0

    return {
        **st,
        'spatial_entropy':    sp_h,
        'temporal_entropy':   te_h,
        'activity':           act,
        'mean_spatial_h':     mean_sp_h,
        'mean_temporal_h':    mean_te_h,
        'transient_activity': trans_act,
        'attractor_activity': attr_act,
        'ic_entropy':         ic_h,
        'attractor_entropy':  attr_h,
    }


def all_st(
    word:          str,
    width:         int = _DEFAULT_WIDTH,
    extra_periods: int = _DEFAULT_EXTRA,
) -> dict[str, dict]:
    """st_dict for all 4 rules."""
    return {r: st_dict(word, r, width, extra_periods) for r in _ALL_RULES}


def build_st_data(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
) -> dict:
    """Space-time summary for the full lexicon × 4 rules.

    Returns dict:
        words    : list[str]
        per_rule : {rule: {word: {transient, period, ic_entropy,
                                   attractor_entropy, mean_spatial_h}}}
        ranking  : {rule: [(word, ic_entropy), …]}  descending
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = st_dict(word, rule, width, extra_periods=1)
            per_rule[rule][word] = {
                'transient':       d['transient'],
                'period':          d['period'],
                'ic_entropy':      d['ic_entropy'],
                'attractor_entropy': d['attractor_entropy'],
                'mean_spatial_h':  d['mean_spatial_h'],
            }
    ranking: dict[str, list] = {
        r: sorted(
            ((w, v['ic_entropy']) for w, v in per_rule[r].items()),
            key=lambda x: -x[1],
        )
        for r in _ALL_RULES
    }
    return {'words': words, 'per_rule': per_rule, 'ranking': ranking}


# ── ANSI colour display ────────────────────────────────────────────────────────

_BG24  = '\033[48;2;{};{};{}m'   # 24-bit background
_FG24  = '\033[38;2;{};{};{}m'   # 24-bit foreground
_BLOCK = '█'
_DARK  = (20, 20, 24)


def _bg(r: int, g: int, b: int) -> str:
    return f'\033[48;2;{r};{g};{b}m'


def _fg(r: int, g: int, b: int) -> str:
    return f'\033[38;2;{r};{g};{b}m'


def print_spacetime(
    word:          str,
    rule:          str  = 'xor3',
    width:         int  = _DEFAULT_WIDTH,
    extra_periods: int  = 1,
    color:         bool = True,
) -> None:
    """Print the space-time diagram as a colour terminal heatmap."""
    d    = st_dict(word, rule, width, extra_periods)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    T    = d['transient']
    P    = d['period']

    print(f"{bold}  ◈ Пространство-время Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  T={T}  P={P}")
    print(f"  {'─' * (width + 12)}")

    grid = d['grid']
    for t, row in enumerate(grid):
        # separator at start of first attractor period
        if t == T and T > 0:
            print(f"  {' ' * 4}{'╌' * width}  ← аттрактор")
        label = f"{t:3d} "
        if color:
            cells_str = ''
            for v in row:
                r_, g_, b_ = _value_to_rgb(v)
                cells_str += _bg(r_, g_, b_) + _fg(0, 0, 0) + _BLOCK + rst
        else:
            # ASCII: map 0-63 to shade chars
            _shade = ' ░▒▓█'
            cells_str = ''.join(
                _shade[min(int(v / _Q6_MAX * (len(_shade) - 1) + 0.5),
                           len(_shade) - 1)]
                for v in row
            )
        h_val = d['spatial_entropy'][t]
        print(f"  {label}{cells_str}  H={h_val:.3f}")

    print(f"  {'─' * (width + 12)}")
    print(f"  IC-энтропия={d['ic_entropy']:.3f}  "
          f"аттр.-энтропия={d['attractor_entropy']:.3f}  "
          f"ср.акт.={d['mean_spatial_h']:.3f}")
    print()


def print_st_stats(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
    color: bool             = True,
) -> None:
    """Summary table: IC entropy + attractor period for each word × rule."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ IC-энтропия Q6 (bits){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = st_dict(word, rule, width, extra_periods=0)
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{d['ic_entropy']:>8.3f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Space-time Diagram Q6 CA')
    parser.add_argument('--word',        default='ТУМАН', help='Русское слово')
    parser.add_argument('--rule',        default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules',   action='store_true')
    parser.add_argument('--stats',       action='store_true')
    parser.add_argument('--extra',       type=int, default=1)
    parser.add_argument('--width',       type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color',    action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_st_stats(color=color, width=args.width)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_spacetime(args.word, rule, args.width, args.extra, color)
    else:
        print_spacetime(args.word, args.rule, args.width, args.extra, color)


if __name__ == '__main__':
    _main()
