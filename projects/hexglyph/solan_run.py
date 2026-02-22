"""solan_run.py — Cell Temporal Run Analysis of Q6 CA Attractor Orbits.

For a period-P attractor each cell i traces a Q6 value sequence across P steps.
This module analyses the *shape* of those trajectories by counting direction
changes (turns), monotone-step types (inc / dec / const), and value range.

    analyze_cell(seq)           → per-cell turn/range statistics
    run_summary(word, rule)     → full per-cell + aggregate statistics
    all_run(word)               → summary for all 4 CA rules

Definitions
──────────────────────────────────────────────────────────────────────────────
  For consecutive steps t and t+1:
    • inc   (increasing) : seq[t+1] > seq[t]
    • dec   (decreasing) : seq[t+1] < seq[t]
    • const (unchanged)  : seq[t+1] = seq[t]

  A *turn* occurs when the non-zero direction changes sign:
    … inc … dec … → one turn   (local maximum)
    … dec … inc … → one turn   (local minimum)
    Consecutive equal values (const) do not count as a direction change.

  A cell is *quasi-frozen* when it reaches its attractor value early:
    n_turns = 0  (no direction reversal after the first inc/dec step)

  value_range = max(seq) − min(seq)

Key discoveries
──────────────────────────────────────────────────────────────────────────────
  РАБОТА XOR3  (P=8):
      cell 1: seq=[63,62,63,1,63,0,63,62]  — 6 turns, range=63
      Maximum turns in the lexicon under XOR3.  Range spans the full Q6 extent.

  МАТ XOR3  (P=8):
      cells 7,8 have 0 turns and n_const=6:
        cell 7: [63,23,23,23,23,23,23,23]
        cell 8: [48,23,23,23,23,23,23,23]
      Cells drop to value 23 after t=0 and stay frozen for 7 steps.
      Turn-count gradient across cells: 4,4,4,3,3,2,1,0,0,1,1,3,3,4,4,4
      (symmetric around the centre, decreasing toward cells 7-8).

  XOR rule  (all 49 words, P=2):
      Every non-zero IC has 1 turn (oscillates once: inc then dec or vice versa).
      Range = |seq[1] − seq[0]|; cells at 0 have range=0, turns=0.

  AND / OR  (mostly P=1 fixed points):
      P=1 → sequence length 1 → no steps → n_turns=n_inc=n_dec=n_const=0.

Запуск:
    python3 -m projects.hexglyph.solan_run --word РАБОТА --rule xor3
    python3 -m projects.hexglyph.solan_run --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_run --table --rule xor3
    python3 -m projects.hexglyph.solan_run --json --word РАБОТА
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES          = tuple(_ALL_RULES)
_DEFAULT_WIDTH = 16


# ── Per-cell analysis ─────────────────────────────────────────────────────────

def analyze_cell(seq: list[int]) -> dict[str, Any]:
    """Turn / monotone-step statistics for a single cell's time series.

    Parameters
    ──────────
    seq  : list[int]  — cell values at orbit steps t = 0 … P−1

    Returns dict with keys:
        n_turns     : int    — number of direction reversals (local extrema)
        n_inc       : int    — steps where value increases
        n_dec       : int    — steps where value decreases
        n_const     : int    — steps where value is unchanged
        value_range : int    — max(seq) − min(seq)
        min_val     : int    — minimum value in sequence
        max_val     : int    — maximum value in sequence
    """
    P = len(seq)
    n_inc = n_dec = n_const = 0
    for t in range(P - 1):
        d = seq[t + 1] - seq[t]
        if d > 0:
            n_inc += 1
        elif d < 0:
            n_dec += 1
        else:
            n_const += 1

    # Count turns: sign reversals in the sequence of non-zero deltas
    n_turns  = 0
    last_sign = 0
    for t in range(P - 1):
        d = seq[t + 1] - seq[t]
        if d > 0:
            if last_sign < 0:
                n_turns += 1
            last_sign = 1
        elif d < 0:
            if last_sign > 0:
                n_turns += 1
            last_sign = -1

    return {
        'n_turns':     n_turns,
        'n_inc':       n_inc,
        'n_dec':       n_dec,
        'n_const':     n_const,
        'value_range': max(seq) - min(seq),
        'min_val':     min(seq),
        'max_val':     max(seq),
    }


# ── Per-word summary ──────────────────────────────────────────────────────────

def run_summary(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Temporal run analysis for all cells of one word/rule attractor.

    Keys
    ────
    word, rule, period, n_cells

    # Per-cell lists (length = width)
    cell_n_turns    : list[int]
    cell_n_inc      : list[int]
    cell_n_dec      : list[int]
    cell_n_const    : list[int]
    cell_range      : list[int]
    cell_min_val    : list[int]
    cell_max_val    : list[int]

    # Aggregate
    max_turns       : int    — max turns across all cells
    max_turns_cell  : int    — cell index with max turns
    min_turns       : int    — min turns across all cells
    min_turns_cell  : int
    mean_turns      : float  — mean turns per cell
    total_inc       : int    — sum of n_inc across all cells
    total_dec       : int    — sum of n_dec across all cells
    total_const     : int    — sum of n_const across all cells
    max_range       : int    — largest value_range across all cells
    max_range_cell  : int
    min_range       : int
    min_range_cell  : int
    mean_range      : float

    # Quasi-frozen cells (n_turns=0)
    quasi_frozen_cells : list[int]  — cell indices with n_turns=0
    n_quasi_frozen     : int
    """
    from projects.hexglyph.solan_transfer import get_orbit

    orbit = get_orbit(word, rule, width)
    P     = len(orbit)
    N     = width

    analyses = []
    for i in range(N):
        seq = [int(orbit[t][i]) for t in range(P)]
        analyses.append(analyze_cell(seq))

    cell_n_turns = [a['n_turns']     for a in analyses]
    cell_n_inc   = [a['n_inc']       for a in analyses]
    cell_n_dec   = [a['n_dec']       for a in analyses]
    cell_n_const = [a['n_const']     for a in analyses]
    cell_range   = [a['value_range'] for a in analyses]
    cell_min_val = [a['min_val']     for a in analyses]
    cell_max_val = [a['max_val']     for a in analyses]

    max_t      = max(cell_n_turns)
    min_t      = min(cell_n_turns)
    max_t_cell = cell_n_turns.index(max_t)
    min_t_cell = cell_n_turns.index(min_t)

    max_r      = max(cell_range)
    min_r      = min(cell_range)
    max_r_cell = cell_range.index(max_r)
    min_r_cell = cell_range.index(min_r)

    qf = [i for i, t in enumerate(cell_n_turns) if t == 0]

    return {
        'word':    word,
        'rule':    rule,
        'period':  P,
        'n_cells': N,

        'cell_n_turns':  cell_n_turns,
        'cell_n_inc':    cell_n_inc,
        'cell_n_dec':    cell_n_dec,
        'cell_n_const':  cell_n_const,
        'cell_range':    cell_range,
        'cell_min_val':  cell_min_val,
        'cell_max_val':  cell_max_val,

        'max_turns':      max_t,
        'max_turns_cell': max_t_cell,
        'min_turns':      min_t,
        'min_turns_cell': min_t_cell,
        'mean_turns':     round(sum(cell_n_turns) / N, 4),
        'total_inc':      sum(cell_n_inc),
        'total_dec':      sum(cell_n_dec),
        'total_const':    sum(cell_n_const),
        'max_range':      max_r,
        'max_range_cell': max_r_cell,
        'min_range':      min_r,
        'min_range_cell': min_r_cell,
        'mean_range':     round(sum(cell_range) / N, 4),

        'quasi_frozen_cells': qf,
        'n_quasi_frozen':     len(qf),
    }


def all_run(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict[str, Any]]:
    """run_summary for all 4 CA rules."""
    return {r: run_summary(word, r, width) for r in RULES}


def build_run_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Full temporal run analysis for the lexicon."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)
    return {
        'words': list(words),
        'data':  {w: {r: run_summary(w, r, width) for r in RULES}
                  for w in words},
    }


def run_dict(s: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of run_summary."""
    return dict(s)


# ── Terminal output ───────────────────────────────────────────────────────────

_BAR_CHARS = ' ▁▂▃▄▅▆▇█'


def _bar(v: float, max_v: float, width: int = 20) -> str:
    if max_v == 0:
        return ' ' * width
    frac = min(v / max_v, 1.0)
    filled = round(frac * width)
    return '█' * filled + ' ' * (width - filled)


def print_run(
    word:  str,
    rule:  str,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print temporal run analysis for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    s = run_summary(word, rule, width)
    P = s['period']
    N = s['n_cells']

    print(bold + f"  ◈ Cell Temporal Runs  {word.upper()}  "
          + col + lbl + reset + bold + f"  (P={P})" + reset)
    print()
    print(f"  Total steps: {N}×{P-1}={N*(P-1)}  "
          f"inc={s['total_inc']}  dec={s['total_dec']}  const={s['total_const']}")
    print(f"  Mean turns per cell: {s['mean_turns']:.2f}   "
          f"Mean range: {s['mean_range']:.1f}")
    print()

    max_t = max(s['cell_n_turns']) if s['cell_n_turns'] else 0
    max_r = max(s['cell_range'])   if s['cell_range']   else 1
    hi_t  = '\033[38;5;214m' if color else ''
    hi_r  = '\033[38;5;117m' if color else ''
    qf_c  = '\033[38;5;240m' if color else ''

    print(f"  {'cell':>4}  {'turns':>5}  {'range':>5}  "
          f"{'inc':>3}  {'dec':>3}  {'const':>5}  turns-bar")
    print('  ' + '─' * 58)

    for i in range(N):
        nt = s['cell_n_turns'][i]
        nr = s['cell_range'][i]
        ni = s['cell_n_inc'][i]
        nd = s['cell_n_dec'][i]
        nc = s['cell_n_const'][i]
        bar = _bar(nt, max_t, 16)
        if nt == max_t and max_t > 0:
            cc = hi_t
        elif nt == 0:
            cc = qf_c
        else:
            cc = dim
        print(f"  {i:4d}  {cc}{nt:5d}{reset}  "
              f"{hi_r if nr == max_r else dim}{nr:5d}{reset}  "
              f"{ni:3d}  {nd:3d}  {nc:5d}  "
              f"{cc}{bar}{reset}")

    print()
    print(f"  Max turns : {s['max_turns']} at cell {s['max_turns_cell']}")
    print(f"  Max range : {s['max_range']} at cell {s['max_range_cell']}")
    if s['quasi_frozen_cells']:
        print(f"  Quasi-frozen (0 turns): {s['quasi_frozen_cells']}  "
              f"({s['n_quasi_frozen']}/{N} cells)")
    print()


def print_run_table(
    words: list[str] | None = None,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Summary table: temporal run stats for all lexicon words."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Cell Temporal Run Summary ({lbl}, n={len(words)})" + reset)
    print()
    print(f"  {'Слово':12s}  {'P':>3}  {'max_t':>5}  {'cell':>4}  "
          f"{'mean_t':>6}  {'max_r':>5}  {'qf':>3}  {'total_inc':>9}  {'total_const':>11}")
    print('  ' + '─' * 80)

    for word in words:
        s = run_summary(word, rule, width)
        print(f"  {word.upper():12s}  {s['period']:>3}  "
              f"{s['max_turns']:>5}  {s['max_turns_cell']:>4}  "
              f"{s['mean_turns']:>6.2f}  {s['max_range']:>5}  "
              f"{s['n_quasi_frozen']:>3}  "
              f"{s['total_inc']:>9}  {s['total_const']:>11}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cell temporal run analysis for Q6 CA attractors')
    parser.add_argument('--word',   metavar='WORD', default='РАБОТА')
    parser.add_argument('--rule',   choices=list(RULES), default='xor3')
    parser.add_argument('--table',  action='store_true')
    parser.add_argument('--json',   action='store_true')
    parser.add_argument('--width',  type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = run_summary(args.word.upper(), args.rule, args.width)
        print(json.dumps(run_dict(s), ensure_ascii=False, indent=2))
    elif args.table:
        print_run_table(rule=args.rule, width=args.width, color=_color)
    else:
        print_run(args.word.upper(), args.rule, args.width, _color)
