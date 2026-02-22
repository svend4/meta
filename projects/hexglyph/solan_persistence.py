"""solan_persistence.py — Run-Length & Persistence Analysis of Q6 Attractor.

On the periodic binary attractor (cells binarised via threshold=32), each
cell i has a temporal sequence b_i(0)…b_i(P−1).  A *run* is a maximal block
of consecutive identical binary symbols (treated as circular, i.e. the last
symbol wraps to the first).

Key quantities per cell i
──────────────────────────
  persistence(i)  : P(b_i(t+1) = b_i(t))  =  1 − (transitions / P)  ∈ [0, 1]
                    1.0 = constant cell (never changes)
                    0.0 = alternating cell (changes every step)
                    Equals the lag-1 self-transition probability.

  mean_run(i)     : mean run-length = P / (number of runs)
                    For persistence p: mean_run = 1 / (1 − p) (theoretically)

  max_run(i)      : longest uninterrupted same-symbol block (circular)

  cv_run(i)       : coefficient of variation = std(runs) / mean(runs)
                    0 = perfectly regular runs (same length each time)
                    ≫1 = bursty (wildly varying run lengths)

  run_dist        : pooled histogram of all run lengths across all cells
                    (how often each run length appears in the attractor)

Interpretation
──────────────
  High persistence, low CV  →  regular slow oscillation (clock-like)
  Low persistence, low CV   →  regular fast switching (alternating clock)
  High persistence, high CV →  bursty / intermittent switching
  Low persistence, high CV  →  irregular fast oscillation

Expected results
────────────────
  XOR  ТУМАН (P=1, all-zero)  : b=[0], persistence=1, mean_run=1 (trivial)
  XOR3 ТУМАН (P=8)            : cell-varying persistence ∈ (0,1); mean run 1–4
  ГОРА XOR3  (P=2, mixed)     : constant cells persist=1, alternating persist=0
  ГОРА AND   (P=2, all-alt)   : all cells persist=0, mean_run=1

Functions
─────────
  run_lengths(seq, circular)         → list[int]       lengths of consecutive runs
  persistence(seq)                   → float           lag-1 self-trans probability
  run_stats(seq)                     → dict            mean/std/max/cv/n_runs/persist
  cell_run_stats(word, rule, width)  → list[dict]      per-cell run statistics
  pooled_run_dist(word, rule, width) → dict[int, int]  run-length histogram
  persistence_dict(word, rule, width)→ dict
  all_persistence(word, width)       → dict[str, dict]
  build_persistence_data(words)      → dict
  print_persistence(word, rule, color)  → None
  print_persistence_stats(words, color) → None

Запуск
──────
  python3 -m projects.hexglyph.solan_persistence --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_persistence --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_persistence --stats --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16
_DEFAULT_THRESH = 32


# ── Run-length core ───────────────────────────────────────────────────────────

def run_lengths(seq: list[int], circular: bool = True) -> list[int]:
    """Extract run lengths from a binary sequence.

    A run is a maximal block of consecutive identical symbols.
    For circular=True, the sequence wraps from end back to start,
    so the last run may merge with the first.

    Returns a list of positive integers that sum to len(seq).

    Examples (circular)
    ────────────────────
    [0,0,0,0]   → [4]          (one constant run)
    [0,1,0,1]   → [1,1,1,1]   (alternating, all length 1)
    [0,1,1,0,0,1] → [2,2,1,1] (two 1-runs and two 0-runs of varying length)
    [v]         → [1]          (single-element, period=1)
    """
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [1]

    if circular:
        # Find positions where symbol transitions (including circular wrap)
        transitions: list[int] = []
        for t in range(n):
            if seq[t] != seq[(t + 1) % n]:
                transitions.append(t)
        if not transitions:
            return [n]           # constant sequence
        nt = len(transitions)
        runs: list[int] = []
        for i in range(nt):
            start = transitions[i]
            end   = transitions[(i + 1) % nt]
            if end > start:
                runs.append(end - start)
            else:
                runs.append(n - start + end)
        return runs
    else:
        # Non-circular (linear) extraction
        runs = []
        cur = 1
        for t in range(1, n):
            if seq[t] == seq[t - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        return runs


def persistence(seq: list[int]) -> float:
    """Lag-1 self-transition probability: P(b(t+1) = b(t)) (circular).

    persistence = 1 − (number of transitions / P)
    0.0 = alternates every step  (minimum persistence)
    1.0 = never changes          (maximum persistence, constant cell)
    """
    n = len(seq)
    if n <= 1:
        return 1.0
    transitions = sum(1 for t in range(n) if seq[t] != seq[(t + 1) % n])
    return round(1.0 - transitions / n, 8)


def run_stats(seq: list[int]) -> dict:
    """Run-length statistics for a single binary temporal sequence.

    Returns dict:
        n_runs      : int     number of distinct runs
        persistence : float   P(b(t+1)=b(t)) = 1 − n_runs/P
        mean_run    : float   mean run length = P / n_runs
        std_run     : float   std of run lengths (0 if n_runs ≤ 1)
        max_run     : int     maximum run length
        min_run     : int     minimum run length
        cv_run      : float   coefficient of variation std/mean (0 if const.)
    """
    rl = run_lengths(seq, circular=True)
    n_runs = len(rl)
    mean_r = sum(rl) / n_runs
    var_r  = sum((r - mean_r) ** 2 for r in rl) / n_runs if n_runs > 1 else 0.0
    std_r  = math.sqrt(var_r)
    return {
        'n_runs':      n_runs,
        'persistence': persistence(seq),
        'mean_run':    round(mean_r, 6),
        'std_run':     round(std_r, 6),
        'max_run':     max(rl),
        'min_run':     min(rl),
        'cv_run':      round(std_r / mean_r, 6) if mean_r > 0 else 0.0,
    }


# ── Cell-level analysis ───────────────────────────────────────────────────────

def _get_binary_grid(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
    threshold: int = _DEFAULT_THRESH,
) -> list[list[int]]:
    """P × N binary grid from the periodic attractor."""
    from projects.hexglyph.solan_symbolic import attractor_binary
    return attractor_binary(word.upper(), rule, width, threshold)


def cell_run_stats(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[dict]:
    """Per-cell run-length statistics: list of N=width dicts.

    Each dict has the same keys as run_stats().
    """
    grid = _get_binary_grid(word, rule, width)
    P    = len(grid)
    result: list[dict] = []
    for i in range(width):
        seq = [grid[t][i] for t in range(P)]
        result.append(run_stats(seq))
    return result


def pooled_run_dist(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[int, int]:
    """Pooled run-length histogram across all N cells.

    Returns dict {run_length: count}.
    """
    grid = _get_binary_grid(word, rule, width)
    P    = len(grid)
    hist: dict[int, int] = {}
    for i in range(width):
        seq = [grid[t][i] for t in range(P)]
        for r in run_lengths(seq):
            hist[r] = hist.get(r, 0) + 1
    return hist


# ── Full analysis ──────────────────────────────────────────────────────────────

def persistence_dict(
    word:  str,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
) -> dict:
    """Full run-length / persistence analysis for one word × rule.

    Returns:
        word, rule, period
        cell_stats       : list[dict]   per-cell run stats (N entries)
        run_dist         : dict         pooled run-length histogram
        mean_persistence : float        mean over all N cells
        mean_run         : float        mean of all cells' mean_run values
        mean_cv          : float        mean of all cells' cv_run values
        max_run_global   : int          longest single run across all cells
        min_persistence  : float        least persistent cell
        max_persistence  : float        most persistent cell
        all_persistent   : bool         True iff all cells have persistence=1
        all_alternating  : bool         True iff all cells have persistence=0
    """
    word = word.upper()
    grid = _get_binary_grid(word, rule, width)
    P    = len(grid)

    stats = cell_run_stats(word, rule, width)
    hist  = pooled_run_dist(word, rule, width)

    persts = [s['persistence'] for s in stats]
    mruns  = [s['mean_run']    for s in stats]
    cvs    = [s['cv_run']      for s in stats]

    return {
        'word':             word,
        'rule':             rule,
        'period':           P,
        'cell_stats':       stats,
        'run_dist':         hist,
        'mean_persistence': round(sum(persts) / len(persts), 6) if persts else 0.0,
        'mean_run':         round(sum(mruns)  / len(mruns),  6) if mruns  else 0.0,
        'mean_cv':          round(sum(cvs)    / len(cvs),    6) if cvs    else 0.0,
        'max_run_global':   max(s['max_run'] for s in stats) if stats else 0,
        'min_persistence':  min(persts) if persts else 0.0,
        'max_persistence':  max(persts) if persts else 0.0,
        'all_persistent':   all(p == 1.0 for p in persts),
        'all_alternating':  all(p == 0.0 for p in persts),
    }


def all_persistence(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict]:
    """persistence_dict for all 4 rules."""
    return {r: persistence_dict(word, r, width) for r in _ALL_RULES}


def build_persistence_data(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
) -> dict:
    """Persistence summary across the lexicon × 4 rules.

    Returns:
        words, width,
        per_rule: {rule: {word: {period, mean_persistence, mean_run, mean_cv,
                                 max_run_global, all_persistent}}}
    """
    words = words if words is not None else list(LEXICON)
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = persistence_dict(word, rule, width)
            per_rule[rule][word] = {
                'period':           d['period'],
                'mean_persistence': d['mean_persistence'],
                'mean_run':         d['mean_run'],
                'mean_cv':          d['mean_cv'],
                'max_run_global':   d['max_run_global'],
                'all_persistent':   d['all_persistent'],
            }
    return {'words': words, 'width': width, 'per_rule': per_rule}


# ── ASCII / ANSI display ───────────────────────────────────────────────────────

_SHADE_6 = ' ░▒▒▓█'


def print_persistence(
    word:  str  = 'ТУМАН',
    rule:  str  = 'xor3',
    color: bool = True,
) -> None:
    """Print per-cell persistence and run-length stats."""
    d    = persistence_dict(word, rule)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    P = d['period']
    print(f"{bold}  ◈ Persistence  {word.upper()}  |  "
          f"{col}{name}{rst}  P={P}  "
          f"persist̄={d['mean_persistence']:.4f}  "
          f"run̄={d['mean_run']:.3f}  "
          f"CV̄={d['mean_cv']:.3f}")
    print(f"  {'─' * 62}")
    print(f"  {'cell':>4}  {'persist':>8}  {'run̄':>6}  {'max':>4}  "
          f"{'CV':>6}  bar")
    print(f"  {'─' * 62}")

    for i, s in enumerate(d['cell_stats']):
        p = s['persistence']
        idx = min(int(p * (len(_SHADE_6) - 1)), len(_SHADE_6) - 1)
        bar_len = round(p * 20)
        bar = (col if color else '') + '█' * bar_len + (rst if color else '') + \
              '░' * (20 - bar_len)
        print(f"  {i:>4}  {p:>8.4f}  {s['mean_run']:>6.2f}  "
              f"{s['max_run']:>4}  {s['cv_run']:>6.3f}  {bar}")

    # Run-length distribution
    hist = d['run_dist']
    if hist:
        print(f"\n  Run-length distribution (pooled):")
        max_cnt = max(hist.values())
        for r in sorted(hist.keys()):
            cnt = hist[r]
            bar_len = round(cnt / max_cnt * 16)
            print(f"  {r:>4}: {'█' * bar_len + '░' * (16 - bar_len)}  {cnt}")
    print()


def print_persistence_stats(
    words: list[str] | None = None,
    color: bool             = True,
) -> None:
    """Table: mean persistence per word × rule."""
    words = words if words is not None else list(LEXICON)
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>9s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Средняя персистентность по лексикону{rst}")
    print('  ' + '─' * (len(header) + 2))
    print('  ' + header)
    print('  ' + '─' * (len(header) + 2))
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = persistence_dict(word, rule)
            v   = d['mean_persistence']
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{v:>9.4f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Run-Length & Persistence — Q6 CA')
    parser.add_argument('--word',      default='ТУМАН')
    parser.add_argument('--rule',      default='xor3', choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_persistence_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_persistence(args.word, rule, color)
    else:
        print_persistence(args.word, args.rule, color)


if __name__ == '__main__':
    _main()
