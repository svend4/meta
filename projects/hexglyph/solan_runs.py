"""
solan_runs.py — Run-Length (Plateau) Analysis of Q6 CA Attractor Cells.

A *plateau* is a maximal consecutive subsequence of identical values in
a cell's temporal orbit {x_t : t = 0…P−1}.  Run-length analysis
identifies and quantifies these constant segments.

Linear run-lengths of series [x_0, …, x_{P-1}]:
    RL = [r_1, r_2, …, r_k]   where Σ r_i = P

Plateau fraction (circular): fraction of time steps t ∈ {0,…,P−1}
where x_t == x_{(t+1) mod P}.  This is the same quantity as the
diagonal fraction in the first-return map (solan_return.py), but
resolved per cell.

Run entropy: H(RL) = −Σ p_r log₂ p_r   where p_r = r / P
(the probability distribution of "which step is in a run of length r")

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, const 0):
      all cells → single run of length 1, plateau_frac=1.0 (self-loop), run_H=0
  ГОРА  AND  (P=2, 47↔1):
      all cells → [1,1], max_run=1, plateau_frac=0.0, no plateaus
  ТУМАН XOR3 (P=8):
      cell 0 → [1,1,4,1,1], max_run=4, plateau_frac=0.500
      cell 1 → [1,1,3,1,1,1], max_run=3, plateau_frac=0.250
      cell 3 → [1,1,1,1,1,1,1,1], max_run=1, plateau_frac=0.000
      global max_run=4, strong spatial heterogeneity

Interpretation
  max_run > 1  : the CA exhibits temporal "stickiness" — the cell
                 remains at the same value for multiple steps
  run_entropy  : diversity of plateau lengths (0 = all runs equal length)
  plateau_frac : fraction of steps that are pure repetitions (diagonal
                 in the return map)

Functions
─────────
  run_lengths(series)                    → list[int]
  plateau_fraction(series)               → float   (circular)
  run_entropy(run_lens)                  → float   (bits)
  cell_run_stats(series)                 → dict
  all_cell_stats(word, rule, width)      → list[dict]  (per cell)
  change_matrix(word, rule, width)       → list[list[int]]  shape (N, P)
  run_summary(word, rule, width)         → dict
  all_run_summaries(word, width)         → dict[str, dict]
  build_run_data(words, width)           → dict
  print_runs(word, rule, color)          → None
  print_run_stats(words, color)          → None

Запуск
──────
  python3 -m projects.hexglyph.solan_runs --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_runs --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_runs --stats --no-color
"""

from __future__ import annotations
import sys
import argparse
import math

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


# ── Core computation ──────────────────────────────────────────────────────────

def run_lengths(series: list[int]) -> list[int]:
    """
    Linear run-lengths of a flat sequence.

    Returns a list of positive integers that sum to len(series).
    Example: [48, 51, 43, 43, 43, 43, 40, 48] → [1, 1, 4, 1, 1]
    """
    if not series:
        return []
    runs: list[int] = []
    cur, cnt = series[0], 1
    for x in series[1:]:
        if x == cur:
            cnt += 1
        else:
            runs.append(cnt)
            cur, cnt = x, 1
    runs.append(cnt)
    return runs


def plateau_fraction(series: list[int]) -> float:
    """
    Circular plateau fraction: fraction of t ∈ {0,…,P−1} with x_t == x_{t+1 mod P}.

    Equals the diagonal fraction in the first-return map (solan_return).
    """
    P = len(series)
    if P == 0:
        return 0.0
    return sum(1 for t in range(P) if series[t] == series[(t + 1) % P]) / P


def run_entropy(run_lens: list[int]) -> float:
    """
    Run-length entropy H(RL) in bits.

    Uses the probability distribution over individual steps:
        p_r = r / Σ r_i  (proportion of steps belonging to a run of length r)
    H = −Σ_r (r/P) log₂(r/P) over distinct run lengths r present.

    This equals the entropy of "which run does a randomly chosen step belong to?"
    Range: 0 (all runs equal length) … log₂(P) (one run of length 1 and rest trivial)
    """
    if not run_lens:
        return 0.0
    total = sum(run_lens)
    if total == 0:
        return 0.0
    # Accumulate probability by run length
    prob: dict[int, float] = {}
    for r in run_lens:
        prob[r] = prob.get(r, 0.0) + r / total
    return max(0.0, -sum(p * math.log2(p) for p in prob.values() if p > 0))


def cell_run_stats(series: list[int]) -> dict:
    """
    Full run-length statistics for a single cell's temporal series.

    Returns:
        runs           : list of run lengths (linear)
        n_runs         : number of runs
        max_run        : longest plateau
        mean_run       : average run length (= P / n_runs)
        plateau_frac   : circular plateau fraction
        run_entropy    : H(RL) bits
    """
    P = len(series)
    if P == 0:
        return {'runs': [], 'n_runs': 0, 'max_run': 0,
                'mean_run': 0.0, 'plateau_frac': 0.0, 'run_entropy': 0.0}
    rl  = run_lengths(series)
    pf  = plateau_fraction(series)
    re  = run_entropy(rl)
    return {
        'runs':          rl,
        'n_runs':        len(rl),
        'max_run':       max(rl),
        'mean_run':      round(P / len(rl), 6),
        'plateau_frac':  round(pf, 6),
        'run_entropy':   round(re, 6),
    }


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Multi-cell functions ───────────────────────────────────────────────────────

def all_cell_stats(word: str, rule: str, width: int = _DEFAULT_W) -> list[dict]:
    """Per-cell run stats (list of length = width)."""
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    out   = []
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        s      = cell_run_stats(series)
        s['cell'] = i
        out.append(s)
    return out


def change_matrix(word: str, rule: str, width: int = _DEFAULT_W) -> list[list[int]]:
    """
    Boolean change matrix: C[i][t] = 1 if x_t[i] ≠ x_{t-1 mod P}[i] else 0.
    Shape: (width, period).

    0 = plateau continuation (x same as previous step)
    1 = transition (x changed from previous step)
    """
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    mat   = []
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        row    = [1 if series[t] != series[(t - 1) % P] else 0 for t in range(P)]
        mat.append(row)
    return mat


# ── Summary ───────────────────────────────────────────────────────────────────

def _safe_stats(vals: list[float]) -> dict:
    if not vals:
        return {'mean': None, 'std': None, 'min': None, 'max': None}
    n   = len(vals)
    mu  = sum(vals) / n
    std = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5
    return {'mean': round(mu, 6), 'std': round(std, 6),
            'min': round(min(vals), 6), 'max': round(max(vals), 6)}


def run_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Aggregate run-length statistics for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']
    css    = all_cell_stats(word, rule, width)

    max_runs  = [c['max_run']       for c in css]
    pfrs      = [c['plateau_frac']  for c in css]
    entrs     = [c['run_entropy']   for c in css]
    n_runs    = [c['n_runs']        for c in css]
    all_rl    = [r for c in css for r in c['runs']]   # flattened

    return {
        'word':           word.upper(),
        'rule':           rule,
        'period':         period,
        'cell_stats':     css,
        'max_run_stats':  _safe_stats([float(v) for v in max_runs]),
        'pf_stats':       _safe_stats(pfrs),
        'entropy_stats':  _safe_stats(entrs),
        'n_runs_stats':   _safe_stats([float(v) for v in n_runs]),
        'global_max_run': max(all_rl) if all_rl else 0,
        'global_mean_run': round(sum(all_rl) / len(all_rl), 6) if all_rl else 0.0,
        'all_run_lengths': all_rl,
    }


def all_run_summaries(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """run_summary for all 4 rules."""
    return {rule: run_summary(word, rule, width) for rule in _RULES}


def build_run_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact run data for a list of words."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = run_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'global_max_run', 'global_mean_run',
                 'max_run_stats', 'pf_stats', 'entropy_stats')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'


def _fmtr(v, d: int = 3) -> str:
    return f'{v:.{d}f}' if v is not None else '—'


def print_runs(word: str = 'ТУМАН', rule: str = 'xor3',
               color: bool = True) -> None:
    d   = run_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    print(f'  {c}◈ Runs  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'global_max_run={d["global_max_run"]}  '
          f'mean_pf={_fmtr(d["pf_stats"]["mean"])}{r}')
    print('  ' + '─' * 62)

    mr_lo = float(d['max_run_stats']['min'] or 0)
    mr_hi = float(d['max_run_stats']['max'] or 1)
    print(f'  {"cell":>4}  {"runs":>20}  {"max":>4}  {"pf":>6}  {"H_RL":>6}')
    for cs in d['cell_stats']:
        runs_str = str(cs['runs'])[:18]
        bar_len  = round((cs['max_run'] - mr_lo) / max(mr_hi - mr_lo, 1) * 10)
        bar      = '█' * bar_len + '░' * (10 - bar_len)
        print(f'  {cs["cell"]:>4}  {runs_str:>20}  {cs["max_run"]:>4}  '
              f'{cs["plateau_frac"]:>6.3f}  {cs["run_entropy"]:>6.3f}  {bar}')

    print(f'\n  all_run_lengths: {sorted(set(d["all_run_lengths"]))[:10]}')
    print(f'  pf:  mean={_fmtr(d["pf_stats"]["mean"])}  '
          f'range=[{_fmtr(d["pf_stats"]["min"])},{_fmtr(d["pf_stats"]["max"])}]')
    print()


def print_run_stats(words: list[str] | None = None, color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_runs(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Run-length / plateau analysis of Q6 CA')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_run_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_runs(args.word, rule, color)
    else:
        print_runs(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
