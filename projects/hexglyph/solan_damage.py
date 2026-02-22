"""solan_damage.py — Damage Spreading Analysis for Q6 CA.

Measures how a localised single-bit perturbation of the initial condition
propagates through the Q6 cellular automaton.

For a word W and rule R:
  1. Compute IC = pad_to(encode_word(W), 16)
  2. Create perturbed IC by flipping bit `bit` in cell `cell`
  3. Evolve both ICs for n_steps steps in parallel
  4. At each step t define the damage vector:
        d(t)[i] = |orig(t)[i] − pert(t)[i]|    (in [0, 63])
  5. Normalise by 63 → d̂(t)[i] ∈ [0, 1]
  6. Total damage at step t: D(t) = mean_i d̂(t)[i]  ∈ [0, 1]

Summary statistics:
    max_damage     : max D(t)
    final_damage   : D(n_steps − 1)
    extinction_step: smallest t with D(t) < 1e-9, or −1 if never
    velocity       : linear slope of the damage-front width W(t)
                     where W(t) = number of cells with d̂(t,i) > 0.02
    phase          : 'ordered'  if extinction_step ≠ −1
                     'chaotic'  if final_damage > 0.05
                     'critical' otherwise

The module also aggregates over all 16 × 6 = 96 (cell, bit) perturbations
to obtain a cell-averaged damage kernel K(t, Δi) showing how damage from a
localised source spreads across the ring.

Functions:
    perturb(cells, cell, bit)              → list[int]
    run_pair(orig, pert, rule, n_steps)    → (grid_orig, grid_pert, damage_grid)
    single_damage(word, rule, cell, bit, n_steps)  → dict
    mean_damage(word, rule, n_steps)       → dict
    damage_dict(word, rule, n_steps)       → dict
    all_damage(word, n_steps)              → dict[str, dict]
    build_damage_data(words, n_steps)      → dict
    print_damage(word, rule, cell, bit, n_steps, color) → None
    print_damage_stats(words, color)       → None

Запуск:
    python3 -m projects.hexglyph.solan_damage --word ТУМАН --rule xor3 --cell 0 --bit 0
    python3 -m projects.hexglyph.solan_damage --word ГОРА --all-rules --no-color
    python3 -m projects.hexglyph.solan_damage --stats --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    step,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16
_DEFAULT_STEPS = 32
_Q6_MAX        = 63
_BITS          = 6          # Q6: bits 0-5
_EPS           = 1e-9       # threshold for "zero damage"
_FRONT_EPS     = 0.02       # threshold for "damaged cell" in width estimate


# ── Core helpers ───────────────────────────────────────────────────────────────

def perturb(
    cells:    list[int],
    cell_idx: int,
    bit:      int,
) -> list[int]:
    """Return copy of cells with bit `bit` (0-5) flipped in cells[cell_idx]."""
    c = cells[:]
    c[cell_idx] = (c[cell_idx] ^ (1 << bit)) & _Q6_MAX
    return c


def run_pair(
    orig:    list[int],
    pert:    list[int],
    rule:    str,
    n_steps: int,
) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
    """Evolve both CAs for n_steps; return (orig_grid, pert_grid, damage_grid).

    damage_grid[t][i] = |orig(t)[i] − pert(t)[i]| / 63   (float in [0, 1])
    """
    orig_grid:   list[list[int]]   = []
    pert_grid:   list[list[int]]   = []
    damage_grid: list[list[float]] = []
    co, cp = orig[:], pert[:]
    for _ in range(n_steps):
        orig_grid.append(co[:])
        pert_grid.append(cp[:])
        damage_grid.append([abs(co[i] - cp[i]) / _Q6_MAX
                            for i in range(len(co))])
        co = step(co, rule)
        cp = step(cp, rule)
    return orig_grid, pert_grid, damage_grid


def _total_damage(damage_grid: list[list[float]]) -> list[float]:
    """Mean damage across all cells at each step."""
    return [sum(row) / max(len(row), 1) for row in damage_grid]


def _damage_width(damage_grid: list[list[float]]) -> list[int]:
    """Number of cells with damage > _FRONT_EPS at each step."""
    return [sum(1 for v in row if v > _FRONT_EPS) for row in damage_grid]


def _velocity(widths: list[int]) -> float:
    """Estimate damage-front velocity as linear slope of width vs step index.

    Uses least-squares over the steps where width is increasing (growth phase).
    Returns 0.0 if fewer than 2 points.
    """
    # keep only the growth phase (until first local max)
    peak = max(range(len(widths)), key=lambda t: widths[t], default=0)
    pts  = widths[:peak + 1]
    n    = len(pts)
    if n < 2:
        return 0.0
    xs   = list(range(n))
    xm   = sum(xs) / n
    ym   = sum(pts) / n
    cov  = sum((xs[i] - xm) * (pts[i] - ym) for i in range(n))
    var  = sum((xs[i] - xm) ** 2           for i in range(n))
    return round(cov / var, 4) if var > 0 else 0.0


def _extinction_step(total: list[float]) -> int:
    """First step where total damage < _EPS, or −1."""
    for t, v in enumerate(total):
        if v < _EPS:
            return t
    return -1


def _phase(extinction: int, final_damage: float) -> str:
    if extinction >= 0:
        return 'ordered'
    if final_damage > 0.05:
        return 'chaotic'
    return 'critical'


# ── Single perturbation ────────────────────────────────────────────────────────

def single_damage(
    word:    str,
    rule:    str,
    cell:    int  = 0,
    bit:     int  = 0,
    n_steps: int  = _DEFAULT_STEPS,
    width:   int  = _DEFAULT_WIDTH,
) -> dict:
    """Full damage analysis for one (word, rule, cell, bit) perturbation.

    Returns dict:
        word, rule, cell, bit, n_steps, width
        orig_grid    : list[list[int]]           n_steps × width
        pert_grid    : list[list[int]]           n_steps × width
        damage_grid  : list[list[float]]         n_steps × width  in [0,1]
        total_damage : list[float]               n_steps
        damage_width : list[int]                 n_steps
        max_damage   : float
        final_damage : float
        extinction_step : int   (−1 if never)
        velocity     : float
        phase        : str  'ordered' | 'critical' | 'chaotic'
    """
    cells = pad_to(encode_word(word.upper()), width)
    pert  = perturb(cells, cell % width, bit % _BITS)
    og, pg, dg = run_pair(cells, pert, rule, n_steps)
    total = _total_damage(dg)
    wids  = _damage_width(dg)
    ext   = _extinction_step(total)
    fin   = total[-1] if total else 0.0
    return {
        'word':          word.upper(),
        'rule':          rule,
        'cell':          cell % width,
        'bit':           bit  % _BITS,
        'n_steps':       n_steps,
        'width':         width,
        'orig_grid':     og,
        'pert_grid':     pg,
        'damage_grid':   dg,
        'total_damage':  total,
        'damage_width':  wids,
        'max_damage':    round(max(total) if total else 0.0, 8),
        'final_damage':  round(fin, 8),
        'extinction_step': ext,
        'velocity':      _velocity(wids),
        'phase':         _phase(ext, fin),
    }


# ── Averaged over all perturbations ───────────────────────────────────────────

def mean_damage(
    word:    str,
    rule:    str,
    n_steps: int = _DEFAULT_STEPS,
    width:   int = _DEFAULT_WIDTH,
) -> dict:
    """Average damage over all width × BITS = 96 (cell, bit) perturbations.

    Returns dict:
        word, rule, n_steps, width, n_perturb
        mean_damage_grid : list[list[float]]   n_steps × width  (mean |δ|/63)
        mean_total       : list[float]         n_steps
        mean_width       : list[float]         n_steps
        max_mean_damage  : float
        final_mean_damage: float
        extinction_step  : int
        velocity         : float
        phase            : str
        kernel           : list[list[float]]   n_steps × width
                           K(t, Δi) = mean damage in cell (source+Δi) at step t
                           averaged over source cells and all bit flips
    """
    cells     = pad_to(encode_word(word.upper()), width)
    n_perturb = width * _BITS
    # accumulate damage_grid sums
    sum_dg = [[0.0] * width for _ in range(n_steps)]
    # for the kernel: accumulate damage aligned to source cell
    sum_kernel = [[0.0] * width for _ in range(n_steps)]
    for src in range(width):
        for b in range(_BITS):
            pert = perturb(cells, src, b)
            _, _, dg = run_pair(cells, pert, rule, n_steps)
            for t in range(n_steps):
                for i in range(width):
                    sum_dg[t][i] += dg[t][i]
                    # kernel: Δi = (i - src) mod width
                    di = (i - src) % width
                    sum_kernel[t][di] += dg[t][i]
    mean_dg  = [[v / n_perturb for v in row] for row in sum_dg]
    mean_k   = [[v / n_perturb for v in row] for row in sum_kernel]
    total    = _total_damage(mean_dg)
    wids_f   = [sum(1 for v in row if v > _FRONT_EPS) for row in mean_dg]
    ext      = _extinction_step(total)
    fin      = total[-1] if total else 0.0
    return {
        'word':             word.upper(),
        'rule':             rule,
        'n_steps':          n_steps,
        'width':            width,
        'n_perturb':        n_perturb,
        'mean_damage_grid': mean_dg,
        'mean_total':       total,
        'mean_width':       wids_f,
        'max_mean_damage':  round(max(total) if total else 0.0, 8),
        'final_mean_damage':round(fin, 8),
        'extinction_step':  ext,
        'velocity':         _velocity(wids_f),
        'phase':            _phase(ext, fin),
        'kernel':           mean_k,
    }


# ── Full analysis dict ────────────────────────────────────────────────────────

def damage_dict(
    word:    str,
    rule:    str,
    n_steps: int = _DEFAULT_STEPS,
    width:   int = _DEFAULT_WIDTH,
) -> dict:
    """Combined single_damage(cell=0,bit=0) + mean_damage summary.

    Returns dict merging both, with keys prefixed where ambiguous.
    Ready for JSON serialisation (no float32/ndarray).
    """
    sd = single_damage(word, rule, cell=0, bit=0, n_steps=n_steps, width=width)
    md = mean_damage(word, rule, n_steps=n_steps, width=width)
    # remove heavy grids from top-level merge to keep size manageable
    result = {k: v for k, v in sd.items()
              if k not in ('orig_grid', 'pert_grid', 'damage_grid')}
    result.update({
        'mean_total':        md['mean_total'],
        'mean_width':        md['mean_width'],
        'max_mean_damage':   md['max_mean_damage'],
        'final_mean_damage': md['final_mean_damage'],
        'mean_ext_step':     md['extinction_step'],
        'mean_velocity':     md['velocity'],
        'mean_phase':        md['phase'],
        'kernel':            md['kernel'],
    })
    return result


def all_damage(
    word:    str,
    n_steps: int = _DEFAULT_STEPS,
) -> dict[str, dict]:
    """damage_dict for all 4 rules."""
    return {r: damage_dict(word, r, n_steps) for r in _ALL_RULES}


def build_damage_data(
    words:   list[str] | None = None,
    n_steps: int              = 24,
) -> dict:
    """Damage summary for lexicon × 4 rules.

    Returns dict:
        words   : list[str]
        per_rule: {rule: {word: {max_damage, final_damage,
                                  extinction_step, velocity, phase}}}
        phase_counts: {rule: {'ordered':n, 'critical':n, 'chaotic':n}}
    """
    words = words if words is not None else list(LEXICON)
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            md = mean_damage(word, rule, n_steps=n_steps)
            per_rule[rule][word] = {
                'max_damage':     md['max_mean_damage'],
                'final_damage':   md['final_mean_damage'],
                'extinction_step':md['extinction_step'],
                'velocity':       md['velocity'],
                'phase':          md['phase'],
            }
    phase_counts: dict[str, dict] = {
        r: {'ordered': 0, 'critical': 0, 'chaotic': 0}
        for r in _ALL_RULES
    }
    for rule in _ALL_RULES:
        for entry in per_rule[rule].values():
            phase_counts[rule][entry['phase']] += 1
    return {'words': words, 'per_rule': per_rule, 'phase_counts': phase_counts}


# ── ASCII / ANSI display ───────────────────────────────────────────────────────

def print_damage(
    word:    str         = 'ТУМАН',
    rule:    str         = 'xor3',
    cell:    int         = 0,
    bit:     int         = 0,
    n_steps: int         = _DEFAULT_STEPS,
    color:   bool        = True,
) -> None:
    """Print damage spacetime diagram + total-damage curve."""
    d    = single_damage(word, rule, cell, bit, n_steps)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    N    = d['width']
    _shade = ' ░▒▓█'

    # Normalise shade by the peak cell-damage in this episode
    all_vals = [v for row in d['damage_grid'] for v in row]
    norm_d   = max(all_vals) if all_vals else 1.0
    norm_d   = max(norm_d, 1e-9)
    max_tot  = max(d['total_damage']) if d['total_damage'] else 1.0
    max_tot  = max(max_tot, 1e-9)

    print(f"{bold}  ◈ Разброс повреждений Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  ячейка={cell}  бит={bit}")
    print(f"  {'─' * (N + 24)}")
    for t, (drow, tot) in enumerate(zip(d['damage_grid'], d['total_damage'])):
        cells_str = ''.join(
            _shade[min(int(v / norm_d * (len(_shade) - 1) + 0.5), len(_shade) - 1)]
            for v in drow
        )
        bar_len  = int(tot / max_tot * 20 + 0.5)
        bar_char = col + '▉' * bar_len + rst if color else '▉' * bar_len
        print(f"  {t:3d} {cells_str}  D={tot:.4f}  {bar_char}")
    print(f"  {'─' * (N + 24)}")
    print(f"  фаза={d['phase']}  max={d['max_damage']:.3f}  "
          f"ext={d['extinction_step']}  скор.={d['velocity']:.3f}")
    print()


def print_damage_stats(
    words: list[str] | None = None,
    color: bool             = True,
) -> None:
    """Table: mean max-damage and phase per word × rule."""
    words = words if words is not None else list(LEXICON)
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Макс. повреждение (ср. по ячейкам/битам){rst}")
    print('  ' + '─' * (len(header) + 2))
    print('  ' + header)
    print('  ' + '─' * (len(header) + 2))
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            md  = mean_damage(word, rule, n_steps=20)
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{md['max_mean_damage']:>8.3f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Damage Spreading Q6 CA')
    parser.add_argument('--word',      default='ТУМАН')
    parser.add_argument('--rule',      default='xor3', choices=_ALL_RULES)
    parser.add_argument('--cell',      type=int, default=0)
    parser.add_argument('--bit',       type=int, default=0)
    parser.add_argument('--steps',     type=int, default=_DEFAULT_STEPS)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--no-color',  action='store_true')
    parser.add_argument('--json',      action='store_true', help='JSON output')
    args  = parser.parse_args()
    color = not args.no_color
    if args.json:
        import json as _json
        print(_json.dumps(damage_dict(args.word, args.rule), ensure_ascii=False, indent=2))
        return

    if args.stats:
        print_damage_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_damage(args.word, rule, args.cell, args.bit, args.steps, color)
    else:
        print_damage(args.word, args.rule, args.cell, args.bit, args.steps, color)


if __name__ == '__main__':
    _main()
