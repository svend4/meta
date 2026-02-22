"""solan_fourier.py — Discrete Fourier Transform of Q6 CA Cell Temporal Sequences.

For a period-P attractor each cell i traces a Q6 value sequence x_i[0…P-1].
Because the orbit is periodic with period P the DFT gives an exact spectral
decomposition into sinusoidal components.

    cell_spectrum(seq)              → per-cell one-sided spectral statistics
    fourier_summary(word, rule)     → full per-cell + aggregate statistics
    all_fourier(word)               → summary for all 4 CA rules

Definitions
──────────────────────────────────────────────────────────────────────────────
  DFT:      F[k] = Σ_{t=0}^{P-1} x[t] · exp(−2πi · k · t / P)

  One-sided normalised power:  S[k] = |F[k]|² / P   for k = 0 … ⌊P/2⌋

  AC power (temporal variation):
      ac_power = Σ_{k=1}^{⌊P/2⌋} S[k]

  Parseval:  Σ_{k=0}^{P-1} |F[k]|² / P  =  Σ_t x[t]²

  Dominant harmonic:   k* = argmax_{k≥1} S[k]   (DC excluded)

  Spectral entropy (bits) over one-sided AC bins k = 1 … ⌊P/2⌋:
      p[k] = S[k] / Σ_{j=1}^{⌊P/2⌋} S[j]
      H    = −Σ p[k] · log₂ p[k]          0 → tonal; log₂(⌊P/2⌋) → flat

  Normalised spectral entropy:  nH = H / log₂(⌊P/2⌋)  ∈ [0, 1]

  Spectral flatness (Wiener entropy):
      SF = geometric_mean(S) / arithmetic_mean(S) ∈ [0, 1]

Key discoveries
──────────────────────────────────────────────────────────────────────────────
  P = 2 words (XOR3, 20/49 lexicon):
      Only k=0 (DC) and k=1 (Nyquist) are non-zero.
      dom_freq = 1 for every non-frozen cell.  spec_entropy = 0.

  РАБОТА XOR3  (P=8), cell 1  [63,62,63,1,63,0,63,62]:
      dom_freq = 4 (Nyquist)  →  strong 2-step alternation within the orbit.
      dom_freq_hist: {1: 8, 2: 1, 3: 1, 4: 6} — Nyquist dominates 6 cells.

  МОНТАЖ XOR3  (P=8):
      mean_spec_entropy ≈ 1.618 bits — highest in the lexicon (closest to
      theoretical maximum log₂(4) = 2.000 bits).

  ГОРОД XOR3  (P=8):
      dom_freq_hist: {1: 4, 3: 12} — 12/16 cells dominated by k=3
      (3rd harmonic, effective period 8/3 ≈ 2.67 steps).  mean_H ≈ 1.163.

  AND / OR rules (P=1 fixed points):
      ac_power = 0 for all cells; spec_entropy = 0; dom_freq = 0.

Запуск:
    python3 -m projects.hexglyph.solan_fourier --word РАБОТА --rule xor3
    python3 -m projects.hexglyph.solan_fourier --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_fourier --table --rule xor3
    python3 -m projects.hexglyph.solan_fourier --json --word РАБОТА
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES          = tuple(_ALL_RULES)
_DEFAULT_WIDTH = 16
_EPS           = 1e-20


# ── Core DFT primitives ───────────────────────────────────────────────────────

def dft1(series: list[int | float]) -> list[complex]:
    """Exact DFT of *series* (O(N²), suitable for period ≤ 128)."""
    N = len(series)
    if N == 0:
        return []
    return [
        sum(series[t] * cmath.exp(-2j * math.pi * k * t / N) for t in range(N))
        for k in range(N)
    ]


def power_spectrum(series: list[int | float]) -> list[float]:
    """One-sided normalised power spectrum S[k] = |F[k]|² / N.

    Returns ⌊N/2⌋+1 values for k = 0, 1, …, ⌊N/2⌋.
    """
    N = len(series)
    if N == 0:
        return []
    X = dft1(series)
    return [(X[k].real ** 2 + X[k].imag ** 2) / N for k in range(N // 2 + 1)]


def spectral_entropy(power: list[float]) -> float:
    """Spectral entropy H = −Σ p[k] log₂ p[k] in bits over all bins.

    Returns 0.0 for empty or all-zero spectrum.
    """
    total = sum(power)
    if total <= 0:
        return 0.0
    return max(0.0, -sum(
        (v / total) * math.log2(v / total)
        for v in power if v > _EPS
    ))


def normalised_spectral_entropy(power: list[float]) -> float:
    """Spectral entropy / log₂(len(power)) ∈ [0, 1]."""
    K1 = len(power)
    if K1 <= 1 or sum(power) <= 0:
        return 0.0
    denom = math.log2(K1)
    return round(min(spectral_entropy(power) / denom, 1.0), 8) if denom > 0 else 0.0


def spectral_flatness(power: list[float]) -> float:
    """Wiener entropy = geometric_mean / arithmetic_mean ∈ [0, 1].

    Returns 0.0 if any component is zero (perfectly tonal).
    """
    n = len(power)
    if n == 0:
        return 0.0
    total = sum(power)
    if total <= 0 or any(v <= 0 for v in power):
        return 0.0
    log_mean   = sum(math.log(v) for v in power) / n
    arith_mean = total / n
    return round(min(math.exp(log_mean) / arith_mean, 1.0), 8)


def dominant_harmonic(power: list[float]) -> int:
    """Index k* ≥ 1 with maximum power (DC at k=0 excluded).

    Returns 0 for spectra with only a DC bin (P=1).
    """
    if len(power) <= 1:
        return 0
    best_k, best_v = 1, power[1] if len(power) > 1 else 0.0
    for k in range(2, len(power)):
        if power[k] > best_v:
            best_v = power[k]
            best_k = k
    return best_k


# ── Per-cell analysis ─────────────────────────────────────────────────────────

def cell_spectrum(seq: list[int]) -> dict[str, Any]:
    """One-sided DFT spectral analysis of a single cell's time series.

    Parameters
    ──────────
    seq : list[int]  — cell values at orbit steps t = 0 … P-1

    Returns dict with keys:
        power       : list[float]  — S[k] = |F[k]|²/P  for k = 0 … ⌊P/2⌋
        dom_freq    : int          — k* ≥ 1 with max S[k]; 0 if P=1
        ac_power    : float        — Σ_{k≥1} S[k]  (total AC energy)
        dc          : float        — S[0]
        dc_frac     : float        — S[0] / Σ S[k]
        h_sp        : float        — spectral entropy over all bins (bits)
        nh_sp       : float        — normalised spectral entropy ∈ [0,1]
        spec_entropy: float        — H over AC-only bins k=1..⌊P/2⌋ (bits)
        sf          : float        — spectral flatness
        period      : int          — P
    """
    P = len(seq)
    if P == 0:
        return {'power': [], 'dom_freq': 0, 'ac_power': 0.0, 'dc': 0.0,
                'dc_frac': 1.0, 'h_sp': 0.0, 'nh_sp': 0.0,
                'spec_entropy': 0.0, 'sf': 0.0, 'period': 0}

    ps     = power_spectrum(seq)     # one-sided, length = P//2 + 1
    total  = sum(ps)
    dc     = ps[0]
    ac     = sum(ps[1:])
    dom    = dominant_harmonic(ps)
    h_sp   = spectral_entropy(ps)
    nh_sp  = normalised_spectral_entropy(ps)
    sf     = spectral_flatness(ps)
    dc_frac = dc / total if total > _EPS else 1.0

    # Spectral entropy over AC-only bins k=1..⌊P/2⌋
    ac_ps  = ps[1:]
    H_ac   = spectral_entropy(ac_ps)

    return {
        'power':        ps,
        'dom_freq':     dom,
        'ac_power':     round(ac, 6),
        'dc':           round(dc, 6),
        'dc_frac':      round(dc_frac, 8),
        'h_sp':         round(h_sp, 8),
        'nh_sp':        nh_sp,
        'spec_entropy': round(H_ac, 8),
        'sf':           sf,
        'period':       P,
    }


# ── Per-word summary ──────────────────────────────────────────────────────────

def fourier_summary(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Fourier spectral analysis for all cells of one word/rule attractor.

    Keys
    ────
    word, rule, period, n_cells

    # Per-cell lists (length = width)
    cell_dom_freq     : list[int]    — dominant harmonic k* per cell
    cell_ac_power     : list[float]  — AC power per cell
    cell_dc           : list[float]  — S[0] per cell
    cell_spec_entropy : list[float]  — AC-bin spectral entropy (bits)
    cell_nh_sp        : list[float]  — normalised spectral entropy
    cell_dc_frac      : list[float]  — DC fraction of total power
    cell_power        : list[list[float]]  — one-sided power spectrum per cell

    # Aggregate
    max_ac_power       : float
    max_ac_cell        : int
    mean_ac_power      : float
    dom_freq_hist      : dict[int, int]   — {k: n_cells}
    most_common_dom_freq: int
    n_nyquist_dom      : int   — cells where k* = ⌊P/2⌋
    n_fundamental_dom  : int   — cells where k* = 1
    mean_spec_entropy  : float
    max_spec_entropy   : float
    max_spec_entropy_cell: int
    mean_nh_sp         : float
    mean_dc_frac       : float
    mean_ps            : list[float]  — mean S[k] across all cells
    dominant_k         : int          — dominant harmonic of mean_ps
    """
    from projects.hexglyph.solan_transfer import get_orbit

    orbit  = get_orbit(word, rule, width)
    P, N   = len(orbit), width

    spectra = []
    for i in range(N):
        seq = [int(orbit[t][i]) for t in range(P)]
        spectra.append(cell_spectrum(seq))

    cell_dom_freq     = [s['dom_freq']     for s in spectra]
    cell_ac_power     = [s['ac_power']     for s in spectra]
    cell_dc           = [s['dc']           for s in spectra]
    cell_spec_entropy = [s['spec_entropy'] for s in spectra]
    cell_nh_sp        = [s['nh_sp']        for s in spectra]
    cell_dc_frac      = [s['dc_frac']      for s in spectra]
    cell_power        = [s['power']        for s in spectra]

    max_ac   = max(cell_ac_power)
    max_ac_c = cell_ac_power.index(max_ac)

    dom_hist: dict[int, int] = {}
    for f in cell_dom_freq:
        dom_hist[f] = dom_hist.get(f, 0) + 1
    most_common = max(dom_hist, key=lambda k: dom_hist[k]) if dom_hist else 0
    half        = P // 2
    n_nyquist   = dom_hist.get(half, 0)
    n_fund      = dom_hist.get(1, 0)

    max_H   = max(cell_spec_entropy)
    max_H_c = cell_spec_entropy.index(max_H)

    # Mean power spectrum across cells
    max_bins = max(len(p) for p in cell_power) if cell_power else 1
    mean_ps: list[float] = []
    for k in range(max_bins):
        vals = [cell_power[i][k] for i in range(N) if k < len(cell_power[i])]
        mean_ps.append(round(sum(vals) / len(vals), 8) if vals else 0.0)
    dom_k = dominant_harmonic(mean_ps)

    return {
        'word':    word,
        'rule':    rule,
        'period':  P,
        'n_cells': N,

        'cell_dom_freq':     cell_dom_freq,
        'cell_ac_power':     cell_ac_power,
        'cell_dc':           cell_dc,
        'cell_spec_entropy': cell_spec_entropy,
        'cell_nh_sp':        cell_nh_sp,
        'cell_dc_frac':      cell_dc_frac,
        'cell_power':        cell_power,

        'max_ac_power':       round(max_ac, 4),
        'max_ac_cell':        max_ac_c,
        'mean_ac_power':      round(sum(cell_ac_power) / N, 4),

        'dom_freq_hist':         dom_hist,
        'most_common_dom_freq':  most_common,
        'n_nyquist_dom':         n_nyquist,
        'n_fundamental_dom':     n_fund,

        'mean_spec_entropy':     round(sum(cell_spec_entropy) / N, 6),
        'max_spec_entropy':      round(max_H, 6),
        'max_spec_entropy_cell': max_H_c,

        'mean_nh_sp':    round(sum(cell_nh_sp) / N, 8),
        'mean_dc_frac':  round(sum(cell_dc_frac) / N, 8),
        'mean_ps':       mean_ps,
        'dominant_k':    dom_k,
    }


def all_fourier(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict[str, Any]]:
    """fourier_summary for all 4 CA rules."""
    return {r: fourier_summary(word, r, width) for r in RULES}


def build_fourier_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Full Fourier spectral analysis for the lexicon."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)
    return {
        'words': list(words),
        'data':  {w: {r: fourier_summary(w, r, width) for r in RULES}
                  for w in words},
    }


def fourier_dict(s: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of fourier_summary.

    cell_power keys are converted to lists of rounded floats.
    dom_freq_hist keys are converted to strings (JSON requirement).
    """
    out: dict[str, Any] = {}
    for k, v in s.items():
        if k == 'dom_freq_hist':
            out[k] = {str(freq): cnt for freq, cnt in v.items()}
        elif k == 'cell_power':
            out[k] = [[round(x, 6) for x in row] for row in v]
        elif k == 'mean_ps':
            out[k] = [round(x, 6) for x in v]
        elif isinstance(v, list) and v and isinstance(v[0], float):
            out[k] = [round(x, 6) for x in v]
        else:
            out[k] = v
    return out


# ── Terminal output ───────────────────────────────────────────────────────────

_FREQ_COLORS = {
    1: '\033[38;5;39m',    # blue  — fundamental
    2: '\033[38;5;118m',   # green — 2nd harmonic
    3: '\033[38;5;214m',   # orange — 3rd harmonic
    4: '\033[38;5;196m',   # red — Nyquist
}


def print_fourier(
    word:  str,
    rule:  str,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print Fourier spectral analysis for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    s = fourier_summary(word, rule, width)
    P = s['period']
    N = s['n_cells']
    half = P // 2

    print(bold + f"  ◈ Cell Fourier Spectrum  {word.upper()}  "
          + col + lbl + reset + bold + f"  (P={P})" + reset)
    print()

    if P == 1:
        print(f"  Fixed point (P=1): no temporal variation — AC power = 0.")
        print()
        return

    # Per-cell table
    ps = s['mean_ps']
    s_max = max(ps) if ps else 1.0

    print(f"  {'cell':>4}  {'k*':>3}  {'H':>5}  {'nH':>5}  "
          f"{'ac_pw':>8}  {'dc%':>5}  AC bins k=1..{half} (normalised per cell)")
    print('  ' + '─' * 72)

    for i in range(N):
        df   = s['cell_dom_freq'][i]
        H    = s['cell_spec_entropy'][i]
        nH   = s['cell_nh_sp'][i]
        ac   = s['cell_ac_power'][i]
        dcf  = s['cell_dc_frac'][i]
        pows = s['cell_power'][i]

        fc = (_FREQ_COLORS.get(df, '') if color and df > 0 else '')
        ac_str = f'{ac:.1f}'
        dc_str = f'{dcf*100:.0f}%'

        # Mini AC spectrum bar (k=1..P//2), normalised per-cell
        ac_pows   = pows[1:]  # exclude DC
        ac_max    = max(ac_pows) if ac_pows else 1.0
        spec_bars = []
        for k, pk in enumerate(ac_pows, start=1):
            frac   = pk / ac_max if ac_max > _EPS else 0
            filled = max(0, min(4, round(frac * 4)))
            bc = (_FREQ_COLORS.get(k, dim) if color else '')
            is_dom = (k == df)
            spec_bars.append(
                f'{bc}{"█" * filled + "·" * (4 - filled)}{reset}'
                + ('←' if is_dom else ' ')
            )
        spec_str = '  '.join(spec_bars)

        print(f"  {i:4d}  {fc}{df:3d}{reset}  {H:5.3f}  {nH:5.3f}  "
              f"{ac_str:>8s}  {dc_str:>5s}  {spec_str}")

    print()
    print(f"  dom_freq hist : {dict(sorted(s['dom_freq_hist'].items()))}")
    print(f"  most common k*: {s['most_common_dom_freq']}   "
          f"Nyquist dom: {s['n_nyquist_dom']} cells   "
          f"Fundamental dom: {s['n_fundamental_dom']} cells")
    print(f"  mean H (AC)   : {s['mean_spec_entropy']:.3f} bits  "
          f"(max={math.log2(half):.3f} bits = log₂({half}))")
    print(f"  max H         : {s['max_spec_entropy']:.3f} at cell "
          f"{s['max_spec_entropy_cell']}")
    print(f"  mean nH_sp    : {s['mean_nh_sp']:.4f}   "
          f"mean DC frac: {s['mean_dc_frac']:.4f}")
    print(f"  max AC power  : {s['max_ac_power']:.1f} at cell {s['max_ac_cell']}")
    print()


def print_fourier_table(
    words: list[str] | None = None,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Summary table: Fourier spectral stats for all lexicon words."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Cell Fourier Spectrum Summary ({lbl}, n={len(words)})" + reset)
    print()
    print(f"  {'Слово':12s}  {'P':>3}  {'dom_hist':26s}  "
          f"{'mean_H':>6}  {'max_H':>5}  {'nyq':>3}  {'nH':>6}  {'dcF':>5}")
    print('  ' + '─' * 80)

    for word in words:
        s = fourier_summary(word, rule, width)
        P2 = s['period']
        if P2 == 1:
            print(f"  {word.upper():12s}  {P2:>3}  {'—':26s}  "
                  f"{'—':>6}  {'—':>5}  {'—':>3}  {'—':>6}  {'—':>5}")
            continue
        hist_str = str(dict(sorted(s['dom_freq_hist'].items())))
        print(f"  {word.upper():12s}  {P2:>3}  {hist_str:26s}  "
              f"{s['mean_spec_entropy']:>6.3f}  {s['max_spec_entropy']:>5.3f}  "
              f"{s['n_nyquist_dom']:>3}  "
              f"{s['mean_nh_sp']:>6.4f}  {s['mean_dc_frac']:>5.4f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DFT spectral analysis of Q6 CA cell temporal sequences')
    parser.add_argument('--word',     metavar='WORD', default='РАБОТА')
    parser.add_argument('--rule',     choices=list(RULES), default='xor3')
    parser.add_argument('--table',    action='store_true')
    parser.add_argument('--json',     action='store_true')
    parser.add_argument('--width',    type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = fourier_summary(args.word.upper(), args.rule, args.width)
        print(json.dumps(fourier_dict(s), ensure_ascii=False, indent=2))
    elif args.table:
        print_fourier_table(rule=args.rule, width=args.width, color=_color)
    else:
        print_fourier(args.word.upper(), args.rule, args.width, _color)
