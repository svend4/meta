"""solan_temporal.py — Per-Cell Temporal DFT of Q6 CA Attractor.

Each cell i of a Q6 CA has a temporal sequence x_i(0)…x_i(P−1) over one
period P of the attractor (Q6 values ∈ [0, 63]).  The Discrete Fourier
Transform (DFT) decomposes this sequence into frequency components,
revealing the hidden temporal oscillation structure of individual cells.

Complementary analyses
───────────────────────
  solan_spectral.py   : DFT in SPACE at fixed time  (N cells → freq, per row)
  solan_correlation.py: spatial autocorrelation in space (r(d) over cells)
  solan_temporal.py   : DFT in TIME for each cell   (P steps → freq, per cell)

Together these give a 2D spectro-temporal portrait:
  rows = cells  (axis 0 = space)
  cols = temporal frequency bins  (axis 1 = time frequency)

Key quantities per cell i
──────────────────────────
  S_i[k]   : normalized power at temporal frequency k/P  ∈ [0, 63²]
               k = 0 : DC  (S_i[0] = (time-mean of cell i)²)
               k = 1 : fundamental oscillation (period P, slowest AC)
               k = j : oscillation at period P/j
               k = P//2 : fastest oscillation (period 2 steps)
             Σ_k S_i[k] = time-mean-square of x_i(t)  (Parseval)

  H_s(i)   : spectral entropy (bits) = −Σ_k p_k log₂ p_k
               p_k = S_i[k] / Σ_k S_i[k]
               0 = single-frequency signal  (all power at one k)
               log₂(P//2+1) = flat spectrum (white temporal noise)

  k*(i)    : dominant oscillation frequency (argmax_{k≥1} S_i[k])
               0 if P=1 (no AC component) or if DC dominates

  DC(i)    : normalised DC fraction = S_i[0] / Σ_k S_i[k]
               1.0 = constant cell   (fixed point)
               0.0 = zero-mean oscillation

Expected results
────────────────
  XOR  ТУМАН (P=1) : S[0] = (mean)² = 0 (all-zero), H_s = 0
  XOR3 ГОРА  (P=2) : 8 constant cells (DC=1) + 8 alternating (k=1 dominates)
  XOR3 ТУМАН (P=8) : 5 freq bins; cell-varying spectral profiles; H_s ∈ (0, 2)
  AND/OR ТУМАН(P=1): trivially constant; H_s = 0

Functions
─────────
  cell_dft_power(seq)                   → list[float]   power spectrum (P//2+1)
  spectral_entropy_of(power)            → float          H_s in bits
  attractor_temporal_spectra(word, rule, width)  → N × (P//2+1) matrix
  temporal_dict(word, rule, width)      → dict
  all_temporal(word, width)             → dict[str, dict]
  build_temporal_data(words, width)     → dict
  print_temporal(word, rule, color)     → None
  print_temporal_stats(words, color)    → None

Запуск
──────
  python3 -m projects.hexglyph.solan_temporal --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_temporal --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_temporal --stats --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16


# ── DFT core ──────────────────────────────────────────────────────────────────

def cell_dft_power(seq: list[int | float]) -> list[float]:
    """One-sided normalized power spectrum of a real temporal sequence.

    Returns P//2 + 1 values  S[k] = |DFT[k]|² / P²  for k = 0 … P//2.

    Properties
    ──────────
    S[0] = (arithmetic mean of seq)²       (DC power)
    Σ_{k=0}^{P//2} S[k] ≈ mean_square(seq)   (Parseval, one-sided approx)
    k=1 : fundamental period-P oscillation  (slowest AC)
    k=P//2 : period-2 oscillation           (fastest, Nyquist)
    """
    n = len(seq)
    if n == 0:
        return []
    power: list[float] = []
    two_pi_over_n = 2.0 * math.pi / n
    for k in range(n // 2 + 1):
        re = sum(seq[j] * math.cos(-two_pi_over_n * k * j) for j in range(n))
        im = sum(seq[j] * math.sin(-two_pi_over_n * k * j) for j in range(n))
        power.append((re * re + im * im) / (n * n))
    return power


def spectral_entropy_of(power: list[float]) -> float:
    """Spectral entropy (bits) of a power spectrum.

    H_s = −Σ_k p_k log₂ p_k  where  p_k = S[k] / Σ_k S[k].
    Returns 0.0 if total power is zero (constant sequence).
    """
    total = sum(power)
    if total <= 0.0:
        return 0.0
    return max(0.0, -sum(
        (p / total) * math.log2(p / total)
        for p in power if p > 0.0
    ))


# ── Attractor extraction ───────────────────────────────────────────────────────

def attractor_temporal_spectra(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[list[float]]:
    """Per-cell power spectra: N × (P//2+1) matrix.

    Row i = one-sided power spectrum of cell i's temporal Q6 sequence
    over one full attractor period.  All rows have the same length.
    """
    from projects.hexglyph.solan_traj import word_trajectory
    traj       = word_trajectory(word.upper(), rule, width)
    transient  = traj['transient']
    period     = max(traj['period'], 1)
    attractor  = traj['rows'][transient:transient + period]   # P × N

    spectra: list[list[float]] = []
    for i in range(width):
        seq = [attractor[t][i] for t in range(period)]
        spectra.append(cell_dft_power(seq))
    return spectra


# ── Full analysis ──────────────────────────────────────────────────────────────

def temporal_dict(
    word:  str,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
) -> dict:
    """Full per-cell temporal spectral analysis for one word × rule.

    Returns:
        word, rule, period, n_freqs
        spectra           : list[list[float]]  N × (P//2+1) power matrix
        spectral_entropy  : list[float]        per-cell H_s
        dominant_freq     : list[int]          per-cell argmax_{k≥1} S_i[k]
        dc_fraction       : list[float]        S_i[0] / total_S per cell
        total_power       : list[float]        Σ_k S_i[k] per cell
        mean_spec_entropy : float              mean H_s over all cells
        mean_dc           : float              mean DC fraction over all cells
        global_dominant   : int               most common dominant_freq across cells
    """
    from projects.hexglyph.solan_traj import word_trajectory
    traj    = word_trajectory(word.upper(), rule, width)
    period  = max(traj['period'], 1)
    spectra = attractor_temporal_spectra(word, rule, width)
    n_freqs = period // 2 + 1

    spec_ent:    list[float] = []
    dom_freqs:   list[int]   = []
    dc_fracs:    list[float] = []
    total_pows:  list[float] = []

    for s in spectra:
        total = sum(s)
        total_pows.append(round(total, 6))
        dc_fracs.append(round(s[0] / total if total > 0 else 0.0, 6))
        spec_ent.append(round(spectral_entropy_of(s), 6))
        if len(s) <= 1:
            dom_freqs.append(0)
        else:
            dom_freqs.append(max(range(1, len(s)), key=lambda k: s[k]))

    cnt        = Counter(dom_freqs)
    global_dom = cnt.most_common(1)[0][0] if cnt else 0
    mean_h     = round(sum(spec_ent) / len(spec_ent), 6) if spec_ent else 0.0
    mean_dc    = round(sum(dc_fracs) / len(dc_fracs), 6) if dc_fracs else 0.0

    return {
        'word':             word.upper(),
        'rule':             rule,
        'period':           period,
        'n_freqs':          n_freqs,
        'spectra':          spectra,
        'spectral_entropy': spec_ent,
        'dominant_freq':    dom_freqs,
        'dc_fraction':      dc_fracs,
        'total_power':      total_pows,
        'mean_spec_entropy': mean_h,
        'mean_dc':          mean_dc,
        'global_dominant':  global_dom,
    }


def all_temporal(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict]:
    """temporal_dict for all 4 rules."""
    return {r: temporal_dict(word, r, width) for r in _ALL_RULES}


def build_temporal_data(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
) -> dict:
    """Temporal spectral summary across the lexicon × 4 rules.

    Returns:
        words, width,
        per_rule: {rule: {word: {period, mean_spec_entropy, mean_dc, global_dominant}}}
    """
    words = words if words is not None else list(LEXICON)
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = temporal_dict(word, rule, width)
            per_rule[rule][word] = {
                'period':            d['period'],
                'mean_spec_entropy': d['mean_spec_entropy'],
                'mean_dc':           d['mean_dc'],
                'global_dominant':   d['global_dominant'],
            }
    return {'words': words, 'width': width, 'per_rule': per_rule}


# ── ASCII / ANSI display ───────────────────────────────────────────────────────

_SHADE = ' ░▒▓█'


def print_temporal(
    word:  str  = 'ТУМАН',
    rule:  str  = 'xor3',
    color: bool = True,
) -> None:
    """Print per-cell temporal power spectra as an ASCII heatmap."""
    d    = temporal_dict(word, rule)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    P = d['period']
    print(f"{bold}  ◈ Temporal DFT  {word.upper()}  |  "
          f"{col}{name}{rst}  P={P}  "
          f"H_s={d['mean_spec_entropy']:.4f}  DC̄={d['mean_dc']:.3f}  "
          f"k*={d['global_dominant']}")
    print(f"  {'─' * 54}")

    # Header: frequency bins
    n_freqs = d['n_freqs']
    hdr = '  cell  ' + ''.join(f'  k={k}' for k in range(n_freqs))
    print(hdr)
    print(f"  {'─' * 54}")

    spectra = d['spectra']
    max_s = max((max(s) if s else 0) for s in spectra) or 1.0

    for i, s in enumerate(spectra):
        bar = ''
        for k in range(n_freqs):
            frac = s[k] / max_s if max_s > 0 else 0
            idx  = min(int(frac * (len(_SHADE) - 1)), len(_SHADE) - 1)
            bar += f'  {col if color else ""}{_SHADE[idx]}{rst if color else ""}    '
        h_s = d['spectral_entropy'][i]
        kstar = d['dominant_freq'][i]
        print(f"  {i:>4}  {bar}  H={h_s:.3f}  k*={kstar}")
    print()


def print_temporal_stats(
    words: list[str] | None = None,
    color: bool             = True,
) -> None:
    """Table: mean spectral entropy per word × rule."""
    words = words if words is not None else list(LEXICON)
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>9s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Средняя спектральная энтропия Hs(k) по лексикону{rst}")
    print('  ' + '─' * (len(header) + 2))
    print('  ' + header)
    print('  ' + '─' * (len(header) + 2))
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = temporal_dict(word, rule)
            v   = d['mean_spec_entropy']
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{v:>9.4f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Per-Cell Temporal DFT — Q6 CA')
    parser.add_argument('--word',      default='ТУМАН')
    parser.add_argument('--rule',      default='xor3', choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_temporal_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_temporal(args.word, rule, color)
    else:
        print_temporal(args.word, args.rule, color)


if __name__ == '__main__':
    _main()
