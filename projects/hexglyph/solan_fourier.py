"""
solan_fourier.py — DFT & Power Spectral Density of Q6 CA Attractor.

For each cell's temporal series on the attractor (exactly one period P),
the Discrete Fourier Transform reveals the frequency-domain structure:

    X[k]  =  Σ_{t=0}^{P−1} x[t] · exp(−2πi·k·t/P)   k = 0,1,…,P−1

One-sided power spectrum (k = 0 … ⌊P/2⌋):

    S[k]  =  |X[k]|² / P

For purely periodic sequences the DFT is exact (no spectral leakage):
energy concentrates at harmonics k = 1/P, 2/P, … of the fundamental.

Derived quantities
──────────────────
  Spectral entropy   H_sp = −Σ p[k] log₂ p[k]
                     p[k] = S[k] / Σ S[k]
                     nH_sp = H_sp / log₂(K+1)  ∈ [0,1]   K = ⌊P/2⌋
                     0 → tonal (one dominant frequency)
                     1 → spectrally flat (uniform power across bins)

  Spectral flatness  SF = exp(Σ log S[k] / (K+1)) / mean(S)  ∈ [0,1]
                     Wiener entropy.  0 → all energy at one bin, 1 → white.

  DC fraction        dc_frac = S[0] / Σ S[k]
                     Fraction of total energy in the mean component.

  Dominant harmonic  k* = argmax_{k≥1} S[k]   (DC excluded)
                     Effective period = P / k*.

Key results  (m=3, width=16)
─────────────────────────────
  XOR  ТУМАН  (P=1)  : all cells → S=[S[0]],  nH_sp=0  (one bin, no spread)
  AND/OR fixed-point : same
  ГОРА AND    (P=2)  : two bins with nearly equal power → nH_sp ≈ 0.999
  ТУМАН XOR3  (P=8)  : DC dominates (88–99%),  mean nH_sp ≈ 0.297

Functions
─────────
  dft1(series)                          → list[complex]
  power_spectrum(series)                → list[float]   one-sided
  spectral_entropy(power)               → float  H_sp in bits
  normalised_spectral_entropy(power)    → float  nH_sp ∈ [0,1]
  spectral_flatness(power)              → float  ∈ [0,1]
  dominant_harmonic(power)              → int    k* ≥ 1
  cell_fourier(series)                  → dict
  fourier_profile(word, rule, width)    → list[dict]  per-cell
  fourier_dict(word, rule, width)       → dict
  all_fourier(word, width)              → dict[str, dict]
  build_fourier_data(words, width)      → dict
  print_fourier(word, rule, color)      → None
  print_fourier_stats(words, color)     → None

Запуск
──────
  python3 -m projects.hexglyph.solan_fourier --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_fourier --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_fourier --stats --no-color
"""

from __future__ import annotations
import math
import cmath
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


# ── DFT ────────────────────────────────────────────────────────────────────────

def dft1(series: list[int | float]) -> list[complex]:
    """Exact DFT of *series* (naive O(N²) — fine for small periods ≤ 128)."""
    N = len(series)
    if N == 0:
        return []
    return [
        sum(series[t] * cmath.exp(-2j * math.pi * k * t / N) for t in range(N))
        for k in range(N)
    ]


def power_spectrum(series: list[int | float]) -> list[float]:
    """One-sided normalised power spectrum: S[k] = |X[k]|² / N.

    Returns K+1 values for k = 0, 1, …, ⌊N/2⌋.
    """
    N = len(series)
    if N == 0:
        return []
    X = dft1(series)
    return [(X[k].real ** 2 + X[k].imag ** 2) / N for k in range(N // 2 + 1)]


# ── Spectral statistics ────────────────────────────────────────────────────────

def spectral_entropy(power: list[float]) -> float:
    """Spectral entropy H_sp = −Σ p[k] log₂ p[k] in bits.

    Returns 0.0 for empty or all-zero power spectrum.
    """
    total = sum(power)
    if total <= 0:
        return 0.0
    return max(0.0, -sum(
        (v / total) * math.log2(v / total)
        for v in power if v > 0
    ))


def normalised_spectral_entropy(power: list[float]) -> float:
    """Spectral entropy normalised by log₂(K+1), K+1 = len(power) ∈ [0, 1]."""
    K1 = len(power)
    if K1 <= 1 or sum(power) <= 0:
        return 0.0
    denom = math.log2(K1)
    return round(min(spectral_entropy(power) / denom, 1.0), 8) if denom > 0 else 0.0


def spectral_flatness(power: list[float]) -> float:
    """Wiener entropy (spectral flatness) = geometric_mean / arithmetic_mean ∈ [0, 1].

    Returns 0.0 if any component is zero (perfectly tonal).
    """
    n = len(power)
    if n == 0:
        return 0.0
    total = sum(power)
    if total <= 0 or any(v <= 0 for v in power):
        return 0.0
    log_mean = sum(math.log(v) for v in power) / n
    arith_mean = total / n
    return round(min(math.exp(log_mean) / arith_mean, 1.0), 8)


def dominant_harmonic(power: list[float]) -> int:
    """Index k* ≥ 1 with maximum power (DC at k=0 excluded).

    Returns 1 if spectrum has ≤ 1 bin.
    """
    if len(power) <= 1:
        return 1
    best_k, best_v = 1, power[1] if len(power) > 1 else 0.0
    for k in range(2, len(power)):
        if power[k] > best_v:
            best_v = power[k]
            best_k = k
    return best_k


# ── Per-cell analysis ──────────────────────────────────────────────────────────

def cell_fourier(series: list[int | float]) -> dict:
    """Full spectral analysis of a single temporal series (one attractor period)."""
    ps = power_spectrum(series)
    total = sum(ps)
    h_sp  = spectral_entropy(ps)
    nh_sp = normalised_spectral_entropy(ps)
    sf    = spectral_flatness(ps)
    dom   = dominant_harmonic(ps)
    dc    = ps[0] if ps else 0.0
    ac    = sum(ps[1:]) if len(ps) > 1 else 0.0
    dc_frac = round(dc / total, 8) if total > 0 else 1.0
    return {
        'n':          len(series),
        'power':      ps,
        'h_sp':       round(h_sp, 8),
        'nh_sp':      nh_sp,
        'sf':         sf,
        'dominant_k': dom,
        'dc':         round(dc, 6),
        'ac_total':   round(ac, 6),
        'dc_frac':    dc_frac,
    }


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_attractor_series(word: str, rule: str,
                          width: int) -> tuple[list[list[float]], int]:
    """Return (cell_series_list, period).  Each series = exactly one period."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P = len(orbit)
    if P == 0:
        return [[0.0]] * width, 0
    series = [[float(orbit[t][i]) for t in range(P)] for i in range(width)]
    return series, P


# ── Profile across cells ──────────────────────────────────────────────────────

def fourier_profile(word: str, rule: str,
                    width: int = _DEFAULT_W) -> list[dict]:
    """Per-cell spectral analysis (one period each)."""
    cell_series, _ = _get_attractor_series(word, rule, width)
    return [cell_fourier(s) for s in cell_series]


# ── Full dictionary ────────────────────────────────────────────────────────────

def fourier_dict(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full spectral analysis for one word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    cells = fourier_profile(word, rule, width)
    N = len(cells)

    mean_nh   = round(sum(c['nh_sp']   for c in cells) / N, 8)
    mean_sf   = round(sum(c['sf']      for c in cells) / N, 8)
    mean_dcf  = round(sum(c['dc_frac'] for c in cells) / N, 8)

    # Mean power spectrum (pad shorter spectra with zeros)
    max_bins = max(len(c['power']) for c in cells) if cells else 1
    mean_ps: list[float] = []
    for k in range(max_bins):
        vals = [c['power'][k] for c in cells if k < len(c['power'])]
        mean_ps.append(round(sum(vals) / len(vals), 8) if vals else 0.0)

    dom_k      = dominant_harmonic(mean_ps)
    eff_period = period // dom_k if dom_k > 0 and period > 0 else period

    return {
        'word':         word.upper(),
        'rule':         rule,
        'period':       period,
        'cell_fourier': cells,
        'mean_nh_sp':   mean_nh,
        'mean_sf':      mean_sf,
        'mean_dc_frac': mean_dcf,
        'mean_ps':      mean_ps,
        'dominant_k':   dom_k,
        'eff_period':   eff_period,
        'n_bins':       max_bins,
    }


def all_fourier(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """fourier_dict for all 4 rules."""
    return {rule: fourier_dict(word, rule, width) for rule in _RULES}


def build_fourier_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Aggregated spectral data for a list of words."""
    per_rule: dict[str, dict[str, dict]] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = fourier_dict(word, rule, width)
            per_rule[rule][word.upper()] = {k: d[k] for k in (
                'period', 'mean_nh_sp', 'mean_sf', 'mean_dc_frac',
                'dominant_k', 'eff_period', 'n_bins')}
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_BAR  = '█'
_SHD  = '░'


def _bar(v: float, w: int = 20) -> str:
    filled = round(min(max(v, 0.0), 1.0) * w)
    return _BAR * filled + _SHD * (w - filled)


def print_fourier(word: str = 'ТУМАН', rule: str = 'xor3',
                  color: bool = True) -> None:
    d = fourier_dict(word, rule)
    c = _RCOL.get(rule, '') if color else ''
    r = _RST if color else ''
    RULE = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    P = d['period']
    ps = d['mean_ps']
    s_max = max(ps) if ps else 1.0
    print(f'  {c}◈ Fourier/PSD  {word.upper()}  |  {RULE}  P={P}  '
          f'k*={d["dominant_k"]}  T_eff={d["eff_period"]}'
          f'  mean_nH_sp={d["mean_nh_sp"]:.4f}{r}')
    print('  ' + '─' * 66)
    print(f'  {"k":>4}  {"freq":>7}  {"mean_S[k]":>11}  bar')
    print('  ' + '─' * 66)
    for k, sk in enumerate(ps):
        freq = k / max(P, 1)
        bar  = _bar(sk / s_max if s_max > 0 else 0.0)
        tag  = '← DC' if k == 0 else ('← k*' if k == d['dominant_k'] else '')
        print(f'  {k:>4}  {freq:>7.4f}  {sk:>11.4f}  {bar}  {tag}')
    print(f'\n  mean_nH_sp={d["mean_nh_sp"]:.4f}  '
          f'mean_SF={d["mean_sf"]:.4f}  '
          f'mean_DC_frac={d["mean_dc_frac"]:.4f}')
    print()


def print_fourier_stats(words: list[str] | None = None,
                        color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import all_words
    if words is None:
        words = all_words()
    for word in words:
        for rule in _RULES:
            print_fourier(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='DFT & Power Spectral Density for Q6 CA attractors')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_fourier_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_fourier(args.word, rule, color)
    else:
        print_fourier(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
