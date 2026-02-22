"""solan_portrait.py — Multi-Dimensional Word Portrait (Radar Fingerprint).

Aggregates 8 normalised metrics from across all analysis modules into a
single fingerprint vector for each word × rule combination.

Axes (all normalised to [0, 1])
────────────────────────────────
  ic_entropy    IC spatial Shannon entropy         / log₂(N)  ≈ 4 bits
  period        attractor period                   / 8
  transient     attractor transient length         / 8
  complexity    LZ76 of attractor trajectory       (already [0,1])
  topological_h Binary topological entropy         (already [0,1])
  te_flow       Total Transfer Entropy             / 300
  sensitivity   Peak Lyapunov divergence mean      / 96
  autocorr      Spatial autocorrelation lag-1      → (1+r)/2

Portrait similarity
───────────────────
  portrait_distance(w1, w2)  normalised L2 distance in 8-dim space
  portrait_cosine(w1, w2)    cosine similarity (1 = identical profile)

Functions
─────────
  portrait_dict(word, rule, width)         → dict with 'metrics' vector
  portrait_compare(word1, word2, rule)     → dict with both + distance
  build_portrait_data(words, rule)         → lexicon summary + ranking
  print_portrait(word, rule, color)        → ASCII bar-chart radar
  print_portrait_ranking(words, color)     → sorted by LZ76 axis

Запуск
──────
  python3 -m projects.hexglyph.solan_portrait --word ТУМАН
  python3 -m projects.hexglyph.solan_portrait --compare ТУМАН ГОРА
  python3 -m projects.hexglyph.solan_portrait --ranking --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    find_orbit,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_entropy import entropy
from projects.hexglyph.solan_complexity import trajectory_complexity
from projects.hexglyph.solan_symbolic import symbolic_dict
from projects.hexglyph.solan_network import network_dict
from projects.hexglyph.solan_lyapunov import lyapunov_profile
from projects.hexglyph.solan_correlation import attractor_autocorr

_ALL_RULES    = ['xor', 'xor3', 'and', 'or']
_DEFAULT_RULE = 'xor3'
_DEFAULT_W    = 16

# Axis names and normalisation denominators
_AXES   = [
    'ic_entropy', 'period', 'transient', 'complexity',
    'topological_h', 'te_flow', 'sensitivity', 'autocorr',
]
_LABELS = ['IC-H', 'Period', 'Trans.', 'LZ76', 'Topo-h', 'TE', 'Lyap.', 'Autocorr']
_N_AXES = len(_AXES)

_MAX_IC_H   = math.log2(16)   # 4.0 bits for N=16
_MAX_PERIOD = 8.0
_MAX_TRANS  = 8.0
_MAX_TE     = 300.0
_MAX_LYA    = 96.0             # N × 6 bits theoretical max


# ── Core portrait computation ─────────────────────────────────────────────────

def portrait_dict(
    word:  str,
    rule:  str = _DEFAULT_RULE,
    width: int = _DEFAULT_W,
) -> dict:
    """8-axis normalised fingerprint of one word × rule.

    Returns dict:
        word, rule, width
        axes      : list[str]    axis identifiers
        labels    : list[str]    short axis labels for display
        metrics   : list[float]  8 normalised values ∈ [0, 1]
        Raw values: ic_entropy, period, transient, complexity,
                    topological_h, te_flow, sensitivity, autocorr
        Normalised (suffix _n): same 8 with _n suffix
    """
    word  = word.upper()
    cells = pad_to(encode_word(word), width)
    transient, period = find_orbit(cells[:], rule)
    period = max(period, 1)

    # 1 ── IC spatial Shannon entropy
    ic_h   = entropy(cells)
    ic_h_n = min(ic_h / _MAX_IC_H, 1.0)

    # 2 ── period
    p_n = min(period / _MAX_PERIOD, 1.0)

    # 3 ── transient
    t_n = min(transient / _MAX_TRANS, 1.0)

    # 4 ── LZ76 complexity of attractor trajectory
    tc  = trajectory_complexity(word, rule, width)
    lz  = tc['traj_norm']              # already ∈ [0, 1]

    # 5 ── topological entropy (binary symbolic dynamics)
    sd   = symbolic_dict(word, rule, width)
    topo = sd['topological_h']         # already ∈ [0, 1]

    # 6 ── total Transfer Entropy
    nd      = network_dict(word, rule, width)
    te_tot  = nd['total_te']
    te_n    = min(te_tot / _MAX_TE, 1.0)

    # 7 ── peak Lyapunov divergence mean
    lp        = lyapunov_profile(word, rule, width)
    peak_mean = lp['peak_mean']
    lyap_n    = min(peak_mean / _MAX_LYA, 1.0)

    # 8 ── spatial autocorrelation at lag 1 → map from [-1,1] to [0,1]
    ac      = attractor_autocorr(word, rule, width)
    acf1    = ac[1] if len(ac) > 1 else 0.0
    acf1_n  = (acf1 + 1.0) / 2.0

    metrics = [ic_h_n, p_n, t_n, lz, topo, te_n, lyap_n, acf1_n]

    return {
        'word':  word, 'rule': rule, 'width': width,
        'axes':  _AXES, 'labels': _LABELS,
        'metrics': metrics,
        # raw
        'ic_entropy': round(ic_h, 6),
        'period':     period,
        'transient':  transient,
        'complexity': round(lz, 6),
        'topological_h': round(topo, 6),
        'te_flow':    round(te_tot, 4),
        'sensitivity':round(peak_mean, 4),
        'autocorr':   round(acf1, 6),
        # normalised
        'ic_entropy_n':    round(ic_h_n,  6),
        'period_n':        round(p_n,     6),
        'transient_n':     round(t_n,     6),
        'complexity_n':    round(lz,      6),
        'topological_h_n': round(topo,    6),
        'te_flow_n':       round(te_n,    6),
        'sensitivity_n':   round(lyap_n,  6),
        'autocorr_n':      round(acf1_n,  6),
    }


# ── Portrait geometry ─────────────────────────────────────────────────────────

def _l2_distance(m1: list[float], m2: list[float]) -> float:
    """Normalised L2 distance between two metric vectors."""
    return round(
        math.sqrt(sum((a - b) ** 2 for a, b in zip(m1, m2)) / len(m1)),
        8,
    )


def _cosine_sim(m1: list[float], m2: list[float]) -> float:
    """Cosine similarity between two metric vectors."""
    dot = sum(a * b for a, b in zip(m1, m2))
    n1  = math.sqrt(sum(v ** 2 for v in m1)) or 1.0
    n2  = math.sqrt(sum(v ** 2 for v in m2)) or 1.0
    return round(dot / (n1 * n2), 8)


def portrait_compare(
    word1: str,
    word2: str,
    rule:  str = _DEFAULT_RULE,
    width: int = _DEFAULT_W,
) -> dict:
    """Compare two word portraits.

    Returns dict: word1, word2, rule, portrait1, portrait2,
                  l2_distance, cosine_sim.
    """
    d1 = portrait_dict(word1, rule, width)
    d2 = portrait_dict(word2, rule, width)
    return {
        'word1': d1['word'], 'word2': d2['word'], 'rule': rule,
        'portrait1':   d1,
        'portrait2':   d2,
        'l2_distance': _l2_distance(d1['metrics'], d2['metrics']),
        'cosine_sim':  _cosine_sim(d1['metrics'], d2['metrics']),
    }


# ── Lexicon summary ───────────────────────────────────────────────────────────

def build_portrait_data(
    words: list[str] | None = None,
    rule:  str               = _DEFAULT_RULE,
    width: int               = _DEFAULT_W,
) -> dict:
    """Portrait data for the full lexicon.

    Returns dict:
        words, rule, axes, labels
        portraits  : {word: metrics_list}
        ranking    : {axis: [word_sorted_desc]}
    """
    words = words if words is not None else list(LEXICON)
    portraits = {w: portrait_dict(w, rule, width) for w in words}
    ranking: dict[str, list[str]] = {}
    for ax, ax_n in zip(_AXES, [a + '_n' for a in _AXES]):
        ranking[ax] = sorted(words, key=lambda w: -portraits[w][ax_n])
    return {
        'words':     words,
        'rule':      rule,
        'axes':      _AXES,
        'labels':    _LABELS,
        'portraits': {w: d['metrics'] for w, d in portraits.items()},
        'ranking':   ranking,
    }


# ── ASCII display ─────────────────────────────────────────────────────────────

_BAR_W = 20


def print_portrait(
    word:  str  = 'ТУМАН',
    rule:  str  = _DEFAULT_RULE,
    color: bool = True,
) -> None:
    d    = portrait_dict(word, rule)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    print(f"{bold}  ◈ Портрет Q6  {word.upper()}  |  {col}{name}{rst}"
          f"  P={d['period']}  T={d['transient']}")
    print(f"  {'─' * 48}")
    for ax, lbl, v in zip(_AXES, _LABELS, d['metrics']):
        filled = round(v * _BAR_W)
        bar    = '█' * filled + '░' * (_BAR_W - filled)
        raw    = d[ax]
        print(f"  {lbl:>9s}  {col if color else ''}{bar}{rst}  {v:.3f}  "
              f"(raw={raw})")
    print()


def print_portrait_ranking(
    words: list[str] | None = None,
    rule:  str              = _DEFAULT_RULE,
    color: bool             = True,
) -> None:
    """Table of all words ranked by LZ76 complexity."""
    words = words if words is not None else list(LEXICON)
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    col   = _RULE_COLOR.get(rule, '') if color else ''

    print(f"\n{bold}  ◈ Рейтинг портрета Q6  {col}{_RULE_NAMES.get(rule, rule)}{rst}"
          f"  (сортировка по LZ76){rst}")
    header = (f"{'Слово':10s}  " +
              '  '.join(f"{lbl:>8s}" for lbl in _LABELS))
    print('  ' + '─' * len(header))
    print('  ' + header)
    print('  ' + '─' * len(header))
    portraits = {w: portrait_dict(w, rule) for w in words}
    for word in sorted(words, key=lambda w: -portraits[w]['complexity_n']):
        d = portraits[word]
        print('  ' + f"{word:10s}  " +
              '  '.join(f"{col}{v:>8.3f}{rst}" for v in d['metrics']))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Word Portrait Q6 CA')
    parser.add_argument('--word',    default='ТУМАН')
    parser.add_argument('--compare', nargs=2, metavar=('W1', 'W2'))
    parser.add_argument('--rule',    default=_DEFAULT_RULE, choices=_ALL_RULES)
    parser.add_argument('--ranking', action='store_true')
    parser.add_argument('--no-color',action='store_true')
    args  = parser.parse_args()
    color = not args.no_color

    if args.compare:
        c = portrait_compare(args.compare[0], args.compare[1], args.rule)
        print_portrait(c['word1'], args.rule, color)
        print_portrait(c['word2'], args.rule, color)
        print(f"  L2-distance={c['l2_distance']:.4f}  "
              f"cosine_sim={c['cosine_sim']:.4f}")
    elif args.ranking:
        print_portrait_ranking(color=color, rule=args.rule)
    else:
        print_portrait(args.word, args.rule, color)


if __name__ == '__main__':
    _main()
