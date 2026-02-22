"""solan_coverage.py — Q6 Value Coverage of CA Attractor Orbits.

For a period-P attractor of width N=16, the full orbit comprises P×N
Q6 cell-value observations.  This module answers: which of the 64 possible
Q6 phonetic codes actually appear, and with what frequency?

    vocab(word, rule)    = {orbit[t][i] : t < P, i < N}    ⊆ {0…63}
    coverage(word, rule) = |vocab| / 64

Key discoveries
──────────────────────────────────────────────────────────────────────
  МАТ   XOR3  (P=8, N=16, 128 observations)
      vocab = {23, 24, 48, 63}  — only 4/64 values!
      Value 23 (=T+B+L+D1) appears 64/128 = 50.0% of all observations.
      All 4 vocabulary values have D1=1 — consistent with the frozen
      D1 bit-plane discovered in solan_bitplane.py.
      Minimum vocabulary in the entire lexicon under XOR3.

  РАБОТА XOR3  (P=8, N=16)
      vocab has 16 distinct values — maximum in the lexicon.

  XOR rule  (all 49 words):
      Every word under XOR converges to the all-zero fixed point.
      Vocabulary = {0} for ALL words — only one Q6 value ever appears.

  XOR3 rule — 4 Q6 values NEVER appear across the entire lexicon:
      6  = B+L   (bilabial + labial, no apical/dorsal/diameter)
      10 = B+R   (bilabial + dorsal)
      45 = T+L+R+D2
      58 = B+R+D1+D2
      These 4 phonetic combinations are structurally forbidden by XOR3.

  AND rule: only 14/64 values ever appear (AND contracts toward 0).
  OR  rule: only 13/64 values ever appear (OR expands toward 63).

Запуск:
    python3 -m projects.hexglyph.solan_coverage --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_coverage --word РАБОТА --rule xor3
    python3 -m projects.hexglyph.solan_coverage --rule xor3 --global
    python3 -m projects.hexglyph.solan_coverage --table --no-color
    python3 -m projects.hexglyph.solan_coverage --json --word МАТ
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES          = tuple(_ALL_RULES)
_DEFAULT_WIDTH = 16
_Q6_N          = 64
_BIT_NAMES     = ['T', 'B', 'L', 'R', 'D1', 'D2']


# ── Phonetic helpers ──────────────────────────────────────────────────────────

def q6_label(v: int) -> str:
    """Return '+'-joined phonetic feature names for Q6 value v."""
    parts = [_BIT_NAMES[b] for b in range(6) if (v >> b) & 1]
    return '+'.join(parts) if parts else '0'


# ── Core computation ──────────────────────────────────────────────────────────

def orbit_frequencies(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> Counter:
    """Counter of Q6 values across the full P×N orbit."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word, rule, width)
    cnt: Counter = Counter()
    for state in orbit:
        cnt.update(int(v) for v in state)
    return cnt


# ── Per-word summary ──────────────────────────────────────────────────────────

def coverage_summary(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Q6 coverage analysis for one word/rule.

    Keys
    ────
    word, rule, period, n_cells, orbit_size (=P×N)

    freq            : dict{v→count}  — Q6 value frequencies across orbit
    vocab           : list[int]      — sorted distinct Q6 values
    n_distinct      : int            — |vocab|
    coverage        : float          — n_distinct / 64
    most_common     : list[(v,count,frac)]  — top values by frequency
    dominant_val    : int            — most frequent Q6 value
    dominant_count  : int            — its occurrence count
    dominant_frac   : float          — its fraction of total observations
    never_seen      : list[int]      — Q6 values absent from orbit
    n_never_seen    : int            — 64 − n_distinct
    vocab_labels    : list[str]      — phonetic feature strings for vocab

    # Per-orbit-step coverage
    step_vocab      : list[set[int]]  — distinct values at each orbit step
    step_n_distinct : list[int]       — per-step vocab size
    step_mode       : list[int]       — most common value per step
    step_mode_count : list[int]       — its count per step
    min_step_n_distinct : int
    max_step_n_distinct : int
    """
    from projects.hexglyph.solan_perm import get_orbit

    orbit = get_orbit(word, rule, width)
    P     = len(orbit)
    N     = width
    size  = P * N

    # Global frequencies
    freq = Counter()
    for state in orbit:
        freq.update(int(v) for v in state)

    vocab   = sorted(freq)
    n_dist  = len(vocab)
    dom_v, dom_c = freq.most_common(1)[0]
    top3 = [(v, c, round(c / size, 4)) for v, c in freq.most_common(5)]

    # Per-step
    step_vocab  = [sorted(set(int(v) for v in state)) for state in orbit]
    step_ndist  = [len(sv) for sv in step_vocab]
    step_modes  = []
    step_mcount = []
    for state in orbit:
        sc = Counter(int(v) for v in state)
        mv, mc = sc.most_common(1)[0]
        step_modes.append(mv)
        step_mcount.append(mc)

    return {
        'word':             word,
        'rule':             rule,
        'period':           P,
        'n_cells':          N,
        'orbit_size':       size,

        'freq':             dict(sorted(freq.items())),
        'vocab':            vocab,
        'n_distinct':       n_dist,
        'coverage':         round(n_dist / _Q6_N, 4),
        'most_common':      top3,
        'dominant_val':     dom_v,
        'dominant_count':   dom_c,
        'dominant_frac':    round(dom_c / size, 4),
        'never_seen':       [v for v in range(_Q6_N) if v not in freq],
        'n_never_seen':     _Q6_N - n_dist,
        'vocab_labels':     [q6_label(v) for v in vocab],

        'step_vocab':       step_vocab,
        'step_n_distinct':  step_ndist,
        'step_mode':        step_modes,
        'step_mode_count':  step_mcount,
        'min_step_n_distinct': min(step_ndist),
        'max_step_n_distinct': max(step_ndist),
    }


def global_coverage(
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Q6 values appearing / never appearing across the full lexicon."""
    from projects.hexglyph.solan_lexicon import LEXICON
    from projects.hexglyph.solan_perm import get_orbit

    total: Counter = Counter()
    for word in LEXICON:
        orbit = get_orbit(word, rule, width)
        for state in orbit:
            total.update(int(v) for v in state)

    seen   = sorted(total)
    absent = [v for v in range(_Q6_N) if v not in total]
    return {
        'rule':       rule,
        'n_words':    len(list(LEXICON)),
        'n_seen':     len(seen),
        'n_absent':   len(absent),
        'seen':       seen,
        'absent':     absent,
        'absent_labels': [q6_label(v) for v in absent],
        'frequencies': dict(sorted(total.items())),
        'most_common': total.most_common(10),
    }


def all_coverage(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict[str, Any]]:
    """Coverage summary for all 4 CA rules."""
    return {r: coverage_summary(word, r, width) for r in RULES}


def build_coverage_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Full Q6 coverage analysis for the lexicon."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)
    return {
        'words': list(words),
        'data':  {w: {r: coverage_summary(w, r, width) for r in RULES}
                  for w in words},
    }


def coverage_dict(s: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of coverage_summary."""
    d = dict(s)
    d['step_vocab'] = [list(sv) for sv in s['step_vocab']]
    return d


# ── Terminal output ───────────────────────────────────────────────────────────

def print_coverage(
    word:  str,
    rule:  str,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print Q6 coverage analysis for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    s = coverage_summary(word, rule, width)
    P = s['period']
    N = s['n_cells']

    print(bold + f"  ◈ Q6 Coverage  {word.upper()}  "
          + col + lbl + reset + bold + f"  (P={P})" + reset)
    print()
    print(f"  Orbit size: {s['orbit_size']} observations ({P}×{N})")
    print(f"  Vocabulary: {s['n_distinct']}/64 distinct Q6 values  "
          f"(coverage = {s['coverage']:.2%})")
    print()

    # Frequency bar chart — only show values that appear
    max_cnt  = max(s['freq'].values()) if s['freq'] else 1
    bar_max  = 30
    dom_col  = '\033[38;5;220m' if color else ''

    print(f"  Q6 value frequencies (top {min(16, s['n_distinct'])} by count):")
    for v, cnt in sorted(s['freq'].items(), key=lambda x: -x[1])[:16]:
        bar_len = max(1, round(cnt / max_cnt * bar_max))
        bar     = '█' * bar_len
        frac    = cnt / s['orbit_size']
        label   = q6_label(v)
        vc      = dom_col if v == s['dominant_val'] else dim
        print(f"  {vc}  v={v:2d} ({label:14s}) {bar:<30s} {cnt:3d} ({frac:.1%}){reset}")

    print()

    # Per-step summary
    if P > 1:
        print(f"  Per-orbit-step:")
        print(f"    {'t':2s}  {'n_distinct':10s}  {'mode':5s}  mode_count")
        print('    ' + '─' * 38)
        for t in range(P):
            print(f"    t{t}  {s['step_n_distinct'][t]:10d}  "
                  f"{s['step_mode'][t]:5d}  {s['step_mode_count'][t]}")
        print()

    print(f"  Dominant value: {s['dominant_val']} ({q6_label(s['dominant_val'])})"
          f"  — {s['dominant_count']}/{s['orbit_size']} ({s['dominant_frac']:.1%})")
    print(f"  Never seen    : {len(s['never_seen'])} values"
          + (f"  {s['never_seen'][:8]}" if s['never_seen'] else "  (all Q6 values appear)"))
    print()


def print_global_coverage(
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print global coverage across the full lexicon."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    g = global_coverage(rule, width)

    print(bold + f"  ◈ Global Q6 Coverage  {col}{lbl}{reset}  "
          + bold + f"(all {g['n_words']} lexicon words)" + reset)
    print()
    print(f"  Q6 values seen   : {g['n_seen']}/64")
    print(f"  Q6 values absent : {g['n_absent']} "
          + (f"→ {list(zip(g['absent'], g['absent_labels']))}"
             if g['absent'] else "(none)"))
    print()
    print(f"  Top-10 most frequent:")
    total = sum(g['frequencies'].values())
    for v, cnt in g['most_common']:
        lbl2 = q6_label(v)
        print(f"    v={v:2d} ({lbl2:14s}) {cnt:5d}  ({cnt/total:.1%})")
    print()


def print_coverage_table(
    words: list[str] | None = None,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Summary table: Q6 coverage for all lexicon words."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Q6 Coverage Summary ({lbl}, n={len(words)})" + reset)
    print()
    print(f"  {'Слово':12s}  {'P':>3}  {'n_dist':>6}  {'cov%':>5}  "
          f"{'dom_v':>5}  {'dom%':>5}  Vocab (first 6)")
    print('  ' + '─' * 74)

    for word in words:
        s  = coverage_summary(word, rule, width)
        vl = ' '.join(str(v) for v in s['vocab'][:6])
        if s['n_distinct'] > 6:
            vl += '…'
        print(f"  {word.upper():12s}  {s['period']:>3}  "
              f"{s['n_distinct']:>6}  {s['coverage']:>5.1%}  "
              f"{s['dominant_val']:>5}  {s['dominant_frac']:>5.1%}  "
              f"{dim}{vl}{reset}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Q6 value coverage of CA attractor orbits')
    parser.add_argument('--word',   metavar='WORD', default='МАТ')
    parser.add_argument('--rule',   choices=list(RULES), default='xor3')
    parser.add_argument('--table',  action='store_true')
    parser.add_argument('--global', action='store_true', dest='global_mode')
    parser.add_argument('--json',   action='store_true')
    parser.add_argument('--width',  type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = coverage_summary(args.word.upper(), args.rule, args.width)
        print(json.dumps(coverage_dict(s), ensure_ascii=False, indent=2))
    elif args.global_mode:
        print_global_coverage(args.rule, args.width, color=_color)
    elif args.table:
        print_coverage_table(rule=args.rule, width=args.width, color=_color)
    else:
        print_coverage(args.word.upper(), args.rule, args.width, color=_color)
