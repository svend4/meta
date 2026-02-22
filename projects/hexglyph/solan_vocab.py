"""
solan_vocab.py — Orbit Vocabulary of Q6 CA Attractors.

The *orbit vocabulary* is the set of distinct Q6 values (∈ {0..63}) that
appear in ANY cell at ANY step of the attractor orbit:

    V = { orbit[t][i]  :  t ∈ {0..P−1}, i ∈ {0..N−1} }

vocab_size     = |V|
vocab_coverage = |V| / 64   (fraction of all possible Q6 values used)

We also compute:
  value_hist   — for each v ∈ V: total count over all (t, i) pairs
  uniform_dist — True when all vocab values appear equally often
  common_bits  — (always_1, always_0): bits that are fixed to 1 (or 0)
                  across ALL vocabulary values
  vocab_hamming — distribution of Hamming weights among vocab values

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      V = {0}  size=1  coverage=1.6%
      Bit profile: all bits always 0.

  ГОРА AND  (P=2, anti-phase {47, 1})
      V = {1, 47}  size=2  coverage=3.1%.
      47=0b101111, 1=0b000001: both have bit 0=1, bit 4=0.
      ★  common_always_1={0}  common_always_0={4}.
      Uniform: each value appears 16 times (P·N/|V| = 16).

  ГОРА XOR3  (P=2, 4 spatial clusters)
      V = {1,15,17,31,33,47,49,63}  size=8  coverage=12.5%.
      ★  ALL 8 values have bit 0=1  → common_always_1={0}.
      Uniform: each appears 4 times (P·N/|V| = 4).
      This directly echoes the bit-0-frozen finding from solan_edge.

  ТУМАН XOR3  (P=8)
      V = {0,3,12,15,20,23,24,27,36,40,43,48,51,60,63}  size=15
      coverage=23.4%.  Non-uniform: values 43 and 60 appear 13× each.
      No common bits (bit 0 is 0 in 8 values, 1 in 7 values).

Functions
─────────
  orbit_vocabulary(word, rule, width)     → list[int]
  vocab_size(word, rule, width)           → int
  vocab_coverage(word, rule, width)       → float
  value_hist(word, rule, width)           → dict[int, int]
  uniform_distribution(word, rule, width) → bool
  vocab_bit_profile(word, rule, width)    → list[float]
  common_bits(word, rule, width)          → tuple[set[int], set[int]]
  vocab_hamming_hist(word, rule, width)   → dict[int, int]
  vocab_summary(word, rule, width)        → dict
  all_vocab(word, width)                  → dict[str, dict]
  build_vocab_data(words, width)          → dict
  print_vocab(word, rule, color)          → None
  print_vocab_table(words, color)         → None

Запуск
──────
  python3 -m projects.hexglyph.solan_vocab --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_vocab --table --no-color
"""

from __future__ import annotations
import sys
import argparse
import math
from collections import Counter

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_Q6_TOTAL:  int = 64   # total possible Q6 values


# ── Core functions ─────────────────────────────────────────────────────────────

def orbit_vocabulary(word: str, rule: str,
                     width: int = _DEFAULT_W) -> list[int]:
    """Sorted list of distinct Q6 values appearing in the attractor orbit."""
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    return sorted({v for state in orbit for v in state})


def vocab_size(word: str, rule: str, width: int = _DEFAULT_W) -> int:
    """Number of distinct Q6 values in the orbit vocabulary."""
    return len(orbit_vocabulary(word, rule, width))


def vocab_coverage(word: str, rule: str, width: int = _DEFAULT_W) -> float:
    """Fraction of all 64 Q6 values used in the orbit (0..1)."""
    return round(vocab_size(word, rule, width) / _Q6_TOTAL, 6)


def value_hist(word: str, rule: str,
               width: int = _DEFAULT_W) -> dict[int, int]:
    """Frequency count of each Q6 value across all (t, i) cell-steps.

    Returns dict sorted by descending frequency.
    """
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    vals  = [v for state in orbit for v in state]
    cnt   = Counter(vals)
    return dict(sorted(cnt.items(), key=lambda x: -x[1]))


def uniform_distribution(word: str, rule: str,
                         width: int = _DEFAULT_W) -> bool:
    """True when every vocab value appears the same number of times."""
    hist = value_hist(word, rule, width)
    counts = list(hist.values())
    return len(set(counts)) == 1 if counts else True


def vocab_bit_profile(word: str, rule: str,
                      width: int = _DEFAULT_W) -> list[float]:
    """For each bit b ∈ {0..5}: fraction of vocab values with bit b set.

    Returns a 6-vector. Values near 0 or 1 signal frozen bits.
    """
    vocab = orbit_vocabulary(word, rule, width)
    if not vocab:
        return [0.0] * 6
    n = len(vocab)
    return [round(sum(1 for v in vocab if (v >> b) & 1) / n, 6)
            for b in range(6)]


def common_bits(word: str, rule: str,
                width: int = _DEFAULT_W) -> tuple[set[int], set[int]]:
    """Bits that are fixed across ALL vocab values.

    Returns (always_1, always_0) — sets of bit indices.
    always_1: bits that are 1 in every vocabulary value.
    always_0: bits that are 0 in every vocabulary value.
    """
    vocab = orbit_vocabulary(word, rule, width)
    if not vocab:
        return (set(), set(range(6)))
    always_1 = {b for b in range(6) if all((v >> b) & 1 for v in vocab)}
    always_0 = {b for b in range(6) if all(not ((v >> b) & 1) for v in vocab)}
    return (always_1, always_0)


def vocab_hamming_hist(word: str, rule: str,
                       width: int = _DEFAULT_W) -> dict[int, int]:
    """Hamming-weight histogram of vocab values: {hw → count_of_vocab_values}."""
    vocab = orbit_vocabulary(word, rule, width)
    hw_cnt: dict[int, int] = {}
    for v in vocab:
        w = bin(v & 63).count('1')
        hw_cnt[w] = hw_cnt.get(w, 0) + 1
    return dict(sorted(hw_cnt.items()))


# ── Summary ────────────────────────────────────────────────────────────────────

def vocab_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full vocabulary summary for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj    = word_trajectory(word.upper(), rule, width)
    period  = traj['period']
    vocab   = orbit_vocabulary(word, rule, width)
    size    = len(vocab)
    hist    = value_hist(word, rule, width)
    unif    = uniform_distribution(word, rule, width)
    bprof   = vocab_bit_profile(word, rule, width)
    al1, al0 = common_bits(word, rule, width)
    hwhist  = vocab_hamming_hist(word, rule, width)
    coverage = round(size / _Q6_TOTAL, 6)
    total_cs = period * width          # total cell-steps

    # Entropy of value histogram (higher = more uniform)
    if hist:
        total = sum(hist.values())
        h_val = -sum((c / total) * math.log2(c / total)
                     for c in hist.values() if c > 0)
    else:
        h_val = 0.0

    # Dominant value: appears most often
    dominant = max(hist, key=hist.get) if hist else None
    dom_frac = round(hist[dominant] / total_cs, 4) if dominant is not None else 0.0

    return {
        'word':            word.upper(),
        'rule':            rule,
        'period':          period,
        'total_cell_steps': total_cs,
        'vocab':           vocab,
        'vocab_size':      size,
        'vocab_coverage':  coverage,
        'hist':            hist,
        'uniform_dist':    unif,
        'hist_entropy':    round(h_val, 6),
        'bit_profile':     bprof,
        'always_1_bits':   sorted(al1),
        'always_0_bits':   sorted(al0),
        'hamming_hist':    hwhist,
        'dominant_value':  dominant,
        'dominant_frac':   dom_frac,
    }


def all_vocab(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """vocab_summary for all 4 rules."""
    return {rule: vocab_summary(word, rule, width) for rule in _RULES}


def build_vocab_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact vocabulary data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = vocab_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'vocab_size', 'vocab_coverage',
                                   'uniform_dist', 'hist_entropy',
                                   'always_1_bits', 'always_0_bits',
                                   'dominant_value', 'dominant_frac',
                                   'hamming_hist')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'q6_total': _Q6_TOTAL, 'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_DIM  = '\033[2m'


def _hw(v: int) -> int:
    return bin(v & 63).count('1')


def print_vocab(word: str = 'ГОРА', rule: str = 'xor3',
                color: bool = True) -> None:
    d   = vocab_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    dim = _DIM if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)
    bar_max = max(d['hist'].values()) if d['hist'] else 1

    print(f'  {c}◈ Vocab  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'size={d["vocab_size"]}  cov={d["vocab_coverage"]:.1%}{r}')
    print('  ' + '─' * 62)
    print(f'  Total cell-steps: {d["total_cell_steps"]}  '
          f'uniform={d["uniform_dist"]}  H={d["hist_entropy"]:.3f} bits')
    print(f'  Always-1 bits: {d["always_1_bits"]}  '
          f'Always-0 bits: {d["always_0_bits"]}')
    print(f'\n  Value histogram (top 10):')
    for v, cnt in list(d['hist'].items())[:10]:
        bar = '█' * int(cnt / bar_max * 16) + '░' * (16 - int(cnt / bar_max * 16))
        print(f'    {v:2d}=0b{v:06b}  hw={_hw(v)}  '
              f'|{bar}|  {cnt}×  ({cnt/d["total_cell_steps"]:.1%})')
    print(f'\n  Hamming-weight distribution of vocab: {d["hamming_hist"]}')
    print()


def print_vocab_table(words: list[str] | None = None,
                      color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    print(f'  {"Слово":10s}  '
          + '  '.join(
              (_RCOL.get(r, '') if color else '') + f'{r.upper():>6s}  sz  cov' + R
              for r in _RULES))
    print('  ' + '─' * 68)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = (_RCOL.get(rule, '') if color else '')
            sz  = vocab_size(word, rule)
            cov = vocab_coverage(word, rule)
            parts.append(f'{col}{sz:>8d}  {cov:.2f}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Orbit Vocabulary of Q6 CA')
    p.add_argument('--word',      default='ГОРА')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(vocab_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.table:
        print_vocab_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_vocab(args.word, rule, color)
    else:
        print_vocab(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
