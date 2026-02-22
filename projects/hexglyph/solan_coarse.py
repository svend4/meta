"""solan_coarse.py — Coarse-Graining & Renormalization Group Analysis.

Maps each Q6 cell value from the 64-level alphabet to a coarser k-level
alphabet and studies two quantities:

1. Commutator error δ(t)
   ─────────────────────
   Define two parallel trajectories:
     Path A : start from original Q6 IC → run t Q6 CA steps → coarsen to k
     Path B : coarsen IC to k → run t *coarse* CA steps
              (each coarse step: dequantize → Q6 CA step → re-coarsen)

   δ(t) = mean_i |PathA[i] - PathB[i]| / (k − 1)   ∈ [0, 1]

   δ(0) = 0 always (both paths start from coarsen(IC)).
   δ(t) = 0 for all t → coarse-graining commutes exactly with the CA rule
            (the rule is RG-consistent at this level).
   δ(t) > 0 → the two trajectories diverge; the rule is RG-inconsistent.

   Mathematical results
   ─────────────────────
   k = 2 (binary, threshold 32 = MSB):
     For XOR:  coarsen(a^b) = bit₅(a^b) = bit₅(a)^bit₅(b) = exact.
     Similarly for XOR3, AND, OR: all four rules are EXACTLY RG-consistent.
     → δ(t) = 0 ∀t for all four rules.

   k = 64 (identity coarsening, dequant is the identity):
     Trivially exact: δ(t) = 0 ∀t.

   k ∈ {3,4,8,16}: generally δ(t) > 0 (inconsistent); the magnitude
     depends on the word's IC and the rule.

2. Coarse-orbit profile
   ─────────────────────
   Run the CA from the coarsened IC at each k → find (transient, period).
   Compare with the Q6 orbit: does coarse-graining preserve the period?

Encoding / decoding
──────────────────
  coarsen(v, k)   : v ∈ [0,63] → bin ∈ {0,...,k−1}
                    = min(v*k // 64, k−1)
  dequantize(i, k): bin → representative Q6 value
                    = round(i * 63 / (k−1))   (k > 1)

Default level set: k ∈ {2, 3, 4, 8, 16, 64}

Functions
─────────
  coarsen(v, k)                            → int
  dequantize(i, k)                         → int
  coarse_step(cells, rule, k)              → list[int]  (k-level step)
  commutator_traj(word, rule, k, n_steps)  → list[float]
  coarse_orbit(word, rule, k)              → dict
  coarse_dict(word, rule, levels)          → dict
  all_coarse(word, levels)                 → dict[rule, dict]
  build_coarse_data(words, levels)         → dict
  print_coarse(word, rule, color)          → None

Запуск
──────
  python3 -m projects.hexglyph.solan_coarse --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_coarse --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_coarse --stats --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    step, find_orbit,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES       = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH   = 16
_DEFAULT_N_STEPS = 20
_DEFAULT_LEVELS  = [2, 3, 4, 8, 16, 64]


# ── Coarsening codec ──────────────────────────────────────────────────────────

def coarsen(v: int, k: int) -> int:
    """Map Q6 value v ∈ [0, 63] to bin index ∈ {0, …, k−1}.

    Uses uniform partitioning: bin = min(v * k // 64, k − 1).
    k = 2  : {0..31} → 0,  {32..63} → 1  (threshold = MSB)
    k = 64 : identity (v → v)
    """
    return min(v * k // 64, k - 1)


def dequantize(i: int, k: int) -> int:
    """Map bin index i ∈ {0, …, k−1} back to a representative Q6 value.

    Representative = round(i * 63 / (k − 1)).
    k = 2  : 0 → 0, 1 → 63
    k = 3  : 0 → 0, 1 → 32, 2 → 63
    k = 64 : i → i   (identity)
    """
    if k <= 1:
        return 0
    return round(i * 63 / (k - 1))


# ── Coarse CA step ────────────────────────────────────────────────────────────

def coarse_step(
    cells: list[int],
    rule:  str,
    k:     int,
) -> list[int]:
    """One coarse-level CA step: dequantize → Q6 rule → re-coarsen.

    Input cells are bin indices ∈ {0, …, k−1}.
    Output cells are bin indices ∈ {0, …, k−1}.
    """
    dq      = [dequantize(v, k) for v in cells]
    stepped = step(dq, rule)
    return [coarsen(v, k) for v in stepped]


# ── Commutator trajectory ─────────────────────────────────────────────────────

def commutator_traj(
    word:    str,
    rule:    str,
    k:       int,
    n_steps: int = _DEFAULT_N_STEPS,
    width:   int = _DEFAULT_WIDTH,
) -> list[float]:
    """δ(t) for t = 0 … n_steps.

    Path A: Q6 IC → Q6 CA^t → coarsen
    Path B: coarsen(IC) → coarse_CA^t

    δ(t) = mean_i |PathA[i] − PathB[i]| / (k − 1)  ∈ [0, 1]
    δ(0) = 0 always.
    δ(t) = 0 ∀t iff the rule is exactly RG-consistent at level k.
    """
    ic        = pad_to(encode_word(word.upper()), width)
    path_a    = ic[:]
    path_b    = [coarsen(v, k) for v in ic]
    denom     = max(k - 1, 1) * width

    deltas = [0.0]
    for _ in range(n_steps):
        path_a = step(path_a, rule)
        path_b = coarse_step(path_b, rule, k)
        ca_coarsened = [coarsen(v, k) for v in path_a]
        err   = sum(abs(ca_coarsened[i] - path_b[i]) for i in range(width))
        deltas.append(round(err / denom, 8))
    return deltas


# ── Coarse orbit ──────────────────────────────────────────────────────────────

def coarse_orbit(
    word:  str,
    rule:  str,
    k:     int,
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """Find the orbit of the CA starting from coarsen(IC).

    Returns:
        transient     : int
        period        : int  (≥ 1)
        entropy       : float  Shannon entropy of attractor values (natural bits)
        entropy_norm  : float  entropy / log₂(k)  ∈ [0, 1]
        n_unique      : int  distinct attractor rows
    """
    ic       = pad_to(encode_word(word.upper()), width)
    c        = [coarsen(v, k) for v in ic]

    # find orbit using hash dict
    seen: dict[tuple, int] = {}
    t = 0
    while True:
        key = tuple(c)
        if key in seen:
            transient = seen[key]
            period    = max(t - transient, 1)
            break
        seen[key] = t
        c = coarse_step(c, rule, k)
        t += 1

    # advance to attractor
    c = [coarsen(v, k) for v in ic]
    for _ in range(transient):
        c = coarse_step(c, rule, k)

    # collect one period
    grid: list[list[int]] = []
    for _ in range(period):
        grid.append(c[:])
        c = coarse_step(c, rule, k)

    # attractor value entropy
    vals   = [v for row in grid for v in row]
    total  = len(vals)
    counts: dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    h = max(0.0, -sum(
        (cnt / total) * math.log2(cnt / total)
        for cnt in counts.values() if cnt > 0
    ))
    h_max = math.log2(k) if k > 1 else 1.0
    h_n   = round(h / h_max, 8) if h_max > 0 else 0.0

    return {
        'transient':    transient,
        'period':       period,
        'entropy':      round(h, 6),
        'entropy_norm': h_n,
        'n_unique':     len(set(tuple(row) for row in grid)),
    }


# ── Full coarse-graining analysis ─────────────────────────────────────────────

def coarse_dict(
    word:    str,
    rule:    str,
    levels:  list[int] = _DEFAULT_LEVELS,
    n_steps: int        = _DEFAULT_N_STEPS,
    width:   int        = _DEFAULT_WIDTH,
) -> dict:
    """Full coarse-graining analysis for one word × rule.

    Returns dict:
        word, rule, levels, n_steps
        q6_period       : int   original Q6 orbit period
        q6_transient    : int   original Q6 orbit transient
        by_level: {k: {
            commutator       : list[float]  δ(t) for t=0..n_steps
            max_commutator   : float        max δ(t)
            mean_commutator  : float        mean δ(t) over t≥1
            is_exact         : bool         max_commutator == 0
            transient        : int          coarse orbit transient
            period           : int          coarse orbit period
            entropy          : float        attractor value entropy
            entropy_norm     : float        entropy / log₂(k)
            n_unique         : int          distinct attractor rows
        }}
        exact_levels    : list[int]  levels with is_exact=True (always ⊇ {2, 64})
        max_inconsistency: dict[int, float]  {k: max δ}
    """
    word  = word.upper()
    cells = pad_to(encode_word(word), width)
    q6_t, q6_p = find_orbit(cells[:], rule)

    by_level: dict[int, dict] = {}
    for k in levels:
        comm  = commutator_traj(word, rule, k, n_steps, width)
        orbit = coarse_orbit(word, rule, k, width)
        mc    = max(comm)
        by_level[k] = {
            'commutator':       comm,
            'max_commutator':   round(mc, 8),
            'mean_commutator':  round(sum(comm[1:]) / max(len(comm) - 1, 1), 8),
            'is_exact':         mc == 0.0,
            **orbit,
        }

    exact   = [k for k in levels if by_level[k]['is_exact']]
    max_inc = {k: by_level[k]['max_commutator'] for k in levels}

    return {
        'word':             word,
        'rule':             rule,
        'levels':           levels,
        'n_steps':          n_steps,
        'q6_period':        max(q6_p, 1),
        'q6_transient':     q6_t,
        'by_level':         by_level,
        'exact_levels':     exact,
        'max_inconsistency': max_inc,
    }


def all_coarse(
    word:   str,
    levels: list[int] = _DEFAULT_LEVELS,
) -> dict[str, dict]:
    """coarse_dict for all 4 rules."""
    return {r: coarse_dict(word, r, levels) for r in _ALL_RULES}


def build_coarse_data(
    words:  list[str] | None = None,
    levels: list[int]        = _DEFAULT_LEVELS,
) -> dict:
    """Coarse-graining summary across the lexicon × 4 rules.

    Returns: words, levels,
             per_rule: {rule: {word: {q6_period, exact_levels, max_inc_3}}}
    """
    words = words if words is not None else list(LEXICON)
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = coarse_dict(word, rule, levels)
            per_rule[rule][word] = {
                'q6_period':     d['q6_period'],
                'exact_levels':  d['exact_levels'],
                'max_inc_3':     d['max_inconsistency'].get(3, 0.0),
                'max_inc_4':     d['max_inconsistency'].get(4, 0.0),
            }
    return {'words': words, 'levels': levels, 'per_rule': per_rule}


# ── ASCII / ANSI display ───────────────────────────────────────────────────────

_BAR_W = 16


def print_coarse(
    word:  str  = 'ТУМАН',
    rule:  str  = 'xor3',
    color: bool = True,
) -> None:
    d    = coarse_dict(word, rule)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    print(f"{bold}  ◈ Огрубление Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  Q6-P={d['q6_period']}  Q6-T={d['q6_transient']}")
    print(f"  {'─' * 62}")
    print(f"  {'k':>4}  {'exact':>5}  {'max-δ':>7}  "
          f"{'cP':>4}  {'cT':>3}  {'H_n':>6}  commutator profile")
    print(f"  {'─' * 62}")

    for k in d['levels']:
        bl  = d['by_level'][k]
        ex  = '✓' if bl['is_exact'] else '✗'
        mc  = bl['max_commutator']
        bar_len = round(mc * _BAR_W)
        bar = (col if color else '') + '█' * bar_len + (rst if color else '') + \
              '░' * (_BAR_W - bar_len)
        print(f"  {k:>4}  {ex:>5}  {mc:>7.4f}  "
              f"{bl['period']:>4}  {bl['transient']:>3}  "
              f"{bl['entropy_norm']:>6.3f}  {bar}")
    print()


def print_coarse_stats(
    words: list[str] | None = None,
    color: bool             = True,
) -> None:
    """Table: max commutator at k=3 per word × rule."""
    words = words if words is not None else list(LEXICON)
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Максимальный коммутатор при k=3{rst}")
    print('  ' + '─' * (len(header) + 2))
    print('  ' + header)
    print('  ' + '─' * (len(header) + 2))
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = coarse_dict(word, rule, levels=[3])
            mc  = d['max_inconsistency'].get(3, 0.0)
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{mc:>8.4f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Coarse-Graining RG Analysis Q6 CA')
    parser.add_argument('--word',      default='ТУМАН')
    parser.add_argument('--rule',      default='xor3', choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--no-color',  action='store_true')
    parser.add_argument('--json',      action='store_true', help='JSON output')
    args  = parser.parse_args()
    color = not args.no_color
    if args.json:
        import json as _json
        print(_json.dumps(coarse_dict(args.word, args.rule), ensure_ascii=False, indent=2))
        return

    if args.stats:
        print_coarse_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_coarse(args.word, rule, color)
    else:
        print_coarse(args.word, args.rule, color)


if __name__ == '__main__':
    _main()
