"""solan_basin.py — Basin of Attraction Analysis for Q6 CA.

For each word + rule, random initial conditions are sampled at increasing
bit-Hamming distances from the word's IC to measure what fraction still
reaches the same attractor (same cycle of states, up to phase shift).

Bit-Hamming distance k = number of bits flipped across all 16 Q6 cells
(each cell has 6 bits → 96 bits total).

Profile interpretation:
    k=0  → fraction = 1.0 by definition (word itself)
    k=1  → 96 single-bit perturbations sampled → attractor robustness
    k=12 → mid-range noise → basin width
    k=48 → half of all bits flipped → approximately random IC

Rule behaviour (typical):
    XOR  → very large basin (all-zeros attractor is global for most topologies)
            → fraction ≈ 1.0 at all distances
    XOR3 → word-dependent; words with period-2 vs period-8 share different basins
    AND  → all-zeros attractor is strong absorber → large basin
    OR   → all-ones attractor is strong absorber → large basin

Функции:
    word_ic(word, width)                         → list[int]
    attractor_sig(word, rule, width)             → frozenset[tuple]
    attractors_match(sig1, sig2)                 → bool
    flip_k_bits(cells, k, rng, n_bits)           → list[int]
    random_ic(width, rng)                        → list[int]
    basin_at_k(word, rule, width, k, n, rng)     → float
    sample_global_basin(word, rule, width, n, rng) → dict
    basin_profile(word, rule, width, max_k, n_per_k, seed) → list[float]
    trajectory_basin(word, rule, width, max_k, n_per_k, seed) → dict
    all_basins(word, width, max_k, n_per_k, seed) → dict[str, dict]
    build_basin_data(words, width, max_k, n_per_k, seed) → dict
    basin_dict(word, width, max_k, n_per_k, seed) → dict
    print_basin(word, rule, width, max_k, n_per_k, seed, color) → None
    print_basin_stats(words, width, n_per_k, seed, color) → None

Запуск:
    python3 -m projects.hexglyph.solan_basin --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_basin --word ТУМАН --all-rules --no-color
    python3 -m projects.hexglyph.solan_basin --stats
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    step, find_orbit,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_N_BITS        = 6                   # bits per Q6 cell
_DEFAULT_WIDTH = 16
_TOTAL_BITS    = _DEFAULT_WIDTH * _N_BITS   # 96
_DEFAULT_MAX_K = 24                  # profile up to 24 bit-flips
_DEFAULT_N     = 100                 # samples per distance
_DEFAULT_SEED  = 42
_DEFAULT_WORDS = list(LEXICON)


# ── IC helpers ────────────────────────────────────────────────────────────────

def word_ic(word: str, width: int = _DEFAULT_WIDTH) -> list[int]:
    """Q6 initial condition for *word* (length = *width*)."""
    return pad_to(encode_word(word.upper()), width)


def flip_k_bits(
    cells:  list[int],
    k:      int,
    rng:    random.Random,
    n_bits: int = _N_BITS,
) -> list[int]:
    """Return a copy of *cells* with exactly *k* random bits flipped.

    Selects *k* distinct bit positions from the total width×n_bits bit space.
    """
    width    = len(cells)
    total    = width * n_bits
    if k == 0:
        return cells[:]
    k        = min(k, total)
    positions = rng.sample(range(total), k)
    result   = cells[:]
    for pos in positions:
        ci       = pos // n_bits
        bi       = pos %  n_bits
        result[ci] = (result[ci] ^ (1 << bi)) & ((1 << n_bits) - 1)
    return result


def random_ic(
    width:  int,
    rng:    random.Random,
    n_bits: int = _N_BITS,
) -> list[int]:
    """Uniformly random Q6 initial condition."""
    top = (1 << n_bits)
    return [rng.randrange(top) for _ in range(width)]


# ── Attractor identity ────────────────────────────────────────────────────────

def attractor_sig(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> frozenset:
    """Canonical attractor signature: frozenset of state tuples.

    A frozenset of all states in the periodic orbit is invariant to phase
    (starting point within the cycle), so two trajectories reaching the
    same cycle will produce identical signatures.
    """
    cells = word_ic(word, width)
    transient, period = find_orbit(cells, rule)
    period = max(period, 1)
    c = cells[:]
    for _ in range(transient):
        c = step(c, rule)
    states: list[tuple] = []
    for _ in range(period):
        states.append(tuple(c))
        c = step(c, rule)
    return frozenset(states)


def _ic_attractor_sig(
    cells: list[int],
    rule:  str,
) -> frozenset:
    """Attractor signature for an arbitrary IC (list of ints)."""
    transient, period = find_orbit(cells[:], rule)
    period = max(period, 1)
    c = cells[:]
    for _ in range(transient):
        c = step(c, rule)
    states: list[tuple] = []
    for _ in range(period):
        states.append(tuple(c))
        c = step(c, rule)
    return frozenset(states)


def attractors_match(sig1: frozenset, sig2: frozenset) -> bool:
    """True iff both attractors have the identical set of cycle states."""
    return sig1 == sig2


# ── Basin sampling ─────────────────────────────────────────────────────────────

def basin_at_k(
    word:   str,
    rule:   str,
    width:  int,
    k:      int,
    n:      int,
    rng:    random.Random,
) -> float:
    """Fraction of *n* ICs at bit-Hamming distance *k* reaching word's attractor.

    For k=0, returns 1.0 (the word itself always reaches its own attractor).
    """
    if k == 0:
        return 1.0
    target = attractor_sig(word, rule, width)
    base   = word_ic(word, width)
    hits   = 0
    for _ in range(n):
        ic  = flip_k_bits(base, k, rng)
        sig = _ic_attractor_sig(ic, rule)
        if attractors_match(sig, target):
            hits += 1
    return round(hits / n, 4)


def sample_global_basin(
    word:   str,
    rule:   str,
    width:  int          = _DEFAULT_WIDTH,
    n:      int          = _DEFAULT_N,
    rng:    random.Random | None = None,
) -> dict:
    """Sample *n* fully-random ICs and count how many reach word's attractor.

    Returns dict: {fraction, n_same, n_samples}.
    """
    if rng is None:
        rng = random.Random(_DEFAULT_SEED)
    target = attractor_sig(word, rule, width)
    hits   = 0
    for _ in range(n):
        ic  = random_ic(width, rng)
        sig = _ic_attractor_sig(ic, rule)
        if attractors_match(sig, target):
            hits += 1
    return {
        'fraction':  round(hits / n, 4),
        'n_same':    hits,
        'n_samples': n,
    }


def basin_profile(
    word:    str,
    rule:    str,
    width:   int = _DEFAULT_WIDTH,
    max_k:   int = _DEFAULT_MAX_K,
    n_per_k: int = _DEFAULT_N,
    seed:    int = _DEFAULT_SEED,
) -> list[float]:
    """Basin fraction at each bit-Hamming distance k = 0 … *max_k*.

    Returns list of length *max_k* + 1; element 0 is always 1.0.
    """
    rng    = random.Random(seed)
    result = [1.0]   # k=0 → always same attractor
    target = attractor_sig(word, rule, width)
    base   = word_ic(word, width)
    for k in range(1, max_k + 1):
        hits = 0
        for _ in range(n_per_k):
            ic  = flip_k_bits(base, k, rng)
            sig = _ic_attractor_sig(ic, rule)
            if attractors_match(sig, target):
                hits += 1
        result.append(round(hits / n_per_k, 4))
    return result


# ── Per-word trajectory dict ───────────────────────────────────────────────────

def trajectory_basin(
    word:    str,
    rule:    str  = 'xor3',
    width:   int  = _DEFAULT_WIDTH,
    max_k:   int  = _DEFAULT_MAX_K,
    n_per_k: int  = _DEFAULT_N,
    seed:    int  = _DEFAULT_SEED,
) -> dict:
    """Full basin analysis for one word + rule.

    Returns dict:
        word          : str
        rule          : str
        width         : int
        max_k         : int
        n_per_k       : int
        profile       : list[float]   — basin fraction at k=0..max_k
        mean_profile  : float         — mean fraction over k=1..max_k
        k50           : int           — smallest k where fraction drops below 0.5
                                        (None if always ≥ 0.5)
        global_basin  : dict          — from sample_global_basin()
    """
    rng = random.Random(seed)
    profile = basin_profile(word, rule, width, max_k, n_per_k, seed)
    mean_p  = round(sum(profile[1:]) / max_k, 4) if max_k > 0 else 1.0

    k50 = None
    for k, f in enumerate(profile):
        if k >= 1 and f < 0.5:
            k50 = k
            break

    gb_rng = random.Random(seed + 1)
    gb = sample_global_basin(word, rule, width, n_per_k, gb_rng)

    return {
        'word':         word.upper(),
        'rule':         rule,
        'width':        width,
        'max_k':        max_k,
        'n_per_k':      n_per_k,
        'profile':      profile,
        'mean_profile': mean_p,
        'k50':          k50,
        'global_basin': gb,
    }


def all_basins(
    word:    str,
    width:   int = _DEFAULT_WIDTH,
    max_k:   int = _DEFAULT_MAX_K,
    n_per_k: int = _DEFAULT_N,
    seed:    int = _DEFAULT_SEED,
) -> dict[str, dict]:
    """trajectory_basin for all 4 rules."""
    return {r: trajectory_basin(word, r, width, max_k, n_per_k, seed)
            for r in _ALL_RULES}


# ── Full dataset ───────────────────────────────────────────────────────────────

def build_basin_data(
    words:   list[str] | None = None,
    width:   int              = _DEFAULT_WIDTH,
    n_per_k: int              = _DEFAULT_N,
    seed:    int              = _DEFAULT_SEED,
) -> dict:
    """Basin summary for the full lexicon × 4 rules.

    Returns dict:
        words    : list[str]
        width    : int
        n_per_k  : int
        per_rule : {rule: {word: {mean_profile, k50, global_fraction}}}
        ranking  : {rule: [(word, mean_profile), …]}  descending
        widest   : {rule: (word, mean_profile)}
        narrowest: {rule: (word, mean_profile)}
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            tr = trajectory_basin(word, rule, width, _DEFAULT_MAX_K, n_per_k, seed)
            per_rule[rule][word] = {
                'mean_profile':    tr['mean_profile'],
                'k50':             tr['k50'],
                'global_fraction': tr['global_basin']['fraction'],
            }

    ranking:  dict[str, list]  = {}
    widest:   dict[str, tuple] = {}
    narrowest:dict[str, tuple] = {}
    for rule in _ALL_RULES:
        by_m = sorted(
            ((w, d['mean_profile']) for w, d in per_rule[rule].items()),
            key=lambda x: -x[1],
        )
        ranking[rule]   = by_m
        widest[rule]    = by_m[0]
        narrowest[rule] = by_m[-1]

    return {
        'words':     words,
        'width':     width,
        'n_per_k':   n_per_k,
        'per_rule':  per_rule,
        'ranking':   ranking,
        'widest':    widest,
        'narrowest': narrowest,
    }


# ── JSON export ────────────────────────────────────────────────────────────────

def basin_dict(
    word:    str,
    width:   int = _DEFAULT_WIDTH,
    max_k:   int = _DEFAULT_MAX_K,
    n_per_k: int = _DEFAULT_N,
    seed:    int = _DEFAULT_SEED,
) -> dict:
    """JSON-serialisable basin analysis for all 4 rules."""
    result: dict[str, object] = {
        'word':    word.upper(),
        'width':   width,
        'max_k':   max_k,
        'n_per_k': n_per_k,
        'rules':   {},
    }
    for rule in _ALL_RULES:
        tr = trajectory_basin(word, rule, width, max_k, n_per_k, seed)
        result['rules'][rule] = {          # type: ignore[index]
            'profile':      tr['profile'],
            'mean_profile': tr['mean_profile'],
            'k50':          tr['k50'],
            'global_fraction': tr['global_basin']['fraction'],
        }
    return result


# ── ASCII display ──────────────────────────────────────────────────────────────

_BAR_FULL  = '█'
_BAR_EMPTY = '·'
_BAR_WIDTH = 20


def print_basin(
    word:    str,
    rule:    str  = 'xor3',
    width:   int  = _DEFAULT_WIDTH,
    max_k:   int  = _DEFAULT_MAX_K,
    n_per_k: int  = _DEFAULT_N,
    seed:    int  = _DEFAULT_SEED,
    color:   bool = True,
) -> None:
    """Print basin profile as a horizontal bar chart."""
    tr   = trajectory_basin(word, rule, width, max_k, n_per_k, seed)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    print(f"{bold}  ◈ Бассейн аттрактора Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  "
          f"(n={n_per_k}, k50={'?' if tr['k50'] is None else tr['k50']})")
    print(f"  {'─' * 40}")
    for k, f in enumerate(tr['profile']):
        bar_len = int(f * _BAR_WIDTH + 0.5)
        bar     = _BAR_FULL * bar_len + _BAR_EMPTY * (_BAR_WIDTH - bar_len)
        marker  = col if f >= 0.5 else dim
        print(f"  k={k:2d}  {marker}{bar}{rst}  {f:.3f}")
    print(f"  {'─' * 40}")
    gb = tr['global_basin']
    print(f"  глобальный бассейн: {gb['fraction']:.3f}  "
          f"({gb['n_same']}/{gb['n_samples']} случайных ИУ)")
    print(f"  ср.профиль: {tr['mean_profile']:.3f}")
    print()


def print_basin_stats(
    words:   list[str] | None = None,
    width:   int              = _DEFAULT_WIDTH,
    n_per_k: int              = _DEFAULT_N,
    seed:    int              = _DEFAULT_SEED,
    color:   bool             = True,
) -> None:
    """Сводная таблица mean_profile для лексикона × 4 правила."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Ширина бассейна Q6 (ср.доля при k=1..{_DEFAULT_MAX_K}){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            tr  = trajectory_basin(word, rule, width, _DEFAULT_MAX_K, n_per_k, seed)
            mp  = tr['mean_profile']
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{mp:>8.3f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Basin of Attraction Analysis Q6 CA')
    parser.add_argument('--word',      default='ГОРА',  help='Русское слово')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--max-k',     type=int, default=_DEFAULT_MAX_K)
    parser.add_argument('--n',         type=int, default=_DEFAULT_N)
    parser.add_argument('--seed',      type=int, default=_DEFAULT_SEED)
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_basin_stats(color=color, n_per_k=args.n, seed=args.seed)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_basin(args.word, rule, args.width, args.max_k, args.n, args.seed, color)
    else:
        print_basin(args.word, args.rule, args.width, args.max_k, args.n, args.seed, color)


if __name__ == '__main__':
    _main()
